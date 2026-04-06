# -*- coding: utf-8 -*-
"""Sleep-EDF preprocessing with manifest-based cache reuse."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from eco_sleep.data.sleep_edf.edf_reader import read_hypnogram_annotations, read_psg_signals
from eco_sleep.data.sleep_edf.labels import SLEEP_EDF_LABEL_MAPPING_TEXT
from eco_sleep.data.sleep_edf.preprocessing import (
    build_epoch_label_trace,
    compute_crop_bounds,
    default_lcs_delta,
    filter_eeg,
    lcs_counts_to_binary,
    lcs_encode_epoch_counts,
    normalize_edf_subset,
    parse_record_info,
    record_in_subset,
    robust_epoch_standardize,
)
from eco_sleep.data.sleep_edf.storage import (
    PROCESSED_SCHEMA_VERSION,
    json_dumps,
    list_processed_records,
    pair_key_from_record_id,
    resolve_existing_path,
)
from eco_sleep.labels import get_num_classes, get_task_name
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio
from eco_sleep.utils.io import ensure_dir, read_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_CACHE_DIRNAME = "_preprocess_last_sleep_edf"
PREPROCESS_MANIFEST_NAME = "preprocess_manifest.json"
PREPROCESS_MANIFEST_VERSION = 1
SKIP_REASON_COLUMNS = ["record_id", "reason"]
PREPROCESS_HASH_FILES = (
    "scripts/preprocess_sleep_edf.py",
    "scripts/_pathfix.py",
    "src/eco_sleep/data/sleep_edf/annotations.py",
    "src/eco_sleep/data/sleep_edf/edf_reader.py",
    "src/eco_sleep/data/sleep_edf/labels.py",
    "src/eco_sleep/data/sleep_edf/preprocessing.py",
    "src/eco_sleep/data/sleep_edf/storage.py",
)


def _record_from_path(path: Path) -> Tuple[str, str, str]:
    return parse_record_info(path.stem.split("-")[0])


def _pair_key_from_path(path: Path) -> str:
    stem = path.stem.split("-")[0]
    return pair_key_from_record_id(stem)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(data).encode("utf-8")).hexdigest()


def _preprocess_manifest_path(out_dir: Path) -> Path:
    return out_dir / PREPROCESS_MANIFEST_NAME


def _load_preprocess_manifest(out_dir: Path) -> dict | None:
    path = _preprocess_manifest_path(out_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _hash_source_files() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for rel_path in PREPROCESS_HASH_FILES:
        path = PROJECT_ROOT / rel_path
        hashes[rel_path] = _sha256_file(path)
    return hashes


def build_preprocess_fingerprint(cfg: dict, raw_dir_cfg: str) -> dict:
    task = get_task_name(cfg, "sleep_edf_5class")
    dataset_cfg = cfg.get("dataset", {})
    edf_subset = normalize_edf_subset(dataset_cfg.get("edf_subset", cfg.get("edf_subset", "edfx_all")))
    lcs_cfg = cfg.get("lcs", {})
    filter_cfg = cfg.get("filters", {})
    band_cfg = filter_cfg.get("bandpass", {})
    notch_cfg = filter_cfg.get("notch", {})
    channels = [str(channel) for channel in cfg.get("channels", ["Fpz-Cz"])]
    target_hz = int(cfg.get("resample_hz", 100))
    crop_policy = str(cfg.get("crop_policy", "30min"))
    epoch_seconds = 30
    lcs_delta = float(lcs_cfg.get("delta", default_lcs_delta(edf_subset)))

    fingerprint = {
        "fingerprint_version": PREPROCESS_MANIFEST_VERSION,
        "task": task,
        "num_classes": int(get_num_classes(task)),
        "schema_version": int(PROCESSED_SCHEMA_VERSION),
        "code_hashes": _hash_source_files(),
        "config": {
            "raw_dir_config": str(raw_dir_cfg),
            "edf_subset": edf_subset,
            "channels": channels,
            "resample_hz": target_hz,
            "epoch_seconds": epoch_seconds,
            "crop_policy": crop_policy,
            "bandpass": {
                "enable": bool(band_cfg.get("enable", True)),
                "low": float(band_cfg.get("low", 0.3)),
                "high": float(band_cfg.get("high", 35.0)),
                "order": int(band_cfg.get("order", 4)),
            },
            "notch": {
                "enable": bool(notch_cfg.get("enable", False)),
                "freq": float(notch_cfg.get("freq", 50.0)),
                "quality": float(notch_cfg.get("quality", 30.0)),
            },
            "lcs": {
                "enabled": True,
                "delta": lcs_delta,
            },
            "label_mapping_text": SLEEP_EDF_LABEL_MAPPING_TEXT,
        },
    }
    return fingerprint


def _expected_output_names(saved_record_ids: List[str]) -> List[str]:
    return sorted(f"{record_id}.npz" for record_id in saved_record_ids)


def _validate_manifest_outputs(out_dir: Path, manifest: dict, num_classes: int) -> dict:
    saved_record_ids = [str(record_id) for record_id in manifest.get("saved_record_ids", [])]
    if not saved_record_ids:
        return {
            "ok": False,
            "reason": "manifest_missing_saved_record_ids",
            "saved_record_ids": [],
            "missing_files": [],
            "extra_files": [],
        }

    actual_names = sorted(path.name for path in out_dir.glob("*.npz"))
    expected_names = _expected_output_names(saved_record_ids)
    missing_files = [name for name in expected_names if name not in actual_names]
    extra_files = [name for name in actual_names if name not in expected_names]
    if missing_files or extra_files:
        return {
            "ok": False,
            "reason": "processed_file_set_mismatch",
            "saved_record_ids": saved_record_ids,
            "missing_files": missing_files,
            "extra_files": extra_files,
        }

    for name in expected_names:
        path = out_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            return {
                "ok": False,
                "reason": "processed_file_missing_or_empty",
                "saved_record_ids": saved_record_ids,
                "missing_files": [name],
                "extra_files": [],
            }

    records = list_processed_records(out_dir, num_classes=num_classes)
    if len(records) != len(saved_record_ids):
        return {
            "ok": False,
            "reason": "processed_record_count_mismatch",
            "saved_record_ids": saved_record_ids,
            "missing_files": [],
            "extra_files": [],
        }

    record_map = {row.record_id: row for row in records}
    missing_records = [record_id for record_id in saved_record_ids if record_id not in record_map]
    if missing_records:
        return {
            "ok": False,
            "reason": "processed_record_id_mismatch",
            "saved_record_ids": saved_record_ids,
            "missing_files": [f"{record_id}.npz" for record_id in missing_records],
            "extra_files": [],
        }

    manifest_cfg = manifest.get("fingerprint", {}).get("config", {})
    expected_subset = str(manifest_cfg.get("edf_subset", ""))
    expected_channels = [str(v) for v in manifest_cfg.get("channels", [])]
    expected_hz = int(manifest_cfg.get("resample_hz", -1))
    expected_epoch_seconds = int(manifest_cfg.get("epoch_seconds", -1))
    expected_crop_policy = str(manifest_cfg.get("crop_policy", ""))
    expected_lcs_delta = float(manifest_cfg.get("lcs", {}).get("delta", np.nan))
    expected_label_mapping = str(manifest_cfg.get("label_mapping_text", ""))

    for record_id in saved_record_ids:
        row = record_map[record_id]
        meta = dict(row.meta or {})
        if row.is_legacy:
            return {
                "ok": False,
                "reason": "legacy_processed_cache_detected",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_subset and str(meta.get("subset", "")) != expected_subset:
            return {
                "ok": False,
                "reason": "processed_subset_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_channels and [str(v) for v in meta.get("channels", [])] != expected_channels:
            return {
                "ok": False,
                "reason": "processed_channels_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_hz > 0 and int(meta.get("resample_hz", -1)) != expected_hz:
            return {
                "ok": False,
                "reason": "processed_resample_hz_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_epoch_seconds > 0 and int(meta.get("epoch_seconds", -1)) != expected_epoch_seconds:
            return {
                "ok": False,
                "reason": "processed_epoch_seconds_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_crop_policy and str(meta.get("crop_policy", "")) != expected_crop_policy:
            return {
                "ok": False,
                "reason": "processed_crop_policy_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if np.isfinite(expected_lcs_delta) and abs(float(meta.get("lcs_delta", np.nan)) - expected_lcs_delta) > 1e-8:
            return {
                "ok": False,
                "reason": "processed_lcs_delta_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }
        if expected_label_mapping and str(meta.get("label_mapping", "")) != expected_label_mapping:
            return {
                "ok": False,
                "reason": "processed_label_mapping_mismatch",
                "saved_record_ids": saved_record_ids,
                "missing_files": [],
                "extra_files": [],
            }

    return {
        "ok": True,
        "reason": "cache_valid",
        "saved_record_ids": saved_record_ids,
        "missing_files": [],
        "extra_files": [],
    }


def get_preprocess_cache_status(cfg: dict, out_dir: Path, num_classes: int) -> dict:
    raw_dir_cfg = str(cfg.get("raw_dir", ""))
    fingerprint = build_preprocess_fingerprint(cfg, raw_dir_cfg=raw_dir_cfg)
    fingerprint_hash = _sha256_json(fingerprint)
    manifest = _load_preprocess_manifest(out_dir)
    if manifest is None:
        return {
            "reuse_available": False,
            "needs_rebuild": True,
            "reason": "manifest_missing",
            "manifest_path": str(_preprocess_manifest_path(out_dir)),
            "manifest": None,
            "fingerprint": fingerprint,
            "fingerprint_hash": fingerprint_hash,
        }

    manifest_hash = str(manifest.get("fingerprint_hash", ""))
    if manifest_hash != fingerprint_hash:
        return {
            "reuse_available": False,
            "needs_rebuild": True,
            "reason": "fingerprint_changed",
            "manifest_path": str(_preprocess_manifest_path(out_dir)),
            "manifest": manifest,
            "fingerprint": fingerprint,
            "fingerprint_hash": fingerprint_hash,
            "manifest_fingerprint_hash": manifest_hash,
        }

    validation = _validate_manifest_outputs(out_dir, manifest, num_classes=num_classes)
    return {
        "reuse_available": bool(validation["ok"]),
        "needs_rebuild": not bool(validation["ok"]),
        "reason": str(validation["reason"]),
        "manifest_path": str(_preprocess_manifest_path(out_dir)),
        "manifest": manifest,
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash,
        "validation": validation,
    }


def scan_raw_pairs(raw_dir: Path) -> Dict[str, object]:
    psg_files = sorted(raw_dir.glob("*PSG.edf"))
    hyp_files = sorted(raw_dir.glob("*Hypnogram.edf"))
    hyp_by_key: Dict[str, List[Path]] = {}
    for hyp_path in hyp_files:
        key = _pair_key_from_path(hyp_path)
        hyp_by_key.setdefault(key, []).append(hyp_path)

    matched: List[Tuple[Path, Path]] = []
    skip_rows: List[Dict[str, object]] = []
    for psg_path in psg_files:
        _, record_id, _ = _record_from_path(psg_path)
        key = _pair_key_from_path(psg_path)
        candidates = hyp_by_key.get(key, [])
        if not candidates:
            skip_rows.append({"record_id": record_id, "reason": "no_matching_hypnogram", "psg_path": str(psg_path)})
            continue
        if len(candidates) > 1:
            skip_rows.append({"record_id": record_id, "reason": "multiple_matching_hypnogram", "psg_path": str(psg_path)})
            continue
        matched.append((psg_path, candidates[0]))

    return {
        "psg_files": psg_files,
        "hyp_files": hyp_files,
        "matched_pairs": matched,
        "skip_rows": skip_rows,
    }


def _encode_record_lcs(raw_epoch: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_epoch, n_t = raw_epoch.shape
    lcs_pos_count = np.zeros((n_epoch, n_t), dtype=np.int16)
    lcs_neg_count = np.zeros((n_epoch, n_t), dtype=np.int16)
    lcs_pos = np.zeros((n_epoch, n_t), dtype=np.uint8)
    lcs_neg = np.zeros((n_epoch, n_t), dtype=np.uint8)
    for idx in range(n_epoch):
        pos_count, neg_count = lcs_encode_epoch_counts(raw_epoch[idx], delta=delta)
        pos_bin, neg_bin = lcs_counts_to_binary(pos_count, neg_count)
        lcs_pos_count[idx] = pos_count
        lcs_neg_count[idx] = neg_count
        lcs_pos[idx] = pos_bin
        lcs_neg[idx] = neg_bin
    return lcs_pos_count, lcs_neg_count, lcs_pos, lcs_neg


def _validate_labels(labels: np.ndarray, record_path: Path, num_classes: int) -> None:
    bad = np.where((labels < 0) | (labels >= num_classes))[0]
    if bad.size == 0:
        return
    idx = int(bad[0])
    raise RuntimeError(
        f"标签越界: file={record_path} epoch_index={idx} "
        f"label={int(labels[idx])} expected=[0,{num_classes - 1}]"
    )


def _preprocess_cache_dir(runs_dir: Path) -> Path:
    return ensure_dir(runs_dir / PREPROCESS_CACHE_DIRNAME)


def _skip_rows_frame(skip_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(skip_rows)
    for col in SKIP_REASON_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    ordered = SKIP_REASON_COLUMNS + [col for col in df.columns if col not in SKIP_REASON_COLUMNS]
    return df.loc[:, ordered]


def _write_preprocess_artifacts(
    runs_dir: Path,
    summary: Dict[str, object],
    skip_rows: List[Dict[str, object]],
    audit_dir: Path | None,
) -> None:
    cache_dir = _preprocess_cache_dir(runs_dir)
    summary_path = cache_dir / "preprocess_summary.json"
    skip_path = cache_dir / "preprocess_skip_reasons.csv"
    skip_df = _skip_rows_frame(skip_rows)
    summary_path.write_text(json_dumps(summary), encoding="utf-8")
    skip_df.to_csv(skip_path, index=False, **csv_utf8_sig_kwargs())
    if audit_dir is not None:
        ensure_dir(audit_dir)
        (audit_dir / "preprocess_summary.json").write_text(json_dumps(summary), encoding="utf-8")
        skip_df.to_csv(audit_dir / "preprocess_skip_reasons.csv", index=False, **csv_utf8_sig_kwargs())


def _remove_existing_processed_files(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob("*.npz"):
        path.unlink(missing_ok=True)
        removed += 1
    manifest_path = _preprocess_manifest_path(out_dir)
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=True)
    return removed


def _save_preprocess_manifest(
    out_dir: Path,
    fingerprint: dict,
    raw_dir_cfg: str,
    raw_dir_resolved: str,
    summary: dict,
    skip_rows: List[Dict[str, object]],
    saved_record_ids: List[str],
) -> dict:
    manifest = {
        "manifest_version": PREPROCESS_MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": int(PROCESSED_SCHEMA_VERSION),
        "raw_dir_config": str(raw_dir_cfg),
        "raw_dir_resolved": str(raw_dir_resolved),
        "processed_dir": str(out_dir),
        "fingerprint": fingerprint,
        "fingerprint_hash": _sha256_json(fingerprint),
        "saved_record_ids": sorted(saved_record_ids),
        "saved_record_count": int(len(saved_record_ids)),
        "summary": summary,
        "skip_rows": skip_rows,
    }
    _preprocess_manifest_path(out_dir).write_text(json_dumps(manifest), encoding="utf-8")
    return manifest


def _reuse_summary_from_manifest(out_dir: Path, manifest: dict, cache_status: dict) -> tuple[dict, List[Dict[str, object]]]:
    saved_record_ids = [str(record_id) for record_id in manifest.get("saved_record_ids", [])]
    skip_rows = list(manifest.get("skip_rows", []))
    base_summary = dict(manifest.get("summary", {}))
    base_summary.update(
        {
            "processed_dir": str(out_dir),
            "out_dir": str(out_dir),
            "reuse_existing": True,
            "force_rebuild": False,
            "cache_decision": "reused",
            "cache_reason": str(cache_status.get("reason", "cache_valid")),
            "newly_saved": 0,
            "newly_saved_records": 0,
            "reused_existing": int(len(saved_record_ids)),
            "reused_existing_records": int(len(saved_record_ids)),
            "valid_saved_records": int(len(saved_record_ids)),
            "manifest_path": str(_preprocess_manifest_path(out_dir)),
            "manifest_fingerprint_hash": str(manifest.get("fingerprint_hash", "")),
        }
    )
    base_summary["skip_reason_counts"] = (
        pd.DataFrame(skip_rows)["reason"].value_counts().sort_index().to_dict() if skip_rows else {}
    )
    return base_summary, skip_rows


def main() -> None:
    setup_utf8_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--audit_dir", type=str, default=None)
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    if num_classes != 5:
        raise RuntimeError("Sleep-EDF 预处理要求 5 类标签。")

    raw_dir_cfg = str(args.raw_dir or cfg["raw_dir"])
    out_dir = Path(args.out_dir or cfg["processed_dir"])
    runs_dir = Path(cfg["runs_dir"])
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    ensure_dir(out_dir)
    ensure_dir(runs_dir)

    override_cfg = dict(cfg)
    override_cfg["raw_dir"] = raw_dir_cfg
    cache_status = get_preprocess_cache_status(override_cfg, out_dir=out_dir, num_classes=num_classes)
    if not args.force_rebuild and cache_status["reuse_available"]:
        manifest = cache_status["manifest"] or {}
        summary, skip_rows = _reuse_summary_from_manifest(out_dir, manifest=manifest, cache_status=cache_status)
        _write_preprocess_artifacts(runs_dir=runs_dir, summary=summary, skip_rows=skip_rows, audit_dir=audit_dir)
        print(
            "preprocess: 配置与代码未变化，直接复用已有 processed 数据 "
            f"reused_existing={summary['reused_existing']} newly_saved=0 out_dir={out_dir}"
        )
        return

    raw_dir = resolve_existing_path(raw_dir_cfg)
    if not raw_dir.exists():
        reason = "force_rebuild" if args.force_rebuild else str(cache_status.get("reason", "manifest_missing"))
        raise RuntimeError(
            "preprocess: 需要重新构建 processed 数据，但 raw_dir 不存在。"
            f" reason={reason} config={raw_dir_cfg} resolved={raw_dir}"
        )

    existing_before = list_processed_records(out_dir, num_classes=num_classes)
    if args.force_rebuild:
        rebuild_reason = "force_rebuild"
    else:
        rebuild_reason = str(cache_status.get("reason", "manifest_missing"))
    removed_files = _remove_existing_processed_files(out_dir)

    dataset_cfg = cfg.get("dataset", {})
    edf_subset = normalize_edf_subset(dataset_cfg.get("edf_subset", cfg.get("edf_subset", "edfx_all")))
    lcs_cfg = cfg.get("lcs", {})
    lcs_delta = float(lcs_cfg.get("delta", default_lcs_delta(edf_subset)))
    filter_cfg = cfg.get("filters", {})
    band_cfg = filter_cfg.get("bandpass", {})
    notch_cfg = filter_cfg.get("notch", {})
    channels = list(cfg.get("channels", ["Fpz-Cz"]))
    target_hz = int(cfg.get("resample_hz", 100))
    crop_policy = str(cfg.get("crop_policy", "30min"))
    epoch_seconds = 30

    scan = scan_raw_pairs(raw_dir)
    skip_rows: List[Dict[str, object]] = list(scan["skip_rows"])
    matched_pairs: List[Tuple[Path, Path]] = list(scan["matched_pairs"])
    newly_saved = 0
    reused_existing = 0
    saved_record_ids: List[str] = []

    print(
        "preprocess: 检测到缓存不可复用，开始重建 "
        f"reason={rebuild_reason} removed_existing={removed_files} out_dir={out_dir}"
    )

    for psg_path, hyp_path in tqdm(matched_pairs, desc="Sleep-EDF preprocess"):
        subject_id, record_id, cohort = _record_from_path(psg_path)
        if not record_in_subset(record_id=record_id, edf_subset=edf_subset):
            skip_rows.append({"record_id": record_id, "reason": "subset_filter_mismatch", "psg_path": str(psg_path)})
            continue

        out_path = out_dir / f"{record_id}.npz"
        signals, _ = read_psg_signals(psg_path, channels=channels, target_hz=target_hz)
        if bool(band_cfg.get("enable", True)) or bool(notch_cfg.get("enable", False)):
            signals = filter_eeg(signals, sfreq=target_hz, band_cfg=band_cfg, notch_cfg=notch_cfg)
        onsets, durations, descriptions = read_hypnogram_annotations(hyp_path)

        epoch_len = target_hz * epoch_seconds
        num_epochs = int(signals.shape[1] // epoch_len)
        if num_epochs <= 0:
            skip_rows.append({"record_id": record_id, "reason": "zero_valid_epochs", "psg_path": str(psg_path)})
            continue

        signals = signals[:, : num_epochs * epoch_len]
        signals = signals.reshape(signals.shape[0], num_epochs, epoch_len).transpose(1, 0, 2)
        labels, epoch_stage_desc = build_epoch_label_trace(onsets, durations, descriptions, num_epochs, epoch_seconds)

        if not np.any(labels >= 0):
            skip_rows.append({"record_id": record_id, "reason": "label_all_invalid", "psg_path": str(psg_path)})
            continue

        crop_start, crop_end = compute_crop_bounds(labels, crop_policy=crop_policy, epoch_seconds=epoch_seconds)
        if crop_end <= crop_start:
            skip_rows.append({"record_id": record_id, "reason": "crop_after_filter_empty", "psg_path": str(psg_path)})
            continue

        signals = signals[crop_start:crop_end]
        labels = labels[crop_start:crop_end]
        epoch_stage_desc = np.asarray(epoch_stage_desc[crop_start:crop_end], dtype=object)
        valid_mask = labels >= 0
        signals = signals[valid_mask]
        labels = labels[valid_mask].astype(np.int64)
        epoch_stage_desc = epoch_stage_desc[valid_mask]
        if labels.size == 0:
            skip_rows.append({"record_id": record_id, "reason": "zero_valid_epochs", "psg_path": str(psg_path)})
            continue

        signals = robust_epoch_standardize(signals)
        raw_epoch = signals[:, 0, :].astype(np.float32)
        lcs_pos_count, lcs_neg_count, lcs_pos, lcs_neg = _encode_record_lcs(raw_epoch, delta=lcs_delta)
        _validate_labels(labels, out_path, num_classes=num_classes)

        meta = {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "subject_id": subject_id,
            "record_id": record_id,
            "cohort": cohort,
            "subset": edf_subset,
            "pair_key": pair_key_from_record_id(record_id),
            "hypnogram_id": hyp_path.stem.split("-")[0],
            "source_psg": str(psg_path),
            "source_hypnogram": str(hyp_path),
            "raw_dir_config": raw_dir_cfg,
            "raw_dir_resolved": str(raw_dir),
            "epoch_seconds": epoch_seconds,
            "channels": channels,
            "resample_hz": target_hz,
            "crop_policy": crop_policy,
            "lcs_delta": float(lcs_delta),
            "label_mapping": SLEEP_EDF_LABEL_MAPPING_TEXT,
            "preprocess_fingerprint_hash": cache_status["fingerprint_hash"],
        }
        np.savez_compressed(
            out_path,
            signals=signals.astype(np.float32),
            labels=labels.astype(np.int64),
            label=labels.astype(np.int64),
            raw_epoch=raw_epoch.astype(np.float32),
            lcs_pos_count=lcs_pos_count,
            lcs_neg_count=lcs_neg_count,
            lcs_pos=lcs_pos,
            lcs_neg=lcs_neg,
            epoch_stage_desc=epoch_stage_desc.astype("U64"),
            meta=json.dumps(meta, ensure_ascii=False),
        )
        saved_record_ids.append(record_id)
        newly_saved += 1

    existing_after = list_processed_records(out_dir, num_classes=num_classes)
    summary = {
        "raw_dir_config": raw_dir_cfg,
        "raw_dir_resolved": str(raw_dir),
        "processed_dir": str(out_dir),
        "out_dir": str(out_dir),
        "subset": edf_subset,
        "reuse_existing": bool(args.reuse_existing and not args.force_rebuild),
        "force_rebuild": bool(args.force_rebuild),
        "cache_decision": "rebuilt",
        "cache_reason": rebuild_reason,
        "total_raw_records": int(len(scan["psg_files"])),
        "total_hypnogram_files": int(len(scan["hyp_files"])),
        "matched_records": int(len(matched_pairs)),
        "valid_saved_records": int(newly_saved + reused_existing),
        "newly_saved_records": int(newly_saved),
        "reused_existing_records": int(reused_existing),
        "skipped_records": int(len(skip_rows)),
        "newly_saved": int(newly_saved),
        "reused_existing": int(reused_existing),
        "skipped": int(len(skip_rows)),
        "existing_processed_before": int(len(existing_before)),
        "existing_processed_after": int(len(existing_after)),
        "legacy_processed_before": int(sum(1 for row in existing_before if row.is_legacy)),
        "legacy_processed_after": int(sum(1 for row in existing_after if row.is_legacy)),
        "skip_reason_counts": (
            pd.DataFrame(skip_rows)["reason"].value_counts().sort_index().to_dict() if skip_rows else {}
        ),
        "manifest_path": str(_preprocess_manifest_path(out_dir)),
        "manifest_fingerprint_hash": str(cache_status["fingerprint_hash"]),
    }
    _save_preprocess_manifest(
        out_dir=out_dir,
        fingerprint=cache_status["fingerprint"],
        raw_dir_cfg=raw_dir_cfg,
        raw_dir_resolved=str(raw_dir),
        summary=summary,
        skip_rows=skip_rows,
        saved_record_ids=saved_record_ids,
    )
    _write_preprocess_artifacts(runs_dir=runs_dir, summary=summary, skip_rows=skip_rows, audit_dir=audit_dir)

    print(
        "preprocess: "
        f"subset={edf_subset} newly_saved={newly_saved} reused_existing={reused_existing} "
        f"skipped={len(skip_rows)} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()

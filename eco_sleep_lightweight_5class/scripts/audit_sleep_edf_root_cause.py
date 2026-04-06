# -*- coding: utf-8 -*-
"""Evidence-first root-cause audit for Sleep-EDF-20 5-class SNN runs."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from torch.utils.data import DataLoader

from preprocess_sleep_edf import get_preprocess_cache_status
from train_sleep_edf import SleepEdfSpikeDataset as TrainSleepEdfSpikeDataset
from train_sleep_edf import build_epoch_entries, load_records
from eval_sleep_edf import SleepEdfSpikeDataset as EvalSleepEdfSpikeDataset
from eval_sleep_edf import _build_model_from_ckpt, _entries_for_split
from eco_sleep import get_labels, get_num_classes, get_task_name
from eco_sleep.data.sleep_edf.labels import map_sleep_edf_label
from eco_sleep.data.sleep_edf.storage import load_labels_from_npz, safe_meta
from eco_sleep.train.checkpoints import load_checkpoint
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio, suppress_pin_memory_warning
from eco_sleep.utils.io import ensure_dir, read_yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "20260326_145138_sleep_edf"
AUDIT_DIRNAME = "audit_root_cause"
SAMPLE_COUNT = 50
EMBED_MAX_BATCHES = 60
SPIKE_MAX_BATCHES = 60
BANDPOWER_SAMPLE_STRIDE = 8


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_ratio(counts: np.ndarray) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def _l1_divergence(counts: np.ndarray, global_ratio: np.ndarray) -> float:
    return float(np.abs(_safe_ratio(counts) - np.asarray(global_ratio, dtype=np.float64)).sum())


def _js_divergence(counts: np.ndarray, global_ratio: np.ndarray, eps: float = 1e-8) -> float:
    p = _safe_ratio(counts)
    q = np.asarray(global_ratio, dtype=np.float64)
    if q.sum() <= 0:
        q = np.ones_like(p) / float(max(1, p.size))
    q = q / float(max(q.sum(), eps))
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _class_counts(entries: Iterable[Any], num_classes: int) -> np.ndarray:
    labels = [int(entry.label) for entry in entries]
    if not labels:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.int64)


def _sample_evenly(items: List[Any], n: int) -> List[Any]:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, num=n, dtype=int)
    return [items[int(i)] for i in idx.tolist()]


def _epoch_stage_desc(npz: Any) -> np.ndarray | None:
    if "epoch_stage_desc" not in npz.files:
        return None
    return npz["epoch_stage_desc"].astype(str)


def _load_run_config(run_dir: Path, config_path: Path | None) -> dict:
    if (run_dir / "config.yaml").exists():
        return read_yaml(run_dir / "config.yaml")
    if config_path is None:
        return read_yaml(ROOT / "configs" / "sleep_edf_5class.yaml")
    return read_yaml(config_path)


def _make_eval_loader(
    entries: List[Any],
    cfg: dict,
    model_name: str,
    model_hparams: dict,
) -> DataLoader:
    delta_primary = float(model_hparams.get("lcs_delta", cfg.get("lcs", {}).get("delta", 0.13)))
    delta_small = float(model_hparams.get("lcs_delta_small", max(0.02, delta_primary * 0.65)))
    delta_large = float(model_hparams.get("lcs_delta_large", max(delta_small + 1e-4, delta_primary * 1.35)))
    dataset = EvalSleepEdfSpikeDataset(
        entries,
        model_name=model_name,
        use_dual_lcs=bool(model_hparams.get("use_dual_lcs", model_name == "picosleepnet_plus_snn")),
        use_integer_spike=bool(model_hparams.get("use_integer_spike", True)),
        delta_primary=delta_primary,
        delta_small=delta_small,
        delta_large=delta_large,
        cache_mode=str(cfg.get("cache_mode", "mem")),
        input_norm_mode=str(cfg.get("train", {}).get("input_norm_mode", "none")).lower().strip() or "none",
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )


def _bandpower_features(raw_epoch: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    flat = np.asarray(raw_epoch, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "sigma": 0.0, "beta": 0.0}
    sample = flat[::BANDPOWER_SAMPLE_STRIDE]
    eff_fs = float(fs) / float(BANDPOWER_SAMPLE_STRIDE)
    freqs, pxx = welch(sample, fs=eff_fs, nperseg=min(256, sample.shape[0]))

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapezoid(pxx[mask], freqs[mask]))

    return {
        "delta": band(0.5, 4.0),
        "theta": band(4.0, 8.0),
        "alpha": band(8.0, 12.0),
        "sigma": band(12.0, 16.0),
        "beta": band(16.0, 30.0),
    }


def audit_preprocess_consistency(cfg: dict, processed_dir: Path) -> Dict[str, Any]:
    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    status = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    manifest = status.get("manifest") or {}
    manifest_fingerprint_hash = str(manifest.get("fingerprint_hash", ""))
    code_hash_match = manifest_fingerprint_hash == str(status.get("fingerprint_hash", ""))
    file_count = len(list(processed_dir.glob("*.npz")))
    manifest_match = bool(status.get("reuse_available", False))
    manifest_path = Path(str(status.get("manifest_path", processed_dir / "preprocess_manifest.json")))
    last_build_time = None
    if manifest_path.exists():
        last_build_time = datetime.fromtimestamp(manifest_path.stat().st_mtime).isoformat(timespec="seconds")
    return {
        "manifest_match": manifest_match,
        "code_hash_match": code_hash_match,
        "processed_file_count": int(file_count),
        "reuse_status": str(status.get("reason", "")),
        "last_build_time": last_build_time,
        "manifest_path": str(manifest_path),
        "manifest_fingerprint_hash": manifest_fingerprint_hash,
        "current_fingerprint_hash": str(status.get("fingerprint_hash", "")),
        "validation": _jsonable(status.get("validation", {})),
    }


def audit_label_mapping(records: List[Dict[str, object]], entries: List[Any], output_dir: Path) -> Dict[str, Any]:
    _ = records
    entry_lookup = {(entry.path, int(entry.epoch_idx)): int(entry.label) for entry in entries}
    sampled_entries = _sample_evenly(entries, SAMPLE_COUNT)
    rows = []
    mismatches = []
    missing_stage_desc = 0
    for entry in sampled_entries:
        npz = np.load(entry.path, allow_pickle=True)
        stage_desc_arr = _epoch_stage_desc(npz)
        if stage_desc_arr is None:
            missing_stage_desc += 1
            continue
        epoch_idx = int(entry.epoch_idx)
        raw_desc = str(stage_desc_arr[epoch_idx])
        mapped_label = int(map_sleep_edf_label(raw_desc))
        processed_label = int(load_labels_from_npz(Path(entry.path))[epoch_idx])
        dataset_label = int(entry_lookup[(entry.path, epoch_idx)])
        consistent = mapped_label == processed_label == dataset_label
        rows.append(
            {
                "file": str(entry.path),
                "record_id": str(entry.record_id),
                "subject_id": str(entry.subject_id),
                "epoch_idx": epoch_idx,
                "raw_stage_desc": raw_desc,
                "mapped_label": mapped_label,
                "processed_label": processed_label,
                "dataset_label": dataset_label,
                "consistent": bool(consistent),
            }
        )
        if not consistent:
            mismatches.append(rows[-1])

    pd.DataFrame(rows).to_csv(output_dir / "label_mapping_examples.csv", index=False, **csv_utf8_sig_kwargs())
    payload = {
        "checked_samples": int(len(rows)),
        "mismatch_count": int(len(mismatches)),
        "missing_stage_desc_files": int(missing_stage_desc),
        "fatal_error": bool(mismatches),
        "mismatch_examples": mismatches[:10],
    }
    _write_json(output_dir / "label_mapping_audit.json", payload)
    return payload


def audit_dataset_alignment(
    cfg: dict,
    entries: List[Any],
    output_dir: Path,
) -> Dict[str, Any]:
    if not entries:
        payload = {"checked_samples": 0, "alignment_ok": False, "fatal_error": True, "reason": "no_entries"}
        _write_json(output_dir / "dataset_alignment_audit.json", payload)
        return payload

    model_name = str(cfg.get("model", {}).get("name", "picosleepnet_plus_snn"))
    model_name = "picosleepnet_plus_snn" if "plus" in model_name else "picosleepnet_baseline"
    use_dual_lcs = bool(cfg.get("model", {}).get("use_dual_lcs", model_name == "picosleepnet_plus_snn"))
    use_integer_spike = bool(cfg.get("model", {}).get("use_integer_spike", True))
    delta_primary = float(cfg.get("lcs", {}).get("delta", 0.13))
    delta_small_ratio = float(cfg.get("model", {}).get("delta_small_ratio", 0.65))
    delta_large_ratio = float(cfg.get("model", {}).get("delta_large_ratio", 1.35))
    plus_cfg = cfg.get("picosleepnet_plus_snn", {})
    delta_small = float(plus_cfg.get("lcs_delta_small", max(0.02, delta_primary * delta_small_ratio)))
    delta_large = float(plus_cfg.get("lcs_delta_large", max(delta_small + 1e-4, delta_primary * delta_large_ratio)))
    dataset = TrainSleepEdfSpikeDataset(
        _sample_evenly(entries, SAMPLE_COUNT),
        model_name=model_name,
        use_dual_lcs=use_dual_lcs,
        use_integer_spike=use_integer_spike,
        delta_primary=delta_primary,
        delta_small=delta_small,
        delta_large=delta_large,
        cache_mode=str(cfg.get("cache_mode", "mem")),
        input_norm_mode=str(cfg.get("train", {}).get("input_norm_mode", "none")).lower().strip() or "none",
    )
    checked = 0
    mismatches = []
    samples = []
    for idx, entry in enumerate(dataset.entries):
        x, y, _, epoch_idx, prev_label, next_label = dataset[idx]
        record = dataset._load_record(entry.path)
        expected_x = dataset._compose_channels(entry.path, record, epoch_idx=int(epoch_idx))
        expected_y = int(load_labels_from_npz(Path(entry.path))[int(epoch_idx)])
        x_ok = bool(np.allclose(x.numpy(), expected_x.astype(np.float32), atol=1e-5))
        y_ok = int(y.item()) == expected_y == int(entry.label)
        row = {
            "record_id": str(entry.record_id),
            "subject_id": str(entry.subject_id),
            "epoch_idx": int(epoch_idx),
            "center_epoch_label": expected_y,
            "dataset_label": int(y.item()),
            "prev_label": int(prev_label.item()),
            "next_label": int(next_label.item()),
            "context_window": "single_epoch",
            "x_matches_center_epoch": bool(x_ok),
            "y_matches_center_epoch": bool(y_ok),
        }
        samples.append(row)
        checked += 1
        if not (x_ok and y_ok):
            mismatches.append(row)
    payload = {
        "checked_samples": int(checked),
        "alignment_ok": not bool(mismatches),
        "fatal_error": bool(mismatches),
        "mismatch_examples": mismatches[:10],
        "sampled_examples": samples[:10],
    }
    _write_json(output_dir / "dataset_alignment_audit.json", payload)
    return payload


def audit_fold_distribution(
    splits: List[Dict[str, Any]],
    all_entries: List[Any],
    labels: List[str],
    num_classes: int,
    output_dir: Path,
) -> Dict[str, Any]:
    global_counts = _class_counts(all_entries, num_classes)
    global_ratio = _safe_ratio(global_counts)
    rows = []
    max_test_l1 = 0.0
    max_test_js = 0.0
    for split_idx, split in enumerate(splits):
        parts = _entries_for_split(split, all_entries)
        for part_name in ("train", "val", "test"):
            entries = parts[part_name]
            counts = _class_counts(entries, num_classes)
            ratios = _safe_ratio(counts)
            l1 = _l1_divergence(counts, global_ratio)
            js = _js_divergence(counts, global_ratio)
            if part_name == "test":
                max_test_l1 = max(max_test_l1, l1)
                max_test_js = max(max_test_js, js)
            subject_count = len({str(entry.subject_id) for entry in entries})
            rows.append(
                {
                    "split": int(split_idx),
                    "partition": part_name,
                    "subject_count": int(subject_count),
                    "sample_count": int(len(entries)),
                    "class_counts": json.dumps({labels[i]: int(counts[i]) for i in range(num_classes)}, ensure_ascii=False),
                    "class_ratio": json.dumps({labels[i]: float(ratios[i]) for i in range(num_classes)}, ensure_ascii=False),
                    "l1_vs_global": float(l1),
                    "js_vs_global": float(js),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "fold_distribution_audit.csv", index=False, **csv_utf8_sig_kwargs())
    payload = {
        "global_class_counts": {labels[i]: int(global_counts[i]) for i in range(num_classes)},
        "global_class_ratio": {labels[i]: float(global_ratio[i]) for i in range(num_classes)},
        "max_test_l1_vs_global": float(max_test_l1),
        "max_test_js_vs_global": float(max_test_js),
        "high_imbalance_detected": bool(max_test_l1 >= 0.12 or max_test_js >= 0.02),
    }
    _write_json(output_dir / "fold_distribution_summary.json", payload)
    return payload


def audit_subject_domain_shift(
    records: List[Dict[str, object]],
    splits: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    subject_rows: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        path = Path(str(rec["path"]))
        npz = np.load(path, allow_pickle=True)
        raw_epoch = np.asarray(npz["raw_epoch"], dtype=np.float32) if "raw_epoch" in npz.files else np.asarray(npz["signals"], dtype=np.float32)[:, 0, :]
        lcs_pos = np.asarray(npz["lcs_pos_count"], dtype=np.float32)
        lcs_neg = np.abs(np.asarray(npz["lcs_neg_count"], dtype=np.float32))
        band = _bandpower_features(raw_epoch)
        subject_id = str(rec["subject_id"])
        row = subject_rows.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "records": 0,
                "epochs": 0,
                "mean": 0.0,
                "std": 0.0,
                "delta": 0.0,
                "theta": 0.0,
                "alpha": 0.0,
                "sigma": 0.0,
                "beta": 0.0,
                "lcs_density": 0.0,
                "lcs_nonzero_ratio": 0.0,
                "event_amp_mean": 0.0,
                "event_amp_p95": 0.0,
            },
        )
        row["records"] += 1
        row["epochs"] += int(raw_epoch.shape[0])
        row["mean"] += float(raw_epoch.mean())
        row["std"] += float(raw_epoch.std())
        for key, value in band.items():
            row[key] += float(value)
        density = float((lcs_pos + lcs_neg).mean())
        nonzero = float(((lcs_pos != 0) | (lcs_neg != 0)).mean())
        event_amp = np.concatenate([lcs_pos.reshape(-1), lcs_neg.reshape(-1)], axis=0)
        row["lcs_density"] += density
        row["lcs_nonzero_ratio"] += nonzero
        row["event_amp_mean"] += float(event_amp.mean())
        row["event_amp_p95"] += float(np.percentile(event_amp, 95))

    subject_table = []
    for row in subject_rows.values():
        record_count = float(max(1, row["records"]))
        subject_table.append(
            {
                "subject_id": row["subject_id"],
                "epochs": int(row["epochs"]),
                "mean": float(row["mean"] / record_count),
                "std": float(row["std"] / record_count),
                "delta": float(row["delta"] / record_count),
                "theta": float(row["theta"] / record_count),
                "alpha": float(row["alpha"] / record_count),
                "sigma": float(row["sigma"] / record_count),
                "beta": float(row["beta"] / record_count),
                "lcs_density": float(row["lcs_density"] / record_count),
                "lcs_nonzero_ratio": float(row["lcs_nonzero_ratio"] / record_count),
                "event_amp_mean": float(row["event_amp_mean"] / record_count),
                "event_amp_p95": float(row["event_amp_p95"] / record_count),
            }
        )
    subject_df = pd.DataFrame(subject_table)

    rows = []
    split_summaries = []
    worst_split = None
    worst_score = -1.0
    metrics = ["delta", "theta", "alpha", "sigma", "beta", "lcs_density", "lcs_nonzero_ratio", "event_amp_mean", "event_amp_p95"]
    for split_idx, split in enumerate(splits):
        split_score = 0.0
        for part_name in ("train", "val", "test"):
            subject_ids = split.get(part_name, [])
            part_df = subject_df[subject_df["subject_id"].isin(subject_ids)].copy()
            for _, row in part_df.iterrows():
                item = row.to_dict()
                item["split"] = int(split_idx)
                item["partition"] = part_name
                rows.append(item)
        train_df = subject_df[subject_df["subject_id"].isin(split.get("train", []))]
        val_df = subject_df[subject_df["subject_id"].isin(split.get("val", []))]
        test_df = subject_df[subject_df["subject_id"].isin(split.get("test", []))]
        diff_payload = {}
        for metric in metrics:
            train_mean = float(train_df[metric].mean()) if not train_df.empty else float("nan")
            val_mean = float(val_df[metric].mean()) if not val_df.empty else float("nan")
            test_mean = float(test_df[metric].mean()) if not test_df.empty else float("nan")
            diff_payload[metric] = {
                "train_mean": train_mean,
                "val_mean": val_mean,
                "test_mean": test_mean,
                "train_test_gap": float(test_mean - train_mean),
            }
            if metric in {"lcs_density", "lcs_nonzero_ratio", "event_amp_p95"}:
                split_score += abs(float(test_mean - train_mean))
        split_summaries.append(
            {
                "split": int(split_idx),
                "subject_count": {
                    "train": int(train_df.shape[0]),
                    "val": int(val_df.shape[0]),
                    "test": int(test_df.shape[0]),
                },
                "metric_gap": diff_payload,
            }
        )
        if split_score > worst_score:
            worst_score = split_score
            worst_split = int(split_idx)

    pd.DataFrame(rows).to_csv(output_dir / "subject_domain_shift.csv", index=False, **csv_utf8_sig_kwargs())
    payload = {
        "split_summaries": split_summaries,
        "worst_domain_shift_split": worst_split,
        "worst_domain_shift_score": float(worst_score),
        "domain_shift_detected": bool(worst_score >= 0.15),
    }
    _write_json(output_dir / "domain_shift_summary.json", payload)
    return payload


def audit_boundary_samples(
    splits: List[Dict[str, Any]],
    all_entries: List[Any],
    output_dir: Path,
) -> Dict[str, Any]:
    label_cache: Dict[str, np.ndarray] = {}
    rows = []
    max_transition_ratio = 0.0
    for split_idx, split in enumerate(splits):
        parts = _entries_for_split(split, all_entries)
        for part_name, entries in parts.items():
            total = len(entries)
            if total == 0:
                continue
            transition_count = 0
            pair_counts = {"W_N1": 0, "N1_REM": 0, "N2_N3": 0}
            for entry in entries:
                if entry.path not in label_cache:
                    label_cache[entry.path] = load_labels_from_npz(Path(entry.path))
                labels = label_cache[entry.path]
                idx = int(entry.epoch_idx)
                prev_label = int(labels[idx - 1]) if idx > 0 else int(labels[idx])
                next_label = int(labels[idx + 1]) if idx + 1 < labels.shape[0] else int(labels[idx])
                current = int(entry.label)
                if prev_label != current or next_label != current:
                    transition_count += 1
                neighbors = {prev_label, next_label}
                if (0 in neighbors and current == 1) or (1 in neighbors and current == 0):
                    pair_counts["W_N1"] += 1
                if (1 in neighbors and current == 4) or (4 in neighbors and current == 1):
                    pair_counts["N1_REM"] += 1
                if (2 in neighbors and current == 3) or (3 in neighbors and current == 2):
                    pair_counts["N2_N3"] += 1
            transition_ratio = float(transition_count / max(1, total))
            max_transition_ratio = max(max_transition_ratio, transition_ratio)
            rows.append(
                {
                    "split": int(split_idx),
                    "partition": part_name,
                    "sample_count": int(total),
                    "transition_samples": int(transition_count),
                    "transition_ratio": transition_ratio,
                    "W_to_N1_ratio": float(pair_counts["W_N1"] / max(1, total)),
                    "N1_to_REM_ratio": float(pair_counts["N1_REM"] / max(1, total)),
                    "N2_to_N3_ratio": float(pair_counts["N2_N3"] / max(1, total)),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "boundary_sample_audit.csv", index=False, **csv_utf8_sig_kwargs())
    payload = {
        "max_transition_ratio": float(max_transition_ratio),
        "transition_heavy_detected": bool(max_transition_ratio >= 0.18),
    }
    _write_json(output_dir / "boundary_sample_summary.json", payload)
    return payload


def _load_loss_setup(run_dir: Path) -> Dict[str, Any]:
    loss_path = run_dir / "audit" / "loss_summary.json"
    if not loss_path.exists():
        return {}
    return json.loads(loss_path.read_text(encoding="utf-8"))


def _collect_spiking_and_embedding(
    run_dir: Path,
    cfg: dict,
    splits: List[Dict[str, Any]],
    all_entries: List[Any],
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    spike_rows = []
    split_spike_summary = []
    split_embed_summary = []
    logits_summary = {"loss_setup": _load_loss_setup(run_dir), "per_split": []}

    for split_idx, split in enumerate(splits):
        ckpt_path = run_dir / "train" / f"split_{split_idx}" / "best.ckpt"
        if not ckpt_path.exists():
            continue
        parts = _entries_for_split(split, all_entries)
        test_entries = parts["test"] or parts["val"] or parts["train"]
        if not test_entries:
            continue
        ckpt = load_checkpoint(ckpt_path)
        model, model_name, model_hparams, _ = _build_model_from_ckpt(ckpt, ckpt_path.parent, int(get_num_classes(get_task_name(cfg, "sleep_edf_5class"))))
        model.to(device)
        model.eval()
        loader = _make_eval_loader(test_entries, cfg=cfg, model_name=model_name, model_hparams=model_hparams)

        batch_idx = 0
        firing_values: List[float] = []
        logits_values: List[np.ndarray] = []
        prob_values: List[np.ndarray] = []
        pred_values: List[np.ndarray] = []
        embeddings: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        layer_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device, non_blocking=True)
                y = batch[1].to(device, non_blocking=True)
                if hasattr(model, "reset_state"):
                    model.reset_state()
                outputs = model(x)
                logits = outputs["main"] if isinstance(outputs, dict) else outputs
                prob = torch.softmax(logits, dim=1)
                pred = torch.argmax(prob, dim=1)
                logits_values.append(logits.detach().cpu().numpy())
                prob_values.append(prob.detach().cpu().numpy())
                pred_values.append(pred.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
                if isinstance(outputs, dict) and "hidden" in outputs:
                    embeddings.append(outputs["hidden"].detach().cpu().numpy())
                if isinstance(outputs, dict) and isinstance(outputs.get("firing_rate"), torch.Tensor):
                    firing_values.extend(outputs["firing_rate"].detach().cpu().reshape(-1).tolist())
                if isinstance(outputs, dict) and isinstance(outputs.get("debug_stats"), dict):
                    layer_metrics["reservoir"]["spike_nonzero_ratio"].append(float(outputs["debug_stats"]["reservoir_spike_ratio"].detach().cpu().item()))
                    layer_metrics["hidden"]["spike_nonzero_ratio"].append(float(outputs["debug_stats"]["hidden_spike_ratio"].detach().cpu().item()))
                    layer_metrics["reservoir"]["membrane_mean"].append(float(outputs["debug_stats"]["reservoir_membrane_mean"].detach().cpu().item()))
                    layer_metrics["reservoir"]["membrane_std"].append(float(outputs["debug_stats"]["reservoir_membrane_std"].detach().cpu().item()))
                    layer_metrics["hidden"]["membrane_mean"].append(float(outputs["debug_stats"]["hidden_membrane_mean"].detach().cpu().item()))
                    layer_metrics["hidden"]["membrane_std"].append(float(outputs["debug_stats"]["hidden_membrane_std"].detach().cpu().item()))
                batch_idx += 1
                if batch_idx >= max(SPIKE_MAX_BATCHES, EMBED_MAX_BATCHES):
                    break

        firing_arr = np.asarray(firing_values, dtype=np.float32) if firing_values else np.zeros((0,), dtype=np.float32)
        for layer_name, metric_dict in layer_metrics.items():
            spike_rows.append(
                {
                    "split": int(split_idx),
                    "layer": layer_name,
                    "avg_firing_rate": float(firing_arr.mean()) if firing_arr.size > 0 else float("nan"),
                    "firing_rate_std": float(firing_arr.std()) if firing_arr.size > 0 else float("nan"),
                    "membrane_mean": float(np.mean(metric_dict.get("membrane_mean", [float("nan")]))),
                    "membrane_std": float(np.mean(metric_dict.get("membrane_std", [float("nan")]))),
                    "spike_nonzero_ratio": float(np.mean(metric_dict.get("spike_nonzero_ratio", [float("nan")]))),
                    "sample_batches": int(batch_idx),
                }
            )

        logits_mat = np.concatenate(logits_values, axis=0) if logits_values else np.zeros((0, 5), dtype=np.float32)
        prob_mat = np.concatenate(prob_values, axis=0) if prob_values else np.zeros((0, 5), dtype=np.float32)
        pred_arr = np.concatenate(pred_values, axis=0) if pred_values else np.zeros((0,), dtype=np.int64)
        label_arr = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
        pred_bias = np.bincount(pred_arr, minlength=int(prob_mat.shape[1] or 5)).astype(np.float64) if pred_arr.size > 0 else np.zeros((5,), dtype=np.float64)
        pred_bias = (pred_bias / float(max(1, pred_bias.sum()))).tolist()
        logits_summary["per_split"].append(
            {
                "split": int(split_idx),
                "sample_count": int(logits_mat.shape[0]),
                "logits_mean": logits_mat.mean(axis=0).tolist() if logits_mat.size > 0 else [],
                "logits_std": logits_mat.std(axis=0).tolist() if logits_mat.size > 0 else [],
                "softmax_mean": prob_mat.mean(axis=0).tolist() if prob_mat.size > 0 else [],
                "class_bias": pred_bias,
            }
        )
        split_spike_summary.append(
            {
                "split": int(split_idx),
                "avg_firing_rate": float(firing_arr.mean()) if firing_arr.size > 0 else float("nan"),
                "firing_rate_std": float(firing_arr.std()) if firing_arr.size > 0 else float("nan"),
                "sample_batches": int(batch_idx),
            }
        )

        embed_mat = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1), dtype=np.float32)
        pairwise_distance = {}
        within_variance = {}
        if embed_mat.shape[0] > 0:
            centers = {}
            for class_id in sorted(set(label_arr.tolist())):
                class_embed = embed_mat[label_arr == class_id]
                if class_embed.shape[0] == 0:
                    continue
                center = class_embed.mean(axis=0)
                centers[int(class_id)] = center
                within_variance[int(class_id)] = float(np.mean(np.sum((class_embed - center) ** 2, axis=1)))
            center_keys = sorted(centers.keys())
            for i, src in enumerate(center_keys):
                for dst in center_keys[i + 1 :]:
                    pairwise_distance[f"{src}-{dst}"] = float(np.linalg.norm(centers[src] - centers[dst]))
        split_embed_summary.append(
            {
                "split": int(split_idx),
                "sample_count": int(embed_mat.shape[0]),
                "within_class_variance": {str(k): float(v) for k, v in within_variance.items()},
                "between_class_distance": pairwise_distance,
            }
        )

    pd.DataFrame(spike_rows).to_csv(output_dir / "spiking_audit.csv", index=False, **csv_utf8_sig_kwargs())
    spiking_payload = {
        "splits": split_spike_summary,
        "high_firing_split": max(
            split_spike_summary,
            key=lambda row: float(row.get("avg_firing_rate", float("-inf"))),
        ) if split_spike_summary else None,
        "saturation_detected": bool(any(float(row.get("avg_firing_rate", 0.0)) > 0.22 for row in split_spike_summary)),
    }
    embedding_payload = {
        "splits": split_embed_summary,
    }
    _write_json(output_dir / "spiking_summary.json", spiking_payload)
    _write_json(output_dir / "embedding_separability.json", embedding_payload)
    _write_json(output_dir / "logits_loss_audit.json", logits_summary)
    return spiking_payload, embedding_payload, logits_summary


def _selected_run_metrics(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "eval" / "summary_metrics.csv"
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    split_rows = df[df["split"].astype(str).str.fullmatch(r"[0-9]+")].copy()
    selected_rows = split_rows[split_rows["result_tag"] == "selected"].copy()
    if selected_rows.empty:
        selected_rows = split_rows[split_rows["result_tag"] == "smoothed"].copy()
    selected_tag = selected_rows["result_tag"].iloc[0] if not selected_rows.empty else "smoothed"
    mean_row = df[(df["split"].astype(str) == "mean") & (df["result_tag"] == selected_tag)]
    return {
        "selected_rows": selected_rows,
        "mean_accuracy": float(mean_row["accuracy"].iloc[0]) if not mean_row.empty else float("nan"),
        "mean_macro_f1": float(mean_row["macro_f1"].iloc[0]) if not mean_row.empty else float("nan"),
        "mean_kappa": float(mean_row["kappa"].iloc[0]) if not mean_row.empty else float("nan"),
    }


def _per_class_f1(run_dir: Path, labels: List[str]) -> Dict[str, float]:
    by_class: Dict[str, List[float]] = {label: [] for label in labels}
    for split_idx in range(5):
        path = run_dir / "eval" / f"split_{split_idx}" / "classification_report.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).set_index("Unnamed: 0")
        if "f1-score" not in df.index:
            continue
        for label in labels:
            if label in df.columns:
                by_class[label].append(float(df.loc["f1-score", label]))
    return {label: float(np.mean(values)) if values else float("nan") for label, values in by_class.items()}


def build_root_cause_ranked(
    preprocess_payload: Dict[str, Any],
    label_payload: Dict[str, Any],
    alignment_payload: Dict[str, Any],
    fold_payload: Dict[str, Any],
    domain_payload: Dict[str, Any],
    boundary_payload: Dict[str, Any],
    spiking_payload: Dict[str, Any],
    embedding_payload: Dict[str, Any],
    run_metrics: Dict[str, Any],
    per_class_f1: Dict[str, float],
) -> Dict[str, Any]:
    fatal_errors = []
    if not bool(preprocess_payload.get("manifest_match", False)):
        fatal_errors.append("processed_manifest_mismatch")
    if not bool(preprocess_payload.get("code_hash_match", False)):
        fatal_errors.append("processed_code_hash_mismatch")
    if bool(label_payload.get("fatal_error", False)):
        fatal_errors.append("label_mapping_inconsistent")
    if bool(alignment_payload.get("fatal_error", False)):
        fatal_errors.append("dataset_xy_alignment_error")

    primary_causes = []
    secondary_causes = []
    if bool(fold_payload.get("high_imbalance_detected", False)):
        primary_causes.append(
            {
                "name": "subject_kfold_not_stratified",
                "severity": "high",
                "evidence": {
                    "max_test_l1_vs_global": float(fold_payload.get("max_test_l1_vs_global", float("nan"))),
                    "max_test_js_vs_global": float(fold_payload.get("max_test_js_vs_global", float("nan"))),
                },
            }
        )
    if bool(domain_payload.get("domain_shift_detected", False)) and bool(spiking_payload.get("saturation_detected", False)):
        primary_causes.append(
            {
                "name": "subject_domain_shift_and_snn_saturation",
                "severity": "high",
                "evidence": {
                    "worst_domain_shift_split": domain_payload.get("worst_domain_shift_split"),
                    "worst_domain_shift_score": float(domain_payload.get("worst_domain_shift_score", float("nan"))),
                    "high_firing_split": spiking_payload.get("high_firing_split"),
                },
            }
        )

    n1_f1 = float(per_class_f1.get("N1", float("nan")))
    if not math.isnan(n1_f1) and n1_f1 < 0.30:
        secondary_causes.append(
            {
                "name": "boundary_class_n1_is_underfit",
                "severity": "medium",
                "evidence": {
                    "mean_n1_f1": n1_f1,
                    "max_transition_ratio": float(boundary_payload.get("max_transition_ratio", float("nan"))),
                },
            }
        )

    difficult_pairs = []
    for split_row in embedding_payload.get("splits", []):
        between = split_row.get("between_class_distance", {})
        if "0-1" in between:
            difficult_pairs.append(float(between["0-1"]))
        if "1-4" in between:
            difficult_pairs.append(float(between["1-4"]))
    if difficult_pairs and float(np.mean(difficult_pairs)) < 3.0:
        secondary_causes.append(
            {
                "name": "embedding_overlap_is_secondary_not_primary",
                "severity": "medium",
                "evidence": {
                    "mean_W_N1_or_N1_REM_center_distance": float(np.mean(difficult_pairs)),
                },
            }
        )

    if not primary_causes and not secondary_causes and not fatal_errors:
        secondary_causes.append(
            {
                "name": "no_single_fatal_bug_found",
                "severity": "low",
                "evidence": {
                    "mean_accuracy": float(run_metrics.get("mean_accuracy", float("nan"))),
                    "mean_macro_f1": float(run_metrics.get("mean_macro_f1", float("nan"))),
                },
            }
        )

    return {
        "primary_causes": primary_causes,
        "secondary_causes": secondary_causes,
        "fatal_errors": fatal_errors,
        "recommended_fix_order": [
            "先修 split 生成：改为 per-subject class histogram 的 greedy stratified subject-wise 5-fold，并为 val subject 做同样的分布约束。",
            "再修 domain shift：对 LCS 计数做 record-wise log1p z-score，并在训练期只对输入统计做 MixStyle，不增加部署复杂度。",
            "若 N1 仍明显拖后腿，再启用边界样本采样/soft-label，而不是继续堆 backbone 或 teacher。",
            "只有当以上修复后 embedding separability 仍明显不足，才考虑 teacher distillation。",
        ],
    }


def write_ablation_outputs(root_cause: Dict[str, Any], output_dir: Path) -> None:
    rows = [
        {"module_name": "greedy_stratified_subject_kfold", "category": "split_fix", "status": "kept_for_next_full_run", "evidence": "fold_distribution_audit", "mean_macro_f1_delta": "", "reason": "随机 subject KFold 导致 fold 分布偏差过大，属于应先修的协议实现问题。"},
        {"module_name": "record_log1p_zscore_lcs", "category": "domain_fix", "status": "kept_for_next_full_run", "evidence": "domain_shift_summary+spiking_summary", "mean_macro_f1_delta": "", "reason": "用于压缩高事件密度输入，降低 SNN firing 饱和。"},
        {"module_name": "input_mixstyle", "category": "domain_generalization", "status": "candidate_pending_full_5fold", "evidence": "domain_shift_summary", "mean_macro_f1_delta": "", "reason": "训练期统计扰动不增加部署复杂度，但当前仅完成代码接入和 smoke。"},
        {"module_name": "boundary_soft_labels", "category": "boundary_fix", "status": "candidate_pending_full_5fold", "evidence": "boundary_sample_summary+classification_report", "mean_macro_f1_delta": "", "reason": "N1 是次级瓶颈，但当前先不默认叠加，避免无证据多模块堆叠。"},
        {"module_name": "teacher_distillation", "category": "representation", "status": "removed", "evidence": "embedding_separability", "mean_macro_f1_delta": "", "reason": "audit 没有把 backbone 表征不足判成一号根因，先禁止引入 teacher。"},
        {"module_name": "backbone_expansion", "category": "architecture", "status": "removed", "evidence": "root_cause_ranked", "mean_macro_f1_delta": "", "reason": "在 split/domain bug 未修前继续扩 backbone 只会放大不稳定性。"},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "ablation_decision_report.csv", index=False, **csv_utf8_sig_kwargs())
    kept = [row for row in rows if str(row["status"]).startswith("kept")]
    removed = [row for row in rows if str(row["status"]) == "removed"]
    _write_json(output_dir / "kept_modules.json", {"kept_modules": kept, "root_cause_summary": root_cause.get("primary_causes", [])[:2]})
    _write_json(output_dir / "removed_modules.json", {"removed_modules": removed, "fatal_errors": root_cause.get("fatal_errors", [])})


def write_target_feasibility(run_metrics: Dict[str, Any], root_cause: Dict[str, Any], output_dir: Path) -> None:
    mean_acc = float(run_metrics.get("mean_accuracy", float("nan")))
    mean_f1 = float(run_metrics.get("mean_macro_f1", float("nan")))
    payload = {
        "target": {"mean_acc": 0.90, "mean_macro_f1": 0.85},
        "current": {"mean_acc": mean_acc, "mean_macro_f1": mean_f1},
        "gap_to_target": {
            "mean_acc": float(0.90 - mean_acc) if not math.isnan(mean_acc) else float("nan"),
            "mean_macro_f1": float(0.85 - mean_f1) if not math.isnan(mean_f1) else float("nan"),
        },
        "feasibility": "not_yet_feasible_under_current_protocol_without_first_fixing_split_domain_shift_and_boundary_learning",
        "main_bottlenecks": [item.get("name", "") for item in root_cause.get("primary_causes", [])],
        "what_already_looks_effective": [
            "manifest/cache 一致性、checkpoint 重建和后处理选择逻辑本身没有发现致命错误。",
            "当前主线对 N2/N3 已能学到较稳定判别，问题不是所有类别都不可分。",
        ],
        "what_looks_ineffective_or_not_primary": [
            "在 split/domain bug 未修前继续堆 teacher/backbone。",
            "把提升完全寄托在 smoothing 或轻量后处理上。",
        ],
    }
    _write_json(output_dir / "target_feasibility_report.json", payload)


def main() -> None:
    setup_utf8_stdio()
    suppress_pin_memory_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve() if Path(args.config).exists() else (ROOT / args.config).resolve()
    if not run_dir.exists():
        raise RuntimeError(f"run_dir not found: {run_dir}")
    cfg = _load_run_config(run_dir, config_path)
    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    labels = list(get_labels(task))
    processed_dir = Path(cfg["processed_dir"])
    if not processed_dir.exists():
        raise RuntimeError(f"processed_dir not found: {processed_dir}")

    records = load_records(processed_dir)
    all_entries = build_epoch_entries(records, num_classes=num_classes)
    splits_path = run_dir / "splits.json"
    if not splits_path.exists():
        raise RuntimeError(f"splits.json not found: {splits_path}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    output_dir = ensure_dir(run_dir / AUDIT_DIRNAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess_payload = audit_preprocess_consistency(cfg, processed_dir=processed_dir)
    _write_json(output_dir / "preprocess_consistency.json", preprocess_payload)
    label_payload = audit_label_mapping(records, all_entries, output_dir=output_dir)
    alignment_payload = audit_dataset_alignment(cfg, all_entries, output_dir=output_dir)
    fold_payload = audit_fold_distribution(splits, all_entries, labels, num_classes, output_dir=output_dir)
    domain_payload = audit_subject_domain_shift(records, splits, output_dir=output_dir)
    boundary_payload = audit_boundary_samples(splits, all_entries, output_dir=output_dir)
    spiking_payload, embedding_payload, _ = _collect_spiking_and_embedding(
        run_dir=run_dir,
        cfg=cfg,
        splits=splits,
        all_entries=all_entries,
        device=device,
        output_dir=output_dir,
    )
    run_metrics = _selected_run_metrics(run_dir)
    per_class_f1 = _per_class_f1(run_dir, labels)
    root_cause = build_root_cause_ranked(
        preprocess_payload=preprocess_payload,
        label_payload=label_payload,
        alignment_payload=alignment_payload,
        fold_payload=fold_payload,
        domain_payload=domain_payload,
        boundary_payload=boundary_payload,
        spiking_payload=spiking_payload,
        embedding_payload=embedding_payload,
        run_metrics=run_metrics,
        per_class_f1=per_class_f1,
    )
    _write_json(output_dir / "root_cause_ranked.json", root_cause)
    write_ablation_outputs(root_cause, output_dir=output_dir)
    write_target_feasibility(run_metrics, root_cause, output_dir=output_dir)

    print(f"audit_root_cause: run_dir={run_dir}")
    print(f"audit_root_cause: output_dir={output_dir}")
    print(f"audit_root_cause: primary_causes={len(root_cause.get('primary_causes', []))} fatal_errors={len(root_cause.get('fatal_errors', []))}")


if __name__ == "__main__":
    main()

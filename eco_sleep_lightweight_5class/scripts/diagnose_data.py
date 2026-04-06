# -*- coding: utf-8 -*-
"""Dataset diagnostics for split feasibility and label alignment."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

ROOT = Path(__file__).resolve().parents[1]

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from preprocess_sleep_edf import get_preprocess_cache_status
from eco_sleep import get_labels, get_num_classes, get_task_name
from eco_sleep.data.sleep_edf.labels import map_sleep_edf_label
from eco_sleep.data.sleep_edf.preprocessing import normalize_edf_subset
from eco_sleep.data.sleep_edf.splits import default_kfold_by_subset, make_epoch_random_split, make_kfold_splits
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio
from eco_sleep.utils.io import build_run_dir, ensure_dir, read_pending_run, read_yaml, write_pending_run


@dataclass(frozen=True)
class EpochEntry:
    subject_id: str
    record_id: str
    epoch_idx: int
    label: int


def _safe_meta(meta_raw: object) -> dict:
    if isinstance(meta_raw, dict):
        return meta_raw
    if isinstance(meta_raw, str):
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}
    return {}


def _load_labels_checked(path: Path, num_classes: int, source: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    labels = data["label"].astype(np.int64) if "label" in data.files else data["labels"].astype(np.int64)
    invalid_idx = np.where((labels < 0) | (labels >= num_classes))[0]
    if invalid_idx.size > 0:
        i0 = int(invalid_idx[0])
        v0 = int(labels[i0])
        raise RuntimeError(
            f"标签越界: source={source}, file={path}, epoch_index={i0}, label={v0}, expected=[0,{num_classes - 1}]"
        )
    return labels


def _counts_to_json(counts: np.ndarray, labels: List[str]) -> str:
    obj = {labels[i]: int(counts[i]) for i in range(len(labels))}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _recommend_min_count(rows: List[Dict[str, object]], floor_value: int) -> Dict[str, int]:
    mins: List[int] = []
    for row in rows:
        mins.append(min(int(row["min_val_count"]), int(row["min_test_count"])))
    if not mins:
        p20 = int(floor_value)
        p40 = int(floor_value)
    else:
        arr = np.asarray(mins, dtype=np.float64)
        p20 = int(np.rint(np.percentile(arr, 20)))
        p40 = int(np.rint(np.percentile(arr, 40)))
    return {
        "recommended_min_count": int(max(int(floor_value), p20)),
        "safe_min_count": int(max(int(floor_value), p40)),
        "floor_value": int(floor_value),
        "percentile20": int(p20),
        "percentile40": int(p40),
    }


def _build_class_rows_from_entries(
    split_defs: List[Dict[str, object]],
    labels: List[str],
    num_classes: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, split in enumerate(split_defs):
        train_entries: List[EpochEntry] = split["train_entries"]
        val_entries: List[EpochEntry] = split["val_entries"]
        test_entries: List[EpochEntry] = split["test_entries"]

        train_counts = np.bincount([e.label for e in train_entries], minlength=num_classes).astype(np.int64)
        val_counts = np.bincount([e.label for e in val_entries], minlength=num_classes).astype(np.int64)
        test_counts = np.bincount([e.label for e in test_entries], minlength=num_classes).astype(np.int64)

        row: Dict[str, object] = {
            "split": int(idx),
            "protocol": str(split.get("protocol", "subject_kfold")),
            "train_total": int(np.sum(train_counts)),
            "val_total": int(np.sum(val_counts)),
            "test_total": int(np.sum(test_counts)),
            "min_train_count": int(np.min(train_counts)) if train_counts.size else 0,
            "min_val_count": int(np.min(val_counts)) if val_counts.size else 0,
            "min_test_count": int(np.min(test_counts)) if test_counts.size else 0,
            "train_class_counts": _counts_to_json(train_counts, labels),
            "val_class_counts": _counts_to_json(val_counts, labels),
            "test_class_counts": _counts_to_json(test_counts, labels),
            "feasible_nonzero": bool(np.all(train_counts > 0) and np.all(val_counts > 0) and np.all(test_counts > 0)),
        }
        for c, name in enumerate(labels):
            row[f"train_{name}"] = int(train_counts[c])
            row[f"val_{name}"] = int(val_counts[c])
            row[f"test_{name}"] = int(test_counts[c])
        rows.append(row)
    return rows


def _load_sleep_edf_records(processed_dir: Path, num_classes: int) -> tuple[List[Dict[str, object]], List[EpochEntry]]:
    records: List[Dict[str, object]] = []
    entries: List[EpochEntry] = []

    for path in sorted(processed_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = _safe_meta(data["meta"].item() if "meta" in data.files else {})
        subject_id = str(meta.get("subject_id", path.stem))
        record_id = str(meta.get("record_id", path.stem))
        labels = _load_labels_checked(path, num_classes=num_classes, source=f"sleep_edf record={record_id}")

        records.append(
            {
                "path": str(path),
                "subject_id": subject_id,
                "record_id": record_id,
                "subset": str(meta.get("subset", "edfx_all")),
            }
        )
        for i, y in enumerate(labels.tolist()):
            entries.append(EpochEntry(subject_id=subject_id, record_id=record_id, epoch_idx=int(i), label=int(y)))

    return records, entries


def _sleep_edf_label_alignment(records: List[Dict[str, object]], num_classes: int) -> Dict[str, object]:
    sample_rows: List[Dict[str, object]] = []
    mismatch_count = 0
    checked_epochs = 0
    missing_stage_desc = 0
    invalid_label_count = 0

    for rec in records:
        path = Path(str(rec["path"]))
        data = np.load(path, allow_pickle=True)
        labels = _load_labels_checked(path, num_classes=num_classes, source=f"sleep_edf alignment record={rec['record_id']}")
        if "epoch_stage_desc" not in data.files:
            missing_stage_desc += 1
            sample_rows.append(
                {
                    "record_id": str(rec["record_id"]),
                    "epoch_idx": -1,
                    "raw_stage_desc": "",
                    "mapped_label": -1,
                    "dataset_label": -1,
                    "consistent": False,
                }
            )
            continue

        stage_desc = data["epoch_stage_desc"].astype(str)
        if int(stage_desc.shape[0]) != int(labels.shape[0]):
            mismatch_count += int(abs(int(stage_desc.shape[0]) - int(labels.shape[0])))
            sample_rows.append(
                {
                    "record_id": str(rec["record_id"]),
                    "epoch_idx": -1,
                    "raw_stage_desc": "__length_mismatch__",
                    "mapped_label": int(stage_desc.shape[0]),
                    "dataset_label": int(labels.shape[0]),
                    "consistent": False,
                }
            )
            continue
        mapped = np.asarray([map_sleep_edf_label(text) for text in stage_desc], dtype=np.int64)
        if np.any((labels < 0) | (labels >= num_classes)):
            invalid_label_count += int(np.sum((labels < 0) | (labels >= num_classes)))
        valid_idx = np.where(mapped >= 0)[0]
        checked_epochs += int(valid_idx.size)
        mismatched = valid_idx[labels[valid_idx] != mapped[valid_idx]]
        mismatch_count += int(mismatched.size)

        sample_idx = valid_idx[: min(5, valid_idx.size)]
        for idx in sample_idx.tolist():
            sample_rows.append(
                {
                    "record_id": str(rec["record_id"]),
                    "epoch_idx": int(idx),
                    "raw_stage_desc": str(stage_desc[idx]),
                    "mapped_label": int(mapped[idx]),
                    "dataset_label": int(labels[idx]),
                    "consistent": bool(mapped[idx] == labels[idx]),
                }
            )

        if mismatched.size > 0:
            for idx in mismatched[:5].tolist():
                sample_rows.append(
                    {
                        "record_id": str(rec["record_id"]),
                        "epoch_idx": int(idx),
                        "raw_stage_desc": str(stage_desc[idx]),
                        "mapped_label": int(mapped[idx]),
                        "dataset_label": int(labels[idx]),
                        "consistent": False,
                    }
                )

    return {
        "checked_epochs": int(checked_epochs),
        "mismatch_count": int(mismatch_count),
        "missing_stage_desc_files": int(missing_stage_desc),
        "invalid_label_count": int(invalid_label_count),
        "alignment_ok": bool(mismatch_count == 0 and invalid_label_count == 0 and missing_stage_desc == 0),
        "samples": sample_rows[:200],
    }


def diagnose_sleep_edf(cfg: dict, run_dir: Path, diagnose_dir: Path) -> Dict[str, object]:
    task = get_task_name(cfg, "sleep_edf_5class")
    labels = get_labels(task)
    num_classes = int(get_num_classes(task))
    if num_classes != 5:
        raise RuntimeError("Sleep-EDF 诊断要求 5 类标签。")

    processed_dir = Path(cfg["processed_dir"])
    cache_status = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    if not cache_status.get("reuse_available", False):
        reason = str(cache_status.get("reason", "unknown"))
        manifest_path = str(cache_status.get("manifest_path", processed_dir / "preprocess_manifest.json"))
        raise RuntimeError(
            "检测到 processed 数据与预处理代码或关键配置不一致，"
            "请先重新运行 preprocess_sleep_edf.py。"
            f" reason={reason} manifest={manifest_path}"
        )

    records, entries = _load_sleep_edf_records(processed_dir, num_classes=num_classes)
    if not records:
        raise RuntimeError("未找到 Sleep-EDF 处理后数据，请先预处理。")
    alignment = _sleep_edf_label_alignment(records, num_classes=num_classes)

    split_cfg = cfg.get("split", {})
    split_constraints = cfg.get("split_constraints", {})
    protocol = str(split_cfg.get("protocol", cfg.get("protocol", cfg.get("split_mode", "subject_kfold")))).lower().strip()
    if protocol == "kfold":
        protocol = "subject_kfold"

    val_ratio = float(cfg.get("val_ratio", split_cfg.get("val_ratio", 0.1)))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    seed = int(cfg.get("seed", 42))

    split_defs: List[Dict[str, object]] = []
    if protocol == "epoch_random":
        key_to_entry = {f"{e.record_id}:{e.epoch_idx}": e for e in entries}
        split_one = make_epoch_random_split(list(key_to_entry.keys()), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

        def by_keys(keys: List[str]) -> List[EpochEntry]:
            return [key_to_entry[k] for k in keys]

        split_defs.append(
            {
                "protocol": "epoch_random",
                "train_entries": by_keys(split_one["train"]),
                "val_entries": by_keys(split_one["val"]),
                "test_entries": by_keys(split_one["test"]),
            }
        )
    else:
        subject_label_counts: Dict[str, np.ndarray] = {}
        for e in entries:
            if e.subject_id not in subject_label_counts:
                subject_label_counts[e.subject_id] = np.zeros(num_classes, dtype=np.int64)
            subject_label_counts[e.subject_id][e.label] += 1

        subject_ids = sorted(subject_label_counts.keys())
        subset_tag = normalize_edf_subset(cfg.get("dataset", {}).get("edf_subset", cfg.get("edf_subset", "edfx_all")))
        k_default = int(default_kfold_by_subset(subject_ids, edf_subset=subset_tag))
        kfold = int(cfg.get("kfold", k_default))
        if kfold <= 0:
            kfold = k_default

        min_count = int(split_constraints.get("min_count_per_class", split_constraints.get("min_count_floor", 1)))
        max_tries = int(split_constraints.get("max_tries", 200))
        subject_splits = make_kfold_splits(
            subject_ids,
            kfold=kfold,
            val_ratio=val_ratio,
            seed=seed,
            label_counts=subject_label_counts,
            num_classes=num_classes,
            min_count_per_class=min_count,
            max_tries=max_tries,
            protocol="subject_kfold",
            edf_subset=subset_tag,
        )

        for sp in subject_splits:
            train_subjects = set(sp["train"])
            val_subjects = set(sp["val"])
            test_subjects = set(sp["test"])
            split_defs.append(
                {
                    "protocol": "subject_kfold",
                    "train": sorted(train_subjects),
                    "val": sorted(val_subjects),
                    "test": sorted(test_subjects),
                    "train_entries": [e for e in entries if e.subject_id in train_subjects],
                    "val_entries": [e for e in entries if e.subject_id in val_subjects],
                    "test_entries": [e for e in entries if e.subject_id in test_subjects],
                }
            )

    class_rows = _build_class_rows_from_entries(split_defs, labels=labels, num_classes=num_classes)
    floor_value = int(split_constraints.get("min_count_per_class", split_constraints.get("min_count_floor", 1)))
    recommendation = _recommend_min_count(class_rows, floor_value=floor_value)
    min_count_used = int(recommendation["recommended_min_count"])

    feasible_count = 0
    for row in class_rows:
        val_ok = int(row["min_val_count"]) >= min_count_used
        test_ok = int(row["min_test_count"]) >= min_count_used
        row["min_count_used"] = min_count_used
        row["recommended_min_count"] = min_count_used
        row["feasible"] = bool(val_ok and test_ok)
        row["feasible_by_min_count"] = bool(val_ok and test_ok)
        row["labels_valid"] = bool(alignment["invalid_label_count"] == 0)
        row["alignment_ok"] = bool(alignment["alignment_ok"])
        if row["feasible"]:
            feasible_count += 1

    recommendation.update(
        {
            "num_splits": int(len(class_rows)),
            "feasible_split_count": int(feasible_count),
            "infeasible_split_count": int(len(class_rows) - feasible_count),
        }
    )

    pd.DataFrame(class_rows).to_csv(diagnose_dir / "class_counts_by_split.csv", index=False, **csv_utf8_sig_kwargs())
    pd.DataFrame(alignment["samples"]).to_csv(diagnose_dir / "label_alignment_samples.csv", index=False, **csv_utf8_sig_kwargs())
    (diagnose_dir / "label_alignment_summary.json").write_text(
        json.dumps({k: v for k, v in alignment.items() if k != "samples"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (diagnose_dir / "recommended_min_count.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{"key": k, "value": v} for k, v in recommendation.items()]).to_csv(
        diagnose_dir / "recommended_min_count.csv", index=False, **csv_utf8_sig_kwargs()
    )

    print(
        f"diagnose: dataset=sleep_edf protocol={protocol} splits={len(class_rows)} "
        f"feasible={feasible_count} recommended_min_count={min_count_used} "
        f"alignment_ok={alignment['alignment_ok']}"
    )

    return {
        "dataset": "sleep_edf",
        "protocol": protocol,
        "preprocess_cache_status": {
            "reason": str(cache_status.get("reason", "")),
            "manifest_path": str(cache_status.get("manifest_path", "")),
            "fingerprint_hash": str(cache_status.get("fingerprint_hash", "")),
        },
        "class_counts_by_split": class_rows,
        "recommended_min_count": recommendation,
        "label_alignment": {k: v for k, v in alignment.items() if k != "samples"},
        "warnings": [],
        "errors": [] if alignment["alignment_ok"] else ["Sleep-EDF 标签映射审计失败或缺少 epoch_stage_desc。"],
    }


def _resolve_default_config_path(dataset: str, task: str | None) -> Path:
    _ = dataset
    _ = task
    return Path("configs/sleep_edf_5class.yaml")


def main() -> None:
    setup_utf8_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sleep_edf", choices=["auto", "sleep_edf"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else _resolve_default_config_path(args.dataset, args.task)
    cfg = read_yaml(cfg_path)

    dataset = "sleep_edf"
    runs_dir = Path(cfg["runs_dir"])
    exp_name = "sleep_edf"
    pending = read_pending_run(runs_dir, exp_name)
    if args.run_dir:
        run_dir = ensure_dir(Path(args.run_dir))
    elif pending is not None and pending.exists():
        run_dir = pending
    else:
        run_dir = build_run_dir(runs_dir, exp_name)
        write_pending_run(runs_dir, run_dir, exp_name)
    diagnose_dir = ensure_dir(run_dir / "diagnose")

    cfg["task"] = get_task_name(cfg, "sleep_edf_5class")
    diagnose = diagnose_sleep_edf(cfg, run_dir, diagnose_dir)

    (diagnose_dir / "diagnose.json").write_text(json.dumps(diagnose, ensure_ascii=False, indent=2), encoding="utf-8")
    if diagnose.get("errors"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

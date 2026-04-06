# -*- coding: utf-8 -*-
"""Generate a local audit index for the Sleep-EDF pipeline."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
from datetime import datetime

from preprocess_sleep_edf import get_preprocess_cache_status
from eco_sleep import get_num_classes, get_task_name
from eco_sleep.data.sleep_edf.storage import list_processed_records, records_signature
from eco_sleep.utils.encoding_fix import setup_utf8_stdio
from eco_sleep.utils.io import ensure_dir, read_yaml, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE_GROUPS = {
    "scripts": [
        "scripts/preprocess_sleep_edf.py",
        "scripts/diagnose_data.py",
        "scripts/train_sleep_edf.py",
        "scripts/eval_sleep_edf.py",
        "scripts/run_all_sleep_edf.ps1",
        "scripts/compile_check.py",
        "scripts/quick_smoke.py",
        "scripts/audit_sleep_edf_pipeline.py",
    ],
    "data_sleep_edf": [
        "src/eco_sleep/data/sleep_edf/annotations.py",
        "src/eco_sleep/data/sleep_edf/dataset.py",
        "src/eco_sleep/data/sleep_edf/edf_reader.py",
        "src/eco_sleep/data/sleep_edf/labels.py",
        "src/eco_sleep/data/sleep_edf/preprocessing.py",
        "src/eco_sleep/data/sleep_edf/splits.py",
        "src/eco_sleep/data/sleep_edf/storage.py",
    ],
    "models": [
        "src/eco_sleep/models/__init__.py",
        "src/eco_sleep/models/picosleepnet_baseline.py",
        "src/eco_sleep/models/picosleepnet_plus_snn.py",
        "src/eco_sleep/models/losses.py",
    ],
    "train_utils": [
        "src/eco_sleep/train/checkpoints.py",
        "src/eco_sleep/train/evaluate.py",
        "src/eco_sleep/train/trainer.py",
    ],
    "runtime_utils": [
        "src/eco_sleep/utils/encoding_fix.py",
        "src/eco_sleep/utils/io.py",
        "src/eco_sleep/utils/model_complexity.py",
        "src/eco_sleep/utils/plots.py",
        "src/eco_sleep/utils/bland_altman.py",
        "src/eco_sleep/utils/logger.py",
    ],
}

DEPENDENCIES = {
    "scripts/preprocess_sleep_edf.py": [
        "src/eco_sleep/data/sleep_edf/edf_reader.py",
        "src/eco_sleep/data/sleep_edf/preprocessing.py",
        "src/eco_sleep/data/sleep_edf/labels.py",
        "src/eco_sleep/data/sleep_edf/storage.py",
    ],
    "scripts/diagnose_data.py": [
        "scripts/preprocess_sleep_edf.py",
        "src/eco_sleep/data/sleep_edf/labels.py",
        "src/eco_sleep/data/sleep_edf/splits.py",
    ],
    "scripts/train_sleep_edf.py": [
        "scripts/preprocess_sleep_edf.py",
        "src/eco_sleep/data/sleep_edf/storage.py",
        "src/eco_sleep/models/picosleepnet_baseline.py",
        "src/eco_sleep/models/picosleepnet_plus_snn.py",
        "src/eco_sleep/models/losses.py",
        "src/eco_sleep/train/checkpoints.py",
        "src/eco_sleep/train/evaluate.py",
    ],
    "scripts/eval_sleep_edf.py": [
        "scripts/preprocess_sleep_edf.py",
        "src/eco_sleep/train/checkpoints.py",
        "src/eco_sleep/train/evaluate.py",
        "src/eco_sleep/utils/plots.py",
        "src/eco_sleep/utils/bland_altman.py",
        "src/eco_sleep/utils/model_complexity.py",
    ],
    "scripts/run_all_sleep_edf.ps1": [
        "scripts/compile_check.py",
        "scripts/preprocess_sleep_edf.py",
        "scripts/diagnose_data.py",
        "scripts/train_sleep_edf.py",
        "scripts/eval_sleep_edf.py",
    ],
}

RISK_POINTS = [
    "processed npz 字段名与 dataset 读取不一致，例如 labels/label、raw_epoch、lcs_pos_count 等错配。",
    "checkpoint 未保存 model_name/model_hparams/task/num_classes/split_id 时，eval 重建模型会发生 state_dict mismatch。",
    "processed_dir 中旧缓存与当前预处理代码/关键配置混用，必须依赖 preprocess manifest 检出并阻止训练/评估。",
    "标签映射来源不统一会导致 preprocess、diagnose、train、eval 的 num_classes 或 label range 不一致。",
    "空 CSV、0 字节 CSV、只有表头的 CSV 如果直接 read_csv，审计脚本容易崩溃。",
    "Windows 下 DataLoader 若偏离 num_workers=0、persistent_workers=False、pin_memory=True，容易引入卡死或行为漂移。",
    "无 GPU 且 allow_cpu!=true 时，训练必须直接中文报错退出，不能静默退回 CPU。",
]


def _build_file_index() -> list[dict]:
    rows: list[dict] = []
    for group, rel_paths in FILE_GROUPS.items():
        for rel_path in rel_paths:
            abs_path = PROJECT_ROOT / rel_path
            rows.append(
                {
                    "group": group,
                    "path": rel_path,
                    "exists": abs_path.exists(),
                    "dependencies": DEPENDENCIES.get(rel_path, []),
                }
            )
    return rows


def main() -> None:
    setup_utf8_stdio()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    processed_dir = Path(cfg["processed_dir"])
    cache_status = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    processed_records = list_processed_records(processed_dir, num_classes=num_classes) if processed_dir.exists() else []

    output_path = Path(args.output) if args.output else (PROJECT_ROOT / "runs_dev_snapshot" / "audit_index.json")
    ensure_dir(output_path.parent)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "config_path": str(Path(args.config)),
        "task": task,
        "num_classes": num_classes,
        "default_model": str(cfg.get("model", {}).get("name", "")),
        "raw_dir": str(cfg.get("raw_dir", "")),
        "processed_dir": str(processed_dir),
        "runs_dir": str(cfg.get("runs_dir", "")),
        "file_index": _build_file_index(),
        "risk_points": RISK_POINTS,
        "preprocess_cache_status": {
            "reuse_available": bool(cache_status.get("reuse_available", False)),
            "reason": str(cache_status.get("reason", "")),
            "manifest_path": str(cache_status.get("manifest_path", "")),
            "fingerprint_hash": str(cache_status.get("fingerprint_hash", "")),
        },
        "processed_records": records_signature(processed_records),
    }
    save_json(output_path, payload)

    if args.run_dir:
        run_audit_path = ensure_dir(Path(args.run_dir) / "audit") / "audit_index.json"
        save_json(run_audit_path, payload)

    print(f"audit_index: {output_path}")


if __name__ == "__main__":
    main()

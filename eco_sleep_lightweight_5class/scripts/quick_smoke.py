# -*- coding: utf-8 -*-
"""Quick smoke for Sleep-EDF manifest, dataset, checkpoint, and eval outputs."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import copy
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess_sleep_edf import PREPROCESS_MANIFEST_NAME, build_preprocess_fingerprint, get_preprocess_cache_status
from eval_sleep_edf import SleepEdfSpikeDataset as EvalSleepEdfSpikeDataset
from eval_sleep_edf import _build_model_from_ckpt, _evaluate_metrics, _save_result_bundle
from train_sleep_edf import SleepEdfSpikeDataset, _build_main_loss, _build_model, build_epoch_entries, load_records
from eco_sleep import get_num_classes, get_task_name
from eco_sleep.train import run_inference
from eco_sleep.train.checkpoints import load_checkpoint, save_checkpoint
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio, suppress_pin_memory_warning
from eco_sleep.utils.io import read_yaml, safe_read_csv
from eco_sleep.utils.plots import ensure_chinese_font


def _fingerprint_hash(fingerprint: dict) -> str:
    payload = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_record(record_id: str, subject_id: str, labels: np.ndarray, delta: float) -> dict:
    n_epoch = int(labels.shape[0])
    time_steps = 3000
    rng = np.random.default_rng(abs(hash(record_id)) % (2**32))
    raw_epoch = rng.normal(0.0, 1.0, size=(n_epoch, time_steps)).astype(np.float32)
    signals = raw_epoch[:, None, :]
    pos = np.clip(np.round(np.maximum(np.diff(np.pad(raw_epoch, ((0, 0), (1, 0))), axis=1), 0) / max(delta, 1e-6)), 0, 4).astype(np.int16)
    neg = -np.clip(np.round(np.maximum(-np.diff(np.pad(raw_epoch, ((0, 0), (1, 0))), axis=1), 0) / max(delta, 1e-6)), 0, 4).astype(np.int16)
    stage_names = np.asarray(["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage R"], dtype=object)
    meta = {
        "schema_version": 2,
        "subject_id": subject_id,
        "record_id": record_id,
        "cohort": "SC",
        "subset": "edf20",
        "pair_key": record_id[:7],
        "epoch_seconds": 30,
        "channels": ["Fpz-Cz"],
        "resample_hz": 100,
        "crop_policy": "30min",
        "lcs_delta": float(delta),
        "label_mapping": "W=0,N1=1,N2=2,N3/N4=3,REM=4",
    }
    return {
        "signals": signals.astype(np.float32),
        "labels": labels.astype(np.int64),
        "label": labels.astype(np.int64),
        "raw_epoch": raw_epoch.astype(np.float32),
        "lcs_pos_count": pos,
        "lcs_neg_count": neg,
        "lcs_pos": (pos > 0).astype(np.uint8),
        "lcs_neg": (neg < 0).astype(np.uint8),
        "epoch_stage_desc": stage_names[labels].astype("U32"),
        "meta": json.dumps(meta, ensure_ascii=False),
    }


def _write_manifest(processed_dir: Path, cfg: dict, record_ids: list[str]) -> None:
    fingerprint = build_preprocess_fingerprint(cfg, raw_dir_cfg=str(cfg["raw_dir"]))
    manifest = {
        "manifest_version": 1,
        "schema_version": 2,
        "processed_dir": str(processed_dir),
        "fingerprint": fingerprint,
        "fingerprint_hash": _fingerprint_hash(fingerprint),
        "saved_record_ids": sorted(record_ids),
        "saved_record_count": len(record_ids),
        "summary": {"valid_saved_records": len(record_ids), "newly_saved": len(record_ids), "reused_existing": 0},
        "skip_rows": [],
    }
    (processed_dir / PREPROCESS_MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    setup_utf8_stdio()
    suppress_pin_memory_warning()
    ensure_chinese_font()

    root = Path(__file__).resolve().parents[1]
    work_dir = root / "runs" / "quick_smoke" / "sleep_edf_smoke"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    processed_dir = work_dir / "processed"
    raw_dir = work_dir / "raw_placeholder"
    eval_dir = work_dir / "eval" / "split_0"
    train_dir = work_dir / "train" / "split_0"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_yaml(root / "configs" / "sleep_edf_5class.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["raw_dir"] = str(raw_dir)
    cfg["processed_dir"] = str(processed_dir)
    cfg["runs_dir"] = str(work_dir / "runs")
    cfg.setdefault("dataset", {})
    cfg["dataset"]["edf_subset"] = "edf20"
    cfg.setdefault("split", {})
    cfg["split"]["protocol"] = "epoch_random"
    cfg["split"]["test_ratio"] = 0.2
    cfg["val_ratio"] = 0.2
    cfg["batch_size"] = 8
    cfg["allow_cpu"] = True
    cfg.setdefault("model", {})
    cfg["model"]["name"] = "picosleepnet_baseline"
    cfg["task"] = get_task_name(cfg, "sleep_edf_5class")

    num_classes = int(get_num_classes(cfg["task"]))
    delta = float(cfg.get("lcs", {}).get("delta", 0.13))
    record_ids: list[str] = []
    for i in range(5):
        record_id = f"SC40{i+1}1E0"
        subject_id = f"SC40{i+1}"
        labels = np.tile(np.arange(num_classes, dtype=np.int64), 6)
        payload = _make_record(record_id, subject_id, labels=labels, delta=delta)
        np.savez_compressed(processed_dir / f"{record_id}.npz", **payload)
        record_ids.append(record_id)

    _write_manifest(processed_dir, cfg, record_ids=record_ids)

    status_ok = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    if not status_ok.get("reuse_available", False):
        raise RuntimeError(f"manifest 复用判定失败: {status_ok}")
    cfg_changed = copy.deepcopy(cfg)
    cfg_changed.setdefault("lcs", {})
    cfg_changed["lcs"]["delta"] = float(delta + 0.01)
    status_changed = get_preprocess_cache_status(cfg_changed, out_dir=processed_dir, num_classes=num_classes)
    if status_changed.get("reuse_available", False):
        raise RuntimeError("manifest 变化后仍被错误判定为可复用。")

    empty_csv = work_dir / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    empty_df = safe_read_csv(empty_csv, default_columns=["record_id", "reason"])
    if list(empty_df.columns) != ["record_id", "reason"] or not empty_df.empty:
        raise RuntimeError("safe_read_csv 未正确处理空 CSV。")

    records = load_records(processed_dir)
    entries = build_epoch_entries(records, num_classes=num_classes)
    train_entries = entries[:80]
    test_entries = entries[80:100]
    if len(test_entries) < 10:
        raise RuntimeError("quick_smoke 切分样本过少。")

    device = torch.device("cpu")
    model, model_hparams = _build_model(cfg=cfg, model_name="picosleepnet_baseline", num_classes=num_classes, subset="edf20")
    model.to(device)
    train_ds = SleepEdfSpikeDataset(train_entries, "picosleepnet_baseline", False, True, delta, delta, delta, "mem")
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    loss_fn, _ = _build_main_loss(cfg=cfg, labels=train_ds.all_labels, num_classes=num_classes, sampler_name="weighted", device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(train_loader))
    x = batch[0].to(device)
    y = batch[1].to(device)
    outputs = model(x)
    logits = outputs["main"] if isinstance(outputs, dict) else outputs
    if int(logits.shape[1]) != num_classes:
        raise RuntimeError(f"logits 维度错误: {tuple(logits.shape)}")
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    ckpt_path = train_dir / "best.ckpt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        None,
        cfg,
        epoch=1,
        best_metric=0.0,
        model_name="picosleepnet_baseline",
        model_hparams=model_hparams,
        task=cfg["task"],
        num_classes=num_classes,
        split_id=0,
        trial_id=0,
    )
    ckpt = load_checkpoint(ckpt_path)
    rebuilt_model, rebuilt_name, rebuilt_hparams, rebuilt_num_classes = _build_model_from_ckpt(ckpt, train_dir, num_classes)
    if rebuilt_name != "picosleepnet_baseline" or rebuilt_num_classes != num_classes or not rebuilt_hparams:
        raise RuntimeError("checkpoint 重建模型失败。")

    test_ds = EvalSleepEdfSpikeDataset(test_entries, rebuilt_name, False, True, delta, delta, delta, "mem")
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
    rebuilt_model.to(device)
    result = run_inference(rebuilt_model, test_loader, device, mixed_precision=False, non_blocking=True, return_logits=True)
    _save_result_bundle(eval_dir, 0, "raw", result, ["W", "N1", "N2", "N3", "REM"], num_classes, wake_label=0, rem_label=4, primary=True)
    metrics = _evaluate_metrics(result["y_true"], result["y_pred"], result["y_prob"], num_classes, wake_label=0, rem_label=4)
    pd.DataFrame(
        [
            {
                "split": 0,
                "result_tag": "raw",
                "model_name": rebuilt_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "kappa": metrics["kappa"],
            }
        ]
    ).to_csv(work_dir / "eval" / "summary_metrics.csv", index=False, **csv_utf8_sig_kwargs())

    required_paths = [
        ckpt_path,
        eval_dir / "confusion_matrix.png",
        eval_dir / "classification_report.csv",
        work_dir / "eval" / "summary_metrics.csv",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"quick_smoke 产物缺失: {missing}")

    print(f"quick_smoke: 通过，输出目录={work_dir}")


if __name__ == "__main__":
    main()

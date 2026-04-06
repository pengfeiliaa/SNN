# -*- coding: utf-8 -*-
"""Sleep-EDF evaluation with stable merged outputs and checkpoint-safe rebuild."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from preprocess_sleep_edf import get_preprocess_cache_status
from eco_sleep import get_labels, get_num_classes, get_task_name, get_wake_label
from eco_sleep.data.sleep_edf.preprocessing import default_lcs_delta, lcs_counts_to_binary, lcs_encode_epoch_counts
from eco_sleep.data.sleep_edf.storage import list_processed_records, load_labels_from_npz, safe_meta
from eco_sleep.models import ContextPicoSNN, ContextPicoSNNV2, PicoSleepNetBaseline, PicoSleepNetPlusSNN
from eco_sleep.train import run_inference
from eco_sleep.train.checkpoints import load_checkpoint, validate_checkpoint_metadata
from eco_sleep.utils.bland_altman import plot_bland_altman
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio, suppress_pin_memory_warning
from eco_sleep.utils.io import ensure_dir, latest_run, read_last_run, read_yaml, save_json
from eco_sleep.utils.metrics_walch2019 import walch_binary_metrics, walch_multiclass_metrics
from eco_sleep.utils.model_complexity import build_complexity_metrics
from eco_sleep.utils.plots import ensure_chinese_font, plot_confusion_matrix, plot_curves
from eco_sleep.utils.roc_pr import multiclass_curves
from eco_sleep.utils.sleep_stats import build_sleep_stats_table


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN_DIR = (REPO_ROOT / "runs" / "20260401_171526_sleep_edf").resolve()
BASELINE_SELECTED_METRICS = {
    0: {"result_tag": "smoothed", "accuracy": 0.7154561017363659, "macro_f1": 0.664538395512947, "kappa": 0.6282419664993963},
    1: {"result_tag": "raw", "accuracy": 0.7410256410256411, "macro_f1": 0.6770970435274625, "kappa": 0.6428224073332747},
    2: {"result_tag": "smoothed", "accuracy": 0.7638501102130786, "macro_f1": 0.7109003211574779, "kappa": 0.6906969097649673},
    3: {"result_tag": "raw", "accuracy": 0.7958273192494423, "macro_f1": 0.7199244814688341, "kappa": 0.7276011548909282},
    4: {"result_tag": "raw", "accuracy": 0.6996180005590236, "macro_f1": 0.6494139151561946, "kappa": 0.6066818677300614},
}
BASELINE_MEAN_METRICS = {
    "mean_acc": 0.7431554345567103,
    "mean_macro_f1": 0.6843748313645832,
    "mean_kappa": 0.6592088612437255,
}


@dataclass(frozen=True)
class EpochEntry:
    path: str
    subject_id: str
    record_id: str
    epoch_idx: int
    label: int


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _ensure_not_baseline_run_dir(path: Path, purpose: str) -> None:
    resolved = path.resolve()
    if resolved == BASELINE_RUN_DIR or BASELINE_RUN_DIR in resolved.parents:
        raise RuntimeError(f"{purpose} points to the locked baseline run_dir and is forbidden: {resolved}")


def _parse_split_indices(raw_value: str | None) -> set[int] | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    out: set[int] = set()
    for part in str(raw_value).split(","):
        item = part.strip()
        if item:
            out.add(int(item))
    return out or None


def _valid_run_candidates(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    candidates = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        score = 0
        if (path / "config.yaml").exists():
            score += 1
        if (path / "splits.json").exists():
            score += 2
        if (path / "train").exists():
            score += 2
        if score > 0:
            candidates.append(path)
    return candidates


def _resolve_run_dir(runs_dir: Path, explicit_run_dir: str | None) -> Path | None:
    if explicit_run_dir:
        candidate = _resolve_repo_path(explicit_run_dir)
        if candidate is None or not candidate.exists():
            raise RuntimeError(f"run_dir not found: {explicit_run_dir}")
        return candidate

    last_run = read_last_run(runs_dir)
    if last_run is not None and last_run.exists():
        return last_run

    candidates = _valid_run_candidates(runs_dir)
    if not candidates:
        fallback = latest_run(runs_dir)
        return fallback if fallback is not None and fallback.exists() else None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_meta(meta_raw: object) -> dict:
    return safe_meta(meta_raw)


def _canonical_model_name(model_name: str | None) -> str:
    raw = str(model_name or "").lower().strip()
    if raw in {"picosleepnet", "picosleepnet_baseline", "picosleepnet_rsnn"}:
        return "picosleepnet_baseline"
    if raw in {"context_pico_snn", "context_pico", "picosleepnet_lite_snn", "multiscale_pico_snn"}:
        return "context_pico_snn"
    if raw in {"", "context_pico_snn_v2", "context_pico_v2", "multiscale_context_pico_snn"}:
        return "context_pico_snn_v2"
    if raw == "picosleepnet_plus_snn":
        return "picosleepnet_plus_snn"
    return raw


def load_records(processed_dir: Path) -> List[Dict[str, object]]:
    return [
        {
            "path": str(row.path),
            "subject_id": row.subject_id,
            "record_id": row.record_id,
            "is_legacy": bool(row.is_legacy),
        }
        for row in list_processed_records(processed_dir)
    ]


def build_entries(records: List[Dict[str, object]], num_classes: int) -> List[EpochEntry]:
    entries: List[EpochEntry] = []
    for rec in records:
        path = Path(str(rec["path"]))
        labels = load_labels_from_npz(path)
        bad = np.where((labels < 0) | (labels >= num_classes))[0]
        if bad.size > 0:
            idx = int(bad[0])
            raise RuntimeError(
                f"label out of range: file={path} epoch_index={idx} "
                f"label={int(labels[idx])} expected=[0,{num_classes - 1}]"
            )
        for idx, label in enumerate(labels.tolist()):
            entries.append(
                EpochEntry(
                    path=str(path),
                    subject_id=str(rec["subject_id"]),
                    record_id=str(rec["record_id"]),
                    epoch_idx=int(idx),
                    label=int(label),
                )
            )
    return entries


def _entry_counts(entries: List[EpochEntry], num_classes: int) -> np.ndarray:
    if not entries:
        return np.zeros(num_classes, dtype=np.int64)
    return np.bincount([int(entry.label) for entry in entries], minlength=num_classes).astype(np.int64)


def _json_counts(counts: np.ndarray, labels: List[str]) -> str:
    return json.dumps({labels[i]: int(counts[i]) for i in range(len(labels))}, ensure_ascii=False, separators=(",", ":"))


def _normalize_lcs_channels(
    x: np.ndarray,
    channel_mean: np.ndarray | None,
    channel_std: np.ndarray | None,
    mode: str,
    eps: float = 1e-5,
) -> np.ndarray:
    if str(mode).lower().strip() in {"", "none", "off"}:
        return x

    out = np.asarray(x, dtype=np.float32)
    if "log1p" in str(mode).lower():
        out = np.log1p(np.maximum(out, 0.0).astype(np.float32))

    if channel_mean is not None and channel_std is not None and "record" in str(mode).lower():
        mean = np.asarray(channel_mean, dtype=np.float32).reshape(-1, 1)
        std = np.asarray(channel_std, dtype=np.float32).reshape(-1, 1)
        return (out - mean) / np.maximum(std, float(eps))

    if "zscore" in str(mode).lower():
        mean = out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, keepdims=True)
        return (out - mean) / np.maximum(std, float(eps))
    return out


class SleepEdfSpikeDataset(Dataset):
    """Per-epoch LCS dataset for Sleep-EDF baseline and plus SNN models."""

    def __init__(
        self,
        entries: List[EpochEntry],
        model_name: str,
        use_dual_lcs: bool,
        use_integer_spike: bool,
        delta_primary: float,
        delta_small: float,
        delta_large: float,
        cache_mode: str = "mem",
        input_norm_mode: str = "none",
        input_norm_eps: float = 1e-5,
        context_len: int = 1,
    ) -> None:
        self.entries = list(entries)
        self.model_name = _canonical_model_name(model_name)
        self.use_dual_lcs = bool(use_dual_lcs)
        self.use_integer_spike = bool(use_integer_spike)
        self.delta_primary = float(delta_primary)
        self.delta_small = float(delta_small)
        self.delta_large = float(delta_large)
        self.cache_mode = str(cache_mode).lower().strip()
        self.input_norm_mode = str(input_norm_mode).lower().strip() or "none"
        self.input_norm_eps = float(max(input_norm_eps, 1e-6))
        self.context_len = int(max(1, context_len))
        if self.context_len % 2 == 0:
            raise ValueError("context_len must be odd for center-epoch evaluation.")
        self.context_half = self.context_len // 2
        self._record_cache: Dict[str, Dict[str, object]] = {}
        self._dynamic_lcs_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self._channel_stats_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._raw_stats_cache: Dict[str, Tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _load_record(self, path: str) -> Dict[str, object]:
        if path in self._record_cache:
            return self._record_cache[path]

        npz = np.load(path, allow_pickle=True)
        meta = _safe_meta(npz["meta"].item() if "meta" in npz.files else {})
        labels = npz["label"].astype(np.int64) if "label" in npz.files else npz["labels"].astype(np.int64)
        raw_epoch = (
            npz["raw_epoch"].astype(np.float32)
            if "raw_epoch" in npz.files
            else npz["signals"].astype(np.float32)[:, 0, :]
        )
        if labels.shape[0] != raw_epoch.shape[0]:
            raise RuntimeError(
                f"processed lengths mismatch: file={path} labels={labels.shape[0]} raw_epoch={raw_epoch.shape[0]}"
            )

        record = {
            "raw_epoch": raw_epoch,
            "lcs_pos_count": npz["lcs_pos_count"].astype(np.int16) if "lcs_pos_count" in npz.files else None,
            "lcs_neg_count": npz["lcs_neg_count"].astype(np.int16) if "lcs_neg_count" in npz.files else None,
            "lcs_pos": npz["lcs_pos"].astype(np.uint8) if "lcs_pos" in npz.files else None,
            "lcs_neg": npz["lcs_neg"].astype(np.uint8) if "lcs_neg" in npz.files else None,
            "lcs_delta": float(meta.get("lcs_delta", self.delta_primary)),
        }
        if self.cache_mode == "mem":
            self._record_cache[path] = record
        return record

    def _compute_lcs_matrix(self, raw_epoch: np.ndarray, delta: float) -> Dict[str, np.ndarray]:
        n_epoch, n_t = raw_epoch.shape
        pos_count = np.zeros((n_epoch, n_t), dtype=np.int16)
        neg_count = np.zeros((n_epoch, n_t), dtype=np.int16)
        pos_bin = np.zeros((n_epoch, n_t), dtype=np.uint8)
        neg_bin = np.zeros((n_epoch, n_t), dtype=np.uint8)
        for idx in range(n_epoch):
            pos, neg = lcs_encode_epoch_counts(raw_epoch[idx], delta=delta)
            pos_b, neg_b = lcs_counts_to_binary(pos, neg)
            pos_count[idx], neg_count[idx] = pos, neg
            pos_bin[idx], neg_bin[idx] = pos_b, neg_b
        return {
            "lcs_pos_count": pos_count,
            "lcs_neg_count": neg_count,
            "lcs_pos": pos_bin,
            "lcs_neg": neg_bin,
        }

    def _get_lcs(self, path: str, record: Dict[str, object], delta: float) -> Dict[str, np.ndarray]:
        has_precomputed = all(record.get(key) is not None for key in ("lcs_pos_count", "lcs_neg_count", "lcs_pos", "lcs_neg"))
        if has_precomputed and abs(float(record.get("lcs_delta", self.delta_primary)) - float(delta)) < 1e-8:
            return {key: record[key] for key in ("lcs_pos_count", "lcs_neg_count", "lcs_pos", "lcs_neg")}

        cache_key = (path, f"{float(delta):.6f}")
        if cache_key not in self._dynamic_lcs_cache:
            raw_epoch = np.asarray(record["raw_epoch"], dtype=np.float32)
            self._dynamic_lcs_cache[cache_key] = self._compute_lcs_matrix(raw_epoch, delta=float(delta))
        return self._dynamic_lcs_cache[cache_key]

    def _record_channel_stats(self, path: str, stat_key: str, arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = (path, stat_key)
        if cache_key in self._channel_stats_cache:
            return self._channel_stats_cache[cache_key]

        means = []
        stds = []
        for arr in arrays:
            values = np.asarray(arr, dtype=np.float32)
            if "log1p" in self.input_norm_mode:
                values = np.log1p(np.maximum(values, 0.0).astype(np.float32))
            means.append(float(values.mean()))
            stds.append(float(max(values.std(), self.input_norm_eps)))
        stats = (np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32))
        self._channel_stats_cache[cache_key] = stats
        return stats

    def _record_raw_stats(self, path: str, raw_epoch: np.ndarray) -> Tuple[float, float]:
        if path in self._raw_stats_cache:
            return self._raw_stats_cache[path]
        values = np.asarray(raw_epoch, dtype=np.float32)
        stats = (float(values.mean()), float(max(values.std(), self.input_norm_eps)))
        self._raw_stats_cache[path] = stats
        return stats

    def _normalize_raw_context(self, path: str, raw_context: np.ndarray, raw_epoch: np.ndarray) -> np.ndarray:
        mode = self.input_norm_mode
        x = np.asarray(raw_context, dtype=np.float32)
        if mode in {"", "none", "off"}:
            return x
        if "record" in mode:
            mean, std = self._record_raw_stats(path, raw_epoch)
            return (x - mean) / std
        if "epoch" in mode or "zscore" in mode:
            mean = x.mean(axis=-1, keepdims=True)
            std = np.maximum(x.std(axis=-1, keepdims=True), self.input_norm_eps)
            return (x - mean) / std
        return x

    def _compose_raw_context(self, path: str, record: Dict[str, object], epoch_idx: int) -> np.ndarray:
        raw_epoch = np.asarray(record["raw_epoch"], dtype=np.float32)
        indices = []
        for offset in range(-self.context_half, self.context_half + 1):
            index = min(max(epoch_idx + offset, 0), raw_epoch.shape[0] - 1)
            indices.append(index)
        context = raw_epoch[np.asarray(indices, dtype=np.int64)]
        context = self._normalize_raw_context(path, context, raw_epoch)
        return context[:, None, :]

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        record = self._load_record(entry.path)
        epoch_idx = int(entry.epoch_idx)

        if self.model_name in {"context_pico_snn", "context_pico_snn_v2"}:
            x = self._compose_raw_context(entry.path, record, epoch_idx)
        elif self.model_name == "picosleepnet_plus_snn" and self.use_dual_lcs:
            small = self._get_lcs(entry.path, record, self.delta_small)
            large = self._get_lcs(entry.path, record, self.delta_large)
            if self.use_integer_spike:
                pos_s = np.asarray(small["lcs_pos_count"][epoch_idx], dtype=np.float32)
                neg_s = np.abs(np.asarray(small["lcs_neg_count"][epoch_idx], dtype=np.float32))
                pos_l = np.asarray(large["lcs_pos_count"][epoch_idx], dtype=np.float32)
                neg_l = np.abs(np.asarray(large["lcs_neg_count"][epoch_idx], dtype=np.float32))
            else:
                pos_s = np.asarray(small["lcs_pos"][epoch_idx], dtype=np.float32)
                neg_s = np.asarray(small["lcs_neg"][epoch_idx], dtype=np.float32)
                pos_l = np.asarray(large["lcs_pos"][epoch_idx], dtype=np.float32)
                neg_l = np.asarray(large["lcs_neg"][epoch_idx], dtype=np.float32)
            x = np.stack([pos_s, neg_s, pos_l, neg_l], axis=0)
            if self.input_norm_mode != "none":
                mean, std = self._record_channel_stats(
                    entry.path,
                    stat_key=f"dual:{float(self.delta_small):.6f}:{float(self.delta_large):.6f}:{self.use_integer_spike}",
                    arrays=[
                        np.asarray(small["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                        np.abs(np.asarray(small["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                        np.asarray(large["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                        np.abs(np.asarray(large["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                    ],
                )
                x = _normalize_lcs_channels(x, channel_mean=mean, channel_std=std, mode=self.input_norm_mode, eps=self.input_norm_eps)
        else:
            lcs = self._get_lcs(entry.path, record, self.delta_primary)
            if self.use_integer_spike:
                pos = np.asarray(lcs["lcs_pos_count"][epoch_idx], dtype=np.float32)
                neg = np.abs(np.asarray(lcs["lcs_neg_count"][epoch_idx], dtype=np.float32))
            else:
                pos = np.asarray(lcs["lcs_pos"][epoch_idx], dtype=np.float32)
                neg = np.asarray(lcs["lcs_neg"][epoch_idx], dtype=np.float32)
            x = np.stack([pos, neg], axis=0)
            if self.input_norm_mode != "none":
                mean, std = self._record_channel_stats(
                    entry.path,
                    stat_key=f"single:{float(self.delta_primary):.6f}:{self.use_integer_spike}",
                    arrays=[
                        np.asarray(lcs["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                        np.abs(np.asarray(lcs["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                    ],
                )
                x = _normalize_lcs_channels(x, channel_mean=mean, channel_std=std, mode=self.input_norm_mode, eps=self.input_norm_eps)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(int(entry.label), dtype=torch.long),
            entry.record_id,
            epoch_idx,
        )


def _model_from_name(model_name: str, hparams: dict, num_classes: int):
    canonical = _canonical_model_name(model_name)
    if canonical == "context_pico_snn_v2":
        model_cls = ContextPicoSNNV2
    elif canonical == "picosleepnet_plus_snn":
        model_cls = PicoSleepNetPlusSNN
    elif canonical == "context_pico_snn":
        model_cls = ContextPicoSNN
    else:
        model_cls = PicoSleepNetBaseline
    allowed = set(inspect.signature(model_cls.__init__).parameters.keys()) - {"self"}
    kwargs = {key: value for key, value in dict(hparams).items() if key in allowed}
    kwargs["num_classes"] = int(num_classes)
    return model_cls(**kwargs)


def _build_model_from_ckpt(ckpt: dict, split_dir: Path, default_num_classes: int):
    validate_checkpoint_metadata(ckpt)
    model_name = _canonical_model_name(ckpt.get("model_name", ""))
    model_hparams = ckpt.get("model_hparams", {})
    num_classes = int(ckpt.get("num_classes", default_num_classes) or default_num_classes)
    if not model_name or not isinstance(model_hparams, dict) or not model_hparams:
        raise RuntimeError(f"checkpoint missing model metadata, please retrain. split_dir={split_dir}")
    model = _model_from_name(model_name, model_hparams, num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, model_name, model_hparams, num_classes


def _entries_for_split(split: Dict[str, object], all_entries: List[EpochEntry]) -> Dict[str, List[EpochEntry]]:
    protocol = str(split.get("protocol", "subject_kfold")).lower().strip()
    if protocol == "epoch_random":
        key_to_entry = {f"{entry.record_id}:{entry.epoch_idx}": entry for entry in all_entries}
        return {
            name: [key_to_entry[key] for key in list(split.get(f"{name}_samples", [])) if key in key_to_entry]
            for name in ("train", "val", "test")
        }

    train_subjects = set(split.get("train", []))
    val_subjects = set(split.get("val", []))
    test_subjects = set(split.get("test", []))
    return {
        "train": [entry for entry in all_entries if entry.subject_id in train_subjects],
        "val": [entry for entry in all_entries if entry.subject_id in val_subjects],
        "test": [entry for entry in all_entries if entry.subject_id in test_subjects],
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(np.sum(exp, axis=1, keepdims=True), 1e-8, None)


def _apply_transition_postprocess(result: dict, model) -> dict | None:
    if not hasattr(model, "smooth_logits") or "logits" not in result:
        return None

    logits = np.asarray(result["logits"], dtype=np.float32)
    record_ids = np.asarray(result["record_ids"])
    epoch_indices = np.asarray(result["epoch_indices"])
    smoothed_logits = logits.copy()

    for record_id in sorted(set(record_ids.tolist())):
        idx = np.where(record_ids == record_id)[0]
        if idx.size <= 1:
            continue
        ordered = idx[np.argsort(epoch_indices[idx])]
        with torch.no_grad():
            device = next(model.parameters()).device
            smoothed_logits[ordered] = (
                model.smooth_logits(torch.from_numpy(logits[ordered]).float().to(device)).cpu().numpy()
            )

    out = dict(result)
    out["logits"] = smoothed_logits
    out["y_prob"] = _softmax_np(smoothed_logits)
    out["y_pred"] = np.argmax(out["y_prob"], axis=1)
    return out


def _result_from_logits(base_result: Dict[str, np.ndarray], logits: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(base_result)
    out["logits"] = np.asarray(logits, dtype=np.float32)
    out["y_prob"] = _softmax_np(out["logits"])
    out["y_pred"] = np.argmax(out["y_prob"], axis=1)
    return out


def _derive_class_bias(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> np.ndarray:
    true_counts = np.bincount(np.asarray(y_true, dtype=np.int64), minlength=num_classes).astype(np.float64)
    true_prior = (true_counts + 1.0) / np.clip(np.sum(true_counts) + num_classes, 1.0, None)
    pred_prior = np.asarray(y_prob, dtype=np.float64).mean(axis=0)
    pred_prior = (pred_prior + 1e-6) / np.clip(np.sum(pred_prior + 1e-6), 1e-6, None)
    bias = np.log(true_prior) - np.log(pred_prior)
    return np.clip(bias.astype(np.float32), -1.5, 1.5)


def _apply_light_postprocess(
    raw_result: Dict[str, np.ndarray],
    smooth_result: Dict[str, np.ndarray] | None,
    temperature: float,
    bias_scale: float,
    bias_vector: np.ndarray,
    smooth_mix: float,
) -> Dict[str, np.ndarray]:
    raw_logits = np.asarray(raw_result["logits"], dtype=np.float32)
    mixed_logits = raw_logits.copy()
    if smooth_result is not None and float(smooth_mix) > 0.0:
        smooth_logits = np.asarray(smooth_result["logits"], dtype=np.float32)
        mixed_logits = (1.0 - float(smooth_mix)) * raw_logits + float(smooth_mix) * smooth_logits
    adjusted = mixed_logits / float(max(temperature, 1e-6))
    adjusted = adjusted + float(bias_scale) * np.asarray(bias_vector, dtype=np.float32).reshape(1, -1)
    return _result_from_logits(raw_result, adjusted)


def _search_light_postprocess(
    raw_result: Dict[str, np.ndarray],
    smooth_result: Dict[str, np.ndarray] | None,
    num_classes: int,
    wake_label: int,
    rem_label: int,
) -> Dict[str, Any]:
    raw_metrics = _evaluate_metrics(raw_result["y_true"], raw_result["y_pred"], raw_result["y_prob"], num_classes, wake_label, rem_label)
    bias_vector = _derive_class_bias(raw_result["y_true"], raw_result["y_prob"], num_classes)
    temperature_candidates = [0.95, 1.0, 1.05]
    bias_scale_candidates = [0.0, 0.5, 1.0]
    smooth_mix_candidates = [0.0]
    if smooth_result is not None:
        smooth_mix_candidates.extend([0.05, 0.10])

    best_result = raw_result
    best_metrics = raw_metrics
    best_params = {
        "temperature": 1.0,
        "bias_scale": 0.0,
        "bias_vector": bias_vector.tolist(),
        "smooth_mix": 0.0,
        "selected_on_validation": False,
    }
    search_rows: List[Dict[str, float]] = []

    for temperature in temperature_candidates:
        for bias_scale in bias_scale_candidates:
            for smooth_mix in smooth_mix_candidates:
                if abs(float(temperature) - 1.0) < 1e-8 and abs(float(bias_scale)) < 1e-8 and abs(float(smooth_mix)) < 1e-8:
                    continue
                candidate = _apply_light_postprocess(
                    raw_result=raw_result,
                    smooth_result=smooth_result,
                    temperature=float(temperature),
                    bias_scale=float(bias_scale),
                    bias_vector=bias_vector,
                    smooth_mix=float(smooth_mix),
                )
                metrics = _evaluate_metrics(
                    candidate["y_true"],
                    candidate["y_pred"],
                    candidate["y_prob"],
                    num_classes,
                    wake_label,
                    rem_label,
                )
                row = {
                    "temperature": float(temperature),
                    "bias_scale": float(bias_scale),
                    "smooth_mix": float(smooth_mix),
                    "macro_f1": float(metrics["macro_f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "kappa": float(metrics["kappa"]),
                }
                search_rows.append(row)
                better = (
                    metrics["macro_f1"] > best_metrics["macro_f1"] + 1e-9
                    or (
                        abs(metrics["macro_f1"] - best_metrics["macro_f1"]) <= 1e-9
                        and metrics["accuracy"] > best_metrics["accuracy"] + 1e-9
                    )
                    or (
                        abs(metrics["macro_f1"] - best_metrics["macro_f1"]) <= 1e-9
                        and abs(metrics["accuracy"] - best_metrics["accuracy"]) <= 1e-9
                        and metrics["kappa"] > best_metrics["kappa"] + 1e-9
                    )
                )
                if better:
                    best_result = candidate
                    best_metrics = metrics
                    best_params = {
                        "temperature": float(temperature),
                        "bias_scale": float(bias_scale),
                        "bias_vector": bias_vector.tolist(),
                        "smooth_mix": float(smooth_mix),
                        "selected_on_validation": True,
                    }

    return {
        "raw_metrics": raw_metrics,
        "best_metrics": best_metrics,
        "best_result": best_result,
        "best_params": best_params,
        "search_rows": search_rows,
    }


def _group_records(result: Dict[str, np.ndarray]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
    true_dict: Dict[str, Dict[int, int]] = {}
    pred_dict: Dict[str, Dict[int, int]] = {}
    for y_t, y_p, record_id, epoch_idx in zip(
        result["y_true"], result["y_pred"], result["record_ids"], result["epoch_indices"]
    ):
        key = str(record_id)
        true_dict.setdefault(key, {})[int(epoch_idx)] = int(y_t)
        pred_dict.setdefault(key, {})[int(epoch_idx)] = int(y_p)

    record_ids = sorted(true_dict.keys())
    y_true_list: List[List[int]] = []
    y_pred_list: List[List[int]] = []
    for record_id in record_ids:
        indices = sorted(true_dict[record_id].keys())
        y_true_list.append([true_dict[record_id][idx] for idx in indices])
        y_pred_list.append([pred_dict[record_id][idx] for idx in indices])
    return record_ids, y_true_list, y_pred_list


def _difficulty_note(
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    test_counts: np.ndarray,
    recommended_min_count: int,
) -> str:
    min_eval = int(min(np.min(val_counts), np.min(test_counts))) if val_counts.size and test_counts.size else 0
    if min_eval < int(recommended_min_count):
        return "hard"
    if min_eval < int(recommended_min_count) * 2:
        return "medium"
    return "easy"


def _save_predictions(path: Path, result: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        y_true=np.asarray(result["y_true"]),
        y_pred=np.asarray(result["y_pred"]),
        y_prob=np.asarray(result["y_prob"]),
        logits=np.asarray(result.get("logits", [])),
        record_ids=np.asarray(result["record_ids"]).astype(str),
        epoch_indices=np.asarray(result["epoch_indices"]),
    )


def _save_confusion_csv(path: Path, cm: np.ndarray, labels: List[str]) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path, index=True, **csv_utf8_sig_kwargs())


def _save_split_comparison(split_dir: Path, raw_result: Dict[str, np.ndarray], smooth_result: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        split_dir / "predictions_compare.npz",
        y_true=np.asarray(raw_result["y_true"]),
        y_pred_raw=np.asarray(raw_result["y_pred"]),
        y_pred_smoothed=np.asarray(smooth_result["y_pred"]),
        record_ids=np.asarray(raw_result["record_ids"]).astype(str),
        epoch_indices=np.asarray(raw_result["epoch_indices"]),
    )


def _error_analysis(cm: np.ndarray, labels: List[str]) -> Dict[str, object]:
    rows = []
    total = int(np.sum(cm))
    for i, true_label in enumerate(labels):
        row_total = int(np.sum(cm[i]))
        for j, pred_label in enumerate(labels):
            count = int(cm[i, j])
            if i == j or count <= 0:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                    "row_ratio": float(count / max(1, row_total)),
                    "global_ratio": float(count / max(1, total)),
                }
            )
    rows.sort(key=lambda row: (-row["count"], -row["row_ratio"]))
    return {"top_confusions": rows[:10], "total_samples": total}


def _evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    wake_label: int,
    rem_label: int,
) -> Dict[str, float]:
    multi, _ = walch_multiclass_metrics(y_true, y_prob, num_classes=num_classes)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    pred_counts = np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=num_classes).astype(np.int64)
    pred_ratio = pred_counts / max(1, int(pred_counts.sum()))
    sleep_true = (y_true != wake_label).astype(int)
    sleep, _ = walch_binary_metrics(sleep_true, 1.0 - y_prob[:, wake_label], positive="sleep")
    rem_true = (y_true == rem_label).astype(int)
    rem, _ = walch_binary_metrics(rem_true, y_prob[:, rem_label], positive="rem")
    metrics = {
        "accuracy": float(multi["accuracy"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(multi["macro_recall"]),
        "macro_specificity": float(multi["macro_specificity"]),
        "macro_roc_auc": float(multi["macro_roc_auc"]),
        "macro_pr_auc": float(multi["macro_pr_auc"]),
        "kappa": float(multi["kappa"]),
        "sleep_auc": float(sleep["roc_auc"]),
        "sleep_pr": float(sleep["pr_auc"]),
        "rem_auc": float(rem["roc_auc"]),
        "rem_pr": float(rem["pr_auc"]),
    }
    for class_idx in range(num_classes):
        metrics[f"class_{class_idx}_precision"] = float(precision[class_idx])
        metrics[f"class_{class_idx}_recall"] = float(recall[class_idx])
        metrics[f"class_{class_idx}_f1"] = float(f1[class_idx])
        metrics[f"class_{class_idx}_pred_ratio"] = float(pred_ratio[class_idx])
    return metrics


def _write_primary_outputs(
    split_dir: Path,
    result: Dict[str, np.ndarray],
    labels: List[str],
    num_classes: int,
    wake_label: int,
    rem_label: int,
) -> None:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    y_prob = result["y_prob"]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).to_csv(split_dir / "classification_report.csv", index=True, **csv_utf8_sig_kwargs())

    multi_metrics, per_class = walch_multiclass_metrics(y_true, y_prob, num_classes=num_classes)
    save_json(split_dir / "walch_multiclass.json", multi_metrics)
    pd.DataFrame(per_class).T.to_csv(split_dir / "walch_per_class.csv", **csv_utf8_sig_kwargs())

    sleep_true = (y_true != wake_label).astype(int)
    sleep_prob = 1.0 - y_prob[:, wake_label]
    sleep_metrics, sleep_curves = walch_binary_metrics(sleep_true, sleep_prob, positive="sleep")
    save_json(split_dir / "walch_binary_sleep.json", sleep_metrics)

    wake_true = (y_true == wake_label).astype(int)
    wake_prob = y_prob[:, wake_label]
    wake_metrics, wake_curves = walch_binary_metrics(wake_true, wake_prob, positive="wake")
    save_json(split_dir / "walch_binary_wake.json", wake_metrics)

    rem_true = (y_true == rem_label).astype(int)
    rem_prob = y_prob[:, rem_label]
    rem_metrics, rem_curves = walch_binary_metrics(rem_true, rem_prob, positive="rem")
    save_json(split_dir / "walch_binary_rem.json", rem_metrics)

    curves = multiclass_curves(y_true, y_prob, num_classes=num_classes)
    plot_curves(
        {key: value["roc"] for key, value in curves.items()},
        split_dir / "roc_multiclass.png",
        "ROC 多分类",
        "FPR",
        "TPR",
        title_fallback="ROC Multiclass",
    )
    plot_curves(
        {key: value["pr"] for key, value in curves.items()},
        split_dir / "pr_multiclass.png",
        "PR 多分类",
        "Recall",
        "Precision",
        title_fallback="PR Multiclass",
    )
    plot_curves(
        {"sleep": (sleep_curves["roc"]["fpr"], sleep_curves["roc"]["tpr"])},
        split_dir / "roc_sleep.png",
        "ROC Sleep",
        "FPR",
        "TPR",
    )
    plot_curves(
        {"sleep": (sleep_curves["pr"]["recall"], sleep_curves["pr"]["precision"])},
        split_dir / "pr_sleep.png",
        "PR Sleep",
        "Recall",
        "Precision",
    )
    plot_curves(
        {"wake": (wake_curves["roc"]["fpr"], wake_curves["roc"]["tpr"])},
        split_dir / "roc_wake.png",
        "ROC Wake",
        "FPR",
        "TPR",
    )
    plot_curves(
        {"wake": (wake_curves["pr"]["recall"], wake_curves["pr"]["precision"])},
        split_dir / "pr_wake.png",
        "PR Wake",
        "Recall",
        "Precision",
    )
    plot_curves(
        {"rem": (rem_curves["roc"]["fpr"], rem_curves["roc"]["tpr"])},
        split_dir / "roc_rem.png",
        "ROC REM",
        "FPR",
        "TPR",
    )
    plot_curves(
        {"rem": (rem_curves["pr"]["recall"], rem_curves["pr"]["precision"])},
        split_dir / "pr_rem.png",
        "PR REM",
        "Recall",
        "Precision",
    )

    record_ids, y_true_list, y_pred_list = _group_records(result)
    stats_df = build_sleep_stats_table(record_ids, y_true_list, y_pred_list, epoch_seconds=30, wake_label=wake_label)
    stats_df.to_csv(split_dir / "sleep_stats.csv", index=False, **csv_utf8_sig_kwargs())

    loa_rows = []
    for variable in ["TIB", "TST", "SOL", "WASO", "SE", "REM", "NREM"]:
        stats = plot_bland_altman(
            stats_df[f"true_{variable}"].values,
            stats_df[f"pred_{variable}"].values,
            split_dir / f"bland_altman_{variable}.png",
            f"{variable} Bland-Altman",
        )
        stats["variable"] = variable
        loa_rows.append(stats)
    pd.DataFrame(loa_rows).to_csv(split_dir / "bland_altman_loa.csv", index=False, **csv_utf8_sig_kwargs())


def _save_result_bundle(
    split_dir: Path,
    split_idx: int,
    result_tag: str,
    result: Dict[str, np.ndarray],
    labels: List[str],
    num_classes: int,
    wake_label: int,
    rem_label: int,
    primary: bool,
) -> None:
    suffix = f"_{result_tag}"
    _save_predictions(split_dir / f"predictions{suffix}.npz", result)

    cm = confusion_matrix(result["y_true"], result["y_pred"], labels=list(range(num_classes)))
    _save_confusion_csv(split_dir / f"confusion_matrix{suffix}.csv", cm, labels)
    plot_confusion_matrix(
        cm,
        labels,
        split_dir / f"confusion_matrix{suffix}.png",
        normalize=False,
        title=f"第{split_idx + 1}折混淆矩阵 ({result_tag})",
        title_fallback=f"Fold {split_idx + 1} Confusion Matrix ({result_tag})",
    )
    save_json(split_dir / f"error_analysis{suffix}.json", _error_analysis(cm, labels))

    if primary:
        shutil.copyfile(split_dir / f"predictions{suffix}.npz", split_dir / "predictions_primary.npz")
        shutil.copyfile(split_dir / f"confusion_matrix{suffix}.csv", split_dir / "confusion_matrix.csv")
        shutil.copyfile(split_dir / f"confusion_matrix{suffix}.png", split_dir / "confusion_matrix.png")
        shutil.copyfile(split_dir / f"error_analysis{suffix}.json", split_dir / "error_analysis.json")
        _write_primary_outputs(split_dir, result, labels, num_classes, wake_label, rem_label)


def _append_group_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    group_cols = ["model_name", "result_tag"]
    skip_cols = {
        "split",
        "train_class_counts",
        "val_class_counts",
        "test_class_counts",
        "difficulty_note",
        "model_hparams",
        "layer_firing_rates",
        "postprocess_used",
        "skipped_reason",
        *group_cols,
    }
    numeric_cols = [col for col in df.columns if col not in skip_cols and pd.api.types.is_numeric_dtype(df[col])]

    rows = []
    group_source = df[df["model_name"].astype(str).str.len() > 0]
    for group_key, group_df in group_source.groupby(group_cols, dropna=False):
        group_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        payload = {group_cols[idx]: group_tuple[idx] for idx in range(len(group_cols))}
        payload["postprocess_used"] = bool(group_df["postprocess_used"].astype(bool).any()) if "postprocess_used" in group_df else False
        mean_row = {"split": "mean", **payload}
        std_row = {"split": "std", **payload}
        for col in numeric_cols:
            mean_row[col] = float(group_df[col].mean())
            std_row[col] = float(group_df[col].std())
        rows.extend([mean_row, std_row])

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def _extract_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "train" / "split_0" / "train_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _preferred_result_tag(df: pd.DataFrame) -> str:
    tags = set(df.get("result_tag", pd.Series(dtype=str)).astype(str).tolist())
    if "selected" in tags:
        return "selected"
    if "smoothed" in tags:
        return "smoothed"
    return "raw"


def _normalize_training_recipe(meta: Dict[str, Any]) -> str:
    recipe = str(meta.get("recipe_name", "")).strip()
    if recipe:
        return recipe
    preset = str(meta.get("preset", "")).strip().lower()
    model_name = str(meta.get("model_name", "")).strip().lower()
    if preset == "plus_full" and model_name == "picosleepnet_plus_snn":
        return "baseline_locked"
    return recipe or preset or "unknown"


def _load_eval_profile(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "eval" / "per_fold_test_metrics.csv"
    if not summary_path.exists():
        raise RuntimeError(f"per_fold_test_metrics.csv not found: {summary_path}")
    primary_rows = pd.read_csv(summary_path)
    if not primary_rows.empty:
        primary_rows["split"] = primary_rows["split"].astype(int)

    per_fold_metrics = []
    for _, row in primary_rows.sort_values("split").iterrows():
        per_fold_metrics.append(
            {
                "split": int(row["split"]),
                "accuracy": float(row["test_acc"]),
                "macro_f1": float(row["test_macro_f1"]),
                "kappa": float(row["test_kappa"]),
            }
        )

    meta = _extract_run_meta(run_dir)
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "result_tag": "selected",
        "model_name": str(meta.get("model_name", "")),
        "model_hparams": meta.get("model_hparams", {}),
        "training_recipe": _normalize_training_recipe(meta),
        "postprocess_selection": json.loads((run_dir / "eval" / "postprocess_selection.json").read_text(encoding="utf-8"))
        if (run_dir / "eval" / "postprocess_selection.json").exists()
        else {},
        "mean_acc": float(pd.to_numeric(primary_rows["test_acc"], errors="coerce").mean()) if not primary_rows.empty else float("nan"),
        "mean_macro_f1": float(pd.to_numeric(primary_rows["test_macro_f1"], errors="coerce").mean()) if not primary_rows.empty else float("nan"),
        "mean_kappa": float(pd.to_numeric(primary_rows["test_kappa"], errors="coerce").mean()) if not primary_rows.empty else float("nan"),
        "min_fold_macro_f1": float(pd.to_numeric(primary_rows["test_macro_f1"], errors="coerce").min()) if not primary_rows.empty else float("nan"),
        "per_fold_metrics": per_fold_metrics,
    }


def _write_baseline_result_check(run_dir: Path, profile: Dict[str, Any]) -> Dict[str, Any]:
    change_proof_dir = ensure_dir(run_dir / "change_proof")
    split_payload = []
    exact_match_full = True
    seen_splits = set()
    for row in sorted(profile.get("per_fold_metrics", []), key=lambda item: int(item.get("split", -1))):
        split_idx = int(row["split"])
        baseline_row = BASELINE_SELECTED_METRICS.get(split_idx)
        if baseline_row is None:
            continue
        seen_splits.add(split_idx)
        exact_match = (
            abs(float(row["accuracy"]) - float(baseline_row["accuracy"])) <= 1e-12
            and abs(float(row["macro_f1"]) - float(baseline_row["macro_f1"])) <= 1e-12
            and abs(float(row["kappa"]) - float(baseline_row["kappa"])) <= 1e-12
        )
        exact_match_full = exact_match_full and exact_match
        split_payload.append(
            {
                "split": split_idx,
                "baseline_result_tag": str(baseline_row["result_tag"]),
                "new_result_tag": str(profile.get("result_tag", "")),
                "baseline_accuracy": float(baseline_row["accuracy"]),
                "new_accuracy": float(row["accuracy"]),
                "delta_accuracy": float(row["accuracy"]) - float(baseline_row["accuracy"]),
                "baseline_macro_f1": float(baseline_row["macro_f1"]),
                "new_macro_f1": float(row["macro_f1"]),
                "delta_macro_f1": float(row["macro_f1"]) - float(baseline_row["macro_f1"]),
                "baseline_kappa": float(baseline_row["kappa"]),
                "new_kappa": float(row["kappa"]),
                "delta_kappa": float(row["kappa"]) - float(baseline_row["kappa"]),
                "exact_match": bool(exact_match),
            }
        )

    full_5fold_present = seen_splits == set(BASELINE_SELECTED_METRICS.keys())
    status = "ok"
    reason = "new_result_differs_from_baseline"
    if full_5fold_present and exact_match_full:
        status = "failed_same_results_as_baseline"
        reason = "training strategy did not enter active path or eval reused locked baseline outputs"

    payload = {
        "status": status,
        "reason": reason,
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "new_run_dir": str(run_dir),
        "mean_acc": float(profile.get("mean_acc", float("nan"))),
        "mean_macro_f1": float(profile.get("mean_macro_f1", float("nan"))),
        "mean_kappa": float(profile.get("mean_kappa", float("nan"))),
        "delta_mean_acc": float(profile.get("mean_acc", float("nan"))) - float(BASELINE_MEAN_METRICS["mean_acc"]),
        "delta_mean_macro_f1": float(profile.get("mean_macro_f1", float("nan"))) - float(BASELINE_MEAN_METRICS["mean_macro_f1"]),
        "delta_mean_kappa": float(profile.get("mean_kappa", float("nan"))) - float(BASELINE_MEAN_METRICS["mean_kappa"]),
        "splits": split_payload,
    }
    save_json(change_proof_dir / "baseline_result_check.json", payload)
    return payload


def _build_baseline_snapshot(run_dir: Path, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "baseline_name": "baseline_locked",
        "run_dir": str(run_dir),
        "model_name": str(profile.get("model_name", "")),
        "model_hparams": profile.get("model_hparams", {}),
        "training_recipe": str(profile.get("training_recipe", "")),
        "postprocess_params": profile.get("postprocess_selection", {}),
        "summary_metrics_csv": str(run_dir / "eval" / "summary_metrics.csv"),
        "summary_metrics": {
            "result_tag": str(profile.get("result_tag", "")),
            "mean_acc": float(profile.get("mean_acc", float("nan"))),
            "mean_macro_f1": float(profile.get("mean_macro_f1", float("nan"))),
            "mean_kappa": float(profile.get("mean_kappa", float("nan"))),
        },
    }


def _safe_ablation_decision(baseline_profile: Dict[str, Any], candidate_profile: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    baseline_macro = float(baseline_profile.get("mean_macro_f1", float("nan")))
    candidate_macro = float(candidate_profile.get("mean_macro_f1", float("nan")))
    baseline_acc = float(baseline_profile.get("mean_acc", float("nan")))
    candidate_acc = float(candidate_profile.get("mean_acc", float("nan")))
    baseline_min = float(baseline_profile.get("min_fold_macro_f1", float("nan")))
    candidate_min = float(candidate_profile.get("min_fold_macro_f1", float("nan")))

    if not candidate_macro > baseline_macro:
        reasons.append("mean_macro_f1_not_improved")
    if candidate_acc < baseline_acc - 0.01:
        reasons.append("mean_acc_drop_exceeds_0.01")
    if candidate_min < baseline_min - 0.02:
        reasons.append("worst_fold_macro_f1_drop_exceeds_0.02")

    baseline_by_split = {int(row["split"]): float(row["macro_f1"]) for row in baseline_profile.get("per_fold_metrics", [])}
    for split_id in (3, 4):
        if split_id not in baseline_by_split:
            continue
        candidate_row = next((row for row in candidate_profile.get("per_fold_metrics", []) if int(row["split"]) == split_id), None)
        if candidate_row is None:
            continue
        if float(candidate_row["macro_f1"]) < baseline_by_split[split_id] - 0.02:
            reasons.append(f"split{split_id}_macro_f1_drop_exceeds_0.02")

    return len(reasons) == 0, reasons


def _write_safe_ablation_outputs(
    eval_root: Path,
    baseline_profile: Dict[str, Any],
    candidate_profile: Dict[str, Any],
) -> None:
    same_profile = (
        str(baseline_profile.get("run_dir", "")) == str(candidate_profile.get("run_dir", ""))
        and str(baseline_profile.get("training_recipe", "")) == str(candidate_profile.get("training_recipe", ""))
    )
    baseline_macro = float(baseline_profile.get("mean_macro_f1", float("nan")))
    baseline_acc = float(baseline_profile.get("mean_acc", float("nan")))
    baseline_kappa = float(baseline_profile.get("mean_kappa", float("nan")))
    kept, reasons = (False, []) if same_profile else _safe_ablation_decision(baseline_profile, candidate_profile)

    rows = []
    row_sources = [("baseline_locked", baseline_profile)]
    if not same_profile:
        row_sources.append((str(candidate_profile.get("training_recipe", "candidate")), candidate_profile))
    for name, profile in row_sources:
        rows.append(
            {
                "recipe_name": name,
                "run_dir": str(profile.get("run_dir", "")),
                "result_tag": str(profile.get("result_tag", "")),
                "mean_acc": float(profile.get("mean_acc", float("nan"))),
                "mean_macro_f1": float(profile.get("mean_macro_f1", float("nan"))),
                "mean_kappa": float(profile.get("mean_kappa", float("nan"))),
                "min_fold_macro_f1": float(profile.get("min_fold_macro_f1", float("nan"))),
                "per_fold_metrics": json.dumps(profile.get("per_fold_metrics", []), ensure_ascii=False, separators=(",", ":")),
                "delta_vs_baseline_acc": float(profile.get("mean_acc", float("nan"))) - baseline_acc,
                "delta_vs_baseline_macro_f1": float(profile.get("mean_macro_f1", float("nan"))) - baseline_macro,
                "delta_vs_baseline_kappa": float(profile.get("mean_kappa", float("nan"))) - baseline_kappa,
                "status": (
                    "kept"
                    if same_profile
                    else (
                        "kept"
                        if (name == "baseline_locked" and not kept) or (name != "baseline_locked" and kept)
                        else ("rejected" if name != "baseline_locked" else "baseline")
                    )
                ),
            }
        )

    pd.DataFrame(rows).to_csv(eval_root / "safe_ablation_report.csv", index=False, **csv_utf8_sig_kwargs())
    kept_payload = {
        "recipe_name": str(candidate_profile.get("training_recipe", "")) if kept else "baseline_locked",
        "run_dir": str(candidate_profile.get("run_dir", "")) if kept else str(baseline_profile.get("run_dir", "")),
        "baseline_run_dir": str(baseline_profile.get("run_dir", "")),
        "result_tag": str(candidate_profile.get("result_tag", "")) if kept else str(baseline_profile.get("result_tag", "")),
        "mean_acc": float(candidate_profile.get("mean_acc", float("nan"))) if kept else float(baseline_profile.get("mean_acc", float("nan"))),
        "mean_macro_f1": float(candidate_profile.get("mean_macro_f1", float("nan"))) if kept else float(baseline_profile.get("mean_macro_f1", float("nan"))),
        "mean_kappa": float(candidate_profile.get("mean_kappa", float("nan"))) if kept else float(baseline_profile.get("mean_kappa", float("nan"))),
        "reason": "candidate_improved_safely" if kept else "fallback_to_baseline_locked",
    }
    rejected_payload = []
    if same_profile:
        rejected_payload = []
    elif kept:
        rejected_payload.append(
            {
                "recipe_name": "baseline_locked",
                "run_dir": str(baseline_profile.get("run_dir", "")),
                "reason": "kept_new_recipe",
            }
        )
    else:
        rejected_payload.append(
            {
                "recipe_name": str(candidate_profile.get("training_recipe", "candidate")),
                "run_dir": str(candidate_profile.get("run_dir", "")),
                "reason": reasons,
            }
        )
    save_json(eval_root / "kept_recipe.json", kept_payload)
    save_json(eval_root / "rejected_recipes.json", rejected_payload)


def _sample_input_from_loader(dataloader) -> torch.Tensor | None:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None
    return batch[0][:1].detach().clone()


def _infer_timesteps(model_hparams: dict[str, Any]) -> int:
    if "t_steps" in model_hparams:
        return int(model_hparams.get("t_steps", 0)) + int(model_hparams.get("context_len", 1))
    if "window_size" in model_hparams:
        window = max(1, int(model_hparams.get("window_size", 40)))
        return int(3000 // window)
    return int(model_hparams.get("context_len", 1))


def _write_change_proof_outputs(run_dir: Path, summary_df: pd.DataFrame, labels: List[str]) -> None:
    change_proof_dir = ensure_dir(run_dir / "change_proof")
    valid_rows = summary_df[pd.to_numeric(summary_df["split"], errors="coerce").notna()].copy()
    if valid_rows.empty:
        return
    valid_rows["split"] = valid_rows["split"].astype(int)
    primary_rows = valid_rows[valid_rows["result_tag"] == "selected"].copy()
    if primary_rows.empty:
        primary_rows = valid_rows.copy()

    per_class_rows = []
    for _, row in primary_rows.sort_values("split").iterrows():
        split_idx = int(row["split"])
        for class_idx, class_name in enumerate(labels):
            per_class_rows.append(
                {
                    "split": split_idx,
                    "class": class_name,
                    "precision": float(row.get(f"class_{class_idx}_precision", float("nan"))),
                    "recall": float(row.get(f"class_{class_idx}_recall", float("nan"))),
                    "f1": float(row.get(f"class_{class_idx}_f1", float("nan"))),
                    "pred_ratio": float(row.get(f"class_{class_idx}_pred_ratio", float("nan"))),
                }
            )
    pd.DataFrame(per_class_rows).to_csv(
        change_proof_dir / "per_fold_per_class_metrics.csv",
        index=False,
        **csv_utf8_sig_kwargs(),
    )

    mean_acc = float(pd.to_numeric(primary_rows["accuracy"], errors="coerce").mean())
    mean_macro_f1 = float(pd.to_numeric(primary_rows["macro_f1"], errors="coerce").mean())
    mean_kappa = float(pd.to_numeric(primary_rows["kappa"], errors="coerce").mean())
    mean_n1_f1 = float(pd.to_numeric(primary_rows["N1_f1"], errors="coerce").mean())
    mean_rem_f1 = float(pd.to_numeric(primary_rows["REM_f1"], errors="coerce").mean())

    model_hparams = {}
    if "model_hparams" in primary_rows.columns and len(primary_rows) > 0:
        try:
            model_hparams = json.loads(str(primary_rows.iloc[0]["model_hparams"]))
        except Exception:
            model_hparams = {}

    baseline_summary = pd.read_csv(BASELINE_RUN_DIR / "eval" / "summary_metrics.csv")
    baseline_primary = baseline_summary[
        (baseline_summary["split"].astype(str) == "mean") & (baseline_summary["result_tag"].astype(str) == "selected")
    ]
    if baseline_primary.empty:
        baseline_primary = baseline_summary[
            (baseline_summary["split"].astype(str) == "mean") & (baseline_summary["result_tag"].astype(str) == "raw")
        ]
    baseline_params = float(baseline_primary["total_params"].iloc[0]) if not baseline_primary.empty else float("nan")
    baseline_macs = float(baseline_primary["estimated_MACs"].iloc[0]) if not baseline_primary.empty else float("nan")
    baseline_latency = (
        float(baseline_primary["inference_latency_ms"].iloc[0]) if not baseline_primary.empty else float("nan")
    )

    current_params = float(pd.to_numeric(primary_rows["total_params"], errors="coerce").mean())
    current_macs = float(pd.to_numeric(primary_rows["estimated_MACs"], errors="coerce").mean())
    current_latency = float(pd.to_numeric(primary_rows["inference_latency_ms"], errors="coerce").mean())
    timesteps_t = int(_infer_timesteps(model_hparams))

    params_ratio = float(current_params / baseline_params) if baseline_params and baseline_params > 0 else float("nan")
    macs_ratio = float(current_macs / baseline_macs) if baseline_macs and baseline_macs > 0 else float("nan")
    latency_ratio = (
        float(current_latency / baseline_latency) if baseline_latency and baseline_latency > 0 else float("nan")
    )

    complexity_report = {
        "params": current_params,
        "MACs_or_FLOPs": {"MACs": current_macs, "FLOPs": float(current_macs * 2.0)},
        "timesteps_T": timesteps_t,
        "estimated_inference_cost": {
            "inference_latency_ms": current_latency,
            "batch_size": 32,
            "runtime_device": "cpu" if not torch.cuda.is_available() else "cuda",
        },
        "complexity_vs_current_baseline": {
            "params_ratio": params_ratio,
            "MACs_ratio": macs_ratio,
            "latency_ratio": latency_ratio,
            "baseline_params": baseline_params,
            "baseline_MACs": baseline_macs,
            "baseline_latency_ms": baseline_latency,
        },
        "complexity_vs_picosleepnet_target": {
            "reference": "in_repo_picosleepnet_plus_snn_baseline_proxy",
            "params_ratio": params_ratio,
            "MACs_ratio": macs_ratio,
        },
        "within_complexity_budget": bool(
            (pd.isna(params_ratio) or params_ratio <= 1.2) and (pd.isna(macs_ratio) or macs_ratio <= 1.3)
        ),
    }
    save_json(change_proof_dir / "complexity_report.json", complexity_report)

    metric_guard = {
        "mean_acc": mean_acc,
        "mean_macro_f1": mean_macro_f1,
        "mean_kappa": mean_kappa,
        "mean_N1_f1": mean_n1_f1,
        "mean_REM_f1": mean_rem_f1,
        "target_acc_reached": bool(mean_acc >= 0.85),
        "target_macro_f1_reached": bool(mean_macro_f1 >= 0.80),
        "final_status": "target_reached" if (mean_acc >= 0.85 and mean_macro_f1 >= 0.80) else "below_target",
    }
    save_json(change_proof_dir / "metric_guard_report.json", metric_guard)


def _append_mean_std_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    mean_row = {"split": "mean"}
    std_row = {"split": "std"}
    for col in numeric_cols:
        mean_row[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        std_row[col] = float(pd.to_numeric(df[col], errors="coerce").std())
    for col in df.columns:
        if col not in mean_row:
            mean_row[col] = ""
            std_row[col] = ""
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def _write_selected_test_reports(run_dir: Path, eval_root: Path, summary_df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    valid_rows = summary_df[pd.to_numeric(summary_df["split"], errors="coerce").notna()].copy()
    if valid_rows.empty:
        return pd.DataFrame()
    valid_rows["split"] = valid_rows["split"].astype(int)
    selected_rows = valid_rows[valid_rows["result_tag"] == "selected"].copy()
    if selected_rows.empty:
        selected_rows = valid_rows[valid_rows["result_tag"] == "raw"].copy()
    selected_rows = selected_rows.sort_values("split")

    test_rows: list[dict[str, object]] = []
    compare_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    confusion_dir = ensure_dir(eval_root / "per_fold_confusion_matrices")

    for _, row in selected_rows.iterrows():
        split_idx = int(row["split"])
        train_meta_path = run_dir / "train" / f"split_{split_idx}" / "train_meta.json"
        meta = json.loads(train_meta_path.read_text(encoding="utf-8")) if train_meta_path.exists() else {}
        best_snapshot = dict(meta.get("best_val_snapshot", {}))
        val_macro = float(best_snapshot.get("val_macro_f1", np.nan))
        test_macro = float(row["macro_f1"])
        val_n1_f1 = float(best_snapshot.get("val_N1_f1", np.nan))
        val_n1_recall = float(best_snapshot.get("val_N1_recall", np.nan))
        test_n1_f1 = float(row["N1_f1"])
        test_n1_recall = float(row["N1_recall"])
        compare_row = {
            "split": split_idx,
            "recipe_name": str(meta.get("recipe_name", "")),
            "chosen_ckpt_epoch": int(meta.get("best_epoch", -1)),
            "val_macro_f1_at_ckpt": val_macro,
            "val_N1_f1_at_ckpt": val_n1_f1,
            "val_N1_recall_at_ckpt": val_n1_recall,
            "test_acc": float(row["accuracy"]),
            "test_macro_f1": test_macro,
            "test_kappa": float(row["kappa"]),
            "test_N1_f1": test_n1_f1,
            "test_N1_recall": test_n1_recall,
            "test_REM_f1": float(row["REM_f1"]),
            "macro_f1_gap_test_minus_val": float(test_macro - val_macro) if not np.isnan(val_macro) else float("nan"),
            "N1_f1_gap_test_minus_val": float(test_n1_f1 - val_n1_f1) if not np.isnan(val_n1_f1) else float("nan"),
            "overfit_or_validation_mismatch": bool((not np.isnan(val_macro)) and (val_macro - test_macro > 0.05)),
        }
        compare_rows.append(compare_row)
        test_rows.append(dict(compare_row))

        for class_idx, class_name in enumerate(labels):
            pred_rows.append(
                {
                    "split": split_idx,
                    "class_name": str(class_name),
                    "pred_ratio": float(row.get(f"class_{class_idx}_pred_ratio", np.nan)),
                }
            )

        src_cm = eval_root / f"split_{split_idx}" / "confusion_matrix.csv"
        dst_cm = confusion_dir / f"split_{split_idx}_confusion_matrix.csv"
        if src_cm.exists():
            shutil.copy2(src_cm, dst_cm)

    detailed_path = eval_root / "summary_metrics_detailed.csv"
    summary_df.to_csv(detailed_path, index=False, **csv_utf8_sig_kwargs())

    per_fold_test_df = pd.DataFrame(test_rows)
    per_fold_compare_df = pd.DataFrame(compare_rows)
    pred_ratio_df = pd.DataFrame(pred_rows)

    per_fold_test_df.to_csv(eval_root / "per_fold_test_metrics.csv", index=False, **csv_utf8_sig_kwargs())
    per_fold_compare_df.to_csv(eval_root / "per_fold_val_test_compare.csv", index=False, **csv_utf8_sig_kwargs())
    pred_ratio_df.to_csv(eval_root / "per_fold_pred_ratio.csv", index=False, **csv_utf8_sig_kwargs())

    summary_test_df = _append_mean_std_test_rows(per_fold_test_df.copy())
    summary_test_df.to_csv(eval_root / "summary_metrics.csv", index=False, **csv_utf8_sig_kwargs())
    ensure_dir(run_dir / "change_proof")
    per_fold_compare_df.to_csv(run_dir / "change_proof" / "val_vs_test_gap_report.csv", index=False, **csv_utf8_sig_kwargs())
    return per_fold_test_df


def _build_class_totals(cm: np.ndarray, labels: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "class_name": labels,
            "true_count": cm.sum(axis=1).astype(int),
            "pred_count": cm.sum(axis=0).astype(int),
            "diagonal_count": np.diag(cm).astype(int),
        }
    )


def _save_merged_confusion(
    eval_root: Path,
    result_tag: str,
    labels: List[str],
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    primary: bool,
) -> np.ndarray | None:
    if not y_true or not y_pred:
        return None

    fold_count = len(y_true)
    cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred), labels=list(range(len(labels))))
    suffix = f"_{result_tag}"
    csv_path = eval_root / f"confusion_matrix_merged_{fold_count}fold{suffix}.csv"
    png_path = eval_root / f"confusion_matrix_merged_{fold_count}fold{suffix}.png"
    totals_path = eval_root / f"merged_class_totals{suffix}.csv"

    _save_confusion_csv(csv_path, cm, labels)
    plot_confusion_matrix(
        cm,
        labels,
        png_path,
        normalize=False,
        title=f"{fold_count}折交叉验证合并混淆矩阵 ({result_tag})",
        title_fallback=f"{fold_count}-Fold Merged Confusion Matrix ({result_tag})",
    )
    _build_class_totals(cm, labels).to_csv(totals_path, index=False, **csv_utf8_sig_kwargs())
    save_json(eval_root / f"error_analysis_merged_{fold_count}fold{suffix}.json", _error_analysis(cm, labels))

    if primary:
        shutil.copyfile(csv_path, eval_root / f"confusion_matrix_merged_{fold_count}fold.csv")
        shutil.copyfile(png_path, eval_root / f"confusion_matrix_merged_{fold_count}fold.png")
        shutil.copyfile(totals_path, eval_root / "merged_class_totals.csv")
    return cm


def _make_skipped_row(
    split_idx: int,
    labels: List[str],
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    test_counts: np.ndarray,
    feasible: bool,
    recommended_min_count: int,
    reason: str,
) -> Dict[str, object]:
    return {
        "split": split_idx,
        "result_tag": "raw",
        "model_name": "",
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "macro_recall": np.nan,
        "macro_specificity": np.nan,
        "macro_roc_auc": np.nan,
        "macro_pr_auc": np.nan,
        "kappa": np.nan,
        "sleep_auc": np.nan,
        "sleep_pr": np.nan,
        "rem_auc": np.nan,
        "rem_pr": np.nan,
        "num_samples": np.nan,
        "expected_test_samples": int(test_counts.sum()),
        "train_class_counts": _json_counts(train_counts, labels),
        "val_class_counts": _json_counts(val_counts, labels),
        "test_class_counts": _json_counts(test_counts, labels),
        "feasible": feasible,
        "recommended_min_count": recommended_min_count,
        "difficulty_note": _difficulty_note(train_counts, val_counts, test_counts, recommended_min_count),
        "model_hparams": "",
        "postprocess_used": False,
        "total_params": np.nan,
        "trainable_params": np.nan,
        "estimated_MACs": np.nan,
        "avg_firing_rate": np.nan,
        "spike_sparsity": np.nan,
        "inference_latency_ms": np.nan,
        "checkpoint_size_mb": np.nan,
        "layer_firing_rates": "",
        "skipped_reason": reason,
    }


def _global_count_l1(counts: np.ndarray, global_counts: np.ndarray) -> float:
    if int(np.sum(counts)) <= 0 or int(np.sum(global_counts)) <= 0:
        return 0.0
    p = counts.astype(np.float64) / float(np.sum(counts))
    q = global_counts.astype(np.float64) / float(np.sum(global_counts))
    return float(np.abs(p - q).sum())


def _build_split3_diagnosis(
    labels: List[str],
    global_counts: np.ndarray,
    split_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if split_payload is None:
        return {
            "available": False,
            "reason": "split3_not_present_in_current_eval",
            "final_conclusion": "需要在完整 5 折评估时再判断，但不建议为缺失 split3 做 fold-aware 补丁。",
        }

    train_counts = np.asarray(split_payload["train_counts"], dtype=np.int64)
    val_counts = np.asarray(split_payload["val_counts"], dtype=np.int64)
    test_counts = np.asarray(split_payload["test_counts"], dtype=np.int64)
    selected_result = split_payload["selected_result"]
    raw_result = split_payload["raw_result"]
    smoothed_result = split_payload.get("smoothed_result")

    selected_cm = confusion_matrix(selected_result["y_true"], selected_result["y_pred"], labels=list(range(len(labels))))
    top_confusions = _error_analysis(selected_cm, labels).get("top_confusions", [])
    raw_metrics = split_payload["raw_metrics"]
    smoothed_metrics = split_payload.get("smoothed_metrics")
    class_issue_score = _global_count_l1(test_counts, global_counts)
    likely_class_distribution_issue = bool(class_issue_score > 0.18 or int(np.min(test_counts)) <= 1)
    selected_macro = float(split_payload["selected_metrics"]["macro_f1"])
    raw_macro = float(raw_metrics["macro_f1"])
    smoothed_macro = float(smoothed_metrics["macro_f1"]) if smoothed_metrics is not None else raw_macro
    likely_decision_issue = bool((selected_macro < 0.62 and raw_macro < 0.62) or not likely_class_distribution_issue)

    return {
        "available": True,
        "split": 3,
        "train_class_counts": {labels[i]: int(train_counts[i]) for i in range(len(labels))},
        "val_class_counts": {labels[i]: int(val_counts[i]) for i in range(len(labels))},
        "test_class_counts": {labels[i]: int(test_counts[i]) for i in range(len(labels))},
        "split3_test_distribution_l1_vs_global": class_issue_score,
        "class_counts_abnormal": likely_class_distribution_issue,
        "major_confusions": top_confusions[:5],
        "raw_metrics": raw_metrics,
        "smoothed_metrics": smoothed_metrics,
        "selected_metrics": split_payload["selected_metrics"],
        "raw_better_than_smoothed": bool(raw_macro > smoothed_macro + 1e-9),
        "smoothed_better_than_raw": bool(smoothed_macro > raw_macro + 1e-9),
        "main_issue": "model_decision_or_boundary_confusion" if likely_decision_issue else "class_distribution",
        "is_pure_class_distribution_problem": bool(likely_class_distribution_issue and not likely_decision_issue),
        "final_conclusion": (
            "split3 更像是全局决策/边界混淆问题，应该优先靠全局保守策略修复，不值得再做 fold-aware 特化。"
            if likely_decision_issue
            else "split3 存在一定类分布扰动，但仍不建议做 fold-aware 补丁，优先采用全局保守策略。"
        ),
    }


def main() -> None:
    setup_utf8_stdio()
    suppress_pin_memory_warning()
    ensure_chinese_font()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--reference_run_dir", type=str, default=None)
    parser.add_argument("--postprocess", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--only_splits", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    if config_path is None or not config_path.exists():
        raise RuntimeError(f"config not found: {args.config}")

    cfg = read_yaml(config_path)
    runs_dir = Path(cfg["runs_dir"])
    run_dir = _resolve_run_dir(runs_dir, args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise RuntimeError("no valid run directory found, please train first or pass --run_dir.")
    _ensure_not_baseline_run_dir(run_dir, "eval run_dir")

    run_cfg = run_dir / "config.yaml"
    if run_cfg.exists():
        cfg = {**cfg, **read_yaml(run_cfg)}

    task = get_task_name(cfg, "sleep_edf_5class")
    labels = get_labels(task)
    num_classes = int(get_num_classes(task))
    wake_label = int(get_wake_label(task))
    rem_label = labels.index("REM") if "REM" in labels else (num_classes - 1)

    processed_dir = Path(cfg["processed_dir"])
    cache_status = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    if not cache_status.get("reuse_available", False):
        reason = str(cache_status.get("reason", "unknown"))
        manifest_path = str(cache_status.get("manifest_path", processed_dir / "preprocess_manifest.json"))
        raise RuntimeError(
            "processed cache does not match the current preprocess code or key config; "
            f"please rerun preprocess_sleep_edf.py. reason={reason} manifest={manifest_path}"
        )

    records = load_records(processed_dir)
    if any(bool(rec.get("is_legacy", False)) for rec in records):
        raise RuntimeError("legacy Sleep-EDF cache detected, please rerun preprocess_sleep_edf.py.")
    all_entries = build_entries(records, num_classes)
    global_counts = _entry_counts(all_entries, num_classes)

    splits_path = run_dir / "splits.json"
    if not splits_path.exists():
        raise RuntimeError(f"splits.json not found: {splits_path}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    eval_root = ensure_dir(run_dir / "eval")
    reference_run_dir = _resolve_repo_path(args.reference_run_dir) if args.reference_run_dir else None
    if reference_run_dir is not None and not reference_run_dir.exists():
        raise RuntimeError(f"reference_run_dir not found: {args.reference_run_dir}")
    if reference_run_dir is not None:
        _ensure_not_baseline_run_dir(reference_run_dir, "eval reference_run_dir")

    summary_rows: List[Dict[str, object]] = []
    merged: Dict[str, Dict[str, List[np.ndarray]]] = {}
    postprocess_selection_rows: List[Dict[str, Any]] = []
    split3_payload: Dict[str, Any] | None = None
    recommended_min_count = 1
    recommend_path = run_dir / "diagnose" / "recommended_min_count.json"
    if recommend_path.exists():
        recommended_min_count = int(
            json.loads(recommend_path.read_text(encoding="utf-8")).get("recommended_min_count", 1)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if os.name == "nt" else int(cfg.get("num_workers", 2))
    pin_memory = True

    ckpt_exists = any((run_dir / "train" / f"split_{split_idx}" / "best.ckpt").exists() for split_idx in range(len(splits)))
    if not ckpt_exists:
        print("eval: no checkpoint found under the selected run directory.")
        raise SystemExit(2)

    print(f"eval: config={config_path}")
    print(f"eval: run_dir={run_dir}")
    selected_splits = _parse_split_indices(args.only_splits)

    for split_idx, split in enumerate(splits):
        if selected_splits is not None and split_idx not in selected_splits:
            continue
        split_dir = ensure_dir(eval_root / f"split_{split_idx}")
        parts = _entries_for_split(split, all_entries)
        train_counts = _entry_counts(parts["train"], num_classes)
        val_counts = _entry_counts(parts["val"], num_classes)
        test_counts = _entry_counts(parts["test"], num_classes)
        feasible = bool(np.all(train_counts > 0) and np.all(val_counts > 0) and np.all(test_counts > 0))

        ckpt_path = run_dir / "train" / f"split_{split_idx}" / "best.ckpt"
        if not ckpt_path.exists() or not parts["test"]:
            reason = "checkpoint missing" if not ckpt_path.exists() else "empty test split"
            summary_rows.append(
                _make_skipped_row(
                    split_idx,
                    labels,
                    train_counts,
                    val_counts,
                    test_counts,
                    feasible,
                    recommended_min_count,
                    reason,
                )
            )
            print(f"eval split={split_idx} skipped: {reason}")
            continue
        _ensure_not_baseline_run_dir(ckpt_path, f"eval checkpoint split={split_idx}")

        ckpt = load_checkpoint(ckpt_path)
        model, model_name, model_hparams, ckpt_num_classes = _build_model_from_ckpt(ckpt, ckpt_path.parent, num_classes)
        if int(ckpt_num_classes) != int(num_classes):
            raise RuntimeError(f"num_classes mismatch: config={num_classes} ckpt={ckpt_num_classes}")

        delta_primary = float(model_hparams.get("lcs_delta", default_lcs_delta("edf20")))
        delta_small = float(model_hparams.get("lcs_delta_small", max(0.02, delta_primary * 0.65)))
        delta_large = float(model_hparams.get("lcs_delta_large", max(delta_small + 1e-4, delta_primary * 1.35)))
        context_len = int(model_hparams.get("context_len", 1))
        input_norm_mode = str(cfg.get("train", {}).get("input_norm_mode", "none")).lower().strip() or "none"
        test_dataset = SleepEdfSpikeDataset(
            parts["test"],
            model_name,
            bool(model_hparams.get("use_dual_lcs", model_name == "picosleepnet_plus_snn")),
            bool(model_hparams.get("use_integer_spike", True)),
            delta_primary,
            delta_small,
            delta_large,
            str(cfg.get("cache_mode", "mem")),
            input_norm_mode=input_norm_mode,
            context_len=context_len,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        val_loader = None
        if parts["val"]:
            val_dataset = SleepEdfSpikeDataset(
                parts["val"],
                model_name,
                bool(model_hparams.get("use_dual_lcs", model_name == "picosleepnet_plus_snn")),
                bool(model_hparams.get("use_integer_spike", True)),
                delta_primary,
                delta_small,
                delta_large,
                str(cfg.get("cache_mode", "mem")),
                input_norm_mode=input_norm_mode,
                context_len=context_len,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(cfg.get("batch_size", 32)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=False,
            )

        model.to(device)
        raw_result = run_inference(
            model,
            test_loader,
            device,
            mixed_precision=bool(cfg.get("mixed_precision", False)) and device.type == "cuda",
            non_blocking=bool(pin_memory),
            return_logits=True,
        )
        expected_test_samples = int(test_counts.sum())
        if int(len(raw_result["y_true"])) != expected_test_samples:
            raise RuntimeError(
                f"split={split_idx} sample count mismatch before merge: "
                f"predictions={len(raw_result['y_true'])} expected_test_entries={expected_test_samples}"
            )

        sample_input = _sample_input_from_loader(test_loader)
        complexity = (
            build_complexity_metrics(model, sample_input, test_loader, device, ckpt_path, non_blocking=bool(pin_memory))
            if sample_input is not None
            else {}
        )

        allow_transition_smoothing = bool(model_hparams.get("use_transition_matrix", model_name == "picosleepnet_plus_snn"))
        allow_postprocess = args.postprocess != "off"
        use_transition_smoothing = args.postprocess == "on" or (
            args.postprocess == "auto" and model_name == "picosleepnet_plus_snn" and allow_transition_smoothing
        )

        raw_metrics = _evaluate_metrics(raw_result["y_true"], raw_result["y_pred"], raw_result["y_prob"], num_classes, wake_label, rem_label)
        smoothed_result = None
        smooth_metrics = None
        selected_result = raw_result
        selected_metrics = raw_metrics
        selected_result_tag = "raw"
        selection_payload: Dict[str, Any] = {
            "split": int(split_idx),
            "selected_result_tag": "raw",
            "raw_metrics": raw_metrics,
            "smoothed_metrics": None,
            "selection_reason": "postprocess_disabled",
            "postprocess_params": {
                "temperature": 1.0,
                "bias_scale": 0.0,
                "bias_vector": [0.0] * num_classes,
                "smooth_mix": 0.0,
                "selected_on_validation": False,
            },
        }

        if allow_postprocess and val_loader is not None:
            val_raw_result = run_inference(
                model,
                val_loader,
                device,
                mixed_precision=bool(cfg.get("mixed_precision", False)) and device.type == "cuda",
                non_blocking=bool(pin_memory),
                return_logits=True,
            )
            val_smooth_result = _apply_transition_postprocess(val_raw_result, model) if use_transition_smoothing else None
            light_search = _search_light_postprocess(
                raw_result=val_raw_result,
                smooth_result=val_smooth_result,
                num_classes=num_classes,
                wake_label=wake_label,
                rem_label=rem_label,
            )
            if bool(light_search["best_params"].get("selected_on_validation", False)):
                test_transition_result = _apply_transition_postprocess(raw_result, model) if use_transition_smoothing else None
                smoothed_result = _apply_light_postprocess(
                    raw_result=raw_result,
                    smooth_result=test_transition_result,
                    temperature=float(light_search["best_params"]["temperature"]),
                    bias_scale=float(light_search["best_params"]["bias_scale"]),
                    bias_vector=np.asarray(light_search["best_params"]["bias_vector"], dtype=np.float32),
                    smooth_mix=float(light_search["best_params"]["smooth_mix"]),
                )
                if int(len(smoothed_result["y_true"])) != expected_test_samples:
                    raise RuntimeError(
                        f"split={split_idx} sample count mismatch after smoothing: "
                        f"predictions={len(smoothed_result['y_true'])} expected_test_entries={expected_test_samples}"
                    )
                smooth_metrics = _evaluate_metrics(
                    smoothed_result["y_true"],
                    smoothed_result["y_pred"],
                    smoothed_result["y_prob"],
                    num_classes,
                    wake_label,
                    rem_label,
                )
                if smooth_metrics["macro_f1"] > raw_metrics["macro_f1"] + 1e-9 or (
                    abs(smooth_metrics["macro_f1"] - raw_metrics["macro_f1"]) <= 1e-9
                    and smooth_metrics["accuracy"] >= raw_metrics["accuracy"] - 1e-9
                ):
                    selected_result = smoothed_result
                    selected_metrics = smooth_metrics
                    selected_result_tag = "smoothed"
                    selection_reason = "validation_selected_and_test_not_worse"
                else:
                    selection_reason = "fallback_to_raw_due_to_macro_f1_or_accuracy_drop"
            else:
                selection_reason = "raw_kept_after_validation_search"
            selection_payload = {
                "split": int(split_idx),
                "selected_result_tag": selected_result_tag,
                "raw_metrics": raw_metrics,
                "smoothed_metrics": smooth_metrics,
                "selection_reason": selection_reason,
                "postprocess_params": light_search["best_params"],
                "validation_search": light_search["search_rows"],
            }
            if smoothed_result is not None:
                _save_split_comparison(split_dir, raw_result, smoothed_result)
        elif allow_postprocess:
            selection_payload["selection_reason"] = "fallback_to_raw_due_to_missing_val_split"

        result_map = {"raw": raw_result}
        if smoothed_result is not None:
            result_map["smoothed"] = smoothed_result
        result_map["selected"] = selected_result
        postprocess_selection_rows.append(selection_payload)

        for result_tag, result in result_map.items():
            metrics = raw_metrics if result_tag == "raw" else (smooth_metrics if result_tag == "smoothed" else selected_metrics)
            n1_idx = labels.index("N1") if "N1" in labels else 1
            rem_idx = labels.index("REM") if "REM" in labels else (num_classes - 1)
            _save_result_bundle(
                split_dir,
                split_idx,
                result_tag,
                result,
                labels,
                num_classes,
                wake_label,
                rem_label,
                primary=(result_tag == "selected"),
            )
            summary_rows.append(
                {
                    "split": split_idx,
                    "result_tag": result_tag,
                    "model_name": model_name,
                    **metrics,
                    "num_samples": int(len(result["y_true"])),
                    "expected_test_samples": expected_test_samples,
                    "train_class_counts": _json_counts(train_counts, labels),
                    "val_class_counts": _json_counts(val_counts, labels),
                    "test_class_counts": _json_counts(test_counts, labels),
                    "feasible": feasible,
                    "recommended_min_count": recommended_min_count,
                    "difficulty_note": _difficulty_note(train_counts, val_counts, test_counts, recommended_min_count),
                    "model_hparams": json.dumps(model_hparams, ensure_ascii=False, separators=(",", ":")),
                    "N1_f1": float(metrics.get(f"class_{n1_idx}_f1", np.nan)),
                    "N1_recall": float(metrics.get(f"class_{n1_idx}_recall", np.nan)),
                    "REM_f1": float(metrics.get(f"class_{rem_idx}_f1", np.nan)),
                    "postprocess_used": bool(result_tag == "smoothed" or (result_tag == "selected" and selected_result_tag != "raw")),
                    "total_params": float(complexity.get("total_params", np.nan)),
                    "trainable_params": float(complexity.get("trainable_params", np.nan)),
                    "estimated_MACs": float(complexity.get("estimated_MACs", np.nan)),
                    "avg_firing_rate": float(complexity.get("avg_firing_rate", np.nan)),
                    "spike_sparsity": float(complexity.get("spike_sparsity", np.nan)),
                    "inference_latency_ms": float(complexity.get("inference_latency_ms", np.nan)),
                    "checkpoint_size_mb": float(complexity.get("checkpoint_size_mb", np.nan)),
                    "layer_firing_rates": json.dumps(complexity.get("layer_firing_rates", {}), ensure_ascii=False),
                    "skipped_reason": "",
                }
            )
            merged.setdefault(result_tag, {"y_true": [], "y_pred": []})
            merged[result_tag]["y_true"].append(np.asarray(result["y_true"], dtype=np.int64))
            merged[result_tag]["y_pred"].append(np.asarray(result["y_pred"], dtype=np.int64))

        if split_idx == 3:
            split3_payload = {
                "train_counts": train_counts.tolist(),
                "val_counts": val_counts.tolist(),
                "test_counts": test_counts.tolist(),
                "raw_result": raw_result,
                "smoothed_result": smoothed_result,
                "selected_result": selected_result,
                "raw_metrics": raw_metrics,
                "smoothed_metrics": smooth_metrics,
                "selected_metrics": selected_metrics,
            }

        primary = selected_result
        print(
            f"eval split={split_idx} tag={selected_result_tag} "
            f"acc={float(np.mean(primary['y_true'] == primary['y_pred'])):.4f} "
            f"macro_f1={float(f1_score(primary['y_true'], primary['y_pred'], average='macro', zero_division=0)):.4f}"
        )

    summary_df = _append_group_mean_std(pd.DataFrame(summary_rows))
    summary_path = eval_root / "summary_metrics_detailed.csv"
    summary_df.to_csv(summary_path, index=False, **csv_utf8_sig_kwargs())
    save_json(eval_root / "postprocess_selection.json", {"folds": postprocess_selection_rows})
    save_json(eval_root / "split3_diagnosis.json", _build_split3_diagnosis(labels, global_counts, split3_payload))
    _write_change_proof_outputs(run_dir, summary_df, labels)
    selected_test_df = _write_selected_test_reports(run_dir, eval_root, summary_df, labels)

    primary_merged_tag = "selected" if "selected" in merged else ("smoothed" if "smoothed" in merged else "raw")
    merged_totals = {}
    for result_tag, payload in merged.items():
        cm = _save_merged_confusion(
            eval_root,
            result_tag,
            labels,
            payload["y_true"],
            payload["y_pred"],
            primary=(result_tag == primary_merged_tag),
        )
        merged_totals[result_tag] = 0 if cm is None else int(cm.sum())

    valid_rows = summary_df[pd.to_numeric(summary_df["split"], errors="coerce").notna()].copy()
    if not valid_rows.empty:
        valid_rows["split"] = valid_rows["split"].astype(int)

    if primary_merged_tag in merged and not valid_rows.empty:
        primary_rows = valid_rows[valid_rows["result_tag"] == primary_merged_tag]
        merged_total = merged_totals.get(primary_merged_tag, 0)
        summary_total = int(pd.to_numeric(primary_rows["num_samples"], errors="coerce").fillna(0).sum())
        if merged_total != summary_total:
            raise RuntimeError(
                f"merged confusion matrix total mismatch: merged={merged_total} summary={summary_total} "
                f"tag={primary_merged_tag}"
            )
    else:
        primary_rows = valid_rows

    if not selected_test_df.empty:
        mean_acc = float(pd.to_numeric(selected_test_df["test_acc"], errors="coerce").mean())
        mean_macro_f1 = float(pd.to_numeric(selected_test_df["test_macro_f1"], errors="coerce").mean())
        mean_kappa = float(pd.to_numeric(selected_test_df["test_kappa"], errors="coerce").mean())
    else:
        mean_acc = float(pd.to_numeric(primary_rows["accuracy"], errors="coerce").mean()) if not primary_rows.empty else float("nan")
        mean_macro_f1 = (
            float(pd.to_numeric(primary_rows["macro_f1"], errors="coerce").mean()) if not primary_rows.empty else float("nan")
        )
        mean_kappa = float(pd.to_numeric(primary_rows["kappa"], errors="coerce").mean()) if not primary_rows.empty else float("nan")

    current_profile = _load_eval_profile(run_dir)
    baseline_result_check = _write_baseline_result_check(run_dir, current_profile)
    print(
        f"eval: run_dir={run_dir} summary={eval_root / 'summary_metrics.csv'} "
        f"mean_acc={mean_acc:.4f} mean_macro_f1={mean_macro_f1:.4f} mean_kappa={mean_kappa:.4f} "
        f"status={baseline_result_check.get('status', 'unknown')}"
    )
    if baseline_result_check.get("status") == "failed_same_results_as_baseline":
        raise RuntimeError(json.dumps(baseline_result_check, ensure_ascii=False))


if __name__ == "__main__":
    main()

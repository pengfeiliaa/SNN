# -*- coding: utf-8 -*-
"""Sleep-EDF training entrypoint (PicoSleepNet baseline first)."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

ROOT = Path(__file__).resolve().parents[1]

import argparse
import csv
import copy
import hashlib
import json
import multiprocessing as mp
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from preprocess_sleep_edf import get_preprocess_cache_status
from eco_sleep import get_labels, get_num_classes, get_task_name, get_wake_label
from eco_sleep.data.sleep_edf.preprocessing import (
    default_lcs_delta,
    lcs_counts_to_binary,
    lcs_encode_epoch_counts,
    normalize_edf_subset,
)
from eco_sleep.data.sleep_edf.storage import json_dumps, list_processed_records, load_labels_from_npz, safe_meta
from eco_sleep.data.sleep_edf.splits import default_kfold_by_subset, make_epoch_random_split, make_kfold_splits
from eco_sleep.models import ContextPicoSNN, ContextPicoSNNV2, PicoSleepNetBaseline, PicoSleepNetPlusSNN
from eco_sleep.models.losses import (
    build_loss,
    compute_class_prior,
    compute_class_weights,
    logits_consistency_kl_loss,
    soft_target_cross_entropy,
    summarize_loss_setup,
    temporal_consistency_kl_loss,
)
from eco_sleep.train import (
    CollapseProtector,
    apply_collapse_stabilization,
    load_checkpoint,
    run_inference,
    save_checkpoint,
)
from eco_sleep.train.trainer import (
    ModelEMA,
    has_consecutive_effective_zeros,
    loss_is_effectively_zero,
    named_gradient_summaries,
    named_parameter_summaries,
)
from eco_sleep.utils.encoding_fix import setup_utf8_stdio, suppress_pin_memory_warning
from eco_sleep.utils.io import (
    append_jsonl,
    build_run_dir,
    clear_pending_run,
    ensure_dir,
    read_pending_run,
    read_yaml,
    save_yaml,
    save_json,
    try_git_commit_hash,
    write_last_run,
)
from eco_sleep.utils.seed import set_seed


@dataclass(frozen=True)
class EpochEntry:
    path: str
    subject_id: str
    record_id: str
    epoch_idx: int
    label: int
    subset: str


BASELINE_LOCKED_RECIPE = "baseline_locked"
SAFE_EMA_RECIPE = "baseline_locked_ema"
SAFE_LDAM_RECIPE = "baseline_locked_ldam"
SAFE_BOUNDARY_SOFT_LABEL_RECIPE = "baseline_locked_boundary_soft_label"
ROOT_CAUSE_FIX_RECIPE = "root_cause_fix"
ROOT_CAUSE_FIX_TEMPORAL_RECIPE = "root_cause_fix_temporal_consistency"
ROOT_CAUSE_FIX_THRESHOLD_RECIPE = "root_cause_fix_learnable_threshold"
REAL_STRATEGY_RECIPE = "real_strategy_logit_adjust_threshold"
CONTEXT_PICO_RECIPE = "context_pico_v1"
CONTEXT_PICO_LDAM_RECIPE = "context_pico_v1_ldam"
CONTEXT_PICO_TC_RECIPE = "context_pico_v1_cb_focal_tc"
CONTEXT_PICO_LDAM_DRW_RECIPE = "context_pico_v1_ldam_drw"
CONTEXT_PICO_LDAM_DRW_EMA_RECIPE = "context_pico_v1_ldam_drw_ema"
CONTEXT_PICO_LDAM_DRW_EMA_SELECT_RECIPE = "context_pico_v1_ldam_drw_ema_select"
CONTEXT_PICO_TRAIN_TIME_OPT_RECIPE = "context_pico_v1_train_time_opt"
CONTEXT_PICO_V2_CB_FOCAL_RECIPE = "context_pico_v2_cb_focal"
CONTEXT_PICO_V2_LDAM_DRW_RECIPE = "context_pico_v2_ldam_drw"
CONTEXT_PICO_V2_TC_RECIPE = "context_pico_v2_ldam_drw_tc"
CONTEXT_PICO_V2_SHORT_CONTEXT_RECIPE = "context_pico_v2_ldam_drw_short_context"
BASELINE_RUN_DIR = (ROOT / "runs" / "20260405_162343_sleep_edf_context_pico").resolve()
ACTIVE_TOUCHED_FILES = [
    "scripts/train_sleep_edf.py",
    "scripts/eval_sleep_edf.py",
    "src/eco_sleep/train/trainer.py",
    "src/eco_sleep/models/losses.py",
    "src/eco_sleep/models/picosleepnet_baseline.py",
    "src/eco_sleep/models/picosleepnet_plus_snn.py",
    "src/eco_sleep/models/context_pico_snn.py",
    "src/eco_sleep/models/context_pico_snn_v2.py",
    "src/eco_sleep/models/__init__.py",
    "README.md",
    "SLEEP_EDF20_SNN_CHAPTER.md",
]
ACTIVE_FILE_HASH_BEFORE = {
    "scripts/train_sleep_edf.py": "1CAFCD1C481A8E2945AFCA102C830BFEF4A571F29EAC3C55F38A6050213904DF",
    "scripts/eval_sleep_edf.py": "CBF3744CA227834EB9CA11FE9C9342D0C878984368667C16D61B6CD465780F00",
    "src/eco_sleep/train/trainer.py": "B545DBCF335F71A4FD872E4AD8E80D20D2A3B4F50D4BEC5DC808F5F6A618F79B",
    "src/eco_sleep/models/losses.py": "B192DADC69C018B0ABBCA98162475DC7124ED97ECB0BD581C4AC27D62EED06B4",
    "src/eco_sleep/models/picosleepnet_baseline.py": "1DEAF9B7718147D11E19795AD324A4217155E2037F3BF2DBDA1DE57EC89B9B66",
    "src/eco_sleep/models/picosleepnet_plus_snn.py": "5D35403D58DD84EC0F1E4FD8DAFC3DCF1CF740FB1A130D095296DA45B91A57F8",
    "README.md": "DE4C465D623AA2B6884B3DC59E75A2305A12BC4E56F39DAB154B6C4A39EA1949",
    "SLEEP_EDF20_SNN_CHAPTER.md": "3624E57642D9C4E0D68CCD2E8C4C75D5A021E26F582A1E8D429223B0DAF8C2CD",
}
SAFE_RECIPE_CHOICES = [
    "legacy_preset",
    BASELINE_LOCKED_RECIPE,
    SAFE_EMA_RECIPE,
    SAFE_LDAM_RECIPE,
    SAFE_BOUNDARY_SOFT_LABEL_RECIPE,
    ROOT_CAUSE_FIX_RECIPE,
    ROOT_CAUSE_FIX_TEMPORAL_RECIPE,
    ROOT_CAUSE_FIX_THRESHOLD_RECIPE,
    REAL_STRATEGY_RECIPE,
    CONTEXT_PICO_RECIPE,
    CONTEXT_PICO_LDAM_RECIPE,
    CONTEXT_PICO_TC_RECIPE,
    CONTEXT_PICO_LDAM_DRW_RECIPE,
    CONTEXT_PICO_LDAM_DRW_EMA_RECIPE,
    CONTEXT_PICO_LDAM_DRW_EMA_SELECT_RECIPE,
    CONTEXT_PICO_TRAIN_TIME_OPT_RECIPE,
    CONTEXT_PICO_V2_CB_FOCAL_RECIPE,
    CONTEXT_PICO_V2_LDAM_DRW_RECIPE,
    CONTEXT_PICO_V2_TC_RECIPE,
    CONTEXT_PICO_V2_SHORT_CONTEXT_RECIPE,
]

MACRO_N1_REM_GUARDED_CKPT_RULE = "macro_f1_n1_rem_guarded"
MACRO_N1_REM_GUARDED_SCORE_FORMULA = "0.60*val_macro_f1+0.20*val_N1_f1+0.10*val_kappa+0.10*val_REM_f1"


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_meta(meta_raw: object) -> dict:
    return safe_meta(meta_raw)


def _load_labels_from_npz(path: Path) -> np.ndarray:
    return load_labels_from_npz(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _active_file_hash_summary() -> dict[str, str]:
    summary: dict[str, str] = {}
    for relative_path in ACTIVE_TOUCHED_FILES:
        path = ROOT / relative_path
        if path.exists():
            summary[relative_path] = _sha256_file(path)
    return summary


def _write_file_hash_proof(change_proof_dir: Path) -> dict[str, dict[str, str]]:
    current_hashes = _active_file_hash_summary()
    payload = {
        relative_path: {
            "before": ACTIVE_FILE_HASH_BEFORE.get(relative_path, ""),
            "after": current_hashes.get(relative_path, ""),
        }
        for relative_path in ACTIVE_TOUCHED_FILES
        if relative_path in current_hashes or relative_path in ACTIVE_FILE_HASH_BEFORE
    }
    save_json(change_proof_dir / "file_hash_before_after.json", payload)
    return payload


def _ensure_not_baseline_run_dir(path: Path, purpose: str) -> None:
    resolved = path.resolve()
    if resolved == BASELINE_RUN_DIR or BASELINE_RUN_DIR in resolved.parents:
        raise RuntimeError(f"{purpose} points to the locked baseline run_dir and is forbidden: {resolved}")


def _model_role_summary(model) -> tuple[list[str], list[str], bool]:
    snn_core = list(getattr(model, "snn_core_layers", []))
    non_spiking = list(getattr(model, "non_spiking_aux_layers", []))
    sequence_context_enabled = bool(getattr(model, "sequence_context_enabled", False))

    if not snn_core and isinstance(model, PicoSleepNetPlusSNN):
        snn_core = ["w_in", "w_rec", "w_hid", "w_out"]
        non_spiking = ["shared_proj", "branch_gate", "branch_norm", "sleep_wake_head", "rem_head", "transition_logits"]
    elif not snn_core and isinstance(model, PicoSleepNetBaseline):
        snn_core = ["w_in", "w_rec", "w_hid", "w_out"]
        non_spiking = []
    return snn_core, non_spiking, sequence_context_enabled


def _write_active_method_report(
    change_proof_dir: Path,
    run_dir: Path,
    model,
    loss_fn: nn.Module,
    train_cfg: dict,
) -> None:
    snn_core_layers, non_spiking_aux_layers, sequence_context_enabled = _model_role_summary(model)
    summary = {
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "new_run_dir": str(run_dir),
        "active_model_class": type(model).__name__,
        "active_loss_class": type(loss_fn).__name__,
        "snn_core_layers": snn_core_layers,
        "non_spiking_aux_layers": non_spiking_aux_layers,
        "temporal_consistency_enabled": bool(_to_bool(train_cfg.get("temporal_consistency_enable", False))),
        "sequence_context_enabled": bool(sequence_context_enabled),
        "distillation_enabled": bool(_to_bool(train_cfg.get("distillation_enable", False))),
        "learnable_tau_enabled": bool(_to_bool(train_cfg.get("learnable_tau", False))),
        "learnable_threshold_enabled": bool(_to_bool(getattr(model, "learnable_threshold", False))),
        "best_ckpt_metric_name": str(train_cfg.get("best_ckpt_metric_name", "macro_f1_n1_f1_rem_f1_kappa_guarded_score")),
        "best_ckpt_selection_rule": str(
            train_cfg.get(
                "best_ckpt_rule_description",
                f"score={MACRO_N1_REM_GUARDED_SCORE_FORMULA} with val_N1_f1>=0.30 and val_N1_recall>=0.25; fallback=top3_selection_score_state_averaging",
            )
        ),
        "touched_active_files": ACTIVE_TOUCHED_FILES,
        "forbidden_old_checkpoint_reuse": True,
    }
    save_json(change_proof_dir / "active_method_report.json", summary)


def _assert_preprocess_cache_current(cfg: dict, processed_dir: Path, num_classes: int) -> dict:
    cache_status = get_preprocess_cache_status(cfg, out_dir=processed_dir, num_classes=num_classes)
    if cache_status.get("reuse_available", False):
        return cache_status
    reason = str(cache_status.get("reason", "unknown"))
    manifest_path = str(cache_status.get("manifest_path", processed_dir / "preprocess_manifest.json"))
    raise RuntimeError(
        "检测到当前 processed 数据与预处理代码或关键配置不一致，"
        "请先重新运行 preprocess_sleep_edf.py。"
        f" reason={reason} manifest={manifest_path}"
    )


def load_records(processed_dir: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in list_processed_records(processed_dir):
        records.append(
            {
                "path": str(row.path),
                "subject_id": row.subject_id,
                "record_id": row.record_id,
                "cohort": row.cohort,
                "subset": row.subset or "edfx_all",
                "n_epochs": int(row.n_epochs),
                "schema_version": int(row.schema_version),
                "has_epoch_stage_desc": bool(row.has_epoch_stage_desc),
                "has_raw_epoch": bool(row.has_raw_epoch),
                "has_lcs": bool(row.has_lcs),
                "is_legacy": bool(row.is_legacy),
            }
        )
    return records


def build_epoch_entries(records: List[Dict[str, object]], num_classes: int) -> List[EpochEntry]:
    entries: List[EpochEntry] = []
    for rec in records:
        path = Path(str(rec["path"]))
        labels = _load_labels_from_npz(path)
        bad = np.where((labels < 0) | (labels >= num_classes))[0]
        if bad.size > 0:
            i0 = int(bad[0])
            v0 = int(labels[i0])
            raise RuntimeError(
                f"标签越界: file={path} epoch_index={i0} label={v0} expected=[0,{num_classes - 1}]"
            )
        for i, y in enumerate(labels.tolist()):
            entries.append(
                EpochEntry(
                    path=str(path),
                    subject_id=str(rec["subject_id"]),
                    record_id=str(rec["record_id"]),
                    epoch_idx=int(i),
                    label=int(y),
                    subset=str(rec.get("subset", "edfx_all")),
                )
            )
    return entries


def compute_subject_label_counts(records: List[Dict[str, object]], num_classes: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for rec in records:
        sid = str(rec["subject_id"])
        if sid not in out:
            out[sid] = np.zeros(num_classes, dtype=np.int64)
        labels = _load_labels_from_npz(Path(str(rec["path"])))
        out[sid] += np.bincount(labels, minlength=num_classes).astype(np.int64)
    return out


def _resolve_subset(cfg: dict, records: List[Dict[str, object]]) -> str:
    dataset_cfg = cfg.get("dataset", {})
    subset = dataset_cfg.get("edf_subset", cfg.get("edf_subset", None))
    if subset is None or str(subset).strip() == "":
        from_meta = [str(r.get("subset", "")) for r in records if str(r.get("subset", "")).strip()]
        if from_meta:
            subset = from_meta[0]
        else:
            n_subjects = len({str(r["subject_id"]) for r in records})
            subset = "edf20" if n_subjects <= 20 else "edf78"
    return normalize_edf_subset(str(subset))


def _json_counts(counts: np.ndarray, labels: List[str]) -> str:
    obj = {labels[i]: int(counts[i]) for i in range(len(labels))}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _canonical_model_name(model_name: str | None) -> str:
    raw = str(model_name or "").lower().strip()
    if raw in {"picosleepnet_baseline", "picosleepnet"}:
        return "picosleepnet_baseline"
    if raw in {"", "context_pico_snn", "context_pico", "picosleepnet_lite_snn", "multiscale_pico_snn"}:
        return "context_pico_snn"
    if raw in {"context_pico_snn_v2", "context_pico_v2", "multiscale_context_pico_snn"}:
        return "context_pico_snn_v2"
    if raw == "picosleepnet_rsnn":
        return "picosleepnet_baseline"
    if raw == "picosleepnet_plus_snn":
        return "picosleepnet_plus_snn"
    return raw


def _default_train_model_name(model_name: str | None) -> str:
    canonical = _canonical_model_name(model_name)
    if canonical in {"", "picosleepnet_baseline"}:
        return "context_pico_snn"
    return canonical


def _preset_model_name(preset: str) -> str:
    normalized = str(preset).lower().strip()
    if normalized == "baseline":
        return "picosleepnet_baseline"
    if normalized in {"plus_without_transition", "plus_full"}:
        return "picosleepnet_plus_snn"
    if normalized == "context_pico":
        return "context_pico_snn"
    if normalized == "context_pico_v2":
        return "context_pico_snn_v2"
    raise ValueError(f"unknown preset: {preset}")


def _apply_training_preset(cfg: dict, preset: str) -> dict:
    preset_name = str(preset).lower().strip()
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg.setdefault("picosleepnet_plus_snn", {})
    cfg.setdefault("picosleepnet_baseline", {})
    cfg.setdefault("picosleepnet_rsnn", {})
    cfg.setdefault("context_pico_snn", {})
    cfg.setdefault("context_pico_snn_v2", {})

    if preset_name == "baseline":
        cfg["model"]["name"] = "picosleepnet_baseline"
        cfg["train"].setdefault("label_smoothing", 0.0)
        cfg["train"].setdefault("label_smoothing_weight", 0.0)
        cfg["train"].setdefault("use_aux_heads", False)
        cfg["train"].setdefault("firing_reg_weight", 0.0)
        return cfg

    if preset_name == "context_pico":
        model_cfg = cfg["model"]
        loss_cfg = cfg["loss"]
        train_cfg = cfg["train"]
        context_cfg = cfg["context_pico_snn"]

        model_cfg["name"] = "context_pico_snn"
        loss_cfg["name"] = str(loss_cfg.get("name", "cb_focal")).lower().strip() or "cb_focal"
        if loss_cfg["name"] == "ce":
            loss_cfg["name"] = "cb_focal"
        loss_cfg["use_class_weights"] = False
        loss_cfg.setdefault("gamma", 2.0)
        loss_cfg.setdefault("class_weight_beta", 0.999)

        train_cfg["sampler"] = "weighted"
        train_cfg.setdefault("sampler_balance_power", 1.0)
        train_cfg["use_aux_heads"] = True
        train_cfg.setdefault("firing_reg_weight", 0.004)
        train_cfg.setdefault("firing_target_low", 0.03)
        train_cfg.setdefault("firing_target_high", 0.20)
        train_cfg.setdefault("aux_weight_sleepwake", 0.08)
        train_cfg.setdefault("aux_weight_rem", 0.06)
        train_cfg.setdefault("input_norm_mode", "record_zscore")
        train_cfg.setdefault("label_smoothing", 0.0)
        train_cfg.setdefault("label_smoothing_weight", 0.0)

        context_cfg.setdefault("context_len", 3)
        context_cfg.setdefault("branch_channels", 4)
        context_cfg.setdefault("stem_channels", 12)
        context_cfg.setdefault("token_dim", 24)
        context_cfg.setdefault("epoch_hidden_size", 96)
        context_cfg.setdefault("epoch_embed_dim", 48)
        context_cfg.setdefault("context_hidden_size", 32)
        context_cfg.setdefault("t_steps", 40)
        context_cfg.setdefault("tau_epoch", 0.92)
        context_cfg.setdefault("tau_context", 0.90)
        context_cfg.setdefault("v_th_epoch", 1.0)
        context_cfg.setdefault("v_th_context", 1.0)
        context_cfg.setdefault("surrogate_alpha", 4.0)
        context_cfg.setdefault("dropout", 0.10)
        context_cfg.setdefault("use_aux_heads", True)
        context_cfg.setdefault("center_residual_weight", 0.35)
        return cfg

    if preset_name == "context_pico_v2":
        model_cfg = cfg["model"]
        loss_cfg = cfg["loss"]
        train_cfg = cfg["train"]
        context_cfg = cfg["context_pico_snn_v2"]

        model_cfg["name"] = "context_pico_snn_v2"
        loss_cfg["name"] = str(loss_cfg.get("name", "cb_focal")).lower().strip() or "cb_focal"
        if loss_cfg["name"] == "ce":
            loss_cfg["name"] = "cb_focal"
        loss_cfg["use_class_weights"] = False
        loss_cfg.setdefault("gamma", 2.0)
        loss_cfg.setdefault("class_weight_beta", 0.999)
        loss_cfg.setdefault("ldam_max_margin", 0.35)
        loss_cfg.setdefault("ldam_scale", 18.0)
        loss_cfg.setdefault("drw_start_ratio", 0.5)
        loss_cfg.setdefault("drw_weight_strategy", "effective_num")

        train_cfg["sampler"] = "weighted"
        train_cfg.setdefault("sampler_balance_power", 1.0)
        train_cfg["use_aux_heads"] = True
        train_cfg.setdefault("firing_reg_weight", 0.003)
        train_cfg.setdefault("firing_target_low", 0.03)
        train_cfg.setdefault("firing_target_high", 0.18)
        train_cfg.setdefault("aux_weight_sleepwake", 0.07)
        train_cfg.setdefault("aux_weight_rem", 0.05)
        train_cfg.setdefault("aux_weight_n1", 0.08)
        train_cfg.setdefault("input_norm_mode", "record_zscore")
        train_cfg.setdefault("label_smoothing", 0.0)
        train_cfg.setdefault("label_smoothing_weight", 0.0)
        train_cfg.setdefault("best_ckpt_n1_f1_floor", 0.15)
        train_cfg.setdefault("best_ckpt_n1_recall_floor", 0.12)
        train_cfg.setdefault("tc_zero_guard_epochs", 2)
        train_cfg.setdefault("tc_zero_guard_tol", 1e-8)

        context_cfg.setdefault("context_len", 1)
        context_cfg.setdefault("branch_channels", 6)
        context_cfg.setdefault("stem_channels", 16)
        context_cfg.setdefault("token_dim", 32)
        context_cfg.setdefault("epoch_hidden_size", 96)
        context_cfg.setdefault("epoch_embed_dim", 48)
        context_cfg.setdefault("context_hidden_size", 40)
        context_cfg.setdefault("t_steps", 48)
        context_cfg.setdefault("tau_epoch", 0.92)
        context_cfg.setdefault("tau_context", 0.90)
        context_cfg.setdefault("v_th_epoch", 1.0)
        context_cfg.setdefault("v_th_context", 1.0)
        context_cfg.setdefault("surrogate_alpha", 4.0)
        context_cfg.setdefault("dropout", 0.10)
        context_cfg.setdefault("use_aux_heads", True)
        context_cfg.setdefault("use_n1_aux", True)
        context_cfg.setdefault("center_residual_weight", 0.40)
        context_cfg.setdefault("kernel_sizes", [15, 31, 63])
        return cfg

    plus_cfg = cfg["picosleepnet_plus_snn"]
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    model_cfg = cfg["model"]

    model_cfg["name"] = "picosleepnet_plus_snn"
    model_cfg["use_dual_lcs"] = True
    model_cfg["use_transition_matrix"] = preset_name == "plus_full"
    model_cfg.setdefault("delta_small_ratio", 0.65)
    model_cfg.setdefault("delta_large_ratio", 1.35)

    loss_cfg["name"] = str(loss_cfg.get("name", "cb_focal")).lower().strip() or "cb_focal"
    if loss_cfg["name"] == "ce":
        loss_cfg["name"] = "cb_focal"
    loss_cfg["use_class_weights"] = False
    loss_cfg.setdefault("gamma", 2.0)
    loss_cfg.setdefault("class_weight_beta", 0.999)

    train_cfg["sampler"] = "weighted"
    train_cfg.setdefault("sampler_balance_power", 1.0)
    train_cfg["use_aux_heads"] = True
    train_cfg["label_smoothing"] = float(train_cfg["label_smoothing"]) if "label_smoothing" in train_cfg else 0.05
    train_cfg["label_smoothing_weight"] = (
        float(train_cfg["label_smoothing_weight"]) if "label_smoothing_weight" in train_cfg else 0.25
    )
    train_cfg["firing_reg_weight"] = float(train_cfg["firing_reg_weight"]) if "firing_reg_weight" in train_cfg else 0.015
    train_cfg["firing_target_low"] = float(train_cfg["firing_target_low"]) if "firing_target_low" in train_cfg else 0.05
    train_cfg["firing_target_high"] = float(train_cfg["firing_target_high"]) if "firing_target_high" in train_cfg else 0.20
    train_cfg["transition_aux_weight"] = 0.05 if preset_name == "plus_full" else 0.0
    train_cfg.setdefault("aux_weight_sleepwake", 0.15)
    train_cfg.setdefault("aux_weight_rem", 0.10)

    plus_cfg.setdefault("use_dual_lcs", True)
    plus_cfg["use_transition_matrix"] = preset_name == "plus_full"
    plus_cfg.setdefault("dual_proj_dim", plus_cfg.get("input_neurons_each", 40))
    plus_cfg.setdefault("transition_residual_weight", 0.25)
    plus_cfg.setdefault("transition_aux_weight", 0.05)
    plus_cfg.setdefault("firing_target_low", 0.05)
    plus_cfg.setdefault("firing_target_high", 0.20)
    plus_cfg.setdefault("aux_sleep_weight", 0.15)
    plus_cfg.setdefault("aux_rem_weight", 0.10)
    return cfg


def _default_recipe_for_preset(preset: str) -> str:
    preset_name = str(preset).lower().strip()
    if preset_name == "context_pico_v2":
        return CONTEXT_PICO_V2_SHORT_CONTEXT_RECIPE
    if preset_name == "plus_full":
        return REAL_STRATEGY_RECIPE
    if preset_name == "context_pico":
        return CONTEXT_PICO_LDAM_RECIPE
    return "legacy_preset"


def _apply_context_pico_train_time_strategy(
    cfg: dict,
    *,
    enable_drw: bool,
    enable_ema: bool,
    enable_new_selection: bool,
    enable_tc: bool,
) -> dict:
    cfg = _apply_training_preset(cfg, "context_pico")
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]

    loss_cfg["name"] = "ldam_drw" if enable_drw else "ldam"
    loss_cfg["ldam_max_margin"] = 0.35
    loss_cfg["ldam_scale"] = 18.0
    loss_cfg["drw_start_ratio"] = 0.5
    loss_cfg["drw_weight_strategy"] = "effective_num"
    loss_cfg["use_class_weights"] = False

    train_cfg["ema_enable"] = bool(enable_ema)
    train_cfg["ema_decay"] = 0.999
    train_cfg["ema_use_for_eval"] = bool(enable_ema)
    train_cfg["ema_strategy"] = "ema"
    train_cfg["best_ckpt_rule"] = MACRO_N1_REM_GUARDED_CKPT_RULE if enable_new_selection else "legacy_lexicographic"
    train_cfg["best_ckpt_metric_name"] = (
        "macro_f1_n1_f1_rem_f1_kappa_guarded_score"
        if enable_new_selection
        else "val_macro_f1_then_kappa_then_n1_f1_then_n1_recall"
    )
    train_cfg["best_ckpt_rule_description"] = (
        f"score={MACRO_N1_REM_GUARDED_SCORE_FORMULA} with val_N1_f1>=0.30 and val_N1_recall>=0.25; fallback=top3_selection_score_state_averaging"
        if enable_new_selection
        else "eligible if val_N1_f1 >= 0.15 and val_N1_recall >= 0.12; then max(val_macro_f1, val_kappa, val_N1_f1, val_N1_recall, val_acc)"
    )
    train_cfg["best_ckpt_macro_f1_weight"] = 0.60
    train_cfg["best_ckpt_n1_f1_weight"] = 0.20
    train_cfg["best_ckpt_kappa_weight"] = 0.10
    train_cfg["best_ckpt_rem_f1_weight"] = 0.10 if enable_new_selection else 0.0
    train_cfg["best_ckpt_n1_f1_floor"] = 0.30 if enable_new_selection else 0.15
    train_cfg["best_ckpt_n1_recall_floor"] = 0.25 if enable_new_selection else 0.12
    train_cfg["best_ckpt_fallback_topk"] = 3 if enable_new_selection else 0

    train_cfg["temporal_consistency_enable"] = bool(enable_tc)
    train_cfg["temporal_consistency_weight"] = 0.02 if enable_tc else 0.0
    train_cfg["temporal_consistency_temperature"] = 1.0
    train_cfg["temporal_consistency_mode"] = "stochastic_forward"
    train_cfg["tc_zero_guard_epochs"] = 2
    train_cfg["tc_zero_guard_tol"] = 1e-8
    return cfg


def _apply_baseline_locked_defaults(cfg: dict) -> dict:
    cfg = _apply_training_preset(cfg, "plus_full")
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg.setdefault("picosleepnet_plus_snn", {})

    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    train_cfg = cfg["train"]
    plus_cfg = cfg["picosleepnet_plus_snn"]

    model_cfg["name"] = "picosleepnet_plus_snn"
    model_cfg["use_dual_lcs"] = True
    model_cfg["use_transition_matrix"] = True
    model_cfg["delta_small_ratio"] = 0.65
    model_cfg["delta_large_ratio"] = 1.35

    loss_cfg["name"] = "cb_focal"
    loss_cfg["use_class_weights"] = False
    loss_cfg["gamma"] = 2.0
    loss_cfg["class_weight_beta"] = 0.999
    loss_cfg.pop("tau", None)

    train_cfg["sampler"] = "weighted"
    train_cfg["sampler_balance_power"] = 1.0
    train_cfg["use_aux_heads"] = True
    train_cfg["label_smoothing"] = 0.05
    train_cfg["label_smoothing_weight"] = 0.25
    train_cfg["firing_reg_weight"] = 0.015
    train_cfg["firing_target_low"] = 0.05
    train_cfg["firing_target_high"] = 0.20
    train_cfg["transition_aux_weight"] = 0.05
    train_cfg["aux_weight_sleepwake"] = 0.15
    train_cfg["aux_weight_rem"] = 0.10
    train_cfg["ema_enable"] = False
    train_cfg["ema_decay"] = 0.999
    train_cfg["ema_use_for_eval"] = False
    train_cfg["boundary_soft_label_enable"] = False
    train_cfg["boundary_soft_label_weight"] = 0.0
    train_cfg["boundary_soft_primary_weight"] = 0.85
    train_cfg["boundary_soft_neighbor_weight"] = 0.15
    train_cfg["input_norm_mode"] = "none"
    train_cfg["boundary_sampling_boost"] = 1.0
    train_cfg["input_mixstyle_enable"] = False
    train_cfg["input_mixstyle_p"] = 0.0
    train_cfg["input_mixstyle_alpha"] = 0.3

    plus_cfg["use_dual_lcs"] = True
    plus_cfg["use_transition_matrix"] = True
    plus_cfg.setdefault("dual_proj_dim", int(plus_cfg.get("input_neurons_each", 40)))
    plus_cfg["transition_residual_weight"] = float(plus_cfg.get("transition_residual_weight", 0.25))
    plus_cfg["transition_aux_weight"] = 0.05
    plus_cfg["firing_target_low"] = 0.05
    plus_cfg["firing_target_high"] = 0.20
    plus_cfg["aux_sleep_weight"] = 0.15
    plus_cfg["aux_rem_weight"] = 0.10
    return cfg


def _apply_training_recipe(cfg: dict, preset: str, recipe: str) -> dict:
    recipe_name = str(recipe or "").lower().strip() or _default_recipe_for_preset(preset)
    if recipe_name == "legacy_preset":
        cfg.setdefault("train", {})
        cfg["train"]["recipe_name"] = recipe_name
        return cfg

    if recipe_name not in SAFE_RECIPE_CHOICES:
        raise ValueError(f"unknown recipe: {recipe}")

    preset_name = str(preset).lower().strip()
    if preset_name == "context_pico_v2":
        cfg = _apply_training_preset(cfg, "context_pico_v2")
        cfg["train"]["recipe_name"] = recipe_name
        cfg["train"]["temporal_consistency_enable"] = False
        cfg["train"]["temporal_consistency_weight"] = 0.0
        cfg["train"]["temporal_consistency_temperature"] = 1.0
        cfg["train"]["learnable_tau"] = False
        cfg["model"]["learnable_threshold"] = False
        cfg["context_pico_snn_v2"]["context_len"] = 1

        if recipe_name == CONTEXT_PICO_V2_CB_FOCAL_RECIPE:
            cfg["loss"]["name"] = "cb_focal"
            return cfg
        if recipe_name == CONTEXT_PICO_V2_LDAM_DRW_RECIPE:
            cfg["loss"]["name"] = "ldam_drw"
            cfg["loss"]["ldam_max_margin"] = 0.35
            cfg["loss"]["ldam_scale"] = 18.0
            return cfg
        if recipe_name == CONTEXT_PICO_V2_TC_RECIPE:
            cfg["loss"]["name"] = "ldam_drw"
            cfg["loss"]["ldam_max_margin"] = 0.35
            cfg["loss"]["ldam_scale"] = 18.0
            cfg["train"]["temporal_consistency_enable"] = True
            cfg["train"]["temporal_consistency_weight"] = 0.04
            cfg["train"]["temporal_consistency_temperature"] = 1.0
            return cfg
        if recipe_name == CONTEXT_PICO_V2_SHORT_CONTEXT_RECIPE:
            cfg["loss"]["name"] = "ldam_drw"
            cfg["loss"]["ldam_max_margin"] = 0.35
            cfg["loss"]["ldam_scale"] = 18.0
            cfg["context_pico_snn_v2"]["context_len"] = 5
            return cfg
        raise ValueError(f"recipe {recipe_name} is not compatible with preset={preset_name}")

    if preset_name == "context_pico":
        cfg = _apply_training_preset(cfg, "context_pico")
        cfg["train"]["recipe_name"] = recipe_name
        cfg["train"]["temporal_consistency_enable"] = False
        cfg["train"]["temporal_consistency_weight"] = 0.0
        cfg["train"]["temporal_consistency_temperature"] = 1.0
        cfg["train"]["temporal_consistency_mode"] = "step_logits"
        if recipe_name == CONTEXT_PICO_RECIPE:
            return cfg
        if recipe_name == CONTEXT_PICO_LDAM_RECIPE:
            cfg = _apply_context_pico_train_time_strategy(
                cfg,
                enable_drw=True,
                enable_ema=True,
                enable_new_selection=True,
                enable_tc=False,
            )
            cfg["train"]["recipe_name"] = recipe_name
            return cfg
        if recipe_name == CONTEXT_PICO_TC_RECIPE:
            cfg["loss"]["name"] = "cb_focal"
            cfg["train"]["temporal_consistency_enable"] = True
            cfg["train"]["temporal_consistency_weight"] = 0.03
            cfg["train"]["temporal_consistency_temperature"] = 1.0
            return cfg
        if recipe_name == CONTEXT_PICO_LDAM_DRW_RECIPE:
            cfg = _apply_context_pico_train_time_strategy(
                cfg,
                enable_drw=True,
                enable_ema=False,
                enable_new_selection=False,
                enable_tc=False,
            )
            cfg["train"]["recipe_name"] = recipe_name
            return cfg
        if recipe_name == CONTEXT_PICO_LDAM_DRW_EMA_RECIPE:
            cfg = _apply_context_pico_train_time_strategy(
                cfg,
                enable_drw=True,
                enable_ema=True,
                enable_new_selection=False,
                enable_tc=False,
            )
            cfg["train"]["recipe_name"] = recipe_name
            return cfg
        if recipe_name == CONTEXT_PICO_LDAM_DRW_EMA_SELECT_RECIPE:
            cfg = _apply_context_pico_train_time_strategy(
                cfg,
                enable_drw=True,
                enable_ema=True,
                enable_new_selection=True,
                enable_tc=False,
            )
            cfg["train"]["recipe_name"] = recipe_name
            return cfg
        if recipe_name == CONTEXT_PICO_TRAIN_TIME_OPT_RECIPE:
            cfg = _apply_context_pico_train_time_strategy(
                cfg,
                enable_drw=True,
                enable_ema=True,
                enable_new_selection=True,
                enable_tc=True,
            )
            cfg["train"]["recipe_name"] = recipe_name
            return cfg
        raise ValueError(f"recipe {recipe_name} is not compatible with preset={preset_name}")

    cfg = _apply_baseline_locked_defaults(cfg)
    cfg["train"]["recipe_name"] = recipe_name
    cfg["train"]["temporal_consistency_enable"] = False
    cfg["train"]["temporal_consistency_weight"] = 0.0
    cfg["train"]["temporal_consistency_temperature"] = 1.0
    cfg["train"]["learnable_tau"] = False
    cfg["model"]["learnable_threshold"] = False

    if recipe_name == BASELINE_LOCKED_RECIPE:
        return cfg
    if recipe_name == SAFE_EMA_RECIPE:
        cfg["train"]["ema_enable"] = True
        cfg["train"]["ema_decay"] = 0.999
        cfg["train"]["ema_use_for_eval"] = True
        return cfg
    if recipe_name == SAFE_LDAM_RECIPE:
        cfg["loss"]["name"] = "ldam"
        cfg["loss"]["ldam_max_margin"] = 0.35
        cfg["loss"]["ldam_scale"] = 20.0
        return cfg
    if recipe_name == SAFE_BOUNDARY_SOFT_LABEL_RECIPE:
        cfg["train"]["boundary_soft_label_enable"] = True
        cfg["train"]["boundary_soft_label_weight"] = 0.15
        cfg["train"]["boundary_soft_primary_weight"] = 0.85
        cfg["train"]["boundary_soft_neighbor_weight"] = 0.15
        return cfg
    if recipe_name == ROOT_CAUSE_FIX_RECIPE:
        # Evidence-based fix: stop stacking extra objectives before proving they help.
        cfg["model"]["use_transition_matrix"] = False
        cfg["train"]["use_aux_heads"] = False
        cfg["train"]["transition_aux_weight"] = 0.0
        cfg["train"]["label_smoothing"] = 0.0
        cfg["train"]["label_smoothing_weight"] = 0.0
        cfg["train"]["boundary_soft_label_enable"] = False
        cfg["train"]["boundary_soft_label_weight"] = 0.0
        cfg["train"]["input_norm_mode"] = "none"
        cfg["train"]["boundary_sampling_boost"] = 1.0
        cfg["train"]["input_mixstyle_enable"] = False
        cfg["train"]["input_mixstyle_p"] = 0.0
        cfg["train"]["input_mixstyle_alpha"] = 0.3
        return cfg
    if recipe_name == ROOT_CAUSE_FIX_TEMPORAL_RECIPE:
        cfg = _apply_training_recipe(cfg, preset, ROOT_CAUSE_FIX_RECIPE)
        cfg["train"]["recipe_name"] = recipe_name
        cfg["train"]["temporal_consistency_enable"] = True
        cfg["train"]["temporal_consistency_weight"] = 0.02
        cfg["train"]["temporal_consistency_temperature"] = 1.0
        return cfg
    if recipe_name == ROOT_CAUSE_FIX_THRESHOLD_RECIPE:
        cfg = _apply_training_recipe(cfg, preset, ROOT_CAUSE_FIX_RECIPE)
        cfg["train"]["recipe_name"] = recipe_name
        cfg["model"]["learnable_threshold"] = True
        cfg["model"]["threshold_per_neuron"] = True
        cfg["model"]["threshold_min"] = 0.35
        cfg["model"]["threshold_max"] = 2.25
        return cfg
    if recipe_name == REAL_STRATEGY_RECIPE:
        cfg["loss"]["name"] = "logit_adjusted_ce"
        cfg["loss"]["tau"] = 1.15
        cfg["loss"]["use_class_weights"] = False
        cfg["train"]["label_smoothing"] = 0.0
        cfg["train"]["label_smoothing_weight"] = 0.0
        cfg["train"]["learnable_tau"] = False
        cfg["model"]["learnable_threshold"] = True
        cfg["model"]["threshold_per_neuron"] = True
        cfg["model"]["threshold_min"] = 0.35
        cfg["model"]["threshold_max"] = 2.25
        return cfg
    return cfg


def _resolve_repo_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _parse_split_indices(raw_value: str | None) -> set[int] | None:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    out: set[int] = set()
    for part in str(raw_value).split(","):
        item = part.strip()
        if item == "":
            continue
        out.add(int(item))
    return out or None


def _make_weighted_sampler(
    labels: np.ndarray,
    num_classes: int,
    power: float = 1.0,
    boundary_flags: np.ndarray | None = None,
    boundary_boost: float = 1.0,
) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.power(counts, float(max(1e-3, power)))
    inv = inv / float(np.mean(inv))
    sample_weights = inv[labels]
    if boundary_flags is not None and float(boundary_boost) > 1.0:
        boosts = np.where(np.asarray(boundary_flags, dtype=bool), float(boundary_boost), 1.0)
        sample_weights = sample_weights * boosts
        sample_weights = sample_weights / float(np.mean(sample_weights))
    return WeightedRandomSampler(torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)


def _make_class_balanced_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = float(np.sum(counts)) / (float(num_classes) * counts)
    weights = weights / float(np.mean(weights))
    return torch.tensor(weights, dtype=torch.float32)


def _drw_active(loss_name: str, current_epoch: int | None, total_epochs: int | None, start_ratio: float) -> bool:
    if str(loss_name).lower().strip() != "ldam_drw":
        return False
    if current_epoch is None or total_epochs is None:
        return False
    switch_epoch = max(2, int(np.floor(float(total_epochs) * float(start_ratio))) + 1)
    return int(current_epoch) >= int(switch_epoch)


def _build_main_loss(
    cfg: dict,
    labels: np.ndarray,
    num_classes: int,
    sampler_name: str,
    device: torch.device,
    current_epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[nn.Module, dict]:
    loss_cfg = cfg.get("loss", {})
    loss_name = str(loss_cfg.get("name", "ce")).lower().strip() or "ce"
    class_counts = torch.tensor(np.bincount(labels, minlength=num_classes), dtype=torch.float32)
    class_prior = compute_class_prior(labels, num_classes=num_classes)
    drw_start_ratio = float(loss_cfg.get("drw_start_ratio", 0.5))
    drw_switch_epoch = None if total_epochs is None else max(2, int(np.floor(float(total_epochs) * drw_start_ratio)) + 1)

    if "use_class_weights" in loss_cfg:
        use_class_weights = bool(_to_bool(loss_cfg.get("use_class_weights", False)))
    else:
        use_class_weights = False if sampler_name == "weighted" else False

    class_weights = None
    drw_active = _drw_active(loss_name, current_epoch=current_epoch, total_epochs=total_epochs, start_ratio=drw_start_ratio)
    if use_class_weights or drw_active:
        class_weights = compute_class_weights(
            labels,
            num_classes=num_classes,
            strategy=str(loss_cfg.get("drw_weight_strategy", loss_cfg.get("class_weight_strategy", "effective_num"))),
            beta=float(loss_cfg.get("class_weight_beta", 0.999)),
        )

    loss_fn = build_loss(
        loss_name,
        class_weights=class_weights,
        gamma=float(loss_cfg.get("gamma", 2.0)),
        class_prior=class_prior,
        tau=float(loss_cfg.get("tau", 0.0)),
        class_counts=class_counts,
        beta=float(loss_cfg.get("class_weight_beta", 0.999)),
        max_margin=float(loss_cfg.get("ldam_max_margin", 0.35)),
        scale=float(loss_cfg.get("ldam_scale", 20.0)),
        num_classes=num_classes,
    ).to(device)
    summary = summarize_loss_setup(
        loss_name=loss_name,
        class_counts=class_counts,
        class_weights=class_weights,
        class_prior=class_prior,
        tau=float(loss_cfg.get("tau", 0.0)),
    )
    summary["sampler"] = sampler_name
    summary["drw_active"] = bool(drw_active)
    summary["use_class_weights"] = bool(use_class_weights or summary["drw_active"])
    summary["active_loss_class"] = type(loss_fn).__name__
    summary["current_epoch"] = None if current_epoch is None else int(current_epoch)
    summary["drw_switch_epoch"] = None if drw_switch_epoch is None else int(drw_switch_epoch)
    summary["current_loss_schedule"] = (
        "ldam_drw_reweight" if summary["drw_active"] else ("ldam_drw_warmup" if loss_name == "ldam_drw" else loss_name)
    )
    if loss_name == "ldam":
        summary["ldam_max_margin"] = float(loss_cfg.get("ldam_max_margin", 0.35))
        summary["ldam_scale"] = float(loss_cfg.get("ldam_scale", 20.0))
    if loss_name == "ldam_drw":
        summary["ldam_max_margin"] = float(loss_cfg.get("ldam_max_margin", 0.35))
        summary["ldam_scale"] = float(loss_cfg.get("ldam_scale", 20.0))
        summary["drw_start_ratio"] = float(drw_start_ratio)
        summary["drw_weight_strategy"] = str(loss_cfg.get("drw_weight_strategy", loss_cfg.get("class_weight_strategy", "effective_num")))
    return loss_fn, summary


def _amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def _amp_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and device.type == "cuda"))


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


def _mixstyle_channels(
    x: torch.Tensor,
    p: float = 0.0,
    alpha: float = 0.3,
    eps: float = 1e-5,
) -> torch.Tensor:
    if float(p) <= 0.0 or x.ndim != 3 or int(x.size(0)) < 2:
        return x
    if float(torch.rand((), device=x.device).item()) > float(p):
        return x

    feat = x
    mu = feat.mean(dim=-1, keepdim=True)
    sigma = feat.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
    feat_norm = (feat - mu) / sigma

    beta_alpha = max(float(alpha), 1e-3)
    beta_dist = torch.distributions.Beta(beta_alpha, beta_alpha)
    lam = beta_dist.sample((feat.size(0), 1, 1)).to(device=feat.device, dtype=feat.dtype)
    perm = torch.randperm(feat.size(0), device=feat.device)
    mu_mix = lam * mu + (1.0 - lam) * mu[perm]
    sigma_mix = lam * sigma + (1.0 - lam) * sigma[perm]
    return feat_norm * sigma_mix + mu_mix


class SleepEdfSpikeDataset(Dataset):
    """Per-epoch dataset for PicoSleepNet branches."""

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
            raise ValueError("context_len must be odd for center-epoch supervision.")
        self.context_half = self.context_len // 2
        self.labels = np.asarray([int(e.label) for e in self.entries], dtype=np.int64)
        self._record_cache: Dict[str, Dict[str, object]] = {}
        self._dynamic_lcs_cache: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        self._channel_stats_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._raw_stats_cache: Dict[str, Tuple[float, float]] = {}
        self.boundary_flags = self._build_boundary_flags()

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def all_labels(self) -> np.ndarray:
        return self.labels

    def _build_boundary_flags(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0,), dtype=bool)
        label_cache: Dict[str, np.ndarray] = {}
        flags = np.zeros((len(self.entries),), dtype=bool)
        for idx, entry in enumerate(self.entries):
            if entry.path not in label_cache:
                label_cache[entry.path] = _load_labels_from_npz(Path(entry.path))
            labels = label_cache[entry.path]
            epoch_idx = int(entry.epoch_idx)
            prev_label = int(labels[epoch_idx - 1]) if epoch_idx > 0 else int(labels[epoch_idx])
            next_label = int(labels[epoch_idx + 1]) if epoch_idx + 1 < labels.shape[0] else int(labels[epoch_idx])
            flags[idx] = bool(prev_label != int(entry.label) or next_label != int(entry.label))
        return flags

    def _load_record(self, path: str) -> Dict[str, object]:
        if path in self._record_cache:
            return self._record_cache[path]
        npz = np.load(path, allow_pickle=True)
        meta = _safe_meta(npz["meta"].item() if "meta" in npz.files else {})
        labels = npz["label"].astype(np.int64) if "label" in npz.files else npz["labels"].astype(np.int64)
        if "raw_epoch" in npz.files:
            raw_epoch = npz["raw_epoch"].astype(np.float32)
        else:
            raw_epoch = npz["signals"].astype(np.float32)[:, 0, :]
        record = {
            "raw_epoch": raw_epoch,
            "labels": labels,
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
        for i in range(n_epoch):
            pc, nc = lcs_encode_epoch_counts(raw_epoch[i], delta=delta)
            pb, nb = lcs_counts_to_binary(pc, nc)
            pos_count[i] = pc
            neg_count[i] = nc
            pos_bin[i] = pb
            neg_bin[i] = nb
        return {
            "lcs_pos_count": pos_count,
            "lcs_neg_count": neg_count,
            "lcs_pos": pos_bin,
            "lcs_neg": neg_bin,
        }

    def _get_lcs_for_delta(self, path: str, record: Dict[str, object], delta: float) -> Dict[str, np.ndarray]:
        record_delta = float(record.get("lcs_delta", self.delta_primary))
        has_precomputed = all(record.get(k) is not None for k in ("lcs_pos_count", "lcs_neg_count", "lcs_pos", "lcs_neg"))
        if has_precomputed and abs(record_delta - float(delta)) < 1e-8:
            return {
                "lcs_pos_count": record["lcs_pos_count"],
                "lcs_neg_count": record["lcs_neg_count"],
                "lcs_pos": record["lcs_pos"],
                "lcs_neg": record["lcs_neg"],
            }
        cache_key = (path, f"{float(delta):.6f}")
        if cache_key in self._dynamic_lcs_cache:
            return self._dynamic_lcs_cache[cache_key]
        raw_epoch = np.asarray(record["raw_epoch"], dtype=np.float32)
        computed = self._compute_lcs_matrix(raw_epoch, delta=float(delta))
        self._dynamic_lcs_cache[cache_key] = computed
        return computed

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
        mean = float(values.mean())
        std = float(max(values.std(), self.input_norm_eps))
        stats = (mean, std)
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

    def _compose_channels(self, path: str, record: Dict[str, object], epoch_idx: int) -> np.ndarray:
        if self.model_name == "picosleepnet_plus_snn" and self.use_dual_lcs:
            lcs_small = self._get_lcs_for_delta(path, record, delta=self.delta_small)
            lcs_large = self._get_lcs_for_delta(path, record, delta=self.delta_large)
            if self.use_integer_spike:
                pos_s = np.asarray(lcs_small["lcs_pos_count"][epoch_idx], dtype=np.float32)
                neg_s = np.abs(np.asarray(lcs_small["lcs_neg_count"][epoch_idx], dtype=np.float32))
                pos_l = np.asarray(lcs_large["lcs_pos_count"][epoch_idx], dtype=np.float32)
                neg_l = np.abs(np.asarray(lcs_large["lcs_neg_count"][epoch_idx], dtype=np.float32))
            else:
                pos_s = np.asarray(lcs_small["lcs_pos"][epoch_idx], dtype=np.float32)
                neg_s = np.asarray(lcs_small["lcs_neg"][epoch_idx], dtype=np.float32)
                pos_l = np.asarray(lcs_large["lcs_pos"][epoch_idx], dtype=np.float32)
                neg_l = np.asarray(lcs_large["lcs_neg"][epoch_idx], dtype=np.float32)
            x = np.stack([pos_s, neg_s, pos_l, neg_l], axis=0)
            if self.input_norm_mode != "none":
                mean, std = self._record_channel_stats(
                    path,
                    stat_key=f"dual:{float(self.delta_small):.6f}:{float(self.delta_large):.6f}:{self.use_integer_spike}",
                    arrays=[
                        np.asarray(lcs_small["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                        np.abs(np.asarray(lcs_small["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                        np.asarray(lcs_large["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                        np.abs(np.asarray(lcs_large["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                    ],
                )
                x = _normalize_lcs_channels(x, channel_mean=mean, channel_std=std, mode=self.input_norm_mode, eps=self.input_norm_eps)
            return x

        lcs = self._get_lcs_for_delta(path, record, delta=self.delta_primary)
        if self.use_integer_spike:
            pos = np.asarray(lcs["lcs_pos_count"][epoch_idx], dtype=np.float32)
            neg = np.abs(np.asarray(lcs["lcs_neg_count"][epoch_idx], dtype=np.float32))
        else:
            pos = np.asarray(lcs["lcs_pos"][epoch_idx], dtype=np.float32)
            neg = np.asarray(lcs["lcs_neg"][epoch_idx], dtype=np.float32)
        x = np.stack([pos, neg], axis=0)
        if self.input_norm_mode != "none":
            mean, std = self._record_channel_stats(
                path,
                stat_key=f"single:{float(self.delta_primary):.6f}:{self.use_integer_spike}",
                arrays=[
                    np.asarray(lcs["lcs_pos_count" if self.use_integer_spike else "lcs_pos"], dtype=np.float32),
                    np.abs(np.asarray(lcs["lcs_neg_count" if self.use_integer_spike else "lcs_neg"], dtype=np.float32)),
                ],
            )
            x = _normalize_lcs_channels(x, channel_mean=mean, channel_std=std, mode=self.input_norm_mode, eps=self.input_norm_eps)
        return x

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        record = self._load_record(entry.path)
        labels = np.asarray(record["labels"], dtype=np.int64)
        epoch_idx = int(entry.epoch_idx)
        if self.model_name in {"context_pico_snn", "context_pico_snn_v2"}:
            x = self._compose_raw_context(entry.path, record, epoch_idx=epoch_idx)
        else:
            x = self._compose_channels(entry.path, record, epoch_idx=epoch_idx)
        prev_label = int(labels[epoch_idx - 1]) if epoch_idx > 0 else -1
        next_label = int(labels[epoch_idx + 1]) if epoch_idx + 1 < labels.shape[0] else -1
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(int(entry.label), dtype=torch.long),
            entry.record_id,
            epoch_idx,
            torch.tensor(prev_label, dtype=torch.long),
            torch.tensor(next_label, dtype=torch.long),
        )


def _lambda_defaults_by_subset(subset: str) -> Tuple[float, float]:
    s = normalize_edf_subset(subset)
    if s == "edf20":
        return 1e-8, 0.01
    return 1e-8, 0.005


def _build_model(cfg: dict, model_name: str, num_classes: int, subset: str):
    subset_delta_default = default_lcs_delta(subset)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    baseline_cfg = {**cfg.get("picosleepnet_rsnn", {}), **cfg.get("picosleepnet_baseline", {})}
    plus_cfg = cfg.get("picosleepnet_plus_snn", {})
    context_cfg = cfg.get("context_pico_snn", {})
    context_v2_cfg = cfg.get("context_pico_snn_v2", {})
    lcs_cfg = cfg.get("lcs", {})

    model_name = _canonical_model_name(model_name)
    use_masked_bpsr = bool(_to_bool(model_cfg.get("use_masked_bpsr", baseline_cfg.get("use_masked_bpsr", True))))
    use_integer_spike = bool(_to_bool(model_cfg.get("use_integer_spike", baseline_cfg.get("use_integer_spike", True))))
    learnable_threshold = bool(_to_bool(model_cfg.get("learnable_threshold", baseline_cfg.get("learnable_threshold", False))))
    threshold_per_neuron = bool(
        _to_bool(model_cfg.get("threshold_per_neuron", baseline_cfg.get("threshold_per_neuron", True)))
    )
    threshold_min = float(model_cfg.get("threshold_min", baseline_cfg.get("threshold_min", 0.25)))
    threshold_max = float(model_cfg.get("threshold_max", baseline_cfg.get("threshold_max", 2.5)))
    lcs_delta = float(lcs_cfg.get("delta", baseline_cfg.get("lcs_delta", subset_delta_default)))

    if model_name == "context_pico_snn_v2":
        model = ContextPicoSNNV2(
            num_classes=num_classes,
            in_channels=int(context_v2_cfg.get("in_channels", 1)),
            context_len=int(context_v2_cfg.get("context_len", 1)),
            branch_channels=int(context_v2_cfg.get("branch_channels", 6)),
            stem_channels=int(context_v2_cfg.get("stem_channels", 16)),
            token_dim=int(context_v2_cfg.get("token_dim", 32)),
            epoch_hidden_size=int(context_v2_cfg.get("epoch_hidden_size", 96)),
            epoch_embed_dim=int(context_v2_cfg.get("epoch_embed_dim", 48)),
            context_hidden_size=int(context_v2_cfg.get("context_hidden_size", 40)),
            t_steps=int(context_v2_cfg.get("t_steps", 48)),
            tau_epoch=float(context_v2_cfg.get("tau_epoch", 0.92)),
            tau_context=float(context_v2_cfg.get("tau_context", 0.90)),
            v_th_epoch=float(context_v2_cfg.get("v_th_epoch", 1.0)),
            v_th_context=float(context_v2_cfg.get("v_th_context", 1.0)),
            surrogate_alpha=float(context_v2_cfg.get("surrogate_alpha", 4.0)),
            dropout=float(context_v2_cfg.get("dropout", 0.10)),
            use_aux_heads=bool(_to_bool(train_cfg.get("use_aux_heads", context_v2_cfg.get("use_aux_heads", True)))),
            use_n1_aux=bool(_to_bool(context_v2_cfg.get("use_n1_aux", True))),
            center_residual_weight=float(context_v2_cfg.get("center_residual_weight", 0.40)),
            kernel_sizes=tuple(int(v) for v in context_v2_cfg.get("kernel_sizes", [15, 31, 63])),
        )
    elif model_name == "context_pico_snn":
        model = ContextPicoSNN(
            num_classes=num_classes,
            in_channels=int(context_cfg.get("in_channels", 1)),
            context_len=int(context_cfg.get("context_len", 3)),
            branch_channels=int(context_cfg.get("branch_channels", 4)),
            stem_channels=int(context_cfg.get("stem_channels", 12)),
            token_dim=int(context_cfg.get("token_dim", 24)),
            epoch_hidden_size=int(context_cfg.get("epoch_hidden_size", 96)),
            epoch_embed_dim=int(context_cfg.get("epoch_embed_dim", 48)),
            context_hidden_size=int(context_cfg.get("context_hidden_size", 32)),
            t_steps=int(context_cfg.get("t_steps", 40)),
            tau_epoch=float(context_cfg.get("tau_epoch", 0.92)),
            tau_context=float(context_cfg.get("tau_context", 0.90)),
            v_th_epoch=float(context_cfg.get("v_th_epoch", 1.0)),
            v_th_context=float(context_cfg.get("v_th_context", 1.0)),
            surrogate_alpha=float(context_cfg.get("surrogate_alpha", 4.0)),
            dropout=float(context_cfg.get("dropout", 0.10)),
            use_aux_heads=bool(_to_bool(train_cfg.get("use_aux_heads", context_cfg.get("use_aux_heads", True)))),
            center_residual_weight=float(context_cfg.get("center_residual_weight", 0.35)),
            kernel_sizes=tuple(int(v) for v in context_cfg.get("kernel_sizes", [31, 63, 125])),
        )
    elif model_name == "picosleepnet_plus_snn":
        delta_small_ratio = float(model_cfg.get("delta_small_ratio", plus_cfg.get("delta_small_ratio", 0.65)))
        delta_large_ratio = float(model_cfg.get("delta_large_ratio", plus_cfg.get("delta_large_ratio", 1.35)))
        delta_small = float(plus_cfg.get("lcs_delta_small", max(0.02, lcs_delta * delta_small_ratio)))
        delta_large = float(plus_cfg.get("lcs_delta_large", max(delta_small + 1e-4, lcs_delta * delta_large_ratio)))
        model = PicoSleepNetPlusSNN(
            num_classes=num_classes,
            window_size=int(plus_cfg.get("window_size", 40)),
            input_neurons_each=int(plus_cfg.get("input_neurons_each", 40)),
            reservoir_size=int(plus_cfg.get("reservoir_size", 150)),
            hidden_size=int(plus_cfg.get("hidden_size", 50)),
            tau=float(plus_cfg.get("tau", 0.95)),
            v_th=float(plus_cfg.get("v_th", 1.0)),
            surrogate_alpha=float(plus_cfg.get("surrogate_alpha", 4.0)),
            use_masked_bpsr=bool(_to_bool(plus_cfg.get("use_masked_bpsr", use_masked_bpsr))),
            use_integer_spike=bool(_to_bool(plus_cfg.get("use_integer_spike", use_integer_spike))),
            lcs_delta=lcs_delta,
            lcs_delta_small=delta_small,
            lcs_delta_large=delta_large,
            use_dual_lcs=bool(_to_bool(model_cfg.get("use_dual_lcs", plus_cfg.get("use_dual_lcs", True)))),
            dual_proj_dim=int(plus_cfg.get("dual_proj_dim", plus_cfg.get("input_neurons_each", 40))),
            use_transition_matrix=bool(
                _to_bool(model_cfg.get("use_transition_matrix", plus_cfg.get("use_transition_matrix", True)))
            ),
            transition_residual_weight=float(
                plus_cfg.get("transition_residual_weight", plus_cfg.get("transition_weight", 0.25))
            ),
            use_aux_heads=bool(_to_bool(train_cfg.get("use_aux_heads", plus_cfg.get("use_aux_heads", True)))),
            firing_target_low=float(train_cfg.get("firing_target_low", plus_cfg.get("firing_target_low", 0.03))),
            firing_target_high=float(train_cfg.get("firing_target_high", plus_cfg.get("firing_target_high", 0.18))),
            postprocess_mode=str(plus_cfg.get("postprocess_mode", "transition_forward")),
            mask_alpha=float(plus_cfg.get("mask_alpha", 4.0)),
            qat_bits=int(plus_cfg.get("qat_bits", 6)),
            learnable_threshold=bool(
                _to_bool(model_cfg.get("learnable_threshold", plus_cfg.get("learnable_threshold", learnable_threshold)))
            ),
            threshold_per_neuron=bool(
                _to_bool(
                    model_cfg.get("threshold_per_neuron", plus_cfg.get("threshold_per_neuron", threshold_per_neuron))
                )
            ),
            threshold_min=float(model_cfg.get("threshold_min", plus_cfg.get("threshold_min", threshold_min))),
            threshold_max=float(model_cfg.get("threshold_max", plus_cfg.get("threshold_max", threshold_max))),
        )
    else:
        model = PicoSleepNetBaseline(
            num_classes=num_classes,
            window_size=int(baseline_cfg.get("window_size", 40)),
            input_neurons_each=int(baseline_cfg.get("input_neurons_each", 40)),
            reservoir_size=int(baseline_cfg.get("reservoir_size", 150)),
            hidden_size=int(baseline_cfg.get("hidden_size", 50)),
            tau=float(baseline_cfg.get("tau", 0.95)),
            v_th=float(baseline_cfg.get("v_th", 1.0)),
            surrogate_alpha=float(baseline_cfg.get("surrogate_alpha", 4.0)),
            use_masked_bpsr=use_masked_bpsr,
            use_integer_spike=use_integer_spike,
            lcs_delta=lcs_delta,
            mask_alpha=float(baseline_cfg.get("mask_alpha", 4.0)),
            qat_bits=int(baseline_cfg.get("qat_bits", 6)),
            learnable_threshold=learnable_threshold,
            threshold_per_neuron=threshold_per_neuron,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    return model, model.get_hparams()


def _entry_counts(entries: List[EpochEntry], num_classes: int) -> np.ndarray:
    if not entries:
        return np.zeros(num_classes, dtype=np.int64)
    arr = np.asarray([int(e.label) for e in entries], dtype=np.int64)
    return np.bincount(arr, minlength=num_classes).astype(np.int64)


def _limit_entries_for_smoke(entries: List[EpochEntry], num_classes: int, per_class: int) -> List[EpochEntry]:
    if per_class <= 0 or not entries:
        return list(entries)
    buckets: Dict[int, List[EpochEntry]] = {c: [] for c in range(num_classes)}
    for entry in entries:
        buckets[int(entry.label)].append(entry)
    limited: List[EpochEntry] = []
    for c in range(num_classes):
        limited.extend(buckets[c][:per_class])
    return limited if limited else list(entries[: max(1, min(len(entries), per_class * num_classes))])


def _build_split_entries(
    entries: List[EpochEntry],
    records: List[Dict[str, object]],
    cfg: dict,
    num_classes: int,
    subset: str,
) -> List[Dict[str, object]]:
    split_cfg = cfg.get("split", {})
    split_constraints = cfg.get("split_constraints", {})
    protocol = str(split_cfg.get("protocol", cfg.get("protocol", cfg.get("split_mode", "subject_kfold")))).lower().strip()
    if protocol == "kfold":
        protocol = "subject_kfold"
    val_ratio = float(cfg.get("val_ratio", split_cfg.get("val_ratio", 0.1)))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    seed = int(cfg.get("seed", 42))

    if protocol == "epoch_random":
        key_to_entry = {f"{e.record_id}:{e.epoch_idx}": e for e in entries}
        split_one = make_epoch_random_split(list(key_to_entry.keys()), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

        def to_entries(keys: List[str]) -> List[EpochEntry]:
            return [key_to_entry[k] for k in keys]

        return [
            {
                "protocol": "epoch_random",
                "train_samples": split_one["train"],
                "val_samples": split_one["val"],
                "test_samples": split_one["test"],
                "train_entries": to_entries(split_one["train"]),
                "val_entries": to_entries(split_one["val"]),
                "test_entries": to_entries(split_one["test"]),
            }
        ]

    subject_label_counts = compute_subject_label_counts(records, num_classes=num_classes)
    subject_ids = sorted(subject_label_counts.keys())
    min_count = int(split_constraints.get("min_count_per_class", 1))
    max_tries = int(split_constraints.get("max_tries", 200))
    k_default = int(default_kfold_by_subset(subject_ids, edf_subset=subset))
    kfold = int(cfg.get("kfold", k_default))
    if kfold <= 0:
        kfold = k_default

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
        edf_subset=subset,
    )

    out: List[Dict[str, object]] = []
    for sp in subject_splits:
        train_subjects = set(sp["train"])
        val_subjects = set(sp["val"])
        test_subjects = set(sp["test"])
        out.append(
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
    return out


def _write_gpu_env(run_dir: Path, device: torch.device) -> None:
    lines = [
        f"device={device}",
        f"torch.cuda.is_available()={torch.cuda.is_available()}",
        f"torch.version.cuda={torch.version.cuda}",
    ]
    if torch.cuda.is_available():
        lines.append(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        lines.append(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
    ensure_dir(run_dir / "diagnose")
    (run_dir / "diagnose" / "env_gpu.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_to_jsonable(split: Dict[str, object]) -> Dict[str, object]:
    out = {k: v for k, v in split.items() if not k.endswith("_entries")}
    out["n_train_epochs"] = len(split["train_entries"])
    out["n_val_epochs"] = len(split["val_entries"])
    out["n_test_epochs"] = len(split["test_entries"])
    return out


def _main_logits(outputs) -> torch.Tensor:
    return outputs["main"] if isinstance(outputs, dict) else outputs


def _output_bias(model) -> List[float] | None:
    head = getattr(model, "w_out", None)
    bias = getattr(head, "bias", None)
    if bias is None:
        return None
    return [float(v) for v in bias.detach().cpu().reshape(-1).tolist()]


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _collect_forward_debug(
    model,
    dataloader,
    device: torch.device,
    mixed_precision: bool,
    pin_memory: bool,
    max_batches: int = 3,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= int(max_batches):
                break
            x = batch[0].to(device, non_blocking=bool(pin_memory))
            y = batch[1].to(device, non_blocking=bool(pin_memory))
            if hasattr(model, "reset_state"):
                model.reset_state()
            if mixed_precision and device.type == "cuda":
                with _amp_autocast(device, enabled=True):
                    outputs = model(x)
            else:
                outputs = model(x)
            logits = _main_logits(outputs)
            prob = torch.softmax(logits, dim=1)
            row = {
                "batch_idx": int(batch_idx),
                "target_counts": np.bincount(y.detach().cpu().numpy(), minlength=int(logits.size(1))).astype(np.int64).tolist(),
                "x": _tensor_stats(x.detach()),
                "logits_mean": [float(v) for v in logits.detach().mean(dim=0).cpu().tolist()],
                "logits_std": [float(v) for v in logits.detach().std(dim=0, unbiased=False).cpu().tolist()],
                "softmax_mean": [float(v) for v in prob.detach().mean(dim=0).cpu().tolist()],
                "output_bias": _output_bias(model),
            }
            if isinstance(outputs, dict) and isinstance(outputs.get("debug_stats"), dict):
                row["spike_debug"] = {
                    key: float(val.detach().cpu().item()) if isinstance(val, torch.Tensor) else float(val)
                    for key, val in outputs["debug_stats"].items()
                }
            if isinstance(outputs, dict) and isinstance(outputs.get("firing_rate"), torch.Tensor):
                row["firing_rate"] = float(outputs["firing_rate"].detach().mean().cpu().item())
            if isinstance(outputs, dict) and isinstance(outputs.get("layer_firing_rates"), dict):
                row["layer_firing_rates"] = {
                    key: float(val.detach().mean().cpu().item()) if isinstance(val, torch.Tensor) else float(val)
                    for key, val in outputs["layer_firing_rates"].items()
                }
            rows.append(row)
    return {"batches": rows}


def _merge_json(path: Path, payload: Dict[str, object]) -> None:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data.update(payload)
    path.write_text(json_dumps(data), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _append_train_log_first_epochs_csv(change_proof_dir: Path, split_idx: int, log_path: Path, max_epochs: int = 3) -> None:
    rows = _read_jsonl(log_path)[: max(1, int(max_epochs))]
    if not rows:
        return
    csv_path = change_proof_dir / "train_log_first_epochs.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "epoch", "loss", "cls_loss", "aux_loss", "tc_loss", "train_acc", "lr"],
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            components = row.get("loss_components", {})
            writer.writerow(
                {
                    "split": int(split_idx),
                    "epoch": int(row.get("epoch", 0)),
                    "loss": float(row.get("train_loss", 0.0)),
                    "cls_loss": float(components.get("cls_loss", components.get("base_loss", 0.0))),
                    "aux_loss": float(components.get("aux_loss", 0.0)),
                    "tc_loss": float(components.get("tc_loss", components.get("temporal_consistency", 0.0))),
                    "train_acc": float(row.get("train_acc", 0.0)),
                    "lr": float(row.get("lr", 0.0)),
                }
            )


def _update_curve_difference_report(change_proof_dir: Path, split_idx: int, log_path: Path) -> None:
    baseline_log_path = BASELINE_RUN_DIR / "train" / f"split_{split_idx}" / "log.jsonl"
    baseline_rows = _read_jsonl(baseline_log_path)[:3]
    new_rows = _read_jsonl(log_path)[:3]
    comparison = {
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "split": int(split_idx),
        "baseline_log_path": str(baseline_log_path),
        "new_log_path": str(log_path),
        "baseline_rows": baseline_rows,
        "new_rows": new_rows,
        "curves_different": True,
        "reason": "baseline log missing" if not baseline_rows else "new curve differs from locked baseline",
    }
    if baseline_rows and len(baseline_rows) == len(new_rows):
        same_rows = True
        for idx, (base_row, new_row) in enumerate(zip(baseline_rows, new_rows)):
            if (
                abs(float(base_row.get("train_loss", 0.0)) - float(new_row.get("train_loss", 0.0))) > 1e-9
                or abs(float(base_row.get("val_macro_f1", 0.0)) - float(new_row.get("val_macro_f1", 0.0))) > 1e-9
                or str(base_row.get("loss_name", "")) != str(new_row.get("loss_name", ""))
            ):
                same_rows = False
                comparison["first_diff_epoch"] = int(new_row.get("epoch", idx + 1))
                break
        comparison["curves_different"] = not same_rows
        comparison["reason"] = "identical_first_epochs_to_baseline" if same_rows else "first_epochs_changed"

    report_path = change_proof_dir / "baseline_vs_new_curve_check.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"baseline_run_dir": str(BASELINE_RUN_DIR), "splits": {}}
    report["splits"][str(split_idx)] = comparison
    save_json(report_path, report)


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in state_dict.items()
    }


def _average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    averaged = {}
    first = state_dicts[0]
    for key, value in first.items():
        if not torch.is_tensor(value):
            averaged[key] = value
            continue
        if not torch.is_floating_point(value):
            averaged[key] = value.detach().clone()
            continue
        stacked = torch.stack([state[key].detach().to(dtype=value.dtype) for state in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0)
    return averaged


def _with_temporary_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    backup = _clone_state_dict(model.state_dict())
    model.load_state_dict(state_dict, strict=True)
    return backup


def _current_best_ckpt_decision(
    *,
    rule_name: str,
    val_macro_f1: float,
    val_kappa: float,
    val_n1_f1: float,
    val_n1_recall: float,
    val_rem_f1: float,
    val_acc: float,
    macro_f1_weight: float,
    n1_f1_weight: float,
    kappa_weight: float,
    rem_f1_weight: float,
    n1_floor: float,
    n1_recall_floor: float,
) -> dict[str, object]:
    eligible = bool(float(val_n1_f1) >= float(n1_floor) and float(val_n1_recall) >= float(n1_recall_floor))
    gate_failures: list[str] = []
    if float(val_n1_f1) < float(n1_floor):
        gate_failures.append("val_N1_f1_below_floor")
    if float(val_n1_recall) < float(n1_recall_floor):
        gate_failures.append("val_N1_recall_below_floor")
    if str(rule_name).lower().strip() == MACRO_N1_REM_GUARDED_CKPT_RULE:
        score_components = {
            "val_macro_f1": float(macro_f1_weight) * float(val_macro_f1),
            "val_N1_f1": float(n1_f1_weight) * float(val_n1_f1),
            "val_kappa": float(kappa_weight) * float(val_kappa),
            "val_REM_f1": float(rem_f1_weight) * float(val_rem_f1),
        }
        score = (
            score_components["val_macro_f1"]
            + score_components["val_N1_f1"]
            + score_components["val_kappa"]
            + score_components["val_REM_f1"]
        )
        sort_key = (
            int(eligible),
            float(score),
            float(val_macro_f1),
            float(val_n1_f1),
            float(val_rem_f1),
            float(val_kappa),
            float(val_acc),
        )
        score_formula = MACRO_N1_REM_GUARDED_SCORE_FORMULA
    else:
        score_components = {
            "val_macro_f1": float(val_macro_f1),
            "val_kappa": float(val_kappa),
            "val_N1_f1": float(val_n1_f1),
            "val_N1_recall": float(val_n1_recall),
            "val_acc": float(val_acc),
        }
        score = float(val_macro_f1)
        sort_key = (
            int(eligible),
            float(val_macro_f1),
            float(val_kappa),
            float(val_n1_f1),
            float(val_n1_recall),
            float(val_acc),
        )
        score_formula = "legacy_lexicographic"
    return {
        "eligible": bool(eligible),
        "score": float(score),
        "sort_key": sort_key,
        "score_formula": str(score_formula),
        "score_components": score_components,
        "gate_failures": gate_failures,
    }


def _compress_drw_schedule(epoch_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for row in epoch_rows:
        current = {
            "epoch_start": int(row["epoch"]),
            "epoch_end": int(row["epoch"]),
            "whether_drw_enabled": bool(row.get("whether_drw_enabled", False)),
            "class_weights": list(row.get("class_weights") or []),
            "current_loss_schedule": str(row.get("current_loss_schedule", "")),
        }
        if segments and (
            segments[-1]["whether_drw_enabled"] == current["whether_drw_enabled"]
            and segments[-1]["current_loss_schedule"] == current["current_loss_schedule"]
            and segments[-1]["class_weights"] == current["class_weights"]
        ):
            segments[-1]["epoch_end"] = current["epoch_end"]
            continue
        segments.append(current)
    return segments


def _build_boundary_soft_targets(
    targets: torch.Tensor,
    prev_labels: torch.Tensor | None,
    next_labels: torch.Tensor | None,
    num_classes: int,
    primary_weight: float,
    neighbor_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    boundary_mask = torch.zeros_like(targets, dtype=torch.bool)
    if prev_labels is None or next_labels is None:
        return None, boundary_mask

    primary = float(min(max(primary_weight, 0.7), 0.95))
    neighbor_total = float(min(max(neighbor_weight, 0.0), 1.0 - primary))
    soft_targets = F.one_hot(targets, num_classes=num_classes).to(dtype=torch.float32)

    for idx in range(int(targets.numel())):
        target = int(targets[idx].item())
        neighbors: List[int] = []
        for label_tensor in (prev_labels[idx], next_labels[idx]):
            label = int(label_tensor.item())
            if 0 <= label < int(num_classes) and label != target and label not in neighbors:
                neighbors.append(label)
        if not neighbors:
            continue
        boundary_mask[idx] = True
        soft_targets[idx].zero_()
        soft_targets[idx, target] = primary
        share = neighbor_total / float(len(neighbors))
        for label in neighbors:
            soft_targets[idx, label] = share
        residual = 1.0 - float(soft_targets[idx].sum().item())
        soft_targets[idx, target] += residual

    if not bool(boundary_mask.any()):
        return None, boundary_mask
    return soft_targets, boundary_mask


def _compute_loss(
    outputs,
    loss_fn: nn.Module,
    targets: torch.Tensor,
    model,
    model_name: str,
    wake_label: int,
    rem_label: int,
    n1_label: int,
    lambda_s: float,
    lambda_w: float,
    aux_sleep_weight: float,
    aux_rem_weight: float,
    aux_n1_weight: float = 0.0,
    transition_weight: float = 0.0,
    firing_reg_weight: float = 0.0,
    firing_target_low: float = 0.03,
    firing_target_high: float = 0.18,
    label_smoothing: float = 0.0,
    label_smoothing_weight: float = 0.0,
    prev_labels: torch.Tensor | None = None,
    next_labels: torch.Tensor | None = None,
    boundary_soft_label_enable: bool = False,
    boundary_soft_label_weight: float = 0.0,
    boundary_soft_primary_weight: float = 0.85,
    boundary_soft_neighbor_weight: float = 0.15,
    temporal_consistency_enable: bool = False,
    temporal_consistency_weight: float = 0.0,
    temporal_consistency_temperature: float = 1.0,
    temporal_consistency_mode: str = "step_logits",
    temporal_reference_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["main"] if isinstance(outputs, dict) else outputs
    cls_loss = loss_fn(logits, targets)
    if float(label_smoothing) > 0.0 and float(label_smoothing_weight) > 0.0:
        smooth_loss = F.cross_entropy(logits, targets, label_smoothing=float(label_smoothing))
        cls_loss = (1.0 - float(label_smoothing_weight)) * cls_loss + float(label_smoothing_weight) * smooth_loss
    total = cls_loss
    aux_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    tc_term = torch.zeros((), device=logits.device, dtype=logits.dtype)
    stats = {"cls_loss": float(cls_loss.detach().item())}
    if float(label_smoothing) > 0.0 and float(label_smoothing_weight) > 0.0:
        stats["label_smoothing"] = float(label_smoothing)
    if bool(boundary_soft_label_enable) and float(boundary_soft_label_weight) > 0.0:
        soft_targets, boundary_mask = _build_boundary_soft_targets(
            targets=targets,
            prev_labels=prev_labels,
            next_labels=next_labels,
            num_classes=int(logits.size(1)),
            primary_weight=float(boundary_soft_primary_weight),
            neighbor_weight=float(boundary_soft_neighbor_weight),
        )
        if soft_targets is not None and bool(boundary_mask.any()):
            boundary_loss = soft_target_cross_entropy(logits[boundary_mask], soft_targets[boundary_mask])
            boundary_term = float(boundary_soft_label_weight) * boundary_loss
            total = total + boundary_term
            aux_loss = aux_loss + boundary_term
            stats["boundary_soft_label"] = float(boundary_term.detach().item())
            stats["boundary_soft_samples"] = float(boundary_mask.sum().item())
    if (
        bool(temporal_consistency_enable)
        and float(temporal_consistency_weight) > 0.0
    ):
        tc_mode = str(temporal_consistency_mode).lower().strip() or "step_logits"
        tc_loss = None
        if tc_mode == "stochastic_forward" and isinstance(temporal_reference_logits, torch.Tensor):
            tc_loss = logits_consistency_kl_loss(
                logits,
                temporal_reference_logits,
                temperature=float(temporal_consistency_temperature),
                detach_target=True,
            )
            stats["tc_mode"] = 1.0
        elif isinstance(outputs, dict) and isinstance(outputs.get("step_logits"), torch.Tensor):
            tc_loss = temporal_consistency_kl_loss(
                outputs["step_logits"],
                logits,
                temperature=float(temporal_consistency_temperature),
                detach_target=True,
            )
            stats["tc_mode"] = 0.0
        if tc_loss is not None:
            tc_term = float(temporal_consistency_weight) * tc_loss
            total = total + tc_term
            stats["tc_loss"] = float(tc_term.detach().item())
    if isinstance(outputs, dict):
        if isinstance(outputs.get("spike_l1"), torch.Tensor):
            spike_reg = float(lambda_s) * outputs["spike_l1"]
            total = total + spike_reg
            aux_loss = aux_loss + spike_reg
            stats["spike_reg"] = float(spike_reg.detach().item())
        if isinstance(outputs.get("mask_l1"), torch.Tensor):
            mask_reg = float(lambda_w) * outputs["mask_l1"]
            total = total + mask_reg
            aux_loss = aux_loss + mask_reg
            stats["mask_reg"] = float(mask_reg.detach().item())
        if isinstance(outputs.get("firing_rate"), torch.Tensor) and float(firing_reg_weight) > 0.0:
            firing_rate = outputs["firing_rate"]
            firing_penalty = torch.relu(torch.as_tensor(float(firing_target_low), device=firing_rate.device, dtype=firing_rate.dtype) - firing_rate)
            firing_penalty = firing_penalty + torch.relu(
                firing_rate - torch.as_tensor(float(firing_target_high), device=firing_rate.device, dtype=firing_rate.dtype)
            )
            firing_term = float(firing_reg_weight) * firing_penalty
            total = total + firing_term
            aux_loss = aux_loss + firing_term
            stats["firing_reg"] = float(firing_term.detach().item())
        if "sleep_wake" in outputs:
            sleep_targets = (targets != int(wake_label)).float()
            aux_sleep = float(aux_sleep_weight) * F.binary_cross_entropy_with_logits(outputs["sleep_wake"], sleep_targets)
            total = total + aux_sleep
            aux_loss = aux_loss + aux_sleep
            stats["aux_sleep"] = float(aux_sleep.detach().item())
        if "rem" in outputs:
            rem_targets = (targets == int(rem_label)).float()
            aux_rem = float(aux_rem_weight) * F.binary_cross_entropy_with_logits(outputs["rem"], rem_targets)
            total = total + aux_rem
            aux_loss = aux_loss + aux_rem
            stats["aux_rem"] = float(aux_rem.detach().item())
        if "n1" in outputs and float(aux_n1_weight) > 0.0:
            n1_targets = (targets == int(n1_label)).float()
            aux_n1 = float(aux_n1_weight) * F.binary_cross_entropy_with_logits(outputs["n1"], n1_targets)
            total = total + aux_n1
            aux_loss = aux_loss + aux_n1
            stats["aux_n1"] = float(aux_n1.detach().item())
        if float(transition_weight) > 0.0 and hasattr(model, "transition_nll"):
            trans_loss = model.transition_nll(prev_labels, targets, next_labels)
            trans_term = float(transition_weight) * trans_loss
            total = total + trans_term
            aux_loss = aux_loss + trans_term
            stats["transition_reg"] = float(trans_term.detach().item())
    stats["base_loss"] = stats["cls_loss"]
    stats["aux_loss"] = float(aux_loss.detach().item())
    stats["tc_loss"] = float(tc_term.detach().item())
    stats["total_loss"] = float(total.detach().item())
    return total, stats


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    rows = []
    by_class: dict[str, dict[str, float]] = {}
    for idx, class_name in enumerate(labels):
        row = {
            "class": str(class_name),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        rows.append(row)
        by_class[str(class_name)] = row
    return {"rows": rows, "by_class": by_class}


def _best_ckpt_key(
    val_macro_f1: float,
    val_kappa: float,
    val_n1_f1: float,
    val_n1_recall: float,
    val_acc: float,
    n1_floor: float,
    n1_recall_floor: float,
) -> tuple[int, float, float, float, float, float]:
    eligible = int(float(val_n1_f1) >= float(n1_floor) and float(val_n1_recall) >= float(n1_recall_floor))
    return (
        eligible,
        float(val_macro_f1),
        float(val_kappa),
        float(val_n1_f1),
        float(val_n1_recall),
        float(val_acc),
    )


def _train_one_split(
    split_idx: int,
    split: Dict[str, object],
    run_dir: Path,
    cfg: dict,
    model_name: str,
    num_classes: int,
    labels: List[str],
    wake_label: int,
    rem_label: int,
    subset: str,
    device: torch.device,
    mixed_precision: bool,
    num_workers: int,
    pin_memory: bool,
    fp32_only: bool,
) -> None:
    split_dir = ensure_dir(run_dir / "train" / f"split_{split_idx}")
    change_proof_dir = ensure_dir(run_dir / "change_proof")
    log_path = split_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)

    train_entries = list(split["train_entries"])
    val_entries = list(split["val_entries"])
    test_entries = list(split["test_entries"])

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    lcs_cfg = cfg.get("lcs", {})
    baseline_cfg = {**cfg.get("picosleepnet_rsnn", {}), **cfg.get("picosleepnet_baseline", {})}
    plus_cfg = cfg.get("picosleepnet_plus_snn", {})
    smoke_mode = bool(_to_bool(train_cfg.get("smoke", False)))
    if smoke_mode:
        smoke_train_per_class = int(train_cfg.get("smoke_train_per_class", 64))
        smoke_eval_per_class = int(train_cfg.get("smoke_eval_per_class", 32))
        train_entries = _limit_entries_for_smoke(train_entries, num_classes=num_classes, per_class=smoke_train_per_class)
        val_entries = _limit_entries_for_smoke(val_entries, num_classes=num_classes, per_class=smoke_eval_per_class)
        test_entries = _limit_entries_for_smoke(test_entries, num_classes=num_classes, per_class=smoke_eval_per_class)
    planned_epochs = int(cfg.get("epochs", 12))
    if smoke_mode:
        planned_epochs = min(planned_epochs, max(2, int(train_cfg.get("smoke_epochs", 3))))

    model_name = _canonical_model_name(model_name)
    model, model_hparams = _build_model(cfg=cfg, model_name=model_name, num_classes=num_classes, subset=subset)
    model.to(device)

    delta_primary = float(model_hparams.get("lcs_delta", lcs_cfg.get("delta", default_lcs_delta(subset))))
    delta_small = float(model_hparams.get("lcs_delta_small", plus_cfg.get("lcs_delta_small", max(0.02, delta_primary * 0.6))))
    delta_large = float(model_hparams.get("lcs_delta_large", plus_cfg.get("lcs_delta_large", delta_primary)))
    use_dual_lcs = bool(_to_bool(model_hparams.get("use_dual_lcs", model_name == "picosleepnet_plus_snn")))
    use_integer_spike = bool(model_hparams.get("use_integer_spike", True))
    cache_mode = str(cfg.get("cache_mode", "mem"))
    input_norm_mode = str(train_cfg.get("input_norm_mode", "none")).lower().strip() or "none"
    default_context_cfg = cfg.get("context_pico_snn_v2", {}) if model_name == "context_pico_snn_v2" else cfg.get("context_pico_snn", {})
    context_len = int(model_hparams.get("context_len", default_context_cfg.get("context_len", 1)))

    train_ds = SleepEdfSpikeDataset(
        train_entries,
        model_name,
        use_dual_lcs,
        use_integer_spike,
        delta_primary,
        delta_small,
        delta_large,
        cache_mode,
        input_norm_mode=input_norm_mode,
        context_len=context_len,
    )
    val_ds = SleepEdfSpikeDataset(
        val_entries,
        model_name,
        use_dual_lcs,
        use_integer_spike,
        delta_primary,
        delta_small,
        delta_large,
        cache_mode,
        input_norm_mode=input_norm_mode,
        context_len=context_len,
    )
    sampler_name = str(train_cfg.get("sampler", "weighted")).lower().strip()
    loss_fn, loss_summary = _build_main_loss(
        cfg=cfg,
        labels=train_ds.all_labels,
        num_classes=num_classes,
        sampler_name=sampler_name,
        device=device,
        current_epoch=1,
        total_epochs=int(planned_epochs),
    )
    sampler_power = float(train_cfg.get("sampler_balance_power", 1.0))
    boundary_sampling_boost = float(train_cfg.get("boundary_sampling_boost", 1.0))

    def _make_train_loader(active_sampler_name: str) -> DataLoader:
        sampler = None
        if active_sampler_name == "weighted":
            sampler = _make_weighted_sampler(
                train_ds.all_labels,
                num_classes=num_classes,
                power=sampler_power,
                boundary_flags=train_ds.boundary_flags,
                boundary_boost=boundary_sampling_boost,
            )
        return DataLoader(
            train_ds,
            batch_size=int(cfg.get("batch_size", 32)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=False,
        )

    train_loader = _make_train_loader(sampler_name)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-2)))
    scaler = _amp_grad_scaler(device, enabled=bool(mixed_precision))

    lambda_s_default, lambda_w_default = _lambda_defaults_by_subset(subset)
    lambda_s = float(model_cfg.get("lambda_s", baseline_cfg.get("lambda_s", lambda_s_default)))
    lambda_w = float(model_cfg.get("lambda_w", baseline_cfg.get("lambda_w", lambda_w_default)))
    use_aux_heads = bool(_to_bool(model_hparams.get("use_aux_heads", model_name != "picosleepnet_baseline")))
    if model_name == "context_pico_snn_v2":
        context_model_cfg = cfg.get("context_pico_snn_v2", {})
    else:
        context_model_cfg = cfg.get("context_pico_snn", {})
    aux_sleep_default = plus_cfg.get("aux_sleep_weight", 0.1) if model_name == "picosleepnet_plus_snn" else train_cfg.get("aux_weight_sleepwake", 0.08)
    aux_rem_default = plus_cfg.get("aux_rem_weight", 0.1) if model_name == "picosleepnet_plus_snn" else train_cfg.get("aux_weight_rem", 0.06)
    aux_sleep_weight = float(train_cfg.get("aux_weight_sleepwake", aux_sleep_default)) if use_aux_heads else 0.0
    aux_rem_weight = float(train_cfg.get("aux_weight_rem", aux_rem_default)) if use_aux_heads else 0.0
    aux_n1_weight = float(train_cfg.get("aux_weight_n1", 0.0)) if use_aux_heads else 0.0
    transition_weight = float(train_cfg.get("transition_aux_weight", plus_cfg.get("transition_aux_weight", 0.05)))
    if not bool(_to_bool(model_hparams.get("use_transition_matrix", model_name == "picosleepnet_plus_snn"))):
        transition_weight = 0.0
    firing_reg_default = plus_cfg.get("firing_reg_weight", 0.02) if model_name == "picosleepnet_plus_snn" else context_model_cfg.get("firing_reg_weight", 0.004)
    firing_reg_weight = float(train_cfg.get("firing_reg_weight", firing_reg_default))
    if model_name == "picosleepnet_baseline":
        firing_reg_weight = 0.0
    firing_target_low = float(train_cfg.get("firing_target_low", model_hparams.get("firing_target_low", 0.03)))
    firing_target_high = float(train_cfg.get("firing_target_high", model_hparams.get("firing_target_high", 0.18)))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05 if model_name == "picosleepnet_plus_snn" else 0.0))
    label_smoothing_weight = float(
        train_cfg.get("label_smoothing_weight", 0.25 if model_name == "picosleepnet_plus_snn" else 0.0)
    )
    ema_enable = bool(_to_bool(train_cfg.get("ema_enable", False)))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_use_for_eval = bool(_to_bool(train_cfg.get("ema_use_for_eval", False))) and ema_enable
    ema_strategy = str(train_cfg.get("ema_strategy", "ema")).lower().strip() or "ema"
    boundary_soft_label_enable = bool(_to_bool(train_cfg.get("boundary_soft_label_enable", False)))
    boundary_soft_label_weight = float(train_cfg.get("boundary_soft_label_weight", 0.0))
    boundary_soft_primary_weight = float(train_cfg.get("boundary_soft_primary_weight", 0.85))
    boundary_soft_neighbor_weight = float(train_cfg.get("boundary_soft_neighbor_weight", 0.15))
    temporal_consistency_requested = bool(_to_bool(train_cfg.get("temporal_consistency_enable", False)))
    temporal_consistency_enable = bool(temporal_consistency_requested)
    temporal_consistency_weight = float(train_cfg.get("temporal_consistency_weight", 0.0))
    temporal_consistency_temperature = float(train_cfg.get("temporal_consistency_temperature", 1.0))
    temporal_consistency_mode = str(train_cfg.get("temporal_consistency_mode", "step_logits")).lower().strip() or "step_logits"
    input_mixstyle_enable = bool(_to_bool(train_cfg.get("input_mixstyle_enable", False)))
    input_mixstyle_p = float(train_cfg.get("input_mixstyle_p", 0.0))
    input_mixstyle_alpha = float(train_cfg.get("input_mixstyle_alpha", 0.3))
    learnable_threshold_enabled = bool(_to_bool(getattr(model, "learnable_threshold", False)))
    threshold_param_names = (
        list(getattr(model, "learnable_threshold_parameter_names")())
        if hasattr(model, "learnable_threshold_parameter_names")
        else []
    )

    _write_active_method_report(
        change_proof_dir=change_proof_dir,
        run_dir=run_dir,
        model=model,
        loss_fn=loss_fn,
        train_cfg=train_cfg,
    )
    epochs = int(planned_epochs)
    patience = int(cfg.get("early_stop_patience", 4))
    min_epochs = int(cfg.get("min_epochs", 3))
    n1_label_idx = labels.index("N1") if "N1" in labels else 1
    rem_label_idx = labels.index("REM") if "REM" in labels else rem_label
    best_ckpt_rule = str(train_cfg.get("best_ckpt_rule", "legacy_lexicographic")).lower().strip() or "legacy_lexicographic"
    best_ckpt_metric_name = str(train_cfg.get("best_ckpt_metric_name", "val_macro_f1_then_kappa_then_n1_f1_then_n1_recall"))
    best_ckpt_rule_description = str(
        train_cfg.get(
            "best_ckpt_rule_description",
            "eligible if val_N1_f1 >= floor and val_N1_recall >= floor; then max(val_macro_f1, val_kappa, val_N1_f1, val_N1_recall, val_acc)",
        )
    )
    best_ckpt_macro_f1_weight = float(train_cfg.get("best_ckpt_macro_f1_weight", 0.60))
    best_ckpt_n1_f1_weight = float(train_cfg.get("best_ckpt_n1_f1_weight", 0.20))
    best_ckpt_kappa_weight = float(train_cfg.get("best_ckpt_kappa_weight", 0.10))
    best_ckpt_rem_f1_weight = float(train_cfg.get("best_ckpt_rem_f1_weight", 0.0))
    best_ckpt_n1_floor = float(train_cfg.get("best_ckpt_n1_f1_floor", 0.15))
    best_ckpt_n1_recall_floor = float(train_cfg.get("best_ckpt_n1_recall_floor", 0.12))
    best_ckpt_fallback_topk = int(train_cfg.get("best_ckpt_fallback_topk", 0))
    tc_zero_guard_epochs = int(train_cfg.get("tc_zero_guard_epochs", 2))
    tc_zero_guard_tol = float(train_cfg.get("tc_zero_guard_tol", 1e-8))
    drw_switch_epoch = int(loss_summary.get("drw_switch_epoch", 0) or 0)
    early_stop_guard_epoch = max(min_epochs, int(drw_switch_epoch + 2 if drw_switch_epoch > 0 else min_epochs))
    print(
        "train: "
        f"active_model_class={type(model).__name__} "
        f"active_loss_class={loss_summary.get('active_loss_class', type(loss_fn).__name__)} "
        f"temporal_consistency_enable={bool(temporal_consistency_requested)} "
        f"learnable_tau_enable={bool(_to_bool(train_cfg.get('learnable_tau', False)))} "
        f"learnable_threshold_enable={learnable_threshold_enabled} "
        f"best_ckpt_rule={str(train_cfg.get('best_ckpt_rule', 'legacy_lexicographic'))} "
        f"drw_switch_epoch={int(drw_switch_epoch) if drw_switch_epoch > 0 else 'off'} "
        f"early_stop_guard_epoch={int(early_stop_guard_epoch)} "
        f"lr={float(cfg.get('lr', 1e-3)):.6f} "
        f"loss_tau={float(cfg.get('loss', {}).get('tau', 0.0)):.3f} "
        f"threshold_param_names={threshold_param_names}"
    )

    collapse_cfg = train_cfg.get("collapse_protection", {})
    collapse_enable = bool(collapse_cfg.get("enable", True))
    collapse_trigger_ratio = float(collapse_cfg.get("trigger_ratio", 0.85))
    collapse_patience = int(collapse_cfg.get("patience_epochs", 2))
    collapse_lr_decay = float(collapse_cfg.get("lr_decay", 0.5))

    audit_dir = ensure_dir(run_dir / "audit")
    if split_idx == 0:
        save_json(audit_dir / "loss_summary.json", loss_summary)
        initial_debug = _collect_forward_debug(
            model=model,
            dataloader=val_loader,
            device=device,
            mixed_precision=bool(mixed_precision),
            pin_memory=bool(pin_memory),
            max_batches=3,
        )
        _merge_json(audit_dir / "logit_debug_split0.json", {"initial_forward": initial_debug})
        spike_rows = []
        for row in initial_debug["batches"]:
            spike_rows.append(
                {
                    "batch_idx": row["batch_idx"],
                    "firing_rate": row.get("firing_rate", 0.0),
                    "layer_firing_rates": row.get("layer_firing_rates", {}),
                    **row.get("spike_debug", {}),
                }
            )
        save_json(audit_dir / "spike_activity_split0.json", {"batches": spike_rows})

    best_metric = -1.0
    best_ckpt_key_value = (-1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    best_state = None
    best_epoch = -1
    best_val_snapshot: dict[str, float] = {}
    best_selection_snapshot: dict[str, object] = {}
    no_improve = 0
    epoch_detail_rows: list[dict[str, object]] = []
    selection_trace_rows: list[dict[str, object]] = []
    topk_macro_candidates: list[dict[str, object]] = []
    tc_epoch_history: list[float] = []
    protector = CollapseProtector(trigger_ratio=collapse_trigger_ratio, patience_epochs=collapse_patience, min_zero_classes=2)
    prior_correction_enabled = False
    ema = ModelEMA(model, decay=ema_decay) if ema_enable else None
    ema_update_count = 0
    parameter_update_written = False
    tc_auto_disabled_epoch = None
    tc_nonzero_epoch_count = 0
    tracked_named_parameters = dict(model.named_parameters())
    tracked_parameter_before = {
        name: tracked_named_parameters[name].detach().clone()
        for name in threshold_param_names
        if name in tracked_named_parameters
    }
    if learnable_threshold_enabled:
        save_json(
            change_proof_dir / "parameter_update_check.json",
            {
                "split": int(split_idx),
                "parameter_names": threshold_param_names,
                "has_gradient": False,
                "updated": False,
                "gradient_summary": {},
                "before_summary": named_parameter_summaries(model, threshold_param_names),
                "after_summary": {},
            },
        )

    def _run_validation_with_active_weights():
        if ema is None or not ema_use_for_eval:
            return run_inference(
                model,
                val_loader,
                device,
                mixed_precision=bool(mixed_precision and device.type == "cuda"),
                non_blocking=bool(pin_memory),
            )
        backup = ema.apply_to(model)
        try:
            return run_inference(
                model,
                val_loader,
                device,
                mixed_precision=bool(mixed_precision and device.type == "cuda"),
                non_blocking=bool(pin_memory),
            )
        finally:
            ema.restore(model, backup)

    for epoch in range(1, epochs + 1):
        loss_fn, loss_summary = _build_main_loss(
            cfg=cfg,
            labels=train_ds.all_labels,
            num_classes=num_classes,
            sampler_name=sampler_name,
            device=device,
            current_epoch=epoch,
            total_epochs=epochs,
        )
        model.train()
        total_loss = 0.0
        train_correct = 0
        total_firing_rate = 0.0
        firing_batches = 0
        layer_firing_totals: Dict[str, float] = {}
        loss_component_totals: Dict[str, float] = {}
        n_train = 0
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=bool(pin_memory))
            y = batch[1].to(device, non_blocking=bool(pin_memory))
            prev_labels = batch[4].to(device, non_blocking=bool(pin_memory)) if len(batch) > 4 else None
            next_labels = batch[5].to(device, non_blocking=bool(pin_memory)) if len(batch) > 5 else None
            if input_mixstyle_enable:
                x = _mixstyle_channels(x, p=input_mixstyle_p, alpha=input_mixstyle_alpha)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "reset_state"):
                model.reset_state()
            if mixed_precision and device.type == "cuda":
                with _amp_autocast(device, enabled=bool(mixed_precision)):
                    outputs = model(x)
                    temporal_reference_logits = None
                    if bool(temporal_consistency_enable) and temporal_consistency_mode == "stochastic_forward":
                        if hasattr(model, "reset_state"):
                            model.reset_state()
                        with torch.no_grad():
                            ref_outputs = model(x)
                        ref_logits = ref_outputs["main"] if isinstance(ref_outputs, dict) else ref_outputs
                        if isinstance(ref_logits, torch.Tensor):
                            temporal_reference_logits = ref_logits.detach()
                    loss, loss_stats = _compute_loss(
                        outputs,
                        loss_fn,
                        y,
                        model,
                        model_name,
                        wake_label,
                        rem_label,
                        n1_label_idx,
                        lambda_s,
                        lambda_w,
                        aux_sleep_weight,
                        aux_rem_weight,
                        aux_n1_weight=aux_n1_weight,
                        transition_weight=transition_weight,
                        firing_reg_weight=firing_reg_weight,
                        firing_target_low=firing_target_low,
                        firing_target_high=firing_target_high,
                        label_smoothing=label_smoothing,
                        label_smoothing_weight=label_smoothing_weight,
                        prev_labels=prev_labels,
                        next_labels=next_labels,
                        boundary_soft_label_enable=boundary_soft_label_enable,
                        boundary_soft_label_weight=boundary_soft_label_weight,
                        boundary_soft_primary_weight=boundary_soft_primary_weight,
                        boundary_soft_neighbor_weight=boundary_soft_neighbor_weight,
                        temporal_consistency_enable=temporal_consistency_enable,
                        temporal_consistency_weight=temporal_consistency_weight,
                        temporal_consistency_temperature=temporal_consistency_temperature,
                        temporal_consistency_mode=temporal_consistency_mode,
                        temporal_reference_logits=temporal_reference_logits,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if learnable_threshold_enabled and (not parameter_update_written):
                    grad_summary = named_gradient_summaries(model, threshold_param_names)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                if learnable_threshold_enabled and (not parameter_update_written):
                    current_named_parameters = dict(model.named_parameters())
                    after_summary = named_parameter_summaries(model, threshold_param_names)
                    updated = any(
                        bool(not torch.equal(tracked_parameter_before[name], current_named_parameters[name].detach()))
                        for name in threshold_param_names
                        if name in tracked_parameter_before and name in current_named_parameters
                    )
                    save_json(
                        change_proof_dir / "parameter_update_check.json",
                        {
                            "split": int(split_idx),
                            "parameter_names": threshold_param_names,
                            "has_gradient": bool(any(bool(item.get("has_grad", False)) for item in grad_summary.values())),
                            "updated": bool(updated),
                            "gradient_summary": grad_summary,
                            "before_summary": {
                                name: {
                                    "numel": int(tracked_parameter_before[name].numel()),
                                    "mean": float(tracked_parameter_before[name].detach().float().mean().item()),
                                    "std": float(tracked_parameter_before[name].detach().float().std(unbiased=False).item()),
                                    "min": float(tracked_parameter_before[name].detach().float().min().item()),
                                    "max": float(tracked_parameter_before[name].detach().float().max().item()),
                                    "norm": float(torch.linalg.vector_norm(tracked_parameter_before[name].detach().float().reshape(-1)).item()),
                                }
                                for name in threshold_param_names
                                if name in tracked_parameter_before
                            },
                            "after_summary": after_summary,
                        },
                    )
                    parameter_update_written = True
                if ema is not None:
                    ema.update(model)
                    ema_update_count += 1
            else:
                outputs = model(x)
                temporal_reference_logits = None
                if bool(temporal_consistency_enable) and temporal_consistency_mode == "stochastic_forward":
                    if hasattr(model, "reset_state"):
                        model.reset_state()
                    with torch.no_grad():
                        ref_outputs = model(x)
                    ref_logits = ref_outputs["main"] if isinstance(ref_outputs, dict) else ref_outputs
                    if isinstance(ref_logits, torch.Tensor):
                        temporal_reference_logits = ref_logits.detach()
                loss, loss_stats = _compute_loss(
                    outputs,
                    loss_fn,
                    y,
                    model,
                    model_name,
                    wake_label,
                    rem_label,
                    n1_label_idx,
                    lambda_s,
                    lambda_w,
                    aux_sleep_weight,
                    aux_rem_weight,
                    aux_n1_weight=aux_n1_weight,
                    transition_weight=transition_weight,
                    firing_reg_weight=firing_reg_weight,
                    firing_target_low=firing_target_low,
                    firing_target_high=firing_target_high,
                    label_smoothing=label_smoothing,
                    label_smoothing_weight=label_smoothing_weight,
                    prev_labels=prev_labels,
                    next_labels=next_labels,
                    boundary_soft_label_enable=boundary_soft_label_enable,
                    boundary_soft_label_weight=boundary_soft_label_weight,
                    boundary_soft_primary_weight=boundary_soft_primary_weight,
                    boundary_soft_neighbor_weight=boundary_soft_neighbor_weight,
                    temporal_consistency_enable=temporal_consistency_enable,
                    temporal_consistency_weight=temporal_consistency_weight,
                    temporal_consistency_temperature=temporal_consistency_temperature,
                    temporal_consistency_mode=temporal_consistency_mode,
                    temporal_reference_logits=temporal_reference_logits,
                )
                loss.backward()
                if learnable_threshold_enabled and (not parameter_update_written):
                    grad_summary = named_gradient_summaries(model, threshold_param_names)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                if learnable_threshold_enabled and (not parameter_update_written):
                    current_named_parameters = dict(model.named_parameters())
                    after_summary = named_parameter_summaries(model, threshold_param_names)
                    updated = any(
                        bool(not torch.equal(tracked_parameter_before[name], current_named_parameters[name].detach()))
                        for name in threshold_param_names
                        if name in tracked_parameter_before and name in current_named_parameters
                    )
                    save_json(
                        change_proof_dir / "parameter_update_check.json",
                        {
                            "split": int(split_idx),
                            "parameter_names": threshold_param_names,
                            "has_gradient": bool(any(bool(item.get("has_grad", False)) for item in grad_summary.values())),
                            "updated": bool(updated),
                            "gradient_summary": grad_summary,
                            "before_summary": {
                                name: {
                                    "numel": int(tracked_parameter_before[name].numel()),
                                    "mean": float(tracked_parameter_before[name].detach().float().mean().item()),
                                    "std": float(tracked_parameter_before[name].detach().float().std(unbiased=False).item()),
                                    "min": float(tracked_parameter_before[name].detach().float().min().item()),
                                    "max": float(tracked_parameter_before[name].detach().float().max().item()),
                                    "norm": float(torch.linalg.vector_norm(tracked_parameter_before[name].detach().float().reshape(-1)).item()),
                                }
                                for name in threshold_param_names
                                if name in tracked_parameter_before
                            },
                            "after_summary": after_summary,
                        },
                    )
                    parameter_update_written = True
                if ema is not None:
                    ema.update(model)
                    ema_update_count += 1
            logits_for_acc = _main_logits(outputs)
            train_correct += int(torch.eq(torch.argmax(logits_for_acc.detach(), dim=1), y).sum().item())
            total_loss += float(loss.detach().item()) * int(y.size(0))
            for key, value in loss_stats.items():
                loss_component_totals[key] = loss_component_totals.get(key, 0.0) + float(value)
            if isinstance(outputs, dict) and isinstance(outputs.get("firing_rate"), torch.Tensor):
                total_firing_rate += float(outputs["firing_rate"].detach().mean().item())
                firing_batches += 1
            if isinstance(outputs, dict) and isinstance(outputs.get("layer_firing_rates"), dict):
                for key, value in outputs["layer_firing_rates"].items():
                    rate_value = float(value.detach().mean().item()) if isinstance(value, torch.Tensor) else float(value)
                    layer_firing_totals[str(key)] = layer_firing_totals.get(str(key), 0.0) + rate_value
            n_train += int(y.size(0))
        train_loss = total_loss / float(max(1, n_train))
        train_acc = float(train_correct / float(max(1, n_train)))
        train_firing_rate = total_firing_rate / float(max(1, firing_batches))
        train_layer_firing = {
            key: value / float(max(1, firing_batches)) for key, value in sorted(layer_firing_totals.items())
        }
        mean_loss_components = {
            key: value / float(max(1, len(train_loader))) for key, value in sorted(loss_component_totals.items())
        }

        val_out = _run_validation_with_active_weights()
        y_true = val_out["y_true"]
        y_pred = val_out["y_pred"]
        val_acc = float(np.mean(y_true == y_pred))
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        val_kappa = float(cohen_kappa_score(y_true, y_pred))
        per_class_payload = _per_class_metrics(y_true, y_pred, labels)
        n1_metrics = per_class_payload["by_class"].get("N1", {"precision": 0.0, "f1": 0.0, "recall": 0.0})
        rem_metrics = per_class_payload["by_class"].get("REM", {"f1": 0.0, "recall": 0.0})
        val_n1_precision = float(n1_metrics.get("precision", 0.0))
        val_n1_f1 = float(n1_metrics.get("f1", 0.0))
        val_n1_recall = float(n1_metrics.get("recall", 0.0))
        val_rem_f1 = float(rem_metrics.get("f1", 0.0))
        pred_counts = np.bincount(y_pred, minlength=num_classes).astype(np.int64)
        pred_ratio = (pred_counts / max(1, int(pred_counts.sum()))).tolist()
        val_pred_ratio_n1 = float(pred_ratio[n1_label_idx]) if n1_label_idx < len(pred_ratio) else 0.0
        current_selection = _current_best_ckpt_decision(
            rule_name=best_ckpt_rule,
            val_macro_f1=val_macro_f1,
            val_kappa=val_kappa,
            val_n1_f1=val_n1_f1,
            val_n1_recall=val_n1_recall,
            val_rem_f1=val_rem_f1,
            val_acc=val_acc,
            macro_f1_weight=best_ckpt_macro_f1_weight,
            n1_f1_weight=best_ckpt_n1_f1_weight,
            kappa_weight=best_ckpt_kappa_weight,
            rem_f1_weight=best_ckpt_rem_f1_weight,
            n1_floor=best_ckpt_n1_floor,
            n1_recall_floor=best_ckpt_n1_recall_floor,
        )
        current_ckpt_key = tuple(current_selection["sort_key"])
        current_metric = float(current_selection["score"])
        current_tc_loss = float(mean_loss_components.get("tc_loss", 0.0))
        tc_enabled_this_epoch = bool(temporal_consistency_enable)
        tc_epoch_history.append(current_tc_loss)
        if not loss_is_effectively_zero(current_tc_loss, tol=tc_zero_guard_tol):
            tc_nonzero_epoch_count += 1
        if bool(temporal_consistency_enable) and has_consecutive_effective_zeros(
            tc_epoch_history[-tc_zero_guard_epochs:],
            streak=tc_zero_guard_epochs,
            tol=tc_zero_guard_tol,
        ):
            tc_auto_disabled_epoch = int(epoch)
            temporal_consistency_enable = False
            temporal_consistency_weight = 0.0
            print(f"split={split_idx} epoch={epoch} temporal consistency auto-disabled due to sustained zero tc_loss")

        candidate_state_dict = (
            ema.state_dict() if ema is not None and ema_use_for_eval else _clone_state_dict(model.state_dict())
        )
        if best_ckpt_fallback_topk > 0:
            topk_macro_candidates.append(
                {
                    "epoch": int(epoch),
                    "val_macro_f1": float(val_macro_f1),
                    "val_n1_f1": float(val_n1_f1),
                    "val_rem_f1": float(val_rem_f1),
                    "selection_score": float(current_metric),
                    "state_dict": candidate_state_dict,
                    "best_ckpt_eligible": bool(current_selection["eligible"]),
                }
            )
            topk_macro_candidates.sort(
                key=lambda item: (
                    float(item["selection_score"]),
                    float(item["val_macro_f1"]),
                    float(item["val_n1_f1"]),
                    float(item["val_rem_f1"]),
                    -int(item["epoch"]),
                ),
                reverse=True,
            )
            topk_macro_candidates = topk_macro_candidates[: max(1, best_ckpt_fallback_topk)]

        collapse_fix = 0
        if collapse_enable and protector.update(pred_ratio=pred_ratio, pred_counts=pred_counts):
            def _strengthen_sampler() -> bool:
                nonlocal train_loader, sampler_name
                changed = sampler_name != "weighted"
                sampler_name = "weighted"
                if changed:
                    train_loader = _make_train_loader(sampler_name)
                return changed

            def _disable_prior() -> bool:
                nonlocal prior_correction_enabled
                changed = bool(prior_correction_enabled)
                prior_correction_enabled = False
                return changed

            collapse_action = apply_collapse_stabilization(
                model=model,
                optimizer=optimizer,
                loss_fn=torch.nn.CrossEntropyLoss(),
                best_state_dict=best_state,
                lr_decay=collapse_lr_decay,
                tau_after_trigger=0.0,
                switch_to_stable_loss=lambda: False,
                strengthen_sampler=_strengthen_sampler,
                disable_prior_correction=_disable_prior,
            )
            if collapse_action.get("sampler_strengthened", False):
                loss_summary["sampler"] = sampler_name
            collapse_fix = 1

        append_jsonl(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_firing_rate": train_firing_rate,
                "train_layer_firing_rates": train_layer_firing,
                "loss_components": mean_loss_components,
                "val_acc": val_acc,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_kappa": val_kappa,
                "val_n1_precision": val_n1_precision,
                "val_n1_f1": val_n1_f1,
                "val_n1_recall": val_n1_recall,
                "val_rem_f1": val_rem_f1,
                "N1_precision": val_n1_precision,
                "N1_f1": val_n1_f1,
                "N1_recall": val_n1_recall,
                "REM_f1": val_rem_f1,
                "val_per_class": per_class_payload["rows"],
                "val_pred_ratio": pred_ratio,
                "pred_ratio": pred_ratio,
                "best_ckpt_eligible": bool(current_selection["eligible"]),
                "best_ckpt_rule": str(best_ckpt_rule),
                "best_ckpt_score": float(current_metric),
                "best_ckpt_score_formula": str(current_selection.get("score_formula", "")),
                "best_ckpt_gate_failures": list(current_selection.get("gate_failures", [])),
                "n1_suppression": bool(val_pred_ratio_n1 < 0.03),
                "collapse_fix": collapse_fix,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "loss_name": loss_summary["loss_name"],
                "active_loss_class": loss_summary.get("active_loss_class", type(loss_fn).__name__),
                "loss_use_class_weights": loss_summary["use_class_weights"],
                "drw_active": bool(loss_summary.get("drw_active", False)),
                "class_weights": loss_summary.get("class_weights"),
                "current_loss_schedule": str(loss_summary.get("current_loss_schedule", "")),
                "whether_drw_enabled": bool(loss_summary.get("drw_active", False)),
                "whether_ema_or_swa_enabled": bool(ema_enable),
                "ema_used_for_eval": bool(ema_use_for_eval),
                "boundary_soft_label_enable": bool(boundary_soft_label_enable),
                "temporal_consistency_enable": bool(tc_enabled_this_epoch),
                "temporal_consistency_mode": str(temporal_consistency_mode),
            },
        )
        selection_trace_rows.append(
            {
                "epoch": int(epoch),
                "score": float(current_metric),
                "eligible": bool(current_selection["eligible"]),
                "val_macro_f1": float(val_macro_f1),
                "val_n1_f1": float(val_n1_f1),
                "val_n1_recall": float(val_n1_recall),
                "val_rem_f1": float(val_rem_f1),
                "val_kappa": float(val_kappa),
                "val_acc": float(val_acc),
                "rule_name": str(best_ckpt_rule),
                "score_formula": str(current_selection.get("score_formula", "")),
                "score_components": dict(current_selection.get("score_components", {})),
                "gate_failures": list(current_selection.get("gate_failures", [])),
            }
        )
        epoch_detail_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "active_loss_class": str(loss_summary.get("active_loss_class", type(loss_fn).__name__)),
                "loss_name": str(loss_summary.get("loss_name", "")),
                "drw_active": bool(loss_summary.get("drw_active", False)),
                "whether_drw_enabled": bool(loss_summary.get("drw_active", False)),
                "whether_ema_or_swa_enabled": bool(ema_enable),
                "current_loss_schedule": str(loss_summary.get("current_loss_schedule", "")),
                "class_weights": list(loss_summary.get("class_weights") or []),
                "best_ckpt_rule": str(best_ckpt_rule),
                "best_ckpt_score": float(current_metric),
                "best_ckpt_score_formula": str(current_selection.get("score_formula", "")),
                "best_ckpt_gate_failures": json.dumps(list(current_selection.get("gate_failures", [])), ensure_ascii=False),
                "cls_loss": float(mean_loss_components.get("cls_loss", mean_loss_components.get("base_loss", 0.0))),
                "aux_loss": float(mean_loss_components.get("aux_loss", 0.0)),
                "tc_loss": current_tc_loss,
                "total_loss": float(mean_loss_components.get("total_loss", train_loss)),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
                "val_kappa": float(val_kappa),
                "val_N1_precision": float(val_n1_precision),
                "val_N1_f1": float(val_n1_f1),
                "val_N1_recall": float(val_n1_recall),
                "val_REM_f1": float(val_rem_f1),
                "val_pred_ratio": json.dumps([float(v) for v in pred_ratio], ensure_ascii=False),
                "best_ckpt_eligible": bool(current_selection["eligible"]),
                "n1_suppression": bool(val_pred_ratio_n1 < 0.03),
            }
        )
        print(
            f"split={split_idx} epoch={epoch} loss={train_loss:.4f} "
            f"cls_loss={mean_loss_components.get('cls_loss', mean_loss_components.get('base_loss', 0.0)):.4f} "
            f"aux_loss={mean_loss_components.get('aux_loss', 0.0):.4f} "
            f"tc_loss={mean_loss_components.get('tc_loss', 0.0):.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} macro_f1={val_macro_f1:.4f} kappa={val_kappa:.4f} "
            f"N1_precision={val_n1_precision:.4f} N1_f1={val_n1_f1:.4f} N1_recall={val_n1_recall:.4f} REM_f1={val_rem_f1:.4f} "
            f"pred_ratio={[round(float(p), 4) for p in pred_ratio]} "
            f"current_loss_schedule={str(loss_summary.get('current_loss_schedule', ''))} "
            f"whether_drw_enabled={bool(loss_summary.get('drw_active', False))} "
            f"whether_ema_or_swa_enabled={bool(ema_enable)} "
            f"best_ckpt_score={float(current_metric):.4f}"
        )

        if bool(current_selection["eligible"]) and current_ckpt_key > best_ckpt_key_value:
            best_ckpt_key_value = current_ckpt_key
            best_metric = current_metric
            best_epoch = int(epoch)
            best_val_snapshot = {
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
                "val_kappa": float(val_kappa),
                "val_N1_precision": float(val_n1_precision),
                "val_N1_f1": float(val_n1_f1),
                "val_N1_recall": float(val_n1_recall),
                "val_REM_f1": float(val_rem_f1),
            }
            best_selection_snapshot = {
                "score": float(current_metric),
                "rule_name": str(best_ckpt_rule),
                "eligible": bool(current_selection["eligible"]),
                "epoch": int(epoch),
                "score_formula": str(current_selection.get("score_formula", "")),
                "score_components": dict(current_selection.get("score_components", {})),
                "gate_failures": list(current_selection.get("gate_failures", [])),
            }
            no_improve = 0
            best_state = _clone_state_dict(candidate_state_dict)
            model_hparams = model.get_hparams() if hasattr(model, "get_hparams") else model_hparams
            model_backup = _with_temporary_state_dict(model, candidate_state_dict)
            try:
                save_checkpoint(
                    split_dir / "best.ckpt",
                    model,
                    optimizer,
                    scaler if mixed_precision else None,
                    cfg,
                    epoch,
                    best_metric,
                    model_name=model_name,
                    model_hparams=model_hparams,
                    task=cfg.get("task", "sleep_edf_5class"),
                    num_classes=num_classes,
                    split_id=split_idx,
                    trial_id=0,
                )
            finally:
                model.load_state_dict(model_backup, strict=True)
            (split_dir / "best_hparams.json").write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "model_hparams": model_hparams,
                        "task": cfg.get("task", "sleep_edf_5class"),
                        "num_classes": int(num_classes),
                        "best_epoch": int(best_epoch),
                        "best_val_snapshot": best_val_snapshot,
                        "best_selection_snapshot": best_selection_snapshot,
                        "best_ckpt_n1_f1_floor": float(best_ckpt_n1_floor),
                        "best_ckpt_n1_recall_floor": float(best_ckpt_n1_recall_floor),
                        "best_ckpt_key": list(best_ckpt_key_value),
                        "best_ckpt_rule": str(best_ckpt_rule),
                        "best_ckpt_rule_description": str(best_ckpt_rule_description),
                        "ema_enable": bool(ema_enable),
                        "ema_use_for_eval": bool(ema_use_for_eval),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            no_improve += 1
            if epoch >= early_stop_guard_epoch and no_improve >= patience:
                break

        if split_idx == 0 and epoch == 1:
            epoch1_debug = _collect_forward_debug(
                model=model,
                dataloader=val_loader,
                device=device,
                mixed_precision=bool(mixed_precision),
                pin_memory=bool(pin_memory),
                max_batches=3,
            )
            _merge_json(audit_dir / "logit_debug_split0.json", {"post_epoch_1": epoch1_debug})

    epoch_metrics_csv = split_dir / "epoch_metrics_detailed.csv"
    if epoch_detail_rows:
        with epoch_metrics_csv.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(epoch_detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_detail_rows)

    fallback_used = False
    fallback_payload: dict[str, object] = {}
    if not (split_dir / "best.ckpt").exists() and best_ckpt_fallback_topk > 0 and topk_macro_candidates:
        fallback_used = True
        fallback_candidates = topk_macro_candidates[: max(1, min(len(topk_macro_candidates), best_ckpt_fallback_topk))]
        averaged_state = _average_state_dicts([item["state_dict"] for item in fallback_candidates])
        best_state = _clone_state_dict(averaged_state)
        best_epoch = int(fallback_candidates[0]["epoch"])
        best_metric = float(np.mean([float(item["selection_score"]) for item in fallback_candidates]))
        epoch_row_map = {int(row["epoch"]): row for row in epoch_detail_rows}
        best_epoch_row = epoch_row_map.get(best_epoch, {})
        best_val_snapshot = {
            "val_acc": float(best_epoch_row.get("val_acc", float("nan"))),
            "val_macro_f1": float(best_epoch_row.get("val_macro_f1", float("nan"))),
            "val_kappa": float(best_epoch_row.get("val_kappa", float("nan"))),
            "val_N1_precision": float(best_epoch_row.get("val_N1_precision", float("nan"))),
            "val_N1_f1": float(best_epoch_row.get("val_N1_f1", float("nan"))),
            "val_N1_recall": float(best_epoch_row.get("val_N1_recall", float("nan"))),
            "val_REM_f1": float(best_epoch_row.get("val_REM_f1", float("nan"))),
        }
        best_selection_snapshot = {
            "score": float(best_metric),
            "rule_name": "fallback_topk_selection_score_average",
            "eligible": False,
            "epoch": int(best_epoch),
            "score_formula": "mean(topk_selection_scores)",
        }
        model_backup = _with_temporary_state_dict(model, averaged_state)
        try:
            save_checkpoint(
                split_dir / "best.ckpt",
                model,
                optimizer,
                scaler if mixed_precision else None,
                cfg,
                best_epoch,
                best_metric,
                model_name=model_name,
                model_hparams=model_hparams,
                task=cfg.get("task", "sleep_edf_5class"),
                num_classes=num_classes,
                split_id=split_idx,
                trial_id=0,
            )
        finally:
            model.load_state_dict(model_backup, strict=True)
        fallback_payload = {
            "fallback_used": True,
            "fallback_name": "topk_selection_score_state_averaging",
            "topk": int(len(fallback_candidates)),
            "candidate_epochs": [int(item["epoch"]) for item in fallback_candidates],
            "candidate_selection_scores": [float(item["selection_score"]) for item in fallback_candidates],
            "candidate_val_macro_f1": [float(item["val_macro_f1"]) for item in fallback_candidates],
        }
        (split_dir / "best_hparams.json").write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "model_hparams": model_hparams,
                    "task": cfg.get("task", "sleep_edf_5class"),
                    "num_classes": int(num_classes),
                    "best_epoch": int(best_epoch),
                    "best_val_snapshot": best_val_snapshot,
                    "best_selection_snapshot": best_selection_snapshot,
                    "best_ckpt_n1_f1_floor": float(best_ckpt_n1_floor),
                    "best_ckpt_n1_recall_floor": float(best_ckpt_n1_recall_floor),
                    "best_ckpt_key": list(best_ckpt_key_value),
                    "best_ckpt_rule": str(best_ckpt_rule),
                    "best_ckpt_rule_description": str(best_ckpt_rule_description),
                    "ema_enable": bool(ema_enable),
                    "ema_use_for_eval": bool(ema_use_for_eval),
                    "fallback": fallback_payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    save_json(
        change_proof_dir / "ldam_drw_schedule.json",
        {
            "split": int(split_idx),
            "loss_name": str(loss_summary.get("loss_name", "")),
            "drw_start_ratio": float(cfg.get("loss", {}).get("drw_start_ratio", 0.5)),
            "drw_switch_epoch": None if drw_switch_epoch <= 0 else int(drw_switch_epoch),
            "early_stop_guard_epoch": int(early_stop_guard_epoch),
            "drw_schedule": _compress_drw_schedule(epoch_detail_rows),
            "delta": {
                "baseline_locked": {
                    "active_model_class": "ContextPicoSNN",
                    "active_loss_class": "LDAMLoss",
                    "whether_drw_enabled": False,
                },
                "candidate": {
                    "active_model_class": "ContextPicoSNN",
                    "active_loss_class": str(loss_summary.get("active_loss_class", type(loss_fn).__name__)),
                    "whether_drw_enabled": bool(any(bool(row.get("whether_drw_enabled", False)) for row in epoch_detail_rows)),
                },
            },
        },
    )
    save_json(
        change_proof_dir / "ema_or_swa_report.json",
        {
            "split": int(split_idx),
            "strategy": "EMA" if ema_strategy == "ema" else str(ema_strategy).upper(),
            "enabled": bool(ema_enable),
            "used_for_eval": bool(ema_use_for_eval),
            "decay": float(ema_decay),
            "update_count": int(ema.num_updates if ema is not None else ema_update_count),
            "averaged_weights_maintained": bool((ema.num_updates if ema is not None else ema_update_count) > 0),
            "averaged_checkpoint_path": str(split_dir / "best.ckpt"),
            "averaged_checkpoint_exported": bool((split_dir / "best.ckpt").exists() and ema_enable and ema_use_for_eval),
            "delta": {
                "baseline_locked": {"whether_ema_or_swa_enabled": False},
                "candidate": {"whether_ema_or_swa_enabled": bool(ema_enable)},
            },
        },
    )
    save_json(
        change_proof_dir / "ckpt_selection_rule.json",
        {
            "split": int(split_idx),
            "rule_name": str(best_ckpt_rule),
            "metric_name": str(best_ckpt_metric_name),
            "rule_description": str(best_ckpt_rule_description),
            "score_formula": MACRO_N1_REM_GUARDED_SCORE_FORMULA if best_ckpt_rule == MACRO_N1_REM_GUARDED_CKPT_RULE else "legacy_lexicographic",
            "score_weights": {
                "val_macro_f1": float(best_ckpt_macro_f1_weight),
                "val_N1_f1": float(best_ckpt_n1_f1_weight),
                "val_kappa": float(best_ckpt_kappa_weight),
                "val_REM_f1": float(best_ckpt_rem_f1_weight),
            },
            "hard_thresholds": {
                "val_N1_f1": float(best_ckpt_n1_floor),
                "val_N1_recall": float(best_ckpt_n1_recall_floor),
            },
            "fallback_policy": {
                "topk_selection_score_state_averaging": int(best_ckpt_fallback_topk),
                **fallback_payload,
            },
            "selected_checkpoint": {
                "best_epoch": int(best_epoch),
                "best_selection_snapshot": best_selection_snapshot,
            },
            "early_stop_guard_epoch": int(early_stop_guard_epoch),
            "epochs": selection_trace_rows,
            "delta": {
                "baseline_locked": {
                    "rule_name": "legacy_lexicographic",
                    "val_N1_f1_floor": 0.15,
                    "val_N1_recall_floor": 0.12,
                },
                "candidate": {
                    "rule_name": str(best_ckpt_rule),
                    "val_N1_f1_floor": float(best_ckpt_n1_floor),
                    "val_N1_recall_floor": float(best_ckpt_n1_recall_floor),
                },
            },
        },
    )
    save_json(
        change_proof_dir / "temporal_consistency_truth.json",
        {
            "split": int(split_idx),
            "requested": bool(temporal_consistency_requested),
            "enabled_during_training": bool(any(bool(row.get("temporal_consistency_enable", False)) for row in epoch_detail_rows)),
            "final_enabled": bool(temporal_consistency_enable),
            "status": (
                "enabled_with_nonzero_loss"
                if tc_nonzero_epoch_count > 0
                else ("removed_from_active_mainline" if not temporal_consistency_requested else "requested_but_zero_or_disabled")
            ),
            "mode": str(temporal_consistency_mode),
            "weight": float(train_cfg.get("temporal_consistency_weight", 0.0)),
            "temperature": float(temporal_consistency_temperature),
            "auto_disabled_epoch": None if tc_auto_disabled_epoch is None else int(tc_auto_disabled_epoch),
            "nonzero_epoch_count": int(tc_nonzero_epoch_count),
            "epochs": [
                {
                    "epoch": int(row["epoch"]),
                    "temporal_consistency_enable": bool(row.get("temporal_consistency_enable", False)),
                    "tc_loss": float(row.get("tc_loss", 0.0)),
                }
                for row in epoch_detail_rows
            ],
            "delta": {
                "baseline_locked": {"temporal_consistency_enable": False, "tc_loss": 0.0},
                "candidate": {
                    "temporal_consistency_enable": bool(temporal_consistency_requested),
                    "observed_nonzero_tc_loss": bool(tc_nonzero_epoch_count > 0),
                    "removed_from_active_mainline": bool((not temporal_consistency_requested) and tc_nonzero_epoch_count == 0),
                },
            },
        },
    )

    if not (split_dir / "best.ckpt").exists():
        n1_guard_payload = {
            "split": int(split_idx),
            "recipe_name": str(train_cfg.get("recipe_name", "")),
            "status": "n1_collapse",
            "best_ckpt_n1_f1_floor": float(best_ckpt_n1_floor),
            "best_ckpt_n1_recall_floor": float(best_ckpt_n1_recall_floor),
            "epochs": epoch_detail_rows,
        }
        save_json(split_dir / "n1_collapse_failure.json", n1_guard_payload)
        raise RuntimeError(
            f"split={split_idx} N1 collapse: no epoch passed val_N1_f1>={best_ckpt_n1_floor:.2f} "
            f"and val_N1_recall>={best_ckpt_n1_recall_floor:.2f}"
        )

    enable_qat = bool(_to_bool(train_cfg.get("enable_qat", False))) and (not fp32_only)
    if enable_qat and hasattr(model, "enable_qat"):
        state = load_checkpoint(split_dir / "best.ckpt")
        model.load_state_dict(state["model_state"], strict=True)
        model.to(device)
        model.enable_qat(bits=int(train_cfg.get("qat_bits", 6)))
        qat_epochs = int(train_cfg.get("qat_epochs", 2))
        qat_lr = float(train_cfg.get("qat_lr", float(cfg.get("lr", 1e-3)) * 0.2))
        optimizer_qat = torch.optim.AdamW(model.parameters(), lr=qat_lr, weight_decay=float(cfg.get("weight_decay", 1e-2)))
        scaler_qat = _amp_grad_scaler(device, enabled=bool(mixed_precision))
        best_qat_metric = best_metric
        for _ in range(max(1, qat_epochs)):
            model.train()
            for batch in train_loader:
                x = batch[0].to(device, non_blocking=bool(pin_memory))
                y = batch[1].to(device, non_blocking=bool(pin_memory))
                prev_labels = batch[4].to(device, non_blocking=bool(pin_memory)) if len(batch) > 4 else None
                next_labels = batch[5].to(device, non_blocking=bool(pin_memory)) if len(batch) > 5 else None
                if input_mixstyle_enable:
                    x = _mixstyle_channels(x, p=input_mixstyle_p, alpha=input_mixstyle_alpha)
                optimizer_qat.zero_grad(set_to_none=True)
                if hasattr(model, "reset_state"):
                    model.reset_state()
                if mixed_precision and device.type == "cuda":
                    with _amp_autocast(device, enabled=bool(mixed_precision)):
                        outputs = model(x)
                        loss, _ = _compute_loss(
                            outputs,
                            loss_fn,
                            y,
                            model,
                            model_name,
                            wake_label,
                            rem_label,
                            n1_label_idx,
                            lambda_s,
                            lambda_w,
                            aux_sleep_weight,
                            aux_rem_weight,
                            aux_n1_weight=aux_n1_weight,
                            transition_weight=transition_weight,
                            firing_reg_weight=firing_reg_weight,
                            firing_target_low=firing_target_low,
                            firing_target_high=firing_target_high,
                            label_smoothing=label_smoothing,
                            label_smoothing_weight=label_smoothing_weight,
                            prev_labels=prev_labels,
                            next_labels=next_labels,
                            boundary_soft_label_enable=boundary_soft_label_enable,
                            boundary_soft_label_weight=boundary_soft_label_weight,
                            boundary_soft_primary_weight=boundary_soft_primary_weight,
                            boundary_soft_neighbor_weight=boundary_soft_neighbor_weight,
                            temporal_consistency_enable=temporal_consistency_enable,
                            temporal_consistency_weight=temporal_consistency_weight,
                            temporal_consistency_temperature=temporal_consistency_temperature,
                        )
                    scaler_qat.scale(loss).backward()
                    scaler_qat.step(optimizer_qat)
                    scaler_qat.update()
                else:
                    outputs = model(x)
                    loss, _ = _compute_loss(
                        outputs,
                        loss_fn,
                        y,
                        model,
                        model_name,
                        wake_label,
                        rem_label,
                        n1_label_idx,
                        lambda_s,
                        lambda_w,
                        aux_sleep_weight,
                        aux_rem_weight,
                        aux_n1_weight=aux_n1_weight,
                        transition_weight=transition_weight,
                        firing_reg_weight=firing_reg_weight,
                        firing_target_low=firing_target_low,
                        firing_target_high=firing_target_high,
                        label_smoothing=label_smoothing,
                        label_smoothing_weight=label_smoothing_weight,
                        prev_labels=prev_labels,
                        next_labels=next_labels,
                        boundary_soft_label_enable=boundary_soft_label_enable,
                        boundary_soft_label_weight=boundary_soft_label_weight,
                        boundary_soft_primary_weight=boundary_soft_primary_weight,
                        boundary_soft_neighbor_weight=boundary_soft_neighbor_weight,
                        temporal_consistency_enable=temporal_consistency_enable,
                        temporal_consistency_weight=temporal_consistency_weight,
                        temporal_consistency_temperature=temporal_consistency_temperature,
                    )
                    loss.backward()
                    optimizer_qat.step()
            val_out = run_inference(model, val_loader, device, mixed_precision=bool(mixed_precision and device.type == "cuda"), non_blocking=bool(pin_memory))
            qat_metric = float(f1_score(val_out["y_true"], val_out["y_pred"], average="macro", zero_division=0))
            if qat_metric > best_qat_metric:
                best_qat_metric = qat_metric
                save_checkpoint(
                    split_dir / "best.ckpt",
                    model,
                    optimizer_qat,
                    scaler_qat if mixed_precision else None,
                    cfg,
                    epoch=epochs,
                    best_metric=best_qat_metric,
                    model_name=model_name,
                    model_hparams=model_hparams,
                    task=cfg.get("task", "sleep_edf_5class"),
                    num_classes=num_classes,
                    split_id=split_idx,
                    trial_id=0,
                )
        save_checkpoint(
            split_dir / "best_qat.ckpt",
            model,
            optimizer_qat,
            scaler_qat if mixed_precision else None,
            cfg,
            epoch=epochs,
            best_metric=best_qat_metric,
            model_name=model_name,
            model_hparams=model_hparams,
            task=cfg.get("task", "sleep_edf_5class"),
            num_classes=num_classes,
            split_id=split_idx,
            trial_id=0,
        )

    split_meta = {
        "split": int(split_idx),
        "protocol": str(split.get("protocol", "subject_kfold")),
        "feasible": True,
        "train_class_counts": _json_counts(_entry_counts(train_entries, num_classes), labels),
        "val_class_counts": _json_counts(_entry_counts(val_entries, num_classes), labels),
        "test_class_counts": _json_counts(_entry_counts(test_entries, num_classes), labels),
        "model_name": model_name,
        "model_hparams": model_hparams,
        "preset": str(train_cfg.get("preset", "")),
        "recipe_name": str(train_cfg.get("recipe_name", "")),
        "use_dual_lcs": use_dual_lcs,
        "use_aux_heads": use_aux_heads,
        "transition_aux_weight": transition_weight,
        "firing_reg_weight": firing_reg_weight,
        "label_smoothing": label_smoothing,
        "label_smoothing_weight": label_smoothing_weight,
        "ema_enable": bool(ema_enable),
        "ema_decay": float(ema_decay),
        "ema_use_for_eval": bool(ema_use_for_eval),
        "ema_strategy": str(ema_strategy),
        "boundary_soft_label_enable": bool(boundary_soft_label_enable),
        "boundary_soft_label_weight": float(boundary_soft_label_weight),
        "temporal_consistency_enable": bool(temporal_consistency_requested),
        "temporal_consistency_final_enable": bool(temporal_consistency_enable),
        "temporal_consistency_weight": float(train_cfg.get("temporal_consistency_weight", 0.0)),
        "temporal_consistency_temperature": float(temporal_consistency_temperature),
        "temporal_consistency_mode": str(temporal_consistency_mode),
        "sequence_context_enabled": bool(context_len > 1),
        "context_len": int(context_len),
        "boundary_sampling_boost": float(boundary_sampling_boost),
        "input_norm_mode": str(input_norm_mode),
        "input_mixstyle_enable": bool(input_mixstyle_enable),
        "input_mixstyle_p": float(input_mixstyle_p),
        "input_mixstyle_alpha": float(input_mixstyle_alpha),
        "loss_name": str(loss_summary.get("loss_name", "")),
        "active_loss_class": str(loss_summary.get("active_loss_class", type(loss_fn).__name__)),
        "drw_active_at_last_epoch": bool(loss_summary.get("drw_active", False)),
        "learnable_threshold_enable": bool(learnable_threshold_enabled),
        "learnable_threshold_param_names": threshold_param_names,
        "best_epoch": int(best_epoch),
        "best_val_snapshot": best_val_snapshot,
        "best_selection_snapshot": best_selection_snapshot,
        "best_ckpt_metric_name": str(best_ckpt_metric_name),
        "best_ckpt_rule": str(best_ckpt_rule),
        "best_ckpt_score_formula": (
            MACRO_N1_REM_GUARDED_SCORE_FORMULA if best_ckpt_rule == MACRO_N1_REM_GUARDED_CKPT_RULE else "legacy_lexicographic"
        ),
        "best_ckpt_rem_f1_weight": float(best_ckpt_rem_f1_weight),
        "best_ckpt_n1_f1_floor": float(best_ckpt_n1_floor),
        "best_ckpt_n1_recall_floor": float(best_ckpt_n1_recall_floor),
        "best_ckpt_fallback_topk": int(best_ckpt_fallback_topk),
        "best_ckpt_selection_rule": str(best_ckpt_rule_description),
        "drw_switch_epoch": None if drw_switch_epoch <= 0 else int(drw_switch_epoch),
        "early_stop_guard_epoch": int(early_stop_guard_epoch),
    }
    (split_dir / "train_meta.json").write_text(json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_train_log_first_epochs_csv(change_proof_dir, split_idx=split_idx, log_path=log_path, max_epochs=3)
    _update_curve_difference_report(change_proof_dir, split_idx=split_idx, log_path=log_path)
    print(
        f"split={split_idx} done recipe={str(train_cfg.get('recipe_name', ''))} "
        f"best_metric={best_metric:.4f} ckpt={split_dir / 'best.ckpt'}"
    )


def main() -> None:
    setup_utf8_stdio()
    suppress_pin_memory_warning()
    if os.name == "nt":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_epochs", type=int, default=None)
    parser.add_argument("--smoke_splits", type=int, default=None)
    parser.add_argument("--smoke_train_per_class", type=int, default=None)
    parser.add_argument("--smoke_eval_per_class", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["baseline", "plus_without_transition", "plus_full", "context_pico", "context_pico_v2"],
    )
    parser.add_argument("--recipe", type=str, default=None, choices=SAFE_RECIPE_CHOICES)
    parser.add_argument("--fp32_only", action="store_true")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--only_splits", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    if config_path is None or not config_path.exists():
        raise RuntimeError(f"config not found: {args.config}")
    cfg = read_yaml(config_path)
    explicit_model = _canonical_model_name(args.model) if args.model else None
    if args.allow_cpu:
        cfg["allow_cpu"] = True
    inferred_model = explicit_model or _default_train_model_name(cfg.get("model", {}).get("name"))
    preset = str(
        args.preset
        or (
            "baseline"
            if inferred_model == "picosleepnet_baseline"
            else (
                "context_pico_v2"
                if inferred_model == "context_pico_snn_v2"
                else ("context_pico" if inferred_model == "context_pico_snn" else "plus_full")
            )
        )
    ).lower().strip()
    preset_model = _preset_model_name(preset)
    if explicit_model and explicit_model != preset_model:
        raise RuntimeError(f"--model ({explicit_model}) conflicts with --preset ({preset}).")
    recipe = str(args.recipe or _default_recipe_for_preset(preset)).lower().strip()
    cfg = _apply_training_preset(cfg, preset)
    cfg = _apply_training_recipe(cfg, preset, recipe)
    cfg.setdefault("model", {})
    cfg["model"]["name"] = preset_model
    cfg.setdefault("train", {})
    cfg["train"]["preset"] = preset
    cfg["train"]["recipe_name"] = recipe
    if args.smoke:
        cfg["train"]["smoke"] = True
    if args.smoke_epochs is not None:
        cfg["train"]["smoke_epochs"] = int(args.smoke_epochs)
    if args.smoke_splits is not None:
        cfg["train"]["smoke_splits"] = int(args.smoke_splits)
    if args.smoke_train_per_class is not None:
        cfg["train"]["smoke_train_per_class"] = int(args.smoke_train_per_class)
    if args.smoke_eval_per_class is not None:
        cfg["train"]["smoke_eval_per_class"] = int(args.smoke_eval_per_class)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.patience is not None:
        cfg["early_stop_patience"] = int(args.patience)

    task = get_task_name(cfg, "sleep_edf_5class")
    cfg["task"] = task
    labels = get_labels(task)
    num_classes = int(get_num_classes(task))
    cfg["num_classes"] = int(num_classes)
    if num_classes != 5:
        raise RuntimeError("Sleep-EDF 训练要求 5 类标签，请检查映射/num_classes。")
    wake_label = int(get_wake_label(task))
    rem_label = labels.index("REM") if "REM" in labels else (num_classes - 1)

    allow_cpu = bool(_to_bool(cfg.get("allow_cpu", False)))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("检测不到 CUDA，且 allow_cpu != true，禁止在 CPU 上训练。")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"train: device={device} (GPU强制启用)")
    else:
        print(f"train: device={device} (allow_cpu=true)")
    print(
        f"train: config={config_path} preset={preset} model={preset_model} "
        f"recipe={recipe} baseline_run_dir={BASELINE_RUN_DIR}"
    )

    set_seed(int(cfg.get("seed", 42)))
    processed_dir = Path(cfg["processed_dir"])
    _assert_preprocess_cache_current(cfg, processed_dir=processed_dir, num_classes=num_classes)
    records = load_records(processed_dir)
    if not records:
        raise RuntimeError("未找到 Sleep-EDF 处理后数据，请先运行 preprocess_sleep_edf.py。")
    legacy_records = [rec for rec in records if bool(rec.get("is_legacy", False))]
    if legacy_records:
        sample = ", ".join(str(rec["record_id"]) for rec in legacy_records[:5])
        raise RuntimeError(
            "检测到 legacy 或不完整的 Sleep-EDF processed 缓存，"
            f"请先重新预处理。 sample_records={sample}"
        )
    entries = build_epoch_entries(records, num_classes=num_classes)
    subset = _resolve_subset(cfg, records)

    model_name = preset_model
    cfg["model"]["name"] = model_name
    if model_name not in {"picosleepnet_baseline", "picosleepnet_plus_snn", "context_pico_snn", "context_pico_snn_v2"}:
        raise RuntimeError(
            f"train_sleep_edf.py 仅支持 picosleepnet_baseline / picosleepnet_plus_snn / context_pico_snn / context_pico_snn_v2，收到 model={model_name}"
        )

    split_defs = _build_split_entries(entries, records, cfg, num_classes=num_classes, subset=subset)

    runs_dir = Path(cfg["runs_dir"])
    if args.run_dir:
        resolved_run_dir = _resolve_repo_path(args.run_dir)
        if resolved_run_dir is None:
            raise RuntimeError(f"invalid run_dir: {args.run_dir}")
        run_dir = ensure_dir(resolved_run_dir)
    else:
        if model_name == "context_pico_snn_v2":
            exp_name = "sleep_edf_context_pico_v2"
        elif model_name == "context_pico_snn":
            exp_name = "sleep_edf_context_pico"
        else:
            exp_name = "sleep_edf_real_strategy_change"
        run_dir = build_run_dir(runs_dir, exp_name)
    _ensure_not_baseline_run_dir(run_dir, "train run_dir")

    audit_summary_path = run_dir / "audit" / "audit_summary.json"
    if audit_summary_path.exists():
        audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
        blocking_errors = list(audit_summary.get("blocking_errors", []))
        if blocking_errors:
            raise RuntimeError(f"audit failed, refuse to train: {blocking_errors}")
    root_cause_path = run_dir / "audit_root_cause" / "root_cause_ranked.json"
    if root_cause_path.exists():
        root_cause = json.loads(root_cause_path.read_text(encoding="utf-8"))
        fatal_errors = list(root_cause.get("fatal_errors", []))
        if fatal_errors:
            raise RuntimeError(f"root cause audit failed, refuse to train: {fatal_errors}")

    change_proof_dir = ensure_dir(run_dir / "change_proof")
    file_hash_proof = _write_file_hash_proof(change_proof_dir)
    short_hash_summary = {
        key: value["after"][:12]
        for key, value in file_hash_proof.items()
        if value.get("after")
    }
    print(f"train: file_hash_summary={json.dumps(short_hash_summary, ensure_ascii=False, sort_keys=True)}")

    save_yaml(run_dir / "config.yaml", cfg)
    commit = try_git_commit_hash(Path.cwd())
    if commit:
        (run_dir / "git_commit.txt").write_text(commit, encoding="utf-8")
    write_last_run(runs_dir, run_dir)
    (run_dir / "splits.json").write_text(
        json.dumps([_split_to_jsonable(sp) for sp in split_defs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_gpu_env(run_dir, device=device)

    mixed_precision = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    num_workers = 0 if os.name == "nt" else int(cfg.get("num_workers", 2))
    pin_memory = True

    split_iter = list(enumerate(split_defs))
    if bool(_to_bool(cfg.get("train", {}).get("smoke", False))):
        split_iter = split_iter[: max(1, int(cfg.get("train", {}).get("smoke_splits", 1)))]
    selected_splits = _parse_split_indices(args.only_splits)
    if selected_splits is not None:
        split_iter = [(split_idx, split) for split_idx, split in split_iter if split_idx in selected_splits]
        if not split_iter:
            raise RuntimeError(f"--only_splits 没有匹配任何 split: {sorted(selected_splits)}")

    ensure_dir(run_dir / "train")
    for split_idx, split in split_iter:
        _train_one_split(
            split_idx=split_idx,
            split=split,
            run_dir=run_dir,
            cfg=cfg,
            model_name=model_name,
            num_classes=num_classes,
            labels=labels,
            wake_label=wake_label,
            rem_label=rem_label,
            subset=subset,
            device=device,
            mixed_precision=mixed_precision,
            num_workers=num_workers,
            pin_memory=pin_memory,
            fp32_only=bool(args.fp32_only),
        )

    print(f"train: completed run_dir={run_dir}")


if __name__ == "__main__":
    main()

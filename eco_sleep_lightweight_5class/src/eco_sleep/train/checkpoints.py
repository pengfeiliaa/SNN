"""Checkpoint save/load helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from eco_sleep.utils.io import ensure_dir

CHECKPOINT_METADATA_VERSION = 1
REQUIRED_CHECKPOINT_FIELDS = ("model_name", "model_hparams", "task", "num_classes", "split_id")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict[str, Any],
    epoch: int,
    best_metric: float,
    model_name: Optional[str] = None,
    model_hparams: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
    num_classes: Optional[int] = None,
    split_id: Optional[int] = None,
    trial_id: Optional[int | str] = None,
) -> None:
    """Save checkpoint together with model reconstruction metadata."""
    ensure_dir(path.parent)

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    hparams = dict(model_hparams or {})
    if hasattr(model, "get_hparams") and callable(getattr(model, "get_hparams")):
        hparams = dict(getattr(model, "get_hparams")())
        hparams.update(dict(model_hparams or {}))

    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_name": str(model_name or model_cfg.get("name", "")),
        "model_hparams": hparams,
        "task": str(task or config.get("task", "")),
        "num_classes": int(num_classes or config.get("num_classes", 0) or 0),
        "split_id": None if split_id is None else int(split_id),
        "trial_id": trial_id,
        "checkpoint_metadata_version": CHECKPOINT_METADATA_VERSION,
    }
    torch.save(ckpt, path)


def _load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="You are using `torch.load` with `weights_only=False`.*",
            )
            return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load checkpoint content only; caller rebuilds objects from metadata."""
    return _load_torch_checkpoint(path)


def validate_checkpoint_metadata(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    missing_fields = [field for field in REQUIRED_CHECKPOINT_FIELDS if field not in ckpt]
    if missing_fields:
        raise RuntimeError(f"checkpoint 缺少关键元信息，请重新训练。 missing={missing_fields}")

    model_name = str(ckpt.get("model_name", "")).strip()
    model_hparams = ckpt.get("model_hparams", {})
    task = str(ckpt.get("task", "")).strip()
    num_classes = int(ckpt.get("num_classes", 0) or 0)

    if not model_name:
        raise RuntimeError("checkpoint 缺少 model_name，请重新训练。")
    if not isinstance(model_hparams, dict) or not model_hparams:
        raise RuntimeError("checkpoint 缺少 model_hparams，请重新训练。")
    if not task:
        raise RuntimeError("checkpoint 缺少 task，请重新训练。")
    if num_classes <= 1:
        raise RuntimeError(f"checkpoint num_classes 非法: {num_classes}，请重新训练。")
    return ckpt


def restore_checkpoint_state(
    ckpt: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore state tensors into an already-constructed model/optimizer."""
    model.load_state_dict(ckpt["model_state"], strict=bool(strict))
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt


def load_checkpoint_raw(path: Path) -> Dict[str, Any]:
    """Backward-compatible alias for raw checkpoint loading."""
    return load_checkpoint(path)

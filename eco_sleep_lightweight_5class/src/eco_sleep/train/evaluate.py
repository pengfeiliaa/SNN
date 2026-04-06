"""Inference helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F


def _reset_state_if_needed(model: torch.nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "reset_state") and callable(getattr(m, "reset_state")):
            m.reset_state()


def _amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    mixed_precision: bool = False,
    non_blocking: bool = False,
    return_logits: bool = False,
) -> Dict[str, np.ndarray]:
    """Batch inference with y_true/y_pred/y_prob and metadata."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    record_ids: List[str] = []
    epoch_indices: List[int] = []
    sleep_prob: List[float] = []
    rem_prob: List[float] = []
    raw_logits: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device, non_blocking=non_blocking)
            y = batch[1].to(device, non_blocking=non_blocking)
            metas = batch[2]
            epoch_idx = batch[3]

            _reset_state_if_needed(model)
            if mixed_precision and device.type == "cuda":
                with _amp_autocast(device, enabled=bool(mixed_precision)):
                    outputs = model(x)
            else:
                outputs = model(x)

            if isinstance(outputs, dict):
                logits = outputs["main"]
                sleep_logits = outputs.get("sleep_wake")
                rem_logits = outputs.get("rem")
            else:
                logits = outputs
                sleep_logits = None
                rem_logits = None

            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_prob.extend(prob.cpu().numpy())
            record_ids.extend(list(metas))
            epoch_indices.extend(epoch_idx.cpu().numpy().tolist())
            if sleep_logits is not None:
                sleep_prob.extend(torch.sigmoid(sleep_logits).cpu().numpy().reshape(-1).tolist())
            if rem_logits is not None:
                rem_prob.extend(torch.sigmoid(rem_logits).cpu().numpy().reshape(-1).tolist())
            if return_logits:
                raw_logits.extend(logits.detach().cpu().numpy())

    result = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
        "record_ids": np.asarray(record_ids),
        "epoch_indices": np.asarray(epoch_indices),
    }
    if sleep_prob:
        result["sleep_prob"] = np.asarray(sleep_prob, dtype=np.float32)
    if rem_prob:
        result["rem_prob"] = np.asarray(rem_prob, dtype=np.float32)
    if return_logits:
        result["logits"] = np.asarray(raw_logits, dtype=np.float32)
    return result

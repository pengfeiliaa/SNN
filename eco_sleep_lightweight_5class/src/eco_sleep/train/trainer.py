# -*- coding: utf-8 -*-
"""Training loop helpers and collapse protection."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eco_sleep.utils.meters import AverageMeter


def _reset_state_if_needed(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "reset_state") and callable(getattr(module, "reset_state")):
            module.reset_state()


def _main_logits(outputs) -> torch.Tensor:
    return outputs["main"] if isinstance(outputs, dict) else outputs


def _clip_logits(outputs, logit_clip: float | None):
    if logit_clip is None or logit_clip <= 0:
        return outputs
    clip_value = float(logit_clip)
    if isinstance(outputs, dict):
        out = dict(outputs)
        out["main"] = torch.clamp(out["main"], min=-clip_value, max=clip_value)
        return out
    return torch.clamp(outputs, min=-clip_value, max=clip_value)


def _entropy_regularization(logits: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    prob_mean = prob.mean(dim=0)
    return -(prob_mean * (prob_mean + 1e-12).log()).sum()


def _extract_firing_rate(outputs, model: nn.Module) -> torch.Tensor | None:
    if isinstance(outputs, dict):
        firing_rate = outputs.get("firing_rate")
        if isinstance(firing_rate, torch.Tensor):
            return firing_rate
    cached_rate = getattr(model, "last_firing_rate", None)
    if isinstance(cached_rate, torch.Tensor):
        return cached_rate
    return None


def extract_firing_rate(outputs, model: nn.Module) -> torch.Tensor | None:
    """Public wrapper used by eval/complexity utilities."""

    return _extract_firing_rate(outputs, model)


def extract_layer_firing_rates(outputs, model: nn.Module) -> dict[str, float]:
    """Return per-layer firing rates if the model exposes them."""

    if isinstance(outputs, dict):
        layer_rates = outputs.get("layer_firing_rates")
        if isinstance(layer_rates, dict):
            out = {}
            for key, value in layer_rates.items():
                if isinstance(value, torch.Tensor):
                    out[str(key)] = float(value.detach().mean().cpu().item())
                else:
                    out[str(key)] = float(value)
            return out
    cached = getattr(model, "last_layer_firing_rates", None)
    if isinstance(cached, dict):
        return {str(key): float(value) for key, value in cached.items()}
    return {}


def summarize_parameter_tensor(tensor: torch.Tensor) -> dict[str, float | int]:
    data = tensor.detach().float().reshape(-1)
    if data.numel() == 0:
        return {"numel": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "norm": 0.0}
    return {
        "numel": int(data.numel()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "norm": float(torch.linalg.vector_norm(data).item()),
    }


def named_parameter_summaries(model: nn.Module, parameter_names: list[str]) -> dict[str, dict[str, float | int]]:
    named = dict(model.named_parameters())
    return {
        name: summarize_parameter_tensor(named[name])
        for name in parameter_names
        if name in named
    }


def named_gradient_summaries(model: nn.Module, parameter_names: list[str]) -> dict[str, dict[str, float | int | bool]]:
    named = dict(model.named_parameters())
    out: dict[str, dict[str, float | int | bool]] = {}
    for name in parameter_names:
        if name not in named:
            continue
        grad = named[name].grad
        if grad is None:
            out[name] = {"has_grad": False, "numel": int(named[name].numel()), "mean": 0.0, "std": 0.0, "norm": 0.0}
            continue
        summary = summarize_parameter_tensor(grad)
        summary["has_grad"] = True
        out[name] = summary
    return out


def _firing_regularization(firing_rate: torch.Tensor, target_low: float, target_high: float) -> torch.Tensor:
    low = torch.as_tensor(float(target_low), dtype=firing_rate.dtype, device=firing_rate.device)
    high = torch.as_tensor(float(target_high), dtype=firing_rate.dtype, device=firing_rate.device)
    return torch.relu(low - firing_rate) + torch.relu(firing_rate - high)


def _amp_autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda" and hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def loss_is_effectively_zero(value: float, tol: float = 1e-8) -> bool:
    return abs(float(value)) <= float(max(tol, 0.0))


def has_consecutive_effective_zeros(values: list[float], streak: int = 2, tol: float = 1e-8) -> bool:
    if int(streak) <= 0:
        return False
    run = 0
    for value in values:
        run = run + 1 if loss_is_effectively_zero(value, tol=tol) else 0
        if run >= int(streak):
            return True
    return False


class ModelEMA:
    """Lightweight EMA shadow for safe evaluation/checkpointing."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(min(max(decay, 0.0), 0.99999))
        self.num_updates = 0
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current_decay = self.decay if self.num_updates > 0 else 0.0
        current_state = model.state_dict()
        for key, value in current_state.items():
            if not torch.is_tensor(value):
                continue
            shadow_value = self.shadow[key]
            source = value.detach()
            if not torch.is_floating_point(source):
                shadow_value.copy_(source)
                continue
            shadow_value.mul_(current_decay).add_(source, alpha=(1.0 - current_decay))
        self.num_updates += 1

    def backup_model_state(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = self.backup_model_state(model)
        model.load_state_dict(self.shadow, strict=True)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}


class CollapseProtector:
    """Validation prediction collapse detector."""

    def __init__(
        self,
        trigger_ratio: float = 0.85,
        patience_epochs: int = 2,
        min_zero_classes: int = 2,
    ) -> None:
        self.trigger_ratio = float(trigger_ratio)
        self.patience_epochs = int(max(1, patience_epochs))
        self.min_zero_classes = int(max(1, min_zero_classes))
        self._streak = 0

    def update(self, pred_ratio: list[float], pred_counts: np.ndarray) -> bool:
        max_ratio = float(max(pred_ratio)) if pred_ratio else 1.0
        zero_classes = int(np.sum(np.asarray(pred_counts) <= 0))
        collapse_now = max_ratio > self.trigger_ratio and zero_classes >= self.min_zero_classes
        self._streak = self._streak + 1 if collapse_now else 0
        if self._streak >= self.patience_epochs:
            self._streak = 0
            return True
        return False


def _set_tau_if_supported(loss_fn: nn.Module, tau: float) -> bool:
    base = getattr(loss_fn, "main_loss", loss_fn)
    if hasattr(base, "set_tau") and callable(getattr(base, "set_tau")):
        base.set_tau(float(tau))
        return True
    if hasattr(base, "tau"):
        setattr(base, "tau", float(tau))
        return True
    return False


def apply_collapse_stabilization(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    best_state_dict: dict | None,
    lr_decay: float = 0.5,
    tau_after_trigger: float = 0.0,
    switch_to_stable_loss: Optional[Callable[[], bool]] = None,
    strengthen_sampler: Optional[Callable[[], bool]] = None,
    disable_prior_correction: Optional[Callable[[], bool]] = None,
) -> dict:
    """Rollback best state, decay LR, and disable unstable options."""
    rolled_back = False
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        rolled_back = True

    decay = float(max(lr_decay, 1e-3))
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * decay, 1e-7)
    new_lr = float(optimizer.param_groups[0]["lr"])

    tau_changed = _set_tau_if_supported(loss_fn, float(tau_after_trigger))
    loss_switched = bool(switch_to_stable_loss()) if switch_to_stable_loss is not None else False
    sampler_strengthened = bool(strengthen_sampler()) if strengthen_sampler is not None else False
    prior_correction_disabled = bool(disable_prior_correction()) if disable_prior_correction is not None else False
    return {
        "rolled_back": rolled_back,
        "lr": new_lr,
        "tau_changed": tau_changed,
        "tau_after_trigger": float(tau_after_trigger),
        "loss_switched": loss_switched,
        "sampler_strengthened": sampler_strengthened,
        "prior_correction_disabled": prior_correction_disabled,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    mixed_precision: bool = False,
    max_grad_norm: float | None = None,
    entropy_reg_weight: float = 0.0,
    entropy_reg_epochs: int | None = None,
    epoch: int | None = None,
    label_smoothing: float = 0.0,
    label_smoothing_weight: float = 0.2,
    logit_clip: float | None = None,
    non_blocking: bool = False,
    firing_reg_weight: float = 0.0,
    firing_target_low: float = 0.05,
    firing_target_high: float = 0.2,
    return_stats: bool = False,
):
    """Train one epoch."""
    model.train()
    loss_meter = AverageMeter()
    firing_meter = AverageMeter()
    firing_penalty_meter = AverageMeter()
    label_smoothing = float(max(label_smoothing, 0.0))
    label_smoothing_weight = float(min(max(label_smoothing_weight, 0.0), 1.0))
    firing_reg_weight = float(max(firing_reg_weight, 0.0))

    def compute_loss(outputs, targets) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        outputs = _clip_logits(outputs, logit_clip=logit_clip)
        loss = loss_fn(outputs, targets)
        logits = _main_logits(outputs)

        if label_smoothing > 0 and label_smoothing_weight > 0:
            smoothed = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
            loss = (1.0 - label_smoothing_weight) * loss + label_smoothing_weight * smoothed

        if entropy_reg_weight > 0:
            if entropy_reg_epochs is None or (epoch is not None and epoch <= entropy_reg_epochs):
                loss = loss - float(entropy_reg_weight) * _entropy_regularization(logits)

        firing_rate = _extract_firing_rate(outputs, model)
        firing_penalty = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if firing_rate is not None and firing_reg_weight > 0:
            firing_penalty = _firing_regularization(firing_rate, firing_target_low, firing_target_high)
            loss = loss + firing_reg_weight * firing_penalty
        return loss, firing_rate, firing_penalty

    for batch in dataloader:
        x = batch[0].to(device, non_blocking=non_blocking)
        y = batch[1].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        _reset_state_if_needed(model)

        if mixed_precision and device.type == "cuda":
            with _amp_autocast(device, enabled=bool(mixed_precision)):
                outputs = model(x)
                loss, firing_rate, firing_penalty = compute_loss(outputs, y)
            if scaler is None:
                raise RuntimeError("mixed_precision=True 但未提供 GradScaler。")
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss, firing_rate, firing_penalty = compute_loss(outputs, y)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_meter.update(loss.item(), n=y.size(0))
        if firing_rate is not None:
            firing_meter.update(float(firing_rate.detach().cpu()), n=y.size(0))
            firing_penalty_meter.update(float(firing_penalty.detach().cpu()), n=y.size(0))

    if return_stats:
        return {
            "loss": loss_meter.avg,
            "firing_rate": firing_meter.avg if firing_meter.count > 0 else float("nan"),
            "firing_penalty": firing_penalty_meter.avg if firing_penalty_meter.count > 0 else float("nan"),
        }
    return loss_meter.avg

# -*- coding: utf-8 -*-
"""Loss functions and class-distribution utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _validate_num_classes(num_classes: int) -> int:
    n = int(num_classes)
    if n <= 1:
        raise ValueError(f"num_classes 非法: {num_classes}")
    return n


def _check_logits(logits: torch.Tensor, num_classes: int, name: str) -> None:
    if logits.ndim != 2:
        raise RuntimeError(
            f"{name} 需要二维 logits=[B,C]，收到 shape={tuple(logits.shape)}。"
            "请检查模型输出维度。"
        )
    if int(logits.size(1)) != int(num_classes):
        raise RuntimeError(
            f"{name} logits 类别维不匹配: logits.shape[1]={int(logits.size(1))}, "
            f"num_classes={int(num_classes)}。"
            "请检查 split 计数、标签映射和 num_classes 配置。"
        )


def _check_vec_length(vec: torch.Tensor, num_classes: int, field_name: str) -> torch.Tensor:
    t = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
    if int(t.numel()) != int(num_classes):
        extra = ""
        if int(num_classes) == 5:
            extra = " (Sleep-EDF 5 类任务要求长度必须为 5)"
        raise ValueError(
            f"{field_name} 长度不匹配: got={int(t.numel())}, expect={int(num_classes)}{extra}。"
            "请检查 split 计数/标签映射/num_classes。"
        )
    return t


def regularization_defaults(cfg: Optional[dict] = None) -> dict:
    """Expose lightweight regularization defaults for trainer."""

    cfg = cfg or {}
    firing_default = float(cfg.get("firing_rate_reg_weight", cfg.get("firing_reg_weight", 1e-3)))
    out = {
        "entropy_reg_weight": float(cfg.get("entropy_reg_weight", 1e-3)),
        "firing_rate_reg": bool(cfg.get("firing_rate_reg", True)),
        "firing_target_low": float(cfg.get("firing_target_low", 0.05)),
        "firing_target_high": float(cfg.get("firing_target_high", 0.2)),
        "firing_rate_reg_weight": firing_default,
        # Backward-compatible alias.
        "firing_reg_weight": firing_default,
    }
    return out


class CrossEntropyLoss(nn.Module):
    """Cross entropy with strict class-dimension check."""

    def __init__(self, num_classes: int, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "CrossEntropyLoss")
        return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Standard focal loss."""

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "FocalLoss")
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class LogitAdjustedFocalLoss(nn.Module):
    """Logit-adjusted focal loss.

    Note:
    - This can be unstable on long-tail data.
    - Keep tau=0 by default unless explicitly tuned.
    """

    def __init__(
        self,
        class_prior: torch.Tensor,
        num_classes: int,
        tau: float = 0.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        self.gamma = float(gamma)
        self.eps = float(max(eps, 1e-12))

        prior = _check_vec_length(class_prior, self.num_classes, "class_prior")
        prior = torch.clamp(prior, min=self.eps)
        prior = prior / torch.clamp(prior.sum(), min=self.eps)
        self.register_buffer("log_prior", torch.log(prior))
        self.register_buffer("weight", weight if weight is not None else None)
        self.tau = float(tau)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "LogitAdjustedFocalLoss")
        adjusted = logits - float(self.tau) * self.log_prior
        logpt = F.log_softmax(adjusted, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """Logit-adjusted cross entropy for long-tail Sleep-EDF training."""

    def __init__(
        self,
        class_prior: torch.Tensor,
        num_classes: int,
        tau: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        self.eps = float(max(eps, 1e-12))

        prior = _check_vec_length(class_prior, self.num_classes, "class_prior")
        prior = torch.clamp(prior, min=self.eps)
        prior = prior / torch.clamp(prior.sum(), min=self.eps)
        self.register_buffer("log_prior", torch.log(prior))
        self.register_buffer("weight", weight if weight is not None else None)
        self.tau = float(tau)

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "LogitAdjustedCrossEntropyLoss")
        adjusted = logits - float(self.tau) * self.log_prior
        return F.cross_entropy(adjusted, targets, weight=self.weight)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax cross entropy."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        counts = _check_vec_length(class_counts, self.num_classes, "class_counts")
        if torch.any(counts <= 0):
            raise ValueError(
                "class_counts 存在 <=0。说明训练集划分不可行，"
                "请先修复 split 计数后再训练。"
            )
        self.register_buffer("log_counts", torch.log(counts))
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "BalancedSoftmaxLoss")
        adjusted = logits + self.log_counts
        return F.cross_entropy(adjusted, targets, weight=self.weight)


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss (effective-number weighting)."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        num_classes: int,
        beta: float = 0.999,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        self.gamma = float(gamma)
        counts = _check_vec_length(class_counts, self.num_classes, "class_counts")
        if torch.any(counts <= 0):
            raise ValueError(
                "class_counts 存在 <=0。说明训练集划分不可行，"
                "请先修复 split 计数后再训练。"
            )
        b = float(beta)
        effective_num = 1.0 - torch.pow(b, counts)
        weights = (1.0 - b) / torch.clamp(effective_num, min=1e-12)
        weights = weights / torch.mean(weights)
        self.register_buffer("alpha", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "ClassBalancedFocalLoss")
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = self.alpha[targets]
        return (-alpha * ((1.0 - pt) ** self.gamma) * logpt).mean()


class LDAMLoss(nn.Module):
    """Conservative LDAM implementation for macro-F1 oriented ablation."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        num_classes: int,
        max_margin: float = 0.35,
        scale: float = 20.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = _validate_num_classes(num_classes)
        counts = _check_vec_length(class_counts, self.num_classes, "class_counts")
        if torch.any(counts <= 0):
            raise ValueError(
                "class_counts 存在 <=0。说明训练集划分不可行，"
                "请先修复 split 计数后再训练。"
            )
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (float(max_margin) / torch.clamp(torch.max(margins), min=1e-12))
        self.register_buffer("margins", margins)
        self.register_buffer("weight", weight if weight is not None else None)
        self.scale = float(max(scale, 1.0))

    def set_weight(self, weight: Optional[torch.Tensor]) -> None:
        self.weight = None if weight is None else torch.as_tensor(weight, dtype=torch.float32, device=self.margins.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_logits(logits, self.num_classes, "LDAMLoss")
        margin_tensor = torch.zeros_like(logits)
        margin_values = self.margins.to(device=logits.device, dtype=logits.dtype)[targets].unsqueeze(1)
        margin_tensor.scatter_(1, targets.unsqueeze(1), margin_values)
        adjusted = self.scale * (logits - margin_tensor)
        return F.cross_entropy(adjusted, targets, weight=self.weight)


class MultiTaskLoss(nn.Module):
    """Main task + sleep/wake + REM auxiliary heads."""

    def __init__(
        self,
        main_loss: nn.Module,
        lambda_sleep: float = 0.3,
        lambda_rem: float = 0.2,
        wake_label: int = 0,
        rem_label: int = 4,
    ) -> None:
        super().__init__()
        self.main_loss = main_loss
        self.lambda_sleep = float(lambda_sleep)
        self.lambda_rem = float(lambda_rem)
        self.wake_label = int(wake_label)
        self.rem_label = int(rem_label)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        if not isinstance(outputs, dict):
            return self.main_loss(outputs, targets)

        logits = outputs["main"]
        loss_main = self.main_loss(logits, targets)
        sleep_targets = (targets != self.wake_label).float()
        rem_targets = (targets == self.rem_label).float()
        sleep_logits = outputs["sleep_wake"].squeeze(1)
        rem_logits = outputs["rem"].squeeze(1)
        loss_sleep = self.bce(sleep_logits, sleep_targets)
        loss_rem = self.bce(rem_logits, rem_targets)
        return loss_main + self.lambda_sleep * loss_sleep + self.lambda_rem * loss_rem


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 5,
    strategy: str = "effective_num",
    beta: float = 0.999,
) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    n = _validate_num_classes(num_classes)
    labels = labels[(labels >= 0) & (labels < n)]
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if strategy == "inverse_freq":
        weights = 1.0 / counts
    elif strategy == "effective_num":
        effective_num = 1.0 - np.power(float(beta), counts)
        weights = (1.0 - float(beta)) / np.maximum(effective_num, 1e-12)
    else:
        weights = np.ones_like(counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_prior(labels: np.ndarray, num_classes: int = 5, eps: float = 1e-8) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    n = _validate_num_classes(num_classes)
    labels = labels[(labels >= 0) & (labels < n)]
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    eps = float(max(eps, 1e-12))
    prior = (counts + eps) / (np.sum(counts) + eps * n)
    return torch.tensor(prior, dtype=torch.float32)


def summarize_loss_setup(
    loss_name: str,
    class_counts: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    class_prior: Optional[torch.Tensor] = None,
    tau: float = 0.0,
) -> dict:
    return {
        "loss_name": str(loss_name),
        "class_counts": None if class_counts is None else [float(v) for v in torch.as_tensor(class_counts).reshape(-1).tolist()],
        "class_weights": None if class_weights is None else [float(v) for v in torch.as_tensor(class_weights).reshape(-1).tolist()],
        "class_prior": None if class_prior is None else [float(v) for v in torch.as_tensor(class_prior).reshape(-1).tolist()],
        "tau": float(tau),
    }


def soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    num_classes = _validate_num_classes(int(logits.size(1)))
    _check_logits(logits, num_classes, "soft_target_cross_entropy")
    if soft_targets.ndim != 2 or tuple(soft_targets.shape) != tuple(logits.shape):
        raise RuntimeError(
            "soft_target_cross_entropy 需要 soft_targets.shape == logits.shape == [B,C]。"
        )
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(soft_targets.to(dtype=log_prob.dtype) * log_prob).sum(dim=1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def temporal_consistency_kl_loss(
    step_logits: torch.Tensor,
    aggregate_logits: torch.Tensor,
    temperature: float = 1.0,
    detach_target: bool = True,
) -> torch.Tensor:
    if step_logits.ndim != 3:
        raise RuntimeError(
            f"temporal_consistency_kl_loss 需要 step_logits=[B,T,C]，收到 shape={tuple(step_logits.shape)}。"
        )
    if aggregate_logits.ndim != 2:
        raise RuntimeError(
            f"temporal_consistency_kl_loss 需要 aggregate_logits=[B,C]，收到 shape={tuple(aggregate_logits.shape)}。"
        )
    if int(step_logits.size(0)) != int(aggregate_logits.size(0)) or int(step_logits.size(-1)) != int(aggregate_logits.size(-1)):
        raise RuntimeError(
            "temporal_consistency_kl_loss 需要 step_logits 与 aggregate_logits 的 batch/classes 维匹配。"
        )
    temp = float(max(temperature, 1e-4))
    step_log_prob = F.log_softmax(step_logits / temp, dim=-1)
    target_prob = F.softmax(aggregate_logits / temp, dim=-1)
    if detach_target:
        target_prob = target_prob.detach()
    target_prob = target_prob.unsqueeze(1).expand_as(step_logits)
    loss = F.kl_div(step_log_prob, target_prob, reduction="none").sum(dim=-1)
    return loss.mean() * (temp * temp)


def logits_consistency_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    detach_target: bool = True,
) -> torch.Tensor:
    if student_logits.ndim != 2:
        raise RuntimeError(
            f"logits_consistency_kl_loss 闇€瑕?student_logits=[B,C]锛屾敹鍒?shape={tuple(student_logits.shape)}銆?"
        )
    if teacher_logits.ndim != 2:
        raise RuntimeError(
            f"logits_consistency_kl_loss 闇€瑕?teacher_logits=[B,C]锛屾敹鍒?shape={tuple(teacher_logits.shape)}銆?"
        )
    if tuple(student_logits.shape) != tuple(teacher_logits.shape):
        raise RuntimeError(
            "logits_consistency_kl_loss 闇€瑕?student_logits 涓?teacher_logits 褰㈢姸瀹屽叏涓€鑷淬€?"
        )
    temp = float(max(temperature, 1e-4))
    student_log_prob = F.log_softmax(student_logits / temp, dim=-1)
    target_prob = F.softmax(teacher_logits / temp, dim=-1)
    if detach_target:
        target_prob = target_prob.detach()
    loss = F.kl_div(student_log_prob, target_prob, reduction="none").sum(dim=-1)
    return loss.mean() * (temp * temp)


def build_loss(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    class_prior: Optional[torch.Tensor] = None,
    tau: float = 0.0,
    class_counts: Optional[torch.Tensor] = None,
    beta: float = 0.999,
    max_margin: float = 0.35,
    scale: float = 20.0,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """Build loss module from config."""

    if num_classes is None:
        raise ValueError("build_loss 缺少 num_classes。")
    n = _validate_num_classes(num_classes)
    name = str(loss_type).lower().strip() or "ce"

    if name in ("balanced_softmax", "balancedsoftmax"):
        if class_counts is None:
            raise ValueError("balanced_softmax 需要 class_counts。")
        return BalancedSoftmaxLoss(class_counts=class_counts, num_classes=n, weight=class_weights)
    if name in ("cb_focal", "class_balanced_focal", "classbalanced_focal"):
        if class_counts is None:
            raise ValueError("cb_focal 需要 class_counts。")
        return ClassBalancedFocalLoss(class_counts=class_counts, num_classes=n, beta=beta, gamma=gamma)
    if name in ("ldam", "ldam_drw"):
        if class_counts is None:
            raise ValueError("ldam 需要 class_counts。")
        return LDAMLoss(
            class_counts=class_counts,
            num_classes=n,
            max_margin=float(max_margin),
            scale=float(scale),
            weight=class_weights,
        )
    if name in ("logit_focal", "logit_adjusted_focal"):
        if class_prior is None:
            raise ValueError("logit_focal 需要 class_prior。")
        return LogitAdjustedFocalLoss(
            class_prior=class_prior,
            num_classes=n,
            tau=float(tau),  # keep default 0.0 for safety
            gamma=gamma,
            weight=class_weights,
        )
    if name in ("logit_adjust_ce", "logit_adjusted_ce", "logit_adjustment_ce", "logit_ce"):
        if class_prior is None:
            raise ValueError("logit_adjusted_ce requires class_prior.")
        return LogitAdjustedCrossEntropyLoss(
            class_prior=class_prior,
            num_classes=n,
            tau=float(tau),
            weight=class_weights,
        )
    if name == "focal":
        return FocalLoss(num_classes=n, gamma=gamma, weight=class_weights)
    if name in ("ce", "cross_entropy"):
        return CrossEntropyLoss(num_classes=n, weight=class_weights)
    # Safer fallback.
    if class_counts is None:
        return CrossEntropyLoss(num_classes=n, weight=class_weights)
    return CrossEntropyLoss(num_classes=n, weight=class_weights)

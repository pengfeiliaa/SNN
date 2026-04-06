"""学习率调度器工具。"""

from __future__ import annotations

import math
from typing import Optional

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-5,
    schedule: str = "cosine",
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if schedule.lower() != "cosine":
        return None

    base_lr = optimizer.param_groups[0]["lr"]
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

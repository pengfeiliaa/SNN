"""分类头模块。"""

from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """简单线性分类头。"""

    def __init__(self, in_dim: int, num_classes: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class MultiTaskHeads(nn.Module):
    """主任务 + 辅助任务头：Sleep/Wake 与 REM。"""

    def __init__(self, in_dim: int, num_classes: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.main = nn.Linear(in_dim, num_classes)
        self.sleep_wake = nn.Linear(in_dim, 1)
        self.rem = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.dropout(x)
        return {
            "main": self.main(x),
            "sleep_wake": self.sleep_wake(x),
            "rem": self.rem(x),
        }

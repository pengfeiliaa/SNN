"""轻量上下文 TCN 聚合器。"""

from __future__ import annotations

import torch
from torch import nn

from .heads import LinearHead, MultiTaskHeads
from .tiny_blocks import DepthwiseSeparableConv1d


class _TinyTCN(nn.Module):
    """若干层深度可分离 TCN。"""

    def __init__(self, channels: int, layers: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        dilations = [2**i for i in range(layers)]
        for d in dilations:
            blocks.append(
                DepthwiseSeparableConv1d(
                    channels,
                    channels,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        return x


class ContextTinyTCN(nn.Module):
    """上下文聚合器（输入为 embedding 序列）。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 5,
        tcn_layers: int = 4,
        dropout: float = 0.2,
        use_multitask: bool = False,
    ) -> None:
        super().__init__()
        self.tcn = _TinyTCN(input_dim, tcn_layers, dropout)
        self.use_multitask = use_multitask
        self.head = (
            MultiTaskHeads(input_dim, num_classes=num_classes, dropout=dropout)
            if use_multitask
            else LinearHead(input_dim, num_classes=num_classes, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]
        x = self.tcn(x)
        center_idx = x.shape[-1] // 2
        center_feat = x[:, :, center_idx]
        return self.head(center_feat)

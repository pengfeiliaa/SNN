"""EEG 轻量编码器。"""

from __future__ import annotations

import torch
from torch import nn

from .tiny_blocks import DepthwiseSeparableConv1d
from .tiny_tcn import ContextTinyTCN


class MultiScaleDepthwiseBlock(nn.Module):
    """多尺度深度可分离卷积块：覆盖不同形态与频段。"""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 16,
        kernel_sizes: tuple[int, int, int] = (15, 31, 63),
        dropout: float = 0.2,
        se_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=k,
                        padding=padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )

        fuse_channels = branch_channels * len(kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv1d(fuse_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )

        hidden = max(4, in_channels // se_ratio)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        scale = self.se_fc(self.se_pool(x).squeeze(-1)).unsqueeze(-1)
        x = x * scale
        x = self.dropout(x)
        return x + residual


class EEGTinyEncoder(nn.Module):
    """将单个 epoch 的 EEG 信号编码为低维 embedding。"""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.ms_block = MultiScaleDepthwiseBlock(16, branch_channels=16, dropout=dropout)
        self.block1 = DepthwiseSeparableConv1d(16, 32, kernel_size=7, stride=2, dropout=dropout)
        self.block2 = DepthwiseSeparableConv1d(32, 48, kernel_size=7, stride=2, dropout=dropout)
        self.block3 = DepthwiseSeparableConv1d(48, embed_dim, kernel_size=5, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 3000]
        x = self.stem(x)
        x = self.ms_block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)  # [B, embed_dim]
        return x


class EEGContextModel(nn.Module):
    """EEG 编码 + 上下文 TCN 聚合的五类模型。"""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_classes: int = 5,
        tcn_layers: int = 4,
        dropout: float = 0.2,
        embedding_norm: str = "layernorm",
        use_multitask: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EEGTinyEncoder(
            in_channels=in_channels, embed_dim=embed_dim, dropout=dropout
        )
        self.embedding_norm = embedding_norm
        self.embed_norm = nn.LayerNorm(embed_dim) if embedding_norm == "layernorm" else None
        if embedding_norm == "feature_std":
            self.register_buffer("embed_mean", torch.zeros(embed_dim))
            self.register_buffer("embed_std", torch.ones(embed_dim))
        else:
            self.embed_mean = None
            self.embed_std = None

        self.context = ContextTinyTCN(
            input_dim=embed_dim,
            num_classes=num_classes,
            tcn_layers=tcn_layers,
            dropout=dropout,
            use_multitask=use_multitask,
        )

    def set_embedding_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if self.embedding_norm == "feature_std":
            self.embed_mean.copy_(mean.detach())
            self.embed_std.copy_(std.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C, 3000]
        b, l, c, t = x.shape
        x = x.reshape(b * l, c, t)
        emb = self.encoder(x)
        # 跨被试幅值差异大：在 embedding 级做归一化更稳
        if self.embedding_norm == "layernorm" and self.embed_norm is not None:
            emb = self.embed_norm(emb)
        elif self.embedding_norm == "feature_std" and self.embed_mean is not None:
            emb = (emb - self.embed_mean) / (self.embed_std + 1e-6)
        emb = emb.view(b, l, -1)
        return self.context(emb)

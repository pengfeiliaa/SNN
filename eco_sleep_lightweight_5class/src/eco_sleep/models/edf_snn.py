"""Lightweight fallback SNN for Sleep-EDF."""

from __future__ import annotations

import torch
from torch import nn

from .snn_layers import LIFCell, StatefulModule
from .tiny_cnn1d import EEGTinyEncoder


class EdfSNN(StatefulModule):
    """Encode each epoch, then aggregate over context using LIF cells."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_classes: int = 5,
        lif_layers: int = 2,
        readout: str = "mem",
        dropout: float = 0.1,
        use_multitask: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.lif_layers = int(max(1, lif_layers))
        self.readout = str(readout).lower()
        self.use_multitask = bool(use_multitask)

        self.encoder = EEGTinyEncoder(in_channels=self.in_channels, embed_dim=self.embed_dim, dropout=dropout)
        self.spike_layers = nn.ModuleList([LIFCell(self.embed_dim) for _ in range(self.lif_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

        if self.use_multitask:
            self.sleep_head = nn.Linear(self.embed_dim, 1)
            self.rem_head = nn.Linear(self.embed_dim, 1)
        else:
            self.sleep_head = None
            self.rem_head = None

        self.last_firing_rate: torch.Tensor | None = None

    def reset_state(self) -> None:
        for cell in self.spike_layers:
            cell.reset_state()
        self.last_firing_rate = None

    def forward(self, x: torch.Tensor):
        # x: [B, L, C, 3000]
        b, l, c, t = x.shape
        x = x.reshape(b * l, c, t)
        emb = self.encoder(x)  # [B*L, E]
        emb = emb.view(b, l, -1)  # [B, L, E]

        spike_sum = torch.zeros((b, self.embed_dim), device=x.device, dtype=emb.dtype)
        mem_sum = torch.zeros((b, self.embed_dim), device=x.device, dtype=emb.dtype)
        firing_acc = torch.zeros((), device=x.device, dtype=emb.dtype)
        firing_steps = 0

        for step in range(l):
            z = emb[:, step, :]
            for cell in self.spike_layers:
                z = cell(z)
                firing_acc = firing_acc + z.mean()
                firing_steps += 1
            spike_sum = spike_sum + z
            mem_sum = mem_sum + self.spike_layers[-1].v

        self.last_firing_rate = firing_acc / max(1, firing_steps)
        pooled = mem_sum / max(1, l) if self.readout == "mem" else spike_sum / max(1, l)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)

        if self.use_multitask:
            return {
                "main": logits,
                "sleep_wake": self.sleep_head(pooled),
                "rem": self.rem_head(pooled),
            }
        return logits

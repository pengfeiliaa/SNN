# -*- coding: utf-8 -*-
"""Lightweight short-context SNN backbone for Sleep-EDF-20 5-class staging."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .picosleepnet_baseline import surrogate_spike
from .tiny_blocks import DepthwiseSeparableConv1d


class _MultiScaleStem(nn.Module):
    """Very-light raw EEG front-end before spiking temporal modeling."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        stem_channels: int,
        kernel_sizes: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=int(kernel_size),
                        stride=4,
                        padding=int(kernel_size) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.SiLU(inplace=True),
                )
                for kernel_size in kernel_sizes
            ]
        )
        fused_channels = int(branch_channels * len(kernel_sizes))
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_channels, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout) * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


class ContextPicoSNN(nn.Module):
    """Multi-scale raw stem + recurrent spiking epoch encoder + bidirectional context SNN."""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        context_len: int = 3,
        branch_channels: int = 4,
        stem_channels: int = 12,
        token_dim: int = 24,
        epoch_hidden_size: int = 96,
        epoch_embed_dim: int = 48,
        context_hidden_size: int = 32,
        t_steps: int = 40,
        tau_epoch: float = 0.92,
        tau_context: float = 0.90,
        v_th_epoch: float = 1.0,
        v_th_context: float = 1.0,
        surrogate_alpha: float = 4.0,
        dropout: float = 0.10,
        use_aux_heads: bool = True,
        center_residual_weight: float = 0.35,
        kernel_sizes: Tuple[int, int, int] = (31, 63, 125),
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.context_len = int(max(1, context_len))
        self.branch_channels = int(branch_channels)
        self.stem_channels = int(stem_channels)
        self.token_dim = int(token_dim)
        self.epoch_hidden_size = int(epoch_hidden_size)
        self.epoch_embed_dim = int(epoch_embed_dim)
        self.context_hidden_size = int(context_hidden_size)
        self.t_steps = int(max(8, t_steps))
        self.tau_epoch = float(tau_epoch)
        self.tau_context = float(tau_context)
        self.v_th_epoch = float(v_th_epoch)
        self.v_th_context = float(v_th_context)
        self.surrogate_alpha = float(surrogate_alpha)
        self.dropout_p = float(dropout)
        self.use_aux_heads = bool(use_aux_heads)
        self.center_residual_weight = float(center_residual_weight)
        self.kernel_sizes = tuple(int(v) for v in kernel_sizes)

        self.sequence_context_enabled = True
        self.snn_core_layers = [
            "epoch_input",
            "epoch_recurrent",
            "epoch_proj",
            "context_fwd_input",
            "context_fwd_recurrent",
            "context_bwd_input",
            "context_bwd_recurrent",
            "fusion",
            "head",
        ]
        self.non_spiking_aux_layers = [
            "multiscale_stem",
            "downsample1",
            "downsample2",
            "token_proj",
            "sleep_wake_head",
            "rem_head",
        ]

        self.multiscale_stem = _MultiScaleStem(
            in_channels=self.in_channels,
            branch_channels=self.branch_channels,
            stem_channels=self.stem_channels,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout_p,
        )
        self.downsample1 = DepthwiseSeparableConv1d(
            in_channels=self.stem_channels,
            out_channels=self.stem_channels * 2,
            kernel_size=9,
            stride=2,
            dropout=self.dropout_p,
        )
        self.downsample2 = DepthwiseSeparableConv1d(
            in_channels=self.stem_channels * 2,
            out_channels=self.token_dim,
            kernel_size=7,
            stride=2,
            dropout=self.dropout_p,
        )
        self.token_norm = nn.LayerNorm(self.token_dim)

        self.epoch_input = nn.Linear(self.token_dim, self.epoch_hidden_size, bias=False)
        self.epoch_recurrent = nn.Linear(self.epoch_hidden_size, self.epoch_hidden_size, bias=False)
        self.epoch_proj = nn.Linear(self.epoch_hidden_size, self.epoch_embed_dim, bias=False)
        self.epoch_embed_norm = nn.LayerNorm(self.epoch_embed_dim)
        self.epoch_dropout = nn.Dropout(self.dropout_p)

        self.context_fwd_input = nn.Linear(self.epoch_embed_dim, self.context_hidden_size, bias=False)
        self.context_fwd_recurrent = nn.Linear(self.context_hidden_size, self.context_hidden_size, bias=False)
        self.context_bwd_input = nn.Linear(self.epoch_embed_dim, self.context_hidden_size, bias=False)
        self.context_bwd_recurrent = nn.Linear(self.context_hidden_size, self.context_hidden_size, bias=False)
        self.context_proj = nn.Linear(self.context_hidden_size, self.context_hidden_size, bias=False)
        self.context_norm = nn.LayerNorm(self.context_hidden_size)

        fusion_dim = self.epoch_embed_dim + self.context_hidden_size * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.epoch_embed_dim, bias=False),
            nn.LayerNorm(self.epoch_embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )
        self.step_head = nn.Linear(fusion_dim, self.num_classes)
        self.head = nn.Linear(self.epoch_embed_dim, self.num_classes)

        if self.use_aux_heads:
            self.sleep_wake_head = nn.Linear(self.epoch_embed_dim, 1)
            self.rem_head = nn.Linear(self.epoch_embed_dim, 1)
        else:
            self.sleep_wake_head = None
            self.rem_head = None

        self.last_firing_rate = torch.tensor(0.0)
        self.last_layer_firing_rates: Dict[str, float] = {}

    def reset_state(self) -> None:
        self.last_firing_rate = torch.tensor(0.0)
        self.last_layer_firing_rates = {}

    def get_hparams(self) -> Dict[str, object]:
        return {
            "active_model_class": type(self).__name__,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "context_len": self.context_len,
            "branch_channels": self.branch_channels,
            "stem_channels": self.stem_channels,
            "token_dim": self.token_dim,
            "epoch_hidden_size": self.epoch_hidden_size,
            "epoch_embed_dim": self.epoch_embed_dim,
            "context_hidden_size": self.context_hidden_size,
            "t_steps": self.t_steps,
            "tau_epoch": self.tau_epoch,
            "tau_context": self.tau_context,
            "v_th_epoch": self.v_th_epoch,
            "v_th_context": self.v_th_context,
            "surrogate_alpha": self.surrogate_alpha,
            "dropout": self.dropout_p,
            "use_aux_heads": self.use_aux_heads,
            "center_residual_weight": self.center_residual_weight,
            "kernel_sizes": list(self.kernel_sizes),
            "sequence_context_enabled": True,
        }

    def _select_steps(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) <= self.t_steps:
            return x
        idx = torch.linspace(0, x.size(-1) - 1, self.t_steps, device=x.device).long()
        return x.index_select(-1, idx)

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.multiscale_stem(x)
        feat = self.downsample1(feat)
        feat = self.downsample2(feat)
        feat = self._select_steps(feat)
        feat = feat.transpose(1, 2).contiguous()
        return self.token_norm(feat)

    def _run_epoch_encoder(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, _ = tokens.shape
        v = torch.zeros(batch_size, self.epoch_hidden_size, device=tokens.device, dtype=tokens.dtype)
        s = torch.zeros_like(v)
        emb_sum = torch.zeros(batch_size, self.epoch_embed_dim, device=tokens.device, dtype=tokens.dtype)
        firing_acc = torch.zeros((), device=tokens.device, dtype=tokens.dtype)

        for step in range(steps):
            current = self.epoch_input(tokens[:, step, :]) + self.epoch_recurrent(s)
            v = self.tau_epoch * v * (1.0 - s) + current
            s = surrogate_spike(v - self.v_th_epoch, alpha=self.surrogate_alpha)
            emb_sum = emb_sum + self.epoch_proj(s)
            firing_acc = firing_acc + s.mean()

        pooled = emb_sum / float(max(1, steps))
        pooled = self.epoch_embed_norm(pooled)
        pooled = self.epoch_dropout(pooled)
        firing_rate = firing_acc / float(max(1, steps))
        return pooled, firing_rate

    def _run_context_direction(self, seq: torch.Tensor, reverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, _ = seq.shape
        v = torch.zeros(batch_size, self.context_hidden_size, device=seq.device, dtype=seq.dtype)
        s = torch.zeros_like(v)
        outputs = torch.zeros(batch_size, steps, self.context_hidden_size, device=seq.device, dtype=seq.dtype)
        firing_acc = torch.zeros((), device=seq.device, dtype=seq.dtype)

        iterator = range(steps - 1, -1, -1) if reverse else range(steps)
        for step in iterator:
            x_t = seq[:, step, :]
            if reverse:
                current = self.context_bwd_input(x_t) + self.context_bwd_recurrent(s)
            else:
                current = self.context_fwd_input(x_t) + self.context_fwd_recurrent(s)
            v = self.tau_context * v * (1.0 - s) + current
            s = surrogate_spike(v - self.v_th_context, alpha=self.surrogate_alpha)
            outputs[:, step, :] = self.context_proj(s)
            firing_acc = firing_acc + s.mean()

        outputs = self.context_norm(outputs)
        firing_rate = firing_acc / float(max(1, steps))
        return outputs, firing_rate

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"ContextPicoSNN expects [B,L,C,T] or [B,C,T], got shape={tuple(x.shape)}")

        batch_size, context_len, channels, time_steps = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"input channel mismatch: got={channels}, expect={self.in_channels}, shape={tuple(x.shape)}"
            )

        tokens = self._prepare_tokens(x.reshape(batch_size * context_len, channels, time_steps))
        epoch_embed, epoch_firing = self._run_epoch_encoder(tokens)
        epoch_embed = epoch_embed.view(batch_size, context_len, self.epoch_embed_dim)

        ctx_fwd, ctx_firing_fwd = self._run_context_direction(epoch_embed, reverse=False)
        ctx_bwd, ctx_firing_bwd = self._run_context_direction(epoch_embed, reverse=True)

        fusion_seq = torch.cat([epoch_embed, ctx_fwd, ctx_bwd], dim=-1)
        step_logits = self.step_head(fusion_seq)

        center_idx = context_len // 2
        center_fusion = self.fusion(fusion_seq[:, center_idx, :])
        center_fusion = center_fusion + float(self.center_residual_weight) * epoch_embed[:, center_idx, :]
        logits = self.head(center_fusion)

        mean_firing = (epoch_firing + ctx_firing_fwd + ctx_firing_bwd) / 3.0
        self.last_firing_rate = mean_firing.detach()
        self.last_layer_firing_rates = {
            "epoch_hidden": float(epoch_firing.detach().cpu().item()),
            "context_forward": float(ctx_firing_fwd.detach().cpu().item()),
            "context_backward": float(ctx_firing_bwd.detach().cpu().item()),
        }

        outputs = {
            "main": logits,
            "hidden": center_fusion,
            "step_logits": step_logits,
            "firing_rate": mean_firing,
            "layer_firing_rates": self.last_layer_firing_rates,
            "spike_sparsity": 1.0 - mean_firing,
        }
        if self.use_aux_heads and self.sleep_wake_head is not None and self.rem_head is not None:
            outputs["sleep_wake"] = self.sleep_wake_head(center_fusion).squeeze(-1)
            outputs["rem"] = self.rem_head(center_fusion).squeeze(-1)
        return outputs

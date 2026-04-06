# -*- coding: utf-8 -*-
"""Lightweight PicoSleepNet+ SNN branch for Sleep-EDF staging."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .picosleepnet_baseline import MaskedLinear, PicoSleepNetBaseline, fake_quant_ste


class PicoSleepNetPlusSNN(PicoSleepNetBaseline):
    """Baseline-preserving SNN with lightweight deploy-oriented refinements."""

    def __init__(
        self,
        num_classes: int = 5,
        window_size: int = 40,
        input_neurons_each: int = 40,
        input_streams: int = 4,
        reservoir_size: int = 150,
        hidden_size: int = 50,
        tau: float = 0.95,
        v_th: float = 1.0,
        surrogate_alpha: float = 4.0,
        use_masked_bpsr: bool = True,
        use_integer_spike: bool = True,
        lcs_delta: float = 0.13,
        lcs_delta_small: float = 0.08,
        lcs_delta_large: float = 0.18,
        use_dual_lcs: bool = True,
        dual_proj_dim: int | None = None,
        use_transition_matrix: bool = True,
        transition_residual_weight: float = 0.25,
        use_aux_heads: bool = True,
        firing_target_low: float = 0.05,
        firing_target_high: float = 0.20,
        postprocess_mode: str = "transition_forward",
        mask_alpha: float = 4.0,
        qat_bits: int = 6,
        learnable_threshold: bool = False,
        threshold_per_neuron: bool = True,
        threshold_min: float = 0.25,
        threshold_max: float = 2.5,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_size=window_size,
            input_neurons_each=input_neurons_each,
            input_streams=4 if bool(use_dual_lcs) else int(input_streams),
            reservoir_size=reservoir_size,
            hidden_size=hidden_size,
            tau=tau,
            v_th=v_th,
            surrogate_alpha=surrogate_alpha,
            use_masked_bpsr=use_masked_bpsr,
            use_integer_spike=use_integer_spike,
            lcs_delta=lcs_delta,
            mask_alpha=mask_alpha,
            qat_bits=qat_bits,
            learnable_threshold=learnable_threshold,
            threshold_per_neuron=threshold_per_neuron,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )

        self.lcs_delta_small = float(lcs_delta_small)
        self.lcs_delta_large = float(lcs_delta_large)
        self.use_dual_lcs = bool(use_dual_lcs)
        self.dual_proj_dim = int(max(8, dual_proj_dim or input_neurons_each))
        self.use_transition_matrix = bool(use_transition_matrix)
        self.use_aux_heads = bool(use_aux_heads)
        self.firing_target_low = float(firing_target_low)
        self.firing_target_high = float(firing_target_high)
        self.postprocess_mode = str(postprocess_mode).strip() or "transition_forward"

        if self.use_dual_lcs:
            branch_input_dim = int(self.window_size * 2)
            fused_input_dim = int(self.dual_proj_dim * 2)

            # Small thresholds preserve weak boundary changes, while large thresholds
            # emphasize stronger transitions; this is useful near W/N1/REM borders.
            self.shared_proj = nn.Linear(branch_input_dim, self.dual_proj_dim, bias=False)
            self.branch_gate = nn.Linear(fused_input_dim, fused_input_dim, bias=True)
            self.branch_norm = nn.LayerNorm(fused_input_dim)
            self.input_dim = fused_input_dim

            if self.use_masked_bpsr:
                self.w_in = MaskedLinear(
                    self.input_dim,
                    self.reservoir_size,
                    bias=False,
                    mask_alpha=self.mask_alpha,
                )
                self._masked_layers = [self.w_in, self.w_rec, self.w_hid, self.w_out]
            else:
                self.w_in = nn.Linear(self.input_dim, self.reservoir_size, bias=False)
        else:
            self.shared_proj = None
            self.branch_gate = None
            self.branch_norm = None

        if self.use_aux_heads:
            self.sleep_wake_head = nn.Linear(self.hidden_size, 1)
            self.rem_head = nn.Linear(self.hidden_size, 1)
        else:
            self.sleep_wake_head = None
            self.rem_head = None

        if self.use_transition_matrix:
            self.transition_logits = nn.Parameter(torch.eye(self.num_classes))
            self.transition_residual = nn.Parameter(torch.tensor(float(transition_residual_weight)))
        else:
            self.register_parameter("transition_logits", None)
            self.register_parameter("transition_residual", None)

    def get_hparams(self) -> Dict[str, object]:
        hparams = super().get_hparams()
        hparams.update(
            {
                "active_model_class": type(self).__name__,
                "lcs_delta_small": self.lcs_delta_small,
                "lcs_delta_large": self.lcs_delta_large,
                "use_dual_lcs": self.use_dual_lcs,
                "dual_proj_dim": self.dual_proj_dim,
                "use_transition_matrix": self.use_transition_matrix,
                "transition_residual_weight": (
                    float(self.transition_residual.detach().cpu().item())
                    if isinstance(self.transition_residual, torch.Tensor)
                    else 0.0
                ),
                "use_aux_heads": self.use_aux_heads,
                "firing_target_low": self.firing_target_low,
                "firing_target_high": self.firing_target_high,
                "postprocess_mode": self.postprocess_mode,
            }
        )
        return hparams

    def _reshape_windows(self, x: torch.Tensor, input_streams: int) -> torch.Tensor:
        total_t = int(x.size(-1))
        steps = total_t // self.window_size
        if steps <= 0:
            raise ValueError(f"epoch length too short: T={total_t}, window_size={self.window_size}")
        x = x[..., : steps * self.window_size]
        x = x.reshape(x.size(0), input_streams, steps, self.window_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(x.size(0), steps, input_streams * self.window_size)

    def _project_dual_streams(self, small_x: torch.Tensor, large_x: torch.Tensor) -> torch.Tensor:
        assert self.shared_proj is not None
        assert self.branch_gate is not None
        assert self.branch_norm is not None

        small_feat = torch.tanh(self.shared_proj(small_x))
        large_feat = torch.tanh(self.shared_proj(large_x))
        if self.qat_enable:
            small_feat = fake_quant_ste(small_feat, bits=self.qat_bits, enable=True)
            large_feat = fake_quant_ste(large_feat, bits=self.qat_bits, enable=True)

        fused = torch.cat([small_feat, large_feat], dim=-1)
        gate = torch.sigmoid(self.branch_gate(fused))
        return self.branch_norm(fused * gate)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x[:, x.size(1) // 2]

        if x.dim() == 2:
            raw = x
            if self.use_dual_lcs:
                small = self._lcs_encode_raw(raw, delta=self.lcs_delta_small)
                large = self._lcs_encode_raw(raw, delta=self.lcs_delta_large)
                x = torch.cat([small, large], dim=1)
            else:
                x = self._lcs_encode_raw(raw, delta=self.lcs_delta)
        elif x.dim() == 3 and x.size(1) == 1:
            raw = x[:, 0]
            if self.use_dual_lcs:
                small = self._lcs_encode_raw(raw, delta=self.lcs_delta_small)
                large = self._lcs_encode_raw(raw, delta=self.lcs_delta_large)
                x = torch.cat([small, large], dim=1)
            else:
                x = self._lcs_encode_raw(raw, delta=self.lcs_delta)

        if not self.use_dual_lcs:
            return super()._prepare_input(x)

        if x.dim() != 3:
            raise ValueError(f"PicoSleepNetPlusSNN expects [B,C,T] or [B,T], got shape={tuple(x.shape)}")
        if x.size(1) < 4:
            raise ValueError(
                "PicoSleepNetPlusSNN dual-LCS expects four channels: "
                "[pos_small, neg_small, pos_large, neg_large]. "
                f"got_shape={tuple(x.shape)}"
            )

        x = x[:, :4, :]
        small_flat = self._reshape_windows(x[:, :2, :], input_streams=2)
        large_flat = self._reshape_windows(x[:, 2:4, :], input_streams=2)
        return self._project_dual_streams(small_flat, large_flat)

    def transition_matrix(self) -> torch.Tensor:
        if not self.use_transition_matrix or self.transition_logits is None:
            device = next(self.parameters()).device
            return torch.eye(self.num_classes, device=device)
        return F.softmax(self.transition_logits, dim=1)

    def transition_nll(
        self,
        prev_labels: torch.Tensor | None,
        current_labels: torch.Tensor,
        next_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_transition_matrix:
            return current_labels.new_zeros(())

        trans_log = torch.log(self.transition_matrix().clamp_min(1e-6))
        losses = []

        if prev_labels is not None:
            prev_mask = prev_labels >= 0
            if torch.any(prev_mask):
                prev_log_prob = trans_log[prev_labels[prev_mask]]
                losses.append(F.nll_loss(prev_log_prob, current_labels[prev_mask], reduction="mean"))

        if next_labels is not None:
            next_mask = next_labels >= 0
            if torch.any(next_mask):
                next_log_prob = trans_log[current_labels[next_mask]]
                losses.append(F.nll_loss(next_log_prob, next_labels[next_mask], reduction="mean"))

        if not losses:
            return current_labels.new_zeros(())
        return torch.stack(losses).mean()

    def smooth_logits(self, logits_seq: torch.Tensor) -> torch.Tensor:
        if logits_seq.ndim != 2:
            raise ValueError(f"logits_seq must be [T,C], got {tuple(logits_seq.shape)}")
        if not self.use_transition_matrix:
            return logits_seq

        emission = F.log_softmax(logits_seq, dim=-1)
        trans = torch.log(self.transition_matrix().clamp_min(1e-6))
        alpha = torch.empty_like(emission)
        alpha[0] = emission[0]
        for idx in range(1, logits_seq.size(0)):
            prev = alpha[idx - 1].unsqueeze(1) + trans
            alpha[idx] = emission[idx] + torch.logsumexp(prev, dim=0)
            alpha[idx] = alpha[idx] - torch.logsumexp(alpha[idx], dim=0)
        return alpha

    def forward(self, x: torch.Tensor):
        step_x = self._prepare_input(x)
        logits, hidden_mean, spike_l1, debug_stats, step_logits = self._forward_steps(step_x)
        mask_l1_raw = self._mask_l1(device=logits.device, dtype=logits.dtype)
        mask_count = max(1, self._mask_param_count())
        mask_l1 = mask_l1_raw / float(mask_count)

        main_logits = logits
        transition_prior = None
        transition_weight = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.use_transition_matrix and self.transition_residual is not None:
            transition_weight = torch.clamp(self.transition_residual, min=0.0, max=1.0)
            transition_log = torch.log(self.transition_matrix().clamp_min(1e-6))
            transition_prior = F.softmax(logits, dim=-1) @ transition_log
            main_logits = main_logits + transition_weight * transition_prior
            debug_stats["transition_residual_weight"] = transition_weight

        firing_rate = spike_l1 / float(self.reservoir_size + self.hidden_size)
        layer_firing_rates = {
            "reservoir": debug_stats["reservoir_spike_ratio"],
            "hidden": debug_stats["hidden_spike_ratio"],
        }

        self.last_firing_rate = firing_rate.detach()
        self.last_layer_firing_rates = {
            name: float(value.detach().cpu().item()) for name, value in layer_firing_rates.items()
        }
        self._last_spike_l1 = spike_l1
        self._last_mask_l1 = mask_l1
        self._last_mask_l1_raw = mask_l1_raw

        outputs = {
            "main": main_logits,
            "hidden": hidden_mean,
            "firing_rate": firing_rate,
            "layer_firing_rates": layer_firing_rates,
            "spike_sparsity": 1.0 - firing_rate,
            "spike_l1": spike_l1,
            "mask_l1": mask_l1,
            "mask_l1_raw": mask_l1_raw,
            "step_logits": step_logits,
            "transition": self.transition_matrix() if self.use_transition_matrix else None,
            "transition_prior": transition_prior,
            "debug_stats": debug_stats,
        }

        if self.use_aux_heads and self.sleep_wake_head is not None and self.rem_head is not None:
            outputs["sleep_wake"] = self.sleep_wake_head(hidden_mean).squeeze(-1)
            outputs["rem"] = self.rem_head(hidden_mean).squeeze(-1)

        return outputs

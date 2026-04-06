# -*- coding: utf-8 -*-
"""PicoSleepNet baseline RSNN for single-channel Sleep-EDF staging."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class GaussianSurrogateSpike(torch.autograd.Function):
    """Heaviside forward with Gaussian-shaped surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = float(ctx.alpha)
        grad = torch.exp(-alpha * x.pow(2))
        return grad_output * grad, None


class BinaryMaskSTE(torch.autograd.Function):
    """Binary mask with straight-through surrogate backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = float(ctx.alpha)
        grad = torch.exp(-alpha * x.pow(2))
        return grad_output * grad, None


def fake_quant_ste(x: torch.Tensor, bits: int, enable: bool) -> torch.Tensor:
    if not enable:
        return x
    q_bits = int(max(2, bits))
    qmax = float(2 ** (q_bits - 1) - 1)
    scale = x.detach().abs().max().clamp(min=1e-8) / qmax
    x_q = torch.round(x / scale).clamp(min=-qmax, max=qmax) * scale
    return x + (x_q - x).detach()


def surrogate_spike(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GaussianSurrogateSpike.apply(x, float(alpha))


def _clamped_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = torch.clamp(x, min=float(eps), max=1.0 - float(eps))
    return torch.log(x) - torch.log1p(-x)


class MaskedLinear(nn.Module):
    """Linear layer with learnable binary mask for Masked-BPSR."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, mask_alpha: float = 4.0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.mask_k = nn.Parameter(torch.full((self.out_features, self.in_features), 1.0))
        self.mask_alpha = float(mask_alpha)
        nn.init.kaiming_uniform_(self.weight, a=5.0**0.5)

    def binary_mask(self) -> torch.Tensor:
        return BinaryMaskSTE.apply(self.mask_k, float(self.mask_alpha))

    def mask_l1(self) -> torch.Tensor:
        return self.binary_mask().abs().sum()

    def forward(self, x: torch.Tensor, qat_enable: bool = False, qat_bits: int = 6) -> torch.Tensor:
        mask = self.binary_mask()
        weight = fake_quant_ste(self.weight * mask, bits=qat_bits, enable=qat_enable)
        bias = self.bias
        if bias is not None:
            bias = fake_quant_ste(bias, bits=qat_bits, enable=qat_enable)
        return F.linear(x, weight, bias)


class PicoSleepNetBaseline(nn.Module):
    """Paper-aligned PicoSleepNet baseline with LCS + recurrent SNN."""

    def __init__(
        self,
        num_classes: int = 5,
        window_size: int = 40,
        input_neurons_each: int = 40,
        input_streams: int = 2,
        reservoir_size: int = 150,
        hidden_size: int = 50,
        tau: float = 0.95,
        v_th: float = 1.0,
        surrogate_alpha: float = 4.0,
        use_masked_bpsr: bool = True,
        use_integer_spike: bool = True,
        lcs_delta: float = 0.13,
        mask_alpha: float = 4.0,
        qat_bits: int = 6,
        learnable_threshold: bool = False,
        threshold_per_neuron: bool = True,
        threshold_min: float = 0.25,
        threshold_max: float = 2.5,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.window_size = int(window_size)
        self.input_neurons_each = int(input_neurons_each)
        self.input_streams = int(input_streams)
        self.input_dim = int(self.window_size * self.input_streams)
        self.reservoir_size = int(reservoir_size)
        self.hidden_size = int(hidden_size)
        self.tau = float(tau)
        self.v_th = float(v_th)
        self.surrogate_alpha = float(surrogate_alpha)
        self.use_masked_bpsr = bool(use_masked_bpsr)
        self.use_integer_spike = bool(use_integer_spike)
        self.lcs_delta = float(lcs_delta)
        self.mask_alpha = float(mask_alpha)
        self.qat_bits = int(qat_bits)
        self.qat_enable = False
        self.learnable_threshold = bool(learnable_threshold)
        self.threshold_per_neuron = bool(threshold_per_neuron)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(max(threshold_max, self.threshold_min + 1e-4))

        if self.use_masked_bpsr:
            self.w_in = MaskedLinear(self.input_dim, self.reservoir_size, bias=False, mask_alpha=self.mask_alpha)
            self.w_rec = MaskedLinear(self.reservoir_size, self.reservoir_size, bias=False, mask_alpha=self.mask_alpha)
            self.w_hid = MaskedLinear(self.reservoir_size, self.hidden_size, bias=False, mask_alpha=self.mask_alpha)
            self.w_out = MaskedLinear(self.hidden_size, self.num_classes, bias=True, mask_alpha=self.mask_alpha)
            self._masked_layers = [self.w_in, self.w_rec, self.w_hid, self.w_out]
        else:
            self.w_in = nn.Linear(self.input_dim, self.reservoir_size, bias=False)
            self.w_rec = nn.Linear(self.reservoir_size, self.reservoir_size, bias=False)
            self.w_hid = nn.Linear(self.reservoir_size, self.hidden_size, bias=False)
            self.w_out = nn.Linear(self.hidden_size, self.num_classes, bias=True)
            self._masked_layers = []

        self.bias_res = nn.Parameter(torch.zeros(self.reservoir_size))
        self.bias_hid = nn.Parameter(torch.zeros(self.hidden_size))

        if self.learnable_threshold:
            res_shape = (self.reservoir_size,) if self.threshold_per_neuron else (1,)
            hid_shape = (self.hidden_size,) if self.threshold_per_neuron else (1,)
            self.reservoir_threshold_param = nn.Parameter(
                self._threshold_to_param(torch.full(res_shape, float(self.v_th), dtype=torch.float32))
            )
            self.hidden_threshold_param = nn.Parameter(
                self._threshold_to_param(torch.full(hid_shape, float(self.v_th), dtype=torch.float32))
            )
        else:
            self.register_parameter("reservoir_threshold_param", None)
            self.register_parameter("hidden_threshold_param", None)

        self._v_res: Optional[torch.Tensor] = None
        self._s_res: Optional[torch.Tensor] = None
        self._v_hid: Optional[torch.Tensor] = None
        self._s_hid: Optional[torch.Tensor] = None

        self.last_firing_rate = torch.tensor(0.0)
        self.last_layer_firing_rates: Dict[str, float] = {}
        self._last_spike_l1 = torch.tensor(0.0)
        self._last_mask_l1 = torch.tensor(0.0)
        self._last_mask_l1_raw = torch.tensor(0.0)
        self.debug_enabled = False

    def get_hparams(self) -> Dict[str, object]:
        hparams = {
            "active_model_class": type(self).__name__,
            "num_classes": self.num_classes,
            "window_size": self.window_size,
            "input_neurons_each": self.input_neurons_each,
            "input_streams": self.input_streams,
            "reservoir_size": self.reservoir_size,
            "hidden_size": self.hidden_size,
            "tau": self.tau,
            "v_th": self.v_th,
            "surrogate_alpha": self.surrogate_alpha,
            "use_masked_bpsr": self.use_masked_bpsr,
            "use_integer_spike": self.use_integer_spike,
            "lcs_delta": self.lcs_delta,
            "mask_alpha": self.mask_alpha,
            "qat_bits": self.qat_bits,
            "learnable_threshold": self.learnable_threshold,
            "threshold_per_neuron": self.threshold_per_neuron,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
        }
        hparams.update(self.threshold_summary())
        return hparams

    def learnable_threshold_parameter_names(self) -> list[str]:
        names: list[str] = []
        if self.reservoir_threshold_param is not None:
            names.append("reservoir_threshold_param")
        if self.hidden_threshold_param is not None:
            names.append("hidden_threshold_param")
        return names

    def threshold_summary(self) -> Dict[str, object]:
        names = self.learnable_threshold_parameter_names()
        summary: Dict[str, object] = {
            "learnable_threshold_param_names": names,
            "learnable_threshold_param_count": int(
                sum(int(param.numel()) for param in (self.reservoir_threshold_param, self.hidden_threshold_param) if param is not None)
            ),
        }
        if not self.learnable_threshold or not names:
            summary.update(
                {
                    "reservoir_threshold_mean": float(self.v_th),
                    "reservoir_threshold_std": 0.0,
                    "hidden_threshold_mean": float(self.v_th),
                    "hidden_threshold_std": 0.0,
                }
            )
            return summary

        with torch.no_grad():
            device = self.reservoir_threshold_param.device if self.reservoir_threshold_param is not None else next(self.parameters()).device
            dtype = self.reservoir_threshold_param.dtype if self.reservoir_threshold_param is not None else torch.float32
            res_threshold, hid_threshold = self.current_thresholds(device=device, dtype=dtype)
            summary.update(
                {
                    "reservoir_threshold_mean": float(res_threshold.mean().item()),
                    "reservoir_threshold_std": float(res_threshold.std(unbiased=False).item()),
                    "hidden_threshold_mean": float(hid_threshold.mean().item()),
                    "hidden_threshold_std": float(hid_threshold.std(unbiased=False).item()),
                }
            )
        return summary

    def enable_qat(self, bits: Optional[int] = None) -> None:
        if bits is not None:
            self.qat_bits = int(bits)
        self.qat_enable = True

    def disable_qat(self) -> None:
        self.qat_enable = False

    def set_debug(self, enabled: bool = True) -> None:
        self.debug_enabled = bool(enabled)

    def reset_state(self) -> None:
        self._v_res = None
        self._s_res = None
        self._v_hid = None
        self._s_hid = None

    def _threshold_to_param(self, threshold: torch.Tensor) -> torch.Tensor:
        scale = float(self.threshold_max - self.threshold_min)
        normalized = (threshold - float(self.threshold_min)) / max(scale, 1e-6)
        return _clamped_logit(normalized)

    def _threshold_from_param(
        self,
        param: Optional[torch.Tensor],
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if param is None:
            return torch.full((size,), float(self.v_th), device=device, dtype=dtype)
        value = float(self.threshold_min) + float(self.threshold_max - self.threshold_min) * torch.sigmoid(param)
        return value.to(device=device, dtype=dtype)

    def current_thresholds(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._threshold_from_param(
                self.reservoir_threshold_param,
                self.reservoir_size,
                device=device,
                dtype=dtype,
            ),
            self._threshold_from_param(
                self.hidden_threshold_param,
                self.hidden_size,
                device=device,
                dtype=dtype,
            ),
        )

    def _linear(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(layer, MaskedLinear):
            return layer(x, qat_enable=self.qat_enable, qat_bits=self.qat_bits)
        assert isinstance(layer, nn.Linear)
        if not self.qat_enable:
            return layer(x)
        bias = layer.bias
        if bias is not None:
            bias = fake_quant_ste(bias, bits=self.qat_bits, enable=True)
        weight = fake_quant_ste(layer.weight, bits=self.qat_bits, enable=True)
        return F.linear(x, weight, bias)

    def _ensure_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._v_res is not None
            and self._v_hid is not None
            and self._v_res.size(0) == batch_size
            and self._v_hid.size(0) == batch_size
            and self._v_res.device == device
            and self._v_hid.device == device
            and self._v_res.dtype == dtype
            and self._v_hid.dtype == dtype
        ):
            return
        self._v_res = torch.zeros(batch_size, self.reservoir_size, device=device, dtype=dtype)
        self._s_res = torch.zeros(batch_size, self.reservoir_size, device=device, dtype=dtype)
        self._v_hid = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        self._s_hid = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _lcs_encode_raw(self, raw: torch.Tensor, delta: float) -> torch.Tensor:
        """Convert raw epoch [B,T] to [B,2,T] LCS channels."""
        d = float(max(1e-6, delta))
        pos = torch.zeros_like(raw)
        neg = torch.zeros_like(raw)
        last = raw[:, 0].clone()
        for idx in range(raw.size(1)):
            spike_level = torch.floor((raw[:, idx] - last) / d)
            update = spike_level != 0
            last = torch.where(update, raw[:, idx], last)
            if self.use_integer_spike:
                pos[:, idx] = torch.where(spike_level > 0, spike_level, torch.zeros_like(spike_level))
                neg[:, idx] = torch.where(spike_level < 0, -spike_level, torch.zeros_like(spike_level))
            else:
                pos[:, idx] = (spike_level > 0).to(raw.dtype)
                neg[:, idx] = (spike_level < 0).to(raw.dtype)
        return torch.stack([pos, neg], dim=1)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare [B, steps, input_dim] from raw or pre-encoded epochs."""
        if x.dim() == 4:
            x = x[:, x.size(1) // 2]
        if x.dim() == 2:
            x = self._lcs_encode_raw(x, delta=self.lcs_delta)
        elif x.dim() == 3 and x.size(1) == 1:
            x = self._lcs_encode_raw(x[:, 0], delta=self.lcs_delta)

        if x.dim() != 3:
            raise ValueError(f"PicoSleepNetBaseline expects [B,C,T] or [B,T], got shape={tuple(x.shape)}")
        if x.size(1) < self.input_streams:
            raise ValueError(
                f"input channels mismatch: got={x.size(1)} expect>={self.input_streams}. "
                "请检查 LCS 编码通道数。"
            )

        x = x[:, : self.input_streams, :]
        total_t = int(x.size(-1))
        steps = total_t // self.window_size
        if steps <= 0:
            raise ValueError(f"epoch length too short: T={total_t}, window_size={self.window_size}")
        x = x[..., : steps * self.window_size]
        x = x.reshape(x.size(0), self.input_streams, steps, self.window_size)
        return x.permute(0, 2, 1, 3).contiguous().reshape(x.size(0), steps, self.input_dim)

    def _forward_steps(
        self,
        step_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, steps, _ = step_x.shape
        self._ensure_state(batch_size, step_x.device, step_x.dtype)
        assert self._v_res is not None
        assert self._s_res is not None
        assert self._v_hid is not None
        assert self._s_hid is not None
        res_threshold, hid_threshold = self.current_thresholds(device=step_x.device, dtype=step_x.dtype)

        logits_acc = torch.zeros(batch_size, self.num_classes, device=step_x.device, dtype=step_x.dtype)
        step_logits = torch.zeros(batch_size, steps, self.num_classes, device=step_x.device, dtype=step_x.dtype)
        hidden_acc = torch.zeros(batch_size, self.hidden_size, device=step_x.device, dtype=step_x.dtype)
        spike_l1 = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        res_spike_ratio = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        hid_spike_ratio = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        res_mem_mean = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        res_mem_std = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        hid_mem_mean = torch.zeros((), device=step_x.device, dtype=step_x.dtype)
        hid_mem_std = torch.zeros((), device=step_x.device, dtype=step_x.dtype)

        for step in range(steps):
            x_t = step_x[:, step]
            input_current = self._linear(self.w_in, x_t)
            recurrent_current = self._linear(self.w_rec, self._s_res)
            self._v_res = self.tau * self._v_res * (1.0 - self._s_res) + input_current + recurrent_current + self.bias_res
            self._v_res = fake_quant_ste(self._v_res, bits=self.qat_bits, enable=self.qat_enable)
            self._s_res = surrogate_spike(self._v_res - res_threshold.unsqueeze(0), alpha=self.surrogate_alpha)

            hidden_current = self._linear(self.w_hid, self._s_res)
            self._v_hid = self.tau * self._v_hid * (1.0 - self._s_hid) + hidden_current + self.bias_hid
            self._v_hid = fake_quant_ste(self._v_hid, bits=self.qat_bits, enable=self.qat_enable)
            self._s_hid = surrogate_spike(self._v_hid - hid_threshold.unsqueeze(0), alpha=self.surrogate_alpha)

            step_logits[:, step] = self._linear(self.w_out, self._s_hid)
            logits_acc = logits_acc + step_logits[:, step]
            hidden_acc = hidden_acc + self._s_hid
            spike_l1 = spike_l1 + self._s_res.abs().sum() + self._s_hid.abs().sum()
            res_spike_ratio = res_spike_ratio + self._s_res.ne(0).to(step_x.dtype).mean()
            hid_spike_ratio = hid_spike_ratio + self._s_hid.ne(0).to(step_x.dtype).mean()
            res_mem_mean = res_mem_mean + self._v_res.mean()
            res_mem_std = res_mem_std + self._v_res.std(unbiased=False)
            hid_mem_mean = hid_mem_mean + self._v_hid.mean()
            hid_mem_std = hid_mem_std + self._v_hid.std(unbiased=False)

        logits = logits_acc / float(steps)
        hidden_mean = hidden_acc / float(steps)
        spike_l1 = spike_l1 / float(max(1, batch_size * steps))
        debug_stats = {
            "reservoir_spike_ratio": res_spike_ratio / float(steps),
            "hidden_spike_ratio": hid_spike_ratio / float(steps),
            "reservoir_membrane_mean": res_mem_mean / float(steps),
            "reservoir_membrane_std": res_mem_std / float(steps),
            "hidden_membrane_mean": hid_mem_mean / float(steps),
            "hidden_membrane_std": hid_mem_std / float(steps),
            "reservoir_threshold_mean": res_threshold.mean(),
            "reservoir_threshold_std": res_threshold.std(unbiased=False),
            "hidden_threshold_mean": hid_threshold.mean(),
            "hidden_threshold_std": hid_threshold.std(unbiased=False),
        }
        return logits, hidden_mean, spike_l1, debug_stats, step_logits

    def bpsr_regularization(self, lambda_s: float, lambda_w: float) -> torch.Tensor:
        return float(lambda_s) * self._last_spike_l1 + float(lambda_w) * self._last_mask_l1

    def _mask_l1(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self._masked_layers:
            return torch.zeros((), device=device, dtype=dtype)
        total = torch.zeros((), device=device, dtype=dtype)
        for layer in self._masked_layers:
            total = total + layer.mask_l1().to(device=device, dtype=dtype)
        return total

    def _mask_param_count(self) -> int:
        if not self._masked_layers:
            return 0
        total = 0
        for layer in self._masked_layers:
            total += int(layer.weight.numel())
        return total

    def forward(self, x: torch.Tensor):
        step_x = self._prepare_input(x)
        logits, hidden_mean, spike_l1, debug_stats, step_logits = self._forward_steps(step_x)
        mask_l1_raw = self._mask_l1(device=logits.device, dtype=logits.dtype)
        mask_count = max(1, self._mask_param_count())
        mask_l1 = mask_l1_raw / float(mask_count)

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

        return {
            "main": logits,
            "hidden": hidden_mean,
            "firing_rate": firing_rate,
            "layer_firing_rates": layer_firing_rates,
            "spike_sparsity": 1.0 - firing_rate,
            "spike_l1": spike_l1,
            "mask_l1": mask_l1,
            "mask_l1_raw": mask_l1_raw,
            "step_logits": step_logits,
            "debug_stats": debug_stats,
        }

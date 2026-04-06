"""SNN 基础层：代理梯度脉冲的 LIF 神经元。"""

from __future__ import annotations

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    """Fast-sigmoid 代理梯度脉冲函数。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        # x: [B, ...]，前向为硬阈值
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # d/dx fast-sigmoid: 1 / (1 + alpha*|x|)^2
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = grad_output / (alpha * torch.abs(x) + 1.0) ** 2
        return grad, None


def surrogate_spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """代理梯度脉冲（前向硬阈值，反向 fast-sigmoid）。"""
    return SurrogateSpike.apply(x, alpha)


class StatefulModule(nn.Module):
    """带状态的模块基类（便于统一 reset_state）。"""

    def reset_state(self) -> None:  # pragma: no cover - 基类占位
        return


class LIFCell(StatefulModule):
    """LIF 神经元单元。

    公式（逐时间步）：
      v = decay * v + x
      s = spike(v - threshold)
      v = v - s * threshold
    其中 x, v, s 形状均为 [B, D]。
    """

    def __init__(
        self,
        size: int,
        decay: float = 0.9,
        threshold: float = 1.0,
        alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        # 使用 buffer 保存膜电位，便于 device 迁移
        self.register_buffer("v", torch.zeros(1), persistent=False)
        self._has_state = False

    def reset_state(self) -> None:
        """清空膜电位状态（必须在每个 batch 前调用）。"""
        self.v = torch.zeros(1, device=self.v.device)
        self._has_state = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self._has_state) or (self.v.shape != x.shape):
            self.v = torch.zeros_like(x)
            self._has_state = True
        # LIF 膜电位更新
        self.v = self.v * self.decay + x
        # 产生脉冲（代理梯度）
        spike = surrogate_spike(self.v - self.threshold, alpha=self.alpha)
        # 软重置：超过阈值则扣除阈值
        self.v = self.v - spike * self.threshold
        return spike

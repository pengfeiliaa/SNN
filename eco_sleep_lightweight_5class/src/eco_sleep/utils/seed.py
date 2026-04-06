"""随机种子与可复现性设置。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，减少跨次运行差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关闭不确定性优化以提升可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

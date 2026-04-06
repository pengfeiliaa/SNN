"""特征标准化工具（仅使用训练集统计，避免泄漏）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .io import ensure_dir


def compute_feature_scaler(
    files: Iterable[Path],
    key: str = "features",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """基于训练集样本计算 mean/std，用于后续归一化。"""
    rng = np.random.default_rng(seed)
    total = 0
    sum_ = None
    sumsq = None

    for path in files:
        data = np.load(path, allow_pickle=True)
        arr = data[key].astype(np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if max_samples is not None:
            remaining = max_samples - total
            if remaining <= 0:
                break
            if arr.shape[0] > remaining:
                idx = rng.choice(arr.shape[0], size=remaining, replace=False)
                arr = arr[idx]

        if sum_ is None:
            sum_ = np.zeros(arr.shape[1], dtype=np.float64)
            sumsq = np.zeros(arr.shape[1], dtype=np.float64)

        sum_ += arr.sum(axis=0)
        sumsq += np.square(arr).sum(axis=0)
        total += arr.shape[0]

    if total == 0 or sum_ is None or sumsq is None:
        raise ValueError("无法计算标准化统计量：训练集样本为空。")

    mean = sum_ / total
    var = sumsq / total - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "count": int(total),
    }


def save_scaler_json(path: Path, scaler: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    payload = {
        "mean": scaler["mean"].tolist(),
        "std": scaler["std"].tolist(),
        "count": int(scaler.get("count", 0)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scaler_json(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "mean": np.asarray(data["mean"], dtype=np.float32),
        "std": np.asarray(data["std"], dtype=np.float32),
        "count": int(data.get("count", 0)),
    }

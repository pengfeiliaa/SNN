"""特征标准化工具（增量统计，避免内存爆）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def compute_feature_scaler(npz_paths: List[str], max_items: Optional[int] = None) -> dict:
    """读取一组 .npz 计算全局 mean/std（按维度 D）。

    - 使用增量统计（sum/sumsq），避免一次性加载全量数据
    - max_items 允许限制累计 frame 数，防止内存/时间爆炸
    """
    total = 0
    sum_ = None
    sumsq = None
    rng = np.random.default_rng(42)

    for p in npz_paths:
        path = Path(p)
        data = np.load(path, allow_pickle=True)
        feats = data["features"].astype(np.float64)
        if feats.ndim == 1:
            feats = feats.reshape(-1, 1)

        if max_items is not None:
            remain = max_items - total
            if remain <= 0:
                break
            if feats.shape[0] > remain:
                idx = rng.choice(feats.shape[0], size=remain, replace=False)
                feats = feats[idx]

        if sum_ is None:
            sum_ = np.zeros(feats.shape[1], dtype=np.float64)
            sumsq = np.zeros(feats.shape[1], dtype=np.float64)

        sum_ += feats.sum(axis=0)
        sumsq += np.square(feats).sum(axis=0)
        total += feats.shape[0]

    if total == 0 or sum_ is None or sumsq is None:
        raise ValueError("无法计算标准化统计量：features 为空。")

    mean = sum_ / total
    var = sumsq / total - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return {"mean": mean.tolist(), "std": std.tolist(), "n": int(total)}


def save_scaler_json(scaler: dict, path: str) -> None:
    """保存 scaler 为 json（mean/std 为 list）。"""
    out = {
        "mean": list(scaler["mean"]),
        "std": list(scaler["std"]),
        "n": int(scaler.get("n", 0)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scaler_json(path: str) -> dict:
    """读取 scaler json，并转成 numpy 数组。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "mean": np.asarray(data["mean"], dtype=np.float32),
        "std": np.asarray(data["std"], dtype=np.float32),
        "n": int(data.get("n", 0)),
    }


def apply_scaler(x: np.ndarray, scaler: dict) -> np.ndarray:
    """标准化：(x - mean) / std，支持 x 形状 [..., D]。"""
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    std = np.maximum(std, 1e-6)
    return (x - mean) / std

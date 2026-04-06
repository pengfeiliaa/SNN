# -*- coding: utf-8 -*-
"""Bland-Altman statistics and plotting."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict

import numpy as np

from .io import ensure_dir
from .plots import ensure_chinese_font

_HAS_MPL = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:
    _HAS_MPL = False
    plt = None

_DUMMY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def _write_dummy_png(out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_bytes(_DUMMY_PNG)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    mean_diff = float(np.mean(diff)) if diff.size else 0.0
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return {"mean_diff": mean_diff, "loa_low": mean_diff - 1.96 * sd, "loa_high": mean_diff + 1.96 * sd}


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_vals = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    stats = bland_altman_stats(y_true, y_pred)
    ensure_dir(out_path.parent)
    if not _HAS_MPL:
        _write_dummy_png(out_path)
        return stats
    has_chinese_font = ensure_chinese_font()
    if has_chinese_font:
        mean_label, low_label, high_label = "均值差", "LoA 下限", "LoA 上限"
        xlabel, ylabel, safe_title = "均值(真实+预测)/2", "差值(预测-真实)", title
    else:
        mean_label, low_label, high_label = "Mean Diff", "LoA Low", "LoA High"
        xlabel, ylabel = "Mean (True+Pred)/2", "Diff (Pred-True)"
        safe_title = "Bland-Altman" if _contains_cjk(title) else title
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean_vals, diff, alpha=0.6)
    ax.axhline(stats["mean_diff"], color="red", linestyle="--", label=mean_label)
    ax.axhline(stats["loa_low"], color="gray", linestyle="--", label=low_label)
    ax.axhline(stats["loa_high"], color="gray", linestyle="--", label=high_label)
    ax.set_title(safe_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats

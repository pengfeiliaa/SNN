# -*- coding: utf-8 -*-
"""Plot helpers with Chinese-font fallback."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .io import ensure_dir

_HAS_MPL = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib import font_manager  # noqa: E402
except Exception:
    _HAS_MPL = False
    plt = None
    font_manager = None

_DUMMY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)
_PREFERRED_CHINESE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
]
_FONT_SETUP_DONE = False
_HAS_CHINESE_FONT = False
_FONT_WARNED = False


def _write_dummy_png(out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_bytes(_DUMMY_PNG)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def _safe_text(text: str, fallback: str | None, has_chinese_font: bool) -> str:
    if has_chinese_font or not _contains_cjk(text):
        return str(text)
    return str(fallback or text)


def ensure_chinese_font() -> bool:
    global _FONT_SETUP_DONE, _HAS_CHINESE_FONT, _FONT_WARNED

    if not _HAS_MPL:
        return False
    if _FONT_SETUP_DONE:
        return _HAS_CHINESE_FONT

    selected_font = None
    try:
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        for candidate in _PREFERRED_CHINESE_FONTS:
            if candidate in available_fonts:
                selected_font = candidate
                break
    except Exception:
        selected_font = None

    if selected_font:
        matplotlib.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        _HAS_CHINESE_FONT = True
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        _HAS_CHINESE_FONT = False
        if not _FONT_WARNED:
            print("WARNING: no Chinese font detected, plot titles will fall back to English.")
            _FONT_WARNED = True

    _FONT_SETUP_DONE = True
    return _HAS_CHINESE_FONT


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    normalize: bool = False,
    title: str | None = None,
    title_fallback: str | None = None,
) -> None:
    ensure_dir(out_path.parent)
    if not _HAS_MPL:
        _write_dummy_png(out_path)
        return

    has_chinese_font = ensure_chinese_font()
    matrix = np.asarray(cm, dtype=np.float64 if normalize else cm.dtype)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum != 0)

    safe_title = _safe_text(title or "混淆矩阵", title_fallback or "Confusion Matrix", has_chinese_font)
    safe_xlabel = _safe_text("预测类别", "Predicted", has_chinese_font)
    safe_ylabel = _safe_text("真实类别", "True", has_chinese_font)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(safe_xlabel)
    ax.set_ylabel(safe_ylabel)
    ax.set_title(safe_title)

    value_fmt = "{:.2f}" if normalize else "{}"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, value_fmt.format(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    title_fallback: str | None = None,
    xlabel_fallback: str | None = None,
    ylabel_fallback: str | None = None,
) -> None:
    ensure_dir(out_path.parent)
    if not _HAS_MPL:
        _write_dummy_png(out_path)
        return

    has_chinese_font = ensure_chinese_font()
    safe_title = _safe_text(title, title_fallback or "Curve", has_chinese_font)
    safe_xlabel = _safe_text(xlabel, xlabel_fallback or "X", has_chinese_font)
    safe_ylabel = _safe_text(ylabel, ylabel_fallback or "Y", has_chinese_font)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for name, (x, y) in curves.items():
        ax.plot(x, y, label=str(name))
    ax.set_xlabel(safe_xlabel)
    ax.set_ylabel(safe_ylabel)
    ax.set_title(safe_title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

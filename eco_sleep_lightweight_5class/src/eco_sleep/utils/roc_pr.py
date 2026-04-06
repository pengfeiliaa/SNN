"""ROC / PR 曲线计算。"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore
except Exception:  # sklearn 不可用时提供 numpy 版本

    def _sorted_scores(y_true: np.ndarray, y_score: np.ndarray):
        order = np.argsort(-y_score)
        return y_true[order], y_score[order]

    def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        if y_true.size == 0:
            raise ValueError("empty")
        y_true, y_score = _sorted_scores(y_true, y_score)
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == 0)
        # 去除重复阈值
        distinct = np.where(np.diff(y_score))[0]
        idx = np.r_[distinct, y_true.size - 1]
        tps = tps[idx]
        fps = fps[idx]
        thresholds = y_score[idx]
        # 补充 (0,0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1e-6, thresholds]
        P = max(1, int((y_true == 1).sum()))
        N = max(1, int((y_true == 0).sum()))
        fpr = fps / N
        tpr = tps / P
        return fpr, tpr, thresholds

    def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        if y_true.size == 0:
            raise ValueError("empty")
        y_true, y_score = _sorted_scores(y_true, y_score)
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == 0)
        distinct = np.where(np.diff(y_score))[0]
        idx = np.r_[distinct, y_true.size - 1]
        tps = tps[idx]
        fps = fps[idx]
        thresholds = y_score[idx]
        P = max(1, int((y_true == 1).sum()))
        precision = tps / np.maximum(1, tps + fps)
        recall = tps / P
        # 补齐端点
        precision = np.r_[precision, 1.0]
        recall = np.r_[recall, 0.0]
        thresholds = np.r_[thresholds, thresholds[-1] - 1e-6]
        return precision, recall, thresholds

from eco_sleep import ID_TO_LABEL


def binary_curves(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """二分类 ROC/PR 曲线。"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
    except ValueError:
        precision, recall = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    return {"roc": (fpr, tpr), "pr": (precision, recall)}


def multiclass_curves(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """one-vs-rest 多分类 ROC/PR 曲线。"""
    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for c in range(num_classes):
        label = ID_TO_LABEL.get(c, str(c))
        y_true_bin = (y_true == c).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, c])
        except ValueError:
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        try:
            precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, c])
        except ValueError:
            precision, recall = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        curves[label] = {"roc": (fpr, tpr), "pr": (precision, recall)}
    return curves

"""Walch2019 风格指标计算。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from eco_sleep import ID_TO_LABEL, NUM_CLASSES


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def walch_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    positive: str = "sleep",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """二分类 Walch 指标。

    - y_true 为 0/1，1 表示“正类”
    - y_prob 为“正类”的概率
    - sensitivity = TP / (TP + FN)
    - specificity = TN / (TN + FP)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    kappa = cohen_kappa_score(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    try:
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    except ValueError:
        fpr, tpr, roc_thr = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5])
    try:
        precision, recall, pr_thr = precision_recall_curve(y_true, y_prob)
    except ValueError:
        precision, recall, pr_thr = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5])

    metrics = {
        "positive": positive,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    curves = {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thr},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thr},
    }
    return metrics, curves


def walch_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """多分类 Walch 指标（one-vs-rest）。

    sensitivity = recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    accuracy = _safe_div(np.trace(cm), np.sum(cm))

    kappa = cohen_kappa_score(y_true, y_pred)

    per_class: Dict[str, Any] = {}
    recalls = []
    specificities = []
    roc_aucs = []
    pr_aucs = []

    for c in range(num_classes):
        label = ID_TO_LABEL.get(c, str(c))
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp

        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        sensitivity = recall

        y_true_bin = (y_true == c).astype(int)
        y_prob_bin = y_prob[:, c]
        try:
            roc_auc = roc_auc_score(y_true_bin, y_prob_bin)
        except ValueError:
            roc_auc = float("nan")
        try:
            pr_auc = average_precision_score(y_true_bin, y_prob_bin)
        except ValueError:
            pr_auc = float("nan")

        per_class[label] = {
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support": int(np.sum(y_true == c)),
        }
        recalls.append(recall)
        specificities.append(specificity)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

    metrics = {
        "accuracy": accuracy,
        "kappa": kappa,
        "macro_recall": float(np.nanmean(recalls)),
        "macro_specificity": float(np.nanmean(specificities)),
        "macro_roc_auc": float(np.nanmean(roc_aucs)),
        "macro_pr_auc": float(np.nanmean(pr_aucs)),
    }
    return metrics, per_class

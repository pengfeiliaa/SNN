from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eco_sleep.utils.metrics_walch2019 import walch_binary_metrics, walch_multiclass_metrics


def test_metrics_shapes() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 5, size=100)
    y_prob = rng.random((100, 5))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    multi_metrics, per_class = walch_multiclass_metrics(y_true, y_prob, num_classes=5)
    assert "accuracy" in multi_metrics
    assert "kappa" in multi_metrics
    assert len(per_class) == 5

    y_true_bin = rng.integers(0, 2, size=50)
    y_prob_bin = rng.random(50)
    bin_metrics, curves = walch_binary_metrics(y_true_bin, y_prob_bin, positive="sleep")
    assert "accuracy" in bin_metrics
    assert "roc" in curves
    assert "pr" in curves

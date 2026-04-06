from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eco_sleep.utils.sleep_stats import compute_sleep_stats


def test_sleep_stats_basic() -> None:
    labels = [0, 0, 1, 2, 0, 3, 0, 4, 0, 0]  # 10 epochs
    stats = compute_sleep_stats(labels, epoch_seconds=30)
    assert abs(stats["TIB"] - 5.0) < 1e-6
    assert abs(stats["TST"] - 2.0) < 1e-6
    assert abs(stats["SOL"] - 1.0) < 1e-6
    assert abs(stats["WASO"] - 1.0) < 1e-6
    assert abs(stats["SE"] - 40.0) < 1e-6
    assert abs(stats["REM"] - 0.5) < 1e-6
    assert abs(stats["NREM"] - 1.5) < 1e-6

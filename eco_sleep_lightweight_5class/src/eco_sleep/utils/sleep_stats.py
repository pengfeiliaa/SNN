"""睡眠统计指标计算。"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pandas 不可用时提供轻量替代
    pd = None


class _SimpleSeries:
    def __init__(self, values):
        self.values = np.asarray(values)


class _SimpleDataFrame:
    """简化版 DataFrame，仅支持 to_csv 与列访问。"""

    def __init__(self, rows: List[Dict[str, float]]):
        self._rows = rows
        self._columns = list(rows[0].keys()) if rows else []

    def to_csv(self, path, index: bool = False):  # noqa: D401
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def __getitem__(self, key: str):
        return _SimpleSeries([row.get(key, 0.0) for row in self._rows])

    @property
    def empty(self) -> bool:
        return len(self._rows) == 0


def _infer_stage_labels(labels_arr: np.ndarray, wake_label: int):
    """根据标签范围推断 REM 与 NREM 的编码。"""
    if labels_arr.size == 0:
        return 4, [1, 2, 3]
    max_label = int(labels_arr.max())
    # Sleep-Accel：REM=3，NREM=1/2
    if max_label <= 3 and wake_label == 0:
        return 3, [1, 2]
    # Sleep-EDF：REM=4，NREM=1/2/3
    return 4, [1, 2, 3]


def compute_sleep_stats(
    labels: Iterable[int],
    epoch_seconds: int = 30,
    wake_label: int = 0,
    handle_no_sleep: str = "tib",
) -> Dict[str, float]:
    """计算 TIB/TST/SOL/WASO/SE/REM/NREM 等统计量。"""
    labels_arr = np.asarray(list(labels), dtype=int)
    total_epochs = len(labels_arr)
    epoch_min = epoch_seconds / 60.0

    tib = total_epochs * epoch_min
    sleep_mask = labels_arr != wake_label  # 非 Wake 记为 Sleep

    if np.any(sleep_mask):
        first_sleep = int(np.argmax(sleep_mask))
        last_sleep = int(total_epochs - 1 - np.argmax(sleep_mask[::-1]))
        sol = first_sleep * epoch_min
        waso = (
            np.sum(
                (labels_arr == wake_label)
                & (np.arange(total_epochs) >= first_sleep)
                & (np.arange(total_epochs) <= last_sleep)
            )
            * epoch_min
        )
    else:
        sol = tib if handle_no_sleep == "tib" else 0.0
        waso = 0.0

    rem_label, nrem_labels = _infer_stage_labels(labels_arr, wake_label)
    tst = np.sum(sleep_mask) * epoch_min
    se = (tst / tib * 100.0) if tib > 0 else 0.0
    rem_time = np.sum(labels_arr == rem_label) * epoch_min
    nrem_time = np.sum(np.isin(labels_arr, nrem_labels)) * epoch_min

    return {
        "TIB": float(tib),
        "TST": float(tst),
        "SOL": float(sol),
        "WASO": float(waso),
        "SE": float(se),
        "REM": float(rem_time),
        "NREM": float(nrem_time),
    }


def build_sleep_stats_table(
    record_ids: List[str],
    y_true_list: List[Iterable[int]],
    y_pred_list: List[Iterable[int]],
    epoch_seconds: int = 30,
    wake_label: int = 0,
):
    """构建逐夜统计表（真实/预测）。"""
    rows = []
    for rid, y_true, y_pred in zip(record_ids, y_true_list, y_pred_list):
        true_stats = compute_sleep_stats(y_true, epoch_seconds=epoch_seconds, wake_label=wake_label)
        pred_stats = compute_sleep_stats(y_pred, epoch_seconds=epoch_seconds, wake_label=wake_label)
        row = {"record_id": rid}
        for k, v in true_stats.items():
            row[f"true_{k}"] = v
        for k, v in pred_stats.items():
            row[f"pred_{k}"] = v
        rows.append(row)
    if pd is not None:
        return pd.DataFrame(rows)
    return _SimpleDataFrame(rows)

# -*- coding: utf-8 -*-
"""Task label helpers for the maintained Sleep-EDF 5-class line."""

from __future__ import annotations

from typing import List

from eco_sleep.data.sleep_edf.labels import SLEEP_EDF_LABELS

LABELS_SLEEP_EDF_5: List[str] = list(SLEEP_EDF_LABELS)
WAKE_LABEL_BY_TASK = {
    "sleep_edf_5class": 0,
    "sleep_edf": 0,
}


def get_task_name(cfg: dict | None, default_task: str) -> str:
    """Read task name from config and default to the maintained Sleep-EDF task."""
    if not cfg:
        return default_task
    task = cfg.get("task")
    return str(task) if task else default_task


def get_labels(task_name: str | None) -> List[str]:
    """Return Sleep-EDF 5-class labels for all maintained paths."""
    _ = task_name
    return LABELS_SLEEP_EDF_5


def get_num_classes(task_name: str | None) -> int:
    """Return the maintained class count."""
    _ = task_name
    return len(LABELS_SLEEP_EDF_5)


def get_wake_label(task_name: str | None) -> int:
    """Wake label id stays fixed at 0 for Sleep-EDF 5-class."""
    _ = task_name
    return 0


def get_rem_label(task_name: str | None) -> int | None:
    """REM label id for the maintained Sleep-EDF task."""
    labels = get_labels(task_name)
    return labels.index("REM") if "REM" in labels else None


__all__ = [
    "LABELS_SLEEP_EDF_5",
    "WAKE_LABEL_BY_TASK",
    "get_labels",
    "get_task_name",
    "get_num_classes",
    "get_wake_label",
    "get_rem_label",
]

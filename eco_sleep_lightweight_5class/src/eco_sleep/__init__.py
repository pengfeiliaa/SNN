"""Top-level package exports for the maintained Sleep-EDF 5-class project."""

from .labels import (
    LABELS_SLEEP_EDF_5,
    WAKE_LABEL_BY_TASK,
    get_labels,
    get_num_classes,
    get_rem_label,
    get_task_name,
    get_wake_label,
)

LABELS = LABELS_SLEEP_EDF_5
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABELS)

__all__ = [
    "LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "NUM_CLASSES",
    "LABELS_SLEEP_EDF_5",
    "WAKE_LABEL_BY_TASK",
    "get_labels",
    "get_task_name",
    "get_num_classes",
    "get_wake_label",
    "get_rem_label",
]

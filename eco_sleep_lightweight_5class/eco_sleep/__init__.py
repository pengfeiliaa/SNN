"""兼容导入：在未设置 PYTHONPATH 时也可 import eco_sleep。"""

from __future__ import annotations

from pathlib import Path

# 将 src/eco_sleep 加入包搜索路径，便于 import eco_sleep.utils.*
_SRC_PKG = Path(__file__).resolve().parent.parent / "src" / "eco_sleep"
if _SRC_PKG.exists():
    __path__.append(str(_SRC_PKG))

LABELS = ["W", "N1", "N2", "N3", "REM"]
LABEL_TO_ID = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = 5

__all__ = ["LABELS", "LABEL_TO_ID", "ID_TO_LABEL", "NUM_CLASSES"]

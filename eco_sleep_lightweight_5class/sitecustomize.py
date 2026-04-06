"""自动将 src 加入 sys.path，方便在仓库根目录直接运行 python。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

"""脚本入口的路径兜底工具。"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    """将仓库根目录下的 src 插入 sys.path（幂等）。"""
    # 入口脚本直接运行时需要兜底，避免找不到 eco_sleep 包
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if not src_dir.exists():
        return
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

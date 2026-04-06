"""轻量 JSONL 训练日志记录器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .io import append_jsonl, ensure_dir


class JsonlLogger:
    """同时打印并写入 jsonl 的日志。"""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        ensure_dir(log_path.parent)

    def log(self, record: Dict[str, Any]) -> None:
        print(record)
        append_jsonl(self.log_path, record)

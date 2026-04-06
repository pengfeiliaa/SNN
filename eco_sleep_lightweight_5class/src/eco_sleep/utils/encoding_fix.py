"""Encoding helpers for Windows console and CSV output."""

from __future__ import annotations

import io
import sys
import warnings
from typing import Dict, Any


def setup_utf8_stdio() -> None:
    """Force stdout/stderr to UTF-8 to avoid mojibake on Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except Exception:
            pass

        # Fallback for environments without reconfigure().
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        wrapped = io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True)
        setattr(sys, stream_name, wrapped)


def csv_utf8_sig_kwargs(**extra: Any) -> Dict[str, Any]:
    """Default kwargs for Excel-friendly CSV output."""
    kwargs: Dict[str, Any] = {"encoding": "utf-8-sig"}
    kwargs.update(extra)
    return kwargs


def suppress_pin_memory_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message="'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.",
    )

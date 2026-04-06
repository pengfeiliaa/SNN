# -*- coding: utf-8 -*-
"""Compatibility aliases for the legacy PicoSleepNetRSNN module path."""

from __future__ import annotations

from .picosleepnet_baseline import PicoSleepNetBaseline
from .picosleepnet_plus_snn import PicoSleepNetPlusSNN


class PicoSleepNetRSNN(PicoSleepNetBaseline):
    """Legacy alias kept for old imports. Default training no longer uses this name."""


__all__ = ["PicoSleepNetRSNN", "PicoSleepNetPlusSNN", "PicoSleepNetBaseline"]

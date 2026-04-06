# -*- coding: utf-8 -*-
"""Model registry."""

from .context_pico_snn import ContextPicoSNN
from .context_pico_snn_v2 import ContextPicoSNNV2
from .edf_snn import EdfSNN
from .heads import LinearHead, MultiTaskHeads
from .picosleepnet_baseline import PicoSleepNetBaseline
from .picosleepnet_plus_snn import PicoSleepNetPlusSNN
from .picosleepnet_rsnn import PicoSleepNetRSNN
from .tiny_cnn1d import EEGContextModel, EEGTinyEncoder
from .tiny_tcn import ContextTinyTCN

# Legacy compatibility import kept only for external smoke/import checks.
WearableSNN = EdfSNN


def count_parameters(model, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "ContextTinyTCN",
    "ContextPicoSNN",
    "ContextPicoSNNV2",
    "EEGTinyEncoder",
    "EEGContextModel",
    "EdfSNN",
    "WearableSNN",
    "PicoSleepNetBaseline",
    "PicoSleepNetRSNN",
    "PicoSleepNetPlusSNN",
    "LinearHead",
    "MultiTaskHeads",
    "count_parameters",
]

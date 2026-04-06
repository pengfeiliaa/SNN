# -*- coding: utf-8 -*-
"""Model complexity helpers for lightweight SNN analysis."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from eco_sleep.train.trainer import extract_firing_rate, extract_layer_firing_rates
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs
from eco_sleep.utils.io import ensure_dir

try:
    from thop import profile as thop_profile  # type: ignore
except Exception:
    thop_profile = None


def _tensor_bytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def parameter_stats(model: nn.Module) -> Dict[str, int]:
    total_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    buffer_bytes = sum(_tensor_bytes(b) for b in model.buffers())
    parameter_bytes = sum(_tensor_bytes(p) for p in model.parameters())
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
    }


def _macs_from_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    x = inputs[0] if inputs else None
    if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
        return 0

    if isinstance(module, nn.Linear) or module.__class__.__name__ == "MaskedLinear":
        batch_mul = int(np.prod(output.shape[:-1])) if output.ndim > 1 else int(output.shape[0])
        return int(batch_mul * module.in_features * module.out_features)

    if isinstance(module, nn.Conv1d):
        out = output
        kernel_mul = int(module.kernel_size[0] * (module.in_channels // module.groups))
        return int(out.shape[0] * out.shape[1] * out.shape[2] * kernel_mul)

    if isinstance(module, nn.Conv2d):
        out = output
        kernel_mul = int(np.prod(module.kernel_size) * (module.in_channels // module.groups))
        return int(out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3] * kernel_mul)

    return 0


def estimate_macs(model: nn.Module, sample_input: torch.Tensor) -> int:
    if thop_profile is not None:
        try:
            macs, _ = thop_profile(copy.deepcopy(model).eval(), inputs=(sample_input.clone(),), verbose=False)
            return int(macs)
        except Exception:
            pass

    hooks = []
    macs = 0

    def hook(module: nn.Module, inputs, output):
        nonlocal macs
        try:
            macs += _macs_from_hook(module, inputs, output)
        except Exception:
            pass

    for module in model.modules():
        if module is model:
            continue
        hooks.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model.eval()
            _ = model(sample_input)
    finally:
        for handle in hooks:
            handle.remove()
    return int(macs)


def estimate_activation_bytes(model: nn.Module, sample_input: torch.Tensor) -> int:
    hooks = []
    activation_bytes = 0

    def hook(_module: nn.Module, _inputs, output):
        nonlocal activation_bytes
        if isinstance(output, torch.Tensor):
            activation_bytes += _tensor_bytes(output)
        elif isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    activation_bytes += _tensor_bytes(item)

    for module in model.modules():
        if module is model:
            continue
        hooks.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model.eval()
            _ = model(sample_input)
    finally:
        for handle in hooks:
            handle.remove()
    return int(activation_bytes)


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup: int = 5,
    repeats: int = 20,
) -> Dict[str, float]:
    bench_model = copy.deepcopy(model).eval().to(device)
    bench_input = sample_input.detach().clone().to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(max(1, warmup)):
            if hasattr(bench_model, "reset_state"):
                bench_model.reset_state()
            _ = bench_model(bench_input)
        _sync_if_needed(device)

        elapsed = []
        for _ in range(max(1, repeats)):
            if hasattr(bench_model, "reset_state"):
                bench_model.reset_state()
            start = time.perf_counter()
            _ = bench_model(bench_input)
            _sync_if_needed(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)

    peak_cuda_memory = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    return {
        "latency_ms_mean": float(np.mean(elapsed)),
        "latency_ms_std": float(np.std(elapsed)),
        "peak_cuda_memory_bytes": peak_cuda_memory,
    }


def collect_firing_statistics(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 8,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    model.eval()
    firing_rates: List[float] = []
    per_layer: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= int(max_batches):
                break
            x = batch[0].to(device, non_blocking=non_blocking)
            if hasattr(model, "reset_state"):
                model.reset_state()
            outputs = model(x)
            firing_rate = extract_firing_rate(outputs, model)
            if isinstance(firing_rate, torch.Tensor):
                firing_rates.append(float(firing_rate.detach().mean().cpu().item()))
            layer_rates = extract_layer_firing_rates(outputs, model)
            for key, value in layer_rates.items():
                per_layer.setdefault(str(key), []).append(float(value))

    avg_firing_rate = float(np.mean(firing_rates)) if firing_rates else float("nan")
    layer_summary = {
        key: float(np.mean(values)) if values else float("nan") for key, values in sorted(per_layer.items())
    }
    return {
        "avg_firing_rate": avg_firing_rate,
        "spike_sparsity": float(1.0 - avg_firing_rate) if not math.isnan(avg_firing_rate) else float("nan"),
        "layer_firing_rates": layer_summary,
    }


def checkpoint_size_stats(ckpt_path: Path | None) -> Dict[str, float]:
    if ckpt_path is None or not ckpt_path.exists():
        return {"checkpoint_size_bytes": float("nan"), "checkpoint_size_mb": float("nan")}
    size_bytes = float(ckpt_path.stat().st_size)
    return {
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_size_mb": size_bytes / (1024.0 * 1024.0),
    }


def build_complexity_metrics(
    model: nn.Module,
    sample_input: torch.Tensor,
    dataloader,
    runtime_device: torch.device,
    ckpt_path: Path | None = None,
    non_blocking: bool = False,
) -> Dict[str, Any]:
    sample_cpu = sample_input.detach().cpu()
    model_cpu = copy.deepcopy(model).cpu().eval()

    params = parameter_stats(model_cpu)
    estimated_macs = estimate_macs(model_cpu, sample_cpu)
    activation_bytes = estimate_activation_bytes(model_cpu, sample_cpu)
    firing = collect_firing_statistics(model, dataloader, device=runtime_device, max_batches=8, non_blocking=non_blocking)
    size_stats = checkpoint_size_stats(ckpt_path)

    cpu_latency = benchmark_latency(model_cpu, sample_cpu, device=torch.device("cpu"))
    gpu_latency = None
    if torch.cuda.is_available():
        gpu_latency = benchmark_latency(model, sample_input, device=torch.device("cuda"))

    latency_ms = (
        float(gpu_latency["latency_ms_mean"])
        if gpu_latency is not None
        else float(cpu_latency["latency_ms_mean"])
    )

    peak_memory_bytes_estimate = (
        int(params["parameter_bytes"])
        + int(params["buffer_bytes"])
        + int(_tensor_bytes(sample_cpu))
        + int(activation_bytes)
    )

    lightweight_index = 1.0 / (
        1.0
        + math.log10(max(10.0, float(params["total_params"])))
        + math.log10(max(10.0, float(estimated_macs)))
        + math.log10(max(1.0, float(latency_ms)))
    )

    complexity_score_summary = {
        "lightweight_index": float(lightweight_index),
        "params_k": float(params["total_params"]) / 1000.0,
        "macs_m": float(estimated_macs) / 1_000_000.0,
        "latency_ms": float(latency_ms),
        "spike_sparsity": float(firing["spike_sparsity"]),
    }

    metrics = {
        **params,
        "estimated_MACs": int(estimated_macs),
        "activation_bytes_estimate": int(activation_bytes),
        "peak_memory_bytes_estimate": int(peak_memory_bytes_estimate),
        "avg_firing_rate": float(firing["avg_firing_rate"]),
        "spike_sparsity": float(firing["spike_sparsity"]),
        "inference_latency_ms": float(latency_ms),
        "cpu_latency_ms": float(cpu_latency["latency_ms_mean"]),
        "cpu_latency_std_ms": float(cpu_latency["latency_ms_std"]),
        "gpu_latency_ms": float(gpu_latency["latency_ms_mean"]) if gpu_latency is not None else float("nan"),
        "gpu_latency_std_ms": float(gpu_latency["latency_ms_std"]) if gpu_latency is not None else float("nan"),
        "peak_cuda_memory_bytes": (
            float(gpu_latency["peak_cuda_memory_bytes"]) if gpu_latency is not None else float("nan")
        ),
        **size_stats,
        "complexity_score_summary": complexity_score_summary,
        "layer_firing_rates": firing["layer_firing_rates"],
    }
    return metrics


def flatten_complexity_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(row)
    layer_rates = flat.pop("layer_firing_rates", {}) or {}
    score_summary = flat.pop("complexity_score_summary", {}) or {}
    for key, value in layer_rates.items():
        flat[f"layer_firing_rate_{key}"] = value
    for key, value in score_summary.items():
        flat[f"complexity_score_{key}"] = value
    return flat


def append_mean_std_rows(rows: List[Dict[str, Any]], group_keys: Iterable[str] | None = None) -> List[Dict[str, Any]]:
    if not rows:
        return rows

    df = pd.DataFrame([flatten_complexity_row(row) for row in rows])
    if df.empty:
        return rows

    group_keys = list(group_keys or [])
    skip_cols = {"split", "result_tag", "postprocess_used", *group_keys}
    numeric_cols = [col for col in df.columns if col not in skip_cols and pd.api.types.is_numeric_dtype(df[col])]

    appended = list(rows)
    grouped = [([], df)] if not group_keys else list(df.groupby(group_keys, dropna=False))
    for group_value, group_df in grouped:
        if group_df.empty:
            continue
        group_tuple = tuple(group_value) if isinstance(group_value, tuple) else (group_value,)
        group_payload = {}
        for idx, key in enumerate(group_keys):
            group_payload[key] = group_tuple[idx]

        mean_row = {"split": "mean", **group_payload}
        std_row = {"split": "std", **group_payload}
        for col in numeric_cols:
            mean_row[col] = float(group_df[col].mean())
            std_row[col] = float(group_df[col].std())
        appended.append(mean_row)
        appended.append(std_row)
    return appended


def save_complexity_reports(output_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    ensure_dir(output_dir)
    json_path = output_dir / "model_complexity.json"
    csv_path = output_dir / "model_complexity.csv"

    json_path.write_text(
        json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([flatten_complexity_row(row) for row in rows]).to_csv(
        csv_path,
        index=False,
        **csv_utf8_sig_kwargs(),
    )

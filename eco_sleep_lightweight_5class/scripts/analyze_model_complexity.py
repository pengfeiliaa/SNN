# -*- coding: utf-8 -*-
"""Sleep-EDF model complexity analysis with robust path resolution."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval_sleep_edf import (
    REPO_ROOT,
    SleepEdfSpikeDataset,
    _build_model_from_ckpt,
    _entries_for_split,
    _resolve_repo_path,
    _resolve_run_dir,
    _sample_input_from_loader,
    build_entries,
    load_records,
)
from train_sleep_edf import _build_model, _canonical_model_name, _default_train_model_name, _resolve_subset
from eco_sleep import get_num_classes, get_task_name
from eco_sleep.train.checkpoints import load_checkpoint
from eco_sleep.utils.encoding_fix import csv_utf8_sig_kwargs, setup_utf8_stdio, suppress_pin_memory_warning
from eco_sleep.utils.io import ensure_dir, read_yaml
from eco_sleep.utils.model_complexity import (
    append_mean_std_rows,
    build_complexity_metrics,
    flatten_complexity_row,
    save_complexity_reports,
)


def _ckpt_candidates(run_dir: Path | None, split_idx: int | None) -> list[tuple[int | str, Path]]:
    if run_dir is None or not run_dir.exists():
        return []
    train_dir = run_dir / "train"
    if not train_dir.exists():
        return []

    if split_idx is not None:
        ckpt_path = train_dir / f"split_{int(split_idx)}" / "best.ckpt"
        return [(int(split_idx), ckpt_path)] if ckpt_path.exists() else []

    out = []
    for split_path in sorted(train_dir.glob("split_*")):
        ckpt_path = split_path / "best.ckpt"
        if ckpt_path.exists():
            try:
                split_name: int | str = int(split_path.name.split("_")[-1])
            except Exception:
                split_name = split_path.name
            out.append((split_name, ckpt_path))
    return out


def _build_dataset_loader(
    entries,
    cfg: dict,
    model_name: str,
    model_hparams: dict,
):
    delta_primary = float(model_hparams.get("lcs_delta", cfg.get("lcs", {}).get("delta", 0.13)))
    delta_small = float(model_hparams.get("lcs_delta_small", max(0.02, delta_primary * 0.65)))
    delta_large = float(model_hparams.get("lcs_delta_large", max(delta_small + 1e-4, delta_primary * 1.35)))

    dataset = SleepEdfSpikeDataset(
        list(entries),
        model_name=model_name,
        use_dual_lcs=bool(model_hparams.get("use_dual_lcs", model_name == "picosleepnet_plus_snn")),
        use_integer_spike=bool(model_hparams.get("use_integer_spike", True)),
        delta_primary=delta_primary,
        delta_small=delta_small,
        delta_large=delta_large,
        cache_mode=str(cfg.get("cache_mode", "mem")),
        input_norm_mode=str(cfg.get("train", {}).get("input_norm_mode", "none")).lower().strip() or "none",
        context_len=int(model_hparams.get("context_len", 1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=0 if os.name == "nt" else int(cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return dataset, loader


def _synthetic_loader(model_name: str):
    channels = 4 if model_name == "picosleepnet_plus_snn" else 2
    sample = torch.zeros(1, channels, 3000, dtype=torch.float32)
    batch = (sample, torch.zeros(1, dtype=torch.long), ["synthetic"], torch.zeros(1, dtype=torch.long))
    return sample[:1].clone(), [batch]


def _augment_complexity_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["estimated_FLOPs"] = int(float(out.get("estimated_MACs", 0.0)) * 2.0)
    return out


def _base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split")) not in {"mean", "std"}]


def _mean_of(rows: list[dict[str, Any]], field: str) -> float:
    values = []
    for row in _base_rows(rows):
        value = row.get(field, float("nan"))
        try:
            values.append(float(value))
        except Exception:
            continue
    values = [value for value in values if not pd.isna(value)]
    return float(sum(values) / len(values)) if values else float("nan")


def _resolve_baseline_run_dir(
    explicit_baseline_run_dir: str | None,
    resolved_run_dir: Path | None,
) -> Path | None:
    if explicit_baseline_run_dir:
        baseline_dir = _resolve_repo_path(explicit_baseline_run_dir)
        if baseline_dir is None or not baseline_dir.exists():
            raise RuntimeError(f"baseline_run_dir not found: {explicit_baseline_run_dir}")
        return baseline_dir
    if resolved_run_dir is None:
        return None
    kept_path = resolved_run_dir / "eval" / "kept_recipe.json"
    if kept_path.exists():
        try:
            payload = json.loads(kept_path.read_text(encoding="utf-8"))
            baseline_dir = _resolve_repo_path(payload.get("baseline_run_dir"))
            if baseline_dir is not None and baseline_dir.exists():
                return baseline_dir
        except Exception:
            return resolved_run_dir
    return resolved_run_dir


def _build_complexity_comparison(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    current_params = _mean_of(current_rows, "total_params")
    baseline_params = _mean_of(baseline_rows, "total_params")
    current_macs = _mean_of(current_rows, "estimated_MACs")
    baseline_macs = _mean_of(baseline_rows, "estimated_MACs")
    current_latency = _mean_of(current_rows, "inference_latency_ms")
    baseline_latency = _mean_of(baseline_rows, "inference_latency_ms")
    relative_ratio = {
        "total_params": float(current_params / baseline_params) if baseline_params and baseline_params > 0 else float("nan"),
        "estimated_MACs": float(current_macs / baseline_macs) if baseline_macs and baseline_macs > 0 else float("nan"),
        "inference_latency_ms": float(current_latency / baseline_latency) if baseline_latency and baseline_latency > 0 else float("nan"),
    }
    warning = any(
        (not pd.isna(value)) and float(value) > 1.05
        for value in relative_ratio.values()
    )
    return {
        "baseline_locked": {
            "total_params": baseline_params,
            "estimated_MACs": baseline_macs,
            "estimated_FLOPs": float(baseline_macs * 2.0) if not pd.isna(baseline_macs) else float("nan"),
            "inference_latency_ms": baseline_latency,
        },
        "kept_recipe": {
            "total_params": current_params,
            "estimated_MACs": current_macs,
            "estimated_FLOPs": float(current_macs * 2.0) if not pd.isna(current_macs) else float("nan"),
            "inference_latency_ms": current_latency,
        },
        "relative_complexity_vs_baseline_locked": relative_ratio,
        "warning": warning,
        "warning_reason": "kept recipe exceeds 5% inference complexity increase" if warning else "",
    }


def _analyze_checkpoint_rows(
    cfg: dict,
    run_dir: Path,
    ckpt_items: list[tuple[int | str, Path]],
    processed_records,
    all_entries,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    splits_path = run_dir / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else []

    for split_name, ckpt_path in ckpt_items:
        ckpt = load_checkpoint(ckpt_path)
        model, model_name, model_hparams, num_classes = _build_model_from_ckpt(ckpt, ckpt_path.parent, get_num_classes(get_task_name(cfg, "sleep_edf_5class")))

        entries = all_entries
        if isinstance(split_name, int) and split_name < len(splits):
            split_entries = _entries_for_split(splits[split_name], all_entries)
            entries = split_entries["test"] or split_entries["val"] or split_entries["train"]

        if entries:
            _, loader = _build_dataset_loader(entries, cfg, model_name, model_hparams)
            sample_input = _sample_input_from_loader(loader)
        else:
            sample_input, loader = _synthetic_loader(model_name)

        if sample_input is None:
            sample_input, loader = _synthetic_loader(model_name)

        model.to(device)
        complexity = build_complexity_metrics(
            model=model,
            sample_input=sample_input,
            dataloader=loader,
            runtime_device=device,
            ckpt_path=ckpt_path,
            non_blocking=True,
        )
        row = _augment_complexity_row(
            {
                "split": split_name,
                "model_name": model_name,
                "result_tag": "raw",
                "postprocess_used": False,
                "checkpoint_path": str(ckpt_path),
                **complexity,
            }
        )
        rows.append(row)
        print(
            f"complexity split={split_name} params={int(row.get('total_params', 0))} "
            f"macs={int(row.get('estimated_MACs', 0))} latency_ms={float(row.get('inference_latency_ms', float('nan'))):.3f}"
        )

    return rows


def _analyze_default_model_rows(
    cfg: dict,
    resolved_run_dir: Path | None,
    model_arg: str | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    records = load_records(Path(cfg["processed_dir"])) if Path(cfg["processed_dir"]).exists() else []
    entries = build_entries(records, num_classes=num_classes) if records else []
    subset = _resolve_subset(cfg, records) if records else str(cfg.get("dataset", {}).get("edf_subset", "edf20"))

    model_name = _canonical_model_name(model_arg or cfg.get("model", {}).get("name") or _default_train_model_name(None))
    model, model_hparams = _build_model(cfg=cfg, model_name=model_name, num_classes=num_classes, subset=subset)

    chosen_entries = entries[: min(len(entries), max(32, int(cfg.get("batch_size", 32)) * 2))]
    if chosen_entries:
        _, loader = _build_dataset_loader(chosen_entries, cfg, model_name, model_hparams)
        sample_input = _sample_input_from_loader(loader)
    else:
        sample_input, loader = _synthetic_loader(model_name)

    if sample_input is None:
        sample_input, loader = _synthetic_loader(model_name)

    model.to(device)
    complexity = build_complexity_metrics(
        model=model,
        sample_input=sample_input,
        dataloader=loader,
        runtime_device=device,
        ckpt_path=None,
        non_blocking=True,
    )
    return [
        _augment_complexity_row(
            {
                "split": "config_default",
                "model_name": model_name,
                "result_tag": "raw",
                "postprocess_used": False,
                "checkpoint_path": "",
                **complexity,
            }
        )
    ]


def main() -> None:
    setup_utf8_stdio()
    suppress_pin_memory_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--baseline_run_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    if config_path is None or not config_path.exists():
        raise RuntimeError(f"config not found: {args.config}")

    cfg = read_yaml(config_path)
    resolved_run_dir = _resolve_run_dir(Path(cfg["runs_dir"]), args.run_dir)
    if resolved_run_dir is not None and (resolved_run_dir / "config.yaml").exists():
        cfg = {**cfg, **read_yaml(resolved_run_dir / "config.yaml")}

    output_dir = (
        ensure_dir(_resolve_repo_path(args.output))
        if args.output
        else ensure_dir((resolved_run_dir / "eval") if resolved_run_dir is not None else (REPO_ROOT / "runs" / "_complexity_default"))
    )

    print(f"resolved_repo_root={REPO_ROOT}")
    print(f"resolved_config_path={config_path}")
    print(f"resolved_run_dir={resolved_run_dir}")

    task = get_task_name(cfg, "sleep_edf_5class")
    num_classes = int(get_num_classes(task))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_records = load_records(Path(cfg["processed_dir"])) if Path(cfg["processed_dir"]).exists() else []
    all_entries = build_entries(processed_records, num_classes=num_classes) if processed_records else []

    rows: list[dict[str, Any]]
    if args.ckpt_path:
        ckpt_path = _resolve_repo_path(args.ckpt_path)
        if ckpt_path is None or not ckpt_path.exists():
            raise RuntimeError(f"ckpt_path not found: {args.ckpt_path}")
        rows = _analyze_checkpoint_rows(cfg, ckpt_path.parents[2], [("manual", ckpt_path)], processed_records, all_entries, device)
    else:
        ckpt_items = _ckpt_candidates(resolved_run_dir, args.split)
        if ckpt_items:
            rows = _analyze_checkpoint_rows(cfg, resolved_run_dir, ckpt_items, processed_records, all_entries, device)
        else:
            rows = _analyze_default_model_rows(cfg, resolved_run_dir, args.model, device)

    baseline_run_dir = _resolve_baseline_run_dir(args.baseline_run_dir, resolved_run_dir)
    baseline_rows_raw: list[dict[str, Any]] = []
    if baseline_run_dir is not None:
        baseline_ckpts = _ckpt_candidates(baseline_run_dir, args.split)
        if baseline_ckpts:
            baseline_rows_raw = _analyze_checkpoint_rows(cfg, baseline_run_dir, baseline_ckpts, processed_records, all_entries, device)
        elif resolved_run_dir is not None and baseline_run_dir == resolved_run_dir:
            baseline_rows_raw = list(rows)

    rows = append_mean_std_rows(rows, group_keys=["model_name"])
    summary = {
        "repo_root": str(REPO_ROOT),
        "config_path": str(config_path),
        "run_dir": "" if resolved_run_dir is None else str(resolved_run_dir),
        "baseline_run_dir": "" if baseline_run_dir is None else str(baseline_run_dir),
        "num_rows": int(len(rows)),
        "device_used": str(device),
    }

    save_complexity_reports(output_dir, rows, summary)
    flat_rows = [flatten_complexity_row(row) for row in rows]
    complexity_summary = {"rows": rows, "summary": summary}
    (output_dir / "complexity_summary.json").write_text(
        json.dumps(complexity_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(flat_rows).to_csv(output_dir / "complexity_summary.csv", index=False, **csv_utf8_sig_kwargs())

    comparison = _build_complexity_comparison(current_rows=_base_rows(rows), baseline_rows=(baseline_rows_raw or _base_rows(rows)))
    (output_dir / "complexity_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "target": "baseline_locked",
                **comparison["baseline_locked"],
                "relative_complexity_vs_baseline_locked": 1.0,
            },
            {
                "target": "kept_recipe",
                **comparison["kept_recipe"],
                "relative_complexity_vs_baseline_locked": comparison["relative_complexity_vs_baseline_locked"].get("estimated_MACs", float("nan")),
            },
        ]
    ).to_csv(output_dir / "complexity_comparison.csv", index=False, **csv_utf8_sig_kwargs())

    primary_row = next((row for row in rows if not isinstance(row.get("split"), str) or row.get("split") not in {"mean", "std"}), rows[0])
    print(
        f"complexity: model={primary_row.get('model_name', '')} "
        f"params={int(float(primary_row.get('total_params', 0)))} "
        f"macs={int(float(primary_row.get('estimated_MACs', 0)))} "
        f"out={output_dir}"
    )
    if comparison.get("warning", False):
        print("complexity: warning kept recipe exceeds 5% inference complexity increase vs baseline_locked")


if __name__ == "__main__":
    main()

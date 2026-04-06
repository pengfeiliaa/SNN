# -*- coding: utf-8 -*-
"""Visual spot-check for Sleep-EDF processed epochs."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from eco_sleep.data.sleep_edf.storage import list_processed_records
from eco_sleep.utils.io import ensure_dir, read_yaml


def save_sample_visual_check(
    processed_dir: Path,
    output_dir: Path,
    seed: int = 42,
    num_samples: int = 6,
) -> list[dict]:
    ensure_dir(output_dir)
    rng = random.Random(int(seed))
    records = list_processed_records(processed_dir)
    if not records:
        raise RuntimeError(f"no processed Sleep-EDF files found: {processed_dir}")

    chosen = rng.sample(records, k=min(len(records), max(1, int(num_samples))))
    rows: list[dict] = []
    for idx, record in enumerate(chosen):
        npz = np.load(record.path, allow_pickle=True)
        labels = npz["label"].astype(np.int64) if "label" in npz.files else npz["labels"].astype(np.int64)
        raw_epoch = npz["raw_epoch"].astype(np.float32) if "raw_epoch" in npz.files else npz["signals"].astype(np.float32)[:, 0, :]
        stage_desc = npz["epoch_stage_desc"].astype(str) if "epoch_stage_desc" in npz.files else np.full(labels.shape, "", dtype=object)
        epoch_idx = rng.randrange(int(labels.shape[0]))
        signal = raw_epoch[epoch_idx]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal, linewidth=0.8)
        ax.set_title(f"{record.record_id} epoch={epoch_idx} label={int(labels[epoch_idx])} raw='{stage_desc[epoch_idx]}'")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
        fig.tight_layout()
        out_path = output_dir / f"{idx:02d}_{record.record_id}_epoch_{epoch_idx}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        rows.append(
            {
                "record_id": record.record_id,
                "epoch_idx": int(epoch_idx),
                "label": int(labels[epoch_idx]),
                "raw_stage_desc": str(stage_desc[epoch_idx]),
                "plot_path": str(out_path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sleep_edf_5class.yaml")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config))
    processed_dir = Path(args.processed_dir or cfg["processed_dir"])
    output_dir = Path(args.output_dir or (Path(cfg["runs_dir"]) / "_sample_visual_check"))
    rows = save_sample_visual_check(
        processed_dir=processed_dir,
        output_dir=output_dir,
        seed=int(args.seed),
        num_samples=int(args.num_samples),
    )
    print(f"saved sample visual checks: count={len(rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()

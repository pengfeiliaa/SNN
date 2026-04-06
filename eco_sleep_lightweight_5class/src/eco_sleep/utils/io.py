"""常用 I/O 工具函数。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def try_git_commit_hash(cwd: Optional[Path] = None) -> Optional[str]:
    """尝试读取当前 git 提交号，失败则返回 None。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def build_run_dir(runs_dir: Path, exp_name: str) -> Path:
    ensure_dir(runs_dir)
    run_dir = runs_dir / f"{timestamp()}_{exp_name}"
    ensure_dir(run_dir)
    return run_dir


def write_last_run(runs_dir: Path, run_dir: Path) -> None:
    ensure_dir(runs_dir)
    (runs_dir / "last_run.txt").write_text(str(run_dir), encoding="utf-8")


def read_last_run(runs_dir: Path) -> Optional[Path]:
    path = runs_dir / "last_run.txt"
    if path.exists():
        return Path(path.read_text(encoding="utf-8").strip())
    return None


def latest_run(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pending_run_path(runs_dir: Path, exp_name: str) -> Path:
    safe_name = exp_name.replace("/", "_").replace("\\", "_")
    return runs_dir / f"pending_run_{safe_name}.txt"


def write_pending_run(runs_dir: Path, run_dir: Path, exp_name: str) -> None:
    """诊断阶段预创建 run_dir，供训练复用。"""
    ensure_dir(runs_dir)
    _pending_run_path(runs_dir, exp_name).write_text(str(run_dir), encoding="utf-8")


def read_pending_run(runs_dir: Path, exp_name: str) -> Optional[Path]:
    path = _pending_run_path(runs_dir, exp_name)
    if path.exists():
        return Path(path.read_text(encoding="utf-8").strip())
    return None


def clear_pending_run(runs_dir: Path, exp_name: str) -> None:
    path = _pending_run_path(runs_dir, exp_name)
    if path.exists():
        path.unlink(missing_ok=True)


def _empty_csv_frame(default_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    columns = list(default_columns or [])
    if not columns:
        return pd.DataFrame()
    return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})


def _ensure_csv_columns(df: pd.DataFrame, default_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    columns = list(default_columns or [])
    if not columns:
        return df
    if len(df.columns) == 0:
        return _empty_csv_frame(columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    ordered = columns + [col for col in out.columns if col not in columns]
    return out.loc[:, ordered]


def safe_read_csv(
    path: Path,
    default_columns: Optional[Iterable[str]] = None,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    columns = list(default_columns or [])
    if not path.exists():
        return _empty_csv_frame(columns)
    try:
        if path.stat().st_size == 0:
            return _empty_csv_frame(columns)
        if not path.read_text(encoding=encoding).strip():
            return _empty_csv_frame(columns)
        df = pd.read_csv(path, encoding=encoding)
    except pd.errors.EmptyDataError:
        return _empty_csv_frame(columns)
    except Exception:
        return _empty_csv_frame(columns)
    return _ensure_csv_columns(df, columns)

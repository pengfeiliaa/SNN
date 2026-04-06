# -*- coding: utf-8 -*-
"""Compile and import smoke check for scripts/src."""

from __future__ import annotations

from pathlib import Path
from _pathfix import ensure_src_on_path

ensure_src_on_path()

import importlib
import py_compile

from eco_sleep.utils.encoding_fix import setup_utf8_stdio


IMPORT_TARGETS = [
    "eco_sleep",
    "eco_sleep.models",
    "eco_sleep.models.losses",
    "eco_sleep.data.sleep_edf.dataset",
    "eco_sleep.train.checkpoints",
    "preprocess_sleep_edf",
    "train_sleep_edf",
    "eval_sleep_edf",
]


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _compile_files(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"编译失败: {path} -> {exc}")
    return errors


def _import_targets() -> list[str]:
    errors: list[str] = []
    for name in IMPORT_TARGETS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f"导入失败: {name} -> {exc}")
    return errors


def main() -> int:
    setup_utf8_stdio()
    root = Path(__file__).resolve().parents[1]
    files = _iter_python_files(root / "scripts") + _iter_python_files(root / "src")
    errors = _compile_files(files)
    errors.extend(_import_targets())
    if errors:
        for line in errors:
            print(line)
        print(f"compile_check: 失败，共 {len(errors)} 处问题。")
        return 1
    print(f"compile_check: 通过，编译 {len(files)} 个文件，导入 {len(IMPORT_TARGETS)} 个模块。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

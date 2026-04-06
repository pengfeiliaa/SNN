from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_imports() -> None:
    import eco_sleep  # noqa: F401
    from eco_sleep.data.sleep_edf import dataset as edf_dataset  # noqa: F401
    from eco_sleep.models import tiny_tcn  # noqa: F401
    from eco_sleep.utils import metrics_walch2019  # noqa: F401

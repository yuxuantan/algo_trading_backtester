from __future__ import annotations

from pathlib import Path
from typing import Any

from quantbt.artifacts import make_strategy_run_dir, write_json as _write_json


def make_limited_run_dir(
    *,
    base: str | Path = "runs",
    strategy: str,
    dataset_tag: str,
    test_name: str,
    scenario_slug: str | None = None,
) -> Path:
    return make_strategy_run_dir(
        base=base,
        strategy=strategy,
        workflow="limited",
        category=scenario_slug or test_name or "limited_test",
        dataset_tag=dataset_tag,
    )


def write_json(path: str | Path, obj: Any) -> None:
    _write_json(path, obj)

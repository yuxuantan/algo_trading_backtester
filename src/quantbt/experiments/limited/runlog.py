from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_limited_run_dir(
    *,
    base: str | Path = "runs/limited",
    strategy: str,
    dataset_tag: str,
    test_name: str,
) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(base) / strategy
    base.mkdir(parents=True, exist_ok=True)

    safe_ds = dataset_tag.replace("/", "_").replace(" ", "_")
    safe_test = test_name.replace(" ", "_")

    run_dir = base / f"{ts}__ds-{safe_ds}__{safe_test}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

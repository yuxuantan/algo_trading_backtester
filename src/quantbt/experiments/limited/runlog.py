from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "_")
        .lower()
    )


def make_limited_run_dir(
    *,
    base: str | Path = "runs/limited",
    strategy: str,
    dataset_tag: str,
    test_name: str,
) -> Path:
    base = Path(base) / _safe(strategy) / _safe(dataset_tag) / _safe(test_name)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%d%m%y_%H%M%S")
    stem = f"run_{ts}"
    run_dir = base / stem
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{stem}_{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

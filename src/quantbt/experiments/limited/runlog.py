from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4


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
    rid = uuid4().hex[:8]
    base = Path(base) / _safe(strategy) / _safe(dataset_tag) / _safe(test_name)
    base.mkdir(parents=True, exist_ok=True)

    run_dir = base / f"run_{rid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

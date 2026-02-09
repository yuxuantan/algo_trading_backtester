from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str
    file_path: str
    file_sha256: str
    rows: int
    start: str
    end: str
    timeframe: str
    symbol: str
    source: str
    created_at: str
    extra: dict[str, Any] | None = None


def write_dataset_meta(csv_path: str | Path, meta: DatasetMeta) -> Path:
    csv_path = Path(csv_path)
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)
    return meta_path


def read_dataset_meta(csv_path: str | Path) -> dict[str, Any] | None:
    csv_path = Path(csv_path)
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def compute_dataset_meta_from_df(
    *,
    csv_path: str | Path,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> DatasetMeta:
    csv_path = Path(csv_path)
    file_hash = sha256_file(csv_path)

    # assumes datetime index
    start = str(df.index.min())
    end = str(df.index.max())

    # dataset_id is stable for the file content
    dataset_id = f"{symbol}_{timeframe}_{file_hash[:12]}"

    return DatasetMeta(
        dataset_id=dataset_id,
        file_path=str(csv_path),
        file_sha256=file_hash,
        rows=int(len(df)),
        start=start,
        end=end,
        timeframe=timeframe,
        symbol=symbol,
        source=source,
        created_at=datetime.now().isoformat(timespec="seconds"),
        extra=extra,
    )

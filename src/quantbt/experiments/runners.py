from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _safe(s: Any) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "_")
        .lower()
    )


def make_run_dir(
    *,
    base: str = "runs",
    mode: str = "optimize",
    strategy: str,
    dataset_tag: str,
    variant: str | None = None,
    label: str | None = None,
) -> Path:
    ts = datetime.now().strftime("%d%m%y_%H%M%S")
    stem = f"run_{ts}_{_safe(label)}" if label else f"run_{ts}"

    parent_dir = Path(base) / _safe(mode) / _safe(strategy) / _safe(dataset_tag)
    if variant:
        parent_dir = parent_dir / _safe(variant)
    parent_dir.mkdir(parents=True, exist_ok=True)

    run_dir = parent_dir / stem
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = parent_dir / f"{stem}_{suffix:02d}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_json(path: Path, obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_artifacts(
    run_dir: Path,
    *,
    results_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    equity_df: pd.DataFrame | None = None,
    config: Any | None = None,
    summary: Any | None = None,
) -> None:
    if config is not None:
        dump_json(run_dir / "config.json", config)
    if summary is not None:
        dump_json(run_dir / "summary.json", summary)
    if results_df is not None:
        results_df.to_csv(run_dir / "results.csv", index=False)
    if trades_df is not None:
        trades_df.to_csv(run_dir / "best_trades.csv", index=False)
    if equity_df is not None:
        equity_df.to_csv(run_dir / "best_equity_curve.csv")


def append_run_index(
    run_dir: Path,
    *,
    strategy: str,
    optimizer: str,
    dataset: str,
    timeframe: str | None,
    objective: str,
    best_row: dict,
    index_path: str = "runs/_index.csv",
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    """
    Maintains a cumulative runs index you can sort/filter later.
    """
    dataset_meta = dataset_meta or {}
    row = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_path": str(run_dir),
        "strategy": strategy,
        "optimizer": optimizer,
        "dataset": dataset,
        "dataset_id": dataset_meta.get("dataset_id"),
        "dataset_sha256": dataset_meta.get("file_sha256"),
        "timeframe": timeframe or "",
        "objective": objective,
        "best_total_return_%": best_row.get("total_return_%"),
        "best_max_drawdown_%": best_row.get("max_drawdown_%"),
        "best_profit_factor": best_row.get("profit_factor"),
        "best_trades": best_row.get("trades"),
        "best_params": json.dumps({k: best_row[k] for k in best_row.keys()
                                  if k in ("fast", "slow", "rr", "sl_buffer_pips")}, default=str),
    }

    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])
    if p.exists():
        df_old = pd.read_csv(p)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(p, index=False)
    else:
        df_new.to_csv(p, index=False)

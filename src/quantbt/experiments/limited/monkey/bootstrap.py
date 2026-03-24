from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantbt.artifacts import limited_trades_path


def _load_filtered_baseline_trades(run_dir: str | Path, *, iter_id: int | None = None) -> tuple[pd.DataFrame, bool]:
    trades_path = limited_trades_path(run_dir)
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing baseline trades file: {trades_path}")

    trades = pd.read_csv(trades_path)
    filtered = trades
    iter_filtered = False
    if iter_id is not None and "iter" in trades.columns:
        iter_series = pd.to_numeric(trades["iter"], errors="coerce")
        filtered = trades.loc[iter_series == int(iter_id)]
        iter_filtered = True
    return filtered, iter_filtered


def load_baseline_hold_bars_values(run_dir: str | Path, *, iter_id: int | None = None) -> list[int]:
    """Load positive integer bars-held samples from a saved limited baseline run."""

    filtered, iter_filtered = _load_filtered_baseline_trades(run_dir, iter_id=iter_id)
    if "bars_held" not in filtered.columns:
        raise ValueError("Baseline trades file is missing `bars_held`.")

    bars = pd.to_numeric(filtered["bars_held"], errors="coerce").dropna()
    values = [int(v) for v in bars.astype(int).tolist() if int(v) > 0]
    if values:
        return values

    scope = f" iteration {int(iter_id)}" if iter_filtered and iter_id is not None else ""
    raise ValueError(f"No positive `bars_held` samples found for baseline{scope}.")


def load_baseline_fixed_units(run_dir: str | Path, *, iter_id: int | None = None) -> float:
    """Load a robust constant-units benchmark from saved baseline trades.

    Uses the median absolute `units` of the selected baseline iteration so monkey
    runs can be compared with a fixed-units benchmark instead of synthetic stop sizing.
    """

    filtered, iter_filtered = _load_filtered_baseline_trades(run_dir, iter_id=iter_id)
    if "units" not in filtered.columns:
        raise ValueError("Baseline trades file is missing `units`.")

    units = pd.to_numeric(filtered["units"], errors="coerce").dropna().abs()
    values = [float(v) for v in units.tolist() if float(v) > 0]
    if values:
        return float(pd.Series(values, dtype=float).median())

    scope = f" iteration {int(iter_id)}" if iter_filtered and iter_id is not None else ""
    raise ValueError(f"No positive `units` samples found for baseline{scope}.")

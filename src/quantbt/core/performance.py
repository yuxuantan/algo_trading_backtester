from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantbt.core.metrics import max_drawdown


_SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0


def _as_equity_series(equity_like: pd.DataFrame | pd.Series | None) -> pd.Series:
    if equity_like is None:
        return pd.Series(dtype=float)
    if isinstance(equity_like, pd.Series):
        return pd.to_numeric(equity_like, errors="coerce").dropna().astype(float)
    if isinstance(equity_like, pd.DataFrame) and "equity" in equity_like.columns:
        return pd.to_numeric(equity_like["equity"], errors="coerce").dropna().astype(float)
    return pd.Series(dtype=float)


def years_from_index(index: pd.Index | None) -> float:
    if index is None or len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
        return float("nan")
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    idx = idx[~pd.isna(idx)]
    if len(idx) < 2:
        return float("nan")
    span_seconds = float((idx.max() - idx.min()).total_seconds())
    if not math.isfinite(span_seconds) or span_seconds <= 0:
        return float("nan")
    return float(span_seconds / _SECONDS_PER_YEAR)


def periods_per_year_from_index(index: pd.Index | None) -> float:
    if index is None or len(index) < 3 or not isinstance(index, pd.DatetimeIndex):
        return float("nan")
    idx = pd.to_datetime(index, utc=True, errors="coerce")
    idx = idx[~pd.isna(idx)]
    if len(idx) < 3:
        return float("nan")
    diffs = pd.Series(idx).diff().dt.total_seconds()
    diffs = pd.to_numeric(diffs, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float("nan")
    median_seconds = float(diffs.median())
    if not math.isfinite(median_seconds) or median_seconds <= 0:
        return float("nan")
    return float(_SECONDS_PER_YEAR / median_seconds)


def cagr_pct(initial_equity: float, final_equity: float, years: float) -> float:
    try:
        initial = float(initial_equity)
        final = float(final_equity)
        yrs = float(years)
    except Exception:
        return float("nan")
    if not (math.isfinite(initial) and math.isfinite(final) and math.isfinite(yrs)):
        return float("nan")
    if initial <= 0 or final <= 0 or yrs <= 0:
        return float("nan")
    return float((((final / initial) ** (1.0 / yrs)) - 1.0) * 100.0)


def sortino_ratio_from_returns(
    returns: pd.Series | np.ndarray,
    *,
    periods_per_year: float,
    target_return_per_period: float = 0.0,
) -> float:
    r = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return float("nan")

    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0:
        return float("nan")

    target = float(target_return_per_period)
    excess = r - target
    downside = np.minimum(0.0, excess.to_numpy(dtype=float))
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else float("nan")
    mean_excess = float(excess.mean())

    if not math.isfinite(downside_dev) or downside_dev <= 0:
        if mean_excess > 0:
            return float("inf")
        return float("nan")

    return float((mean_excess / downside_dev) * np.sqrt(ppy))


def sortino_ratio_from_equity(equity_like: pd.DataFrame | pd.Series | None) -> float:
    eq = _as_equity_series(equity_like)
    if len(eq) < 3:
        return float("nan")
    ppy = periods_per_year_from_index(eq.index)
    if not math.isfinite(ppy) or ppy <= 0:
        return float("nan")
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return float("nan")
    return sortino_ratio_from_returns(returns, periods_per_year=ppy)


def common_performance_metrics(
    *,
    equity_like: pd.DataFrame | pd.Series | None,
    trades_df: pd.DataFrame | None,
    initial_equity: float,
) -> dict[str, Any]:
    eq = _as_equity_series(equity_like)
    final_equity = float(eq.iloc[-1]) if not eq.empty else float("nan")
    years = years_from_index(eq.index if not eq.empty else None)

    cagr = cagr_pct(float(initial_equity), final_equity, years)
    mdd_frac = abs(max_drawdown(eq)) if not eq.empty else float("nan")
    mdd_abs_pct = float(mdd_frac * 100.0) if math.isfinite(mdd_frac) else float("nan")
    mar = float(cagr / mdd_abs_pct) if math.isfinite(cagr) and math.isfinite(mdd_abs_pct) and mdd_abs_pct > 0 else float("nan")
    sortino = sortino_ratio_from_equity(eq)

    wins = 0
    losses = 0
    win_rate = float("nan")
    if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
        pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not pnl.empty:
            wins = int((pnl > 0).sum())
            losses = int((pnl < 0).sum())
            win_rate = float((pnl > 0).mean() * 100.0)

    return {
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_%": float(win_rate) if math.isfinite(win_rate) else float("nan"),
        "equity_years": float(years) if math.isfinite(years) else float("nan"),
        "cagr_%": float(cagr) if math.isfinite(cagr) else float("nan"),
        "sortino": float(sortino) if math.isfinite(sortino) else (float("inf") if sortino == float("inf") else float("nan")),
        "max_drawdown_abs_%": float(mdd_abs_pct) if math.isfinite(mdd_abs_pct) else float("nan"),
        "mar": float(mar) if math.isfinite(mar) else float("nan"),
    }

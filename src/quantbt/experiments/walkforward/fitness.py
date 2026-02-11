from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown_abs(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return float("nan")
    dd_abs = equity.cummax() - equity
    return float(dd_abs.max())


def required_margin_from_trades(trades_df: pd.DataFrame, *, margin_rate: float) -> float:
    if trades_df is None or trades_df.empty or margin_rate <= 0:
        return 0.0
    if "entry" not in trades_df.columns or "units" not in trades_df.columns:
        return 0.0
    notionals = trades_df["entry"].astype(float).abs() * trades_df["units"].astype(float).abs()
    return float(notionals.max() * float(margin_rate))


def equity_linearity_metrics(equity: pd.Series) -> dict:
    if equity is None or len(equity) < 2:
        return {
            "equity_linearity_r2": float("nan"),
            "equity_linearity_score": float("nan"),
            "equity_slope_per_bar": float("nan"),
        }

    y = equity.astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        r2 = 1.0
    else:
        r2 = max(0.0, 1.0 - (ss_res / ss_tot))

    score = float(r2 if slope >= 0 else -r2)
    return {
        "equity_linearity_r2": float(r2),
        "equity_linearity_score": score,
        "equity_slope_per_bar": float(slope),
    }


def enrich_summary_with_fitness(
    summary: dict,
    *,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_equity: float,
    margin_rate: float = 0.0,
    required_margin_abs_override: float | None = None,
) -> dict:
    out = dict(summary)

    if equity_df is None or equity_df.empty or "equity" not in equity_df.columns:
        out["net_profit_abs"] = float("nan")
        out["max_drawdown_abs"] = float("nan")
        out["max_drawdown_abs_%"] = float("nan")
        out["required_margin_abs"] = float("nan")
        out["return_on_account"] = float("nan")
        out["roa"] = float("nan")
        out.update(equity_linearity_metrics(pd.Series(dtype=float)))
        return out

    equity = equity_df["equity"].astype(float)
    final_equity = float(equity.iloc[-1])
    net_profit_abs = float(final_equity - float(initial_equity))
    mdd_abs = max_drawdown_abs(equity)
    mdd_abs_pct = (mdd_abs / float(initial_equity)) * 100.0 if initial_equity > 0 else float("nan")

    if required_margin_abs_override is None:
        req_margin_abs = required_margin_from_trades(trades_df, margin_rate=margin_rate)
    else:
        req_margin_abs = float(required_margin_abs_override)

    denom = float(mdd_abs + req_margin_abs)
    roa = float(net_profit_abs / denom) if denom > 0 else float("nan")
    linearity = equity_linearity_metrics(equity)

    out["net_profit_abs"] = net_profit_abs
    out["max_drawdown_abs"] = float(mdd_abs)
    out["max_drawdown_abs_%"] = float(mdd_abs_pct)
    out["required_margin_abs"] = float(req_margin_abs)
    out["return_on_account"] = roa
    out["roa"] = roa
    out.update(linearity)

    for k, v in list(out.items()):
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = float("nan")
    return out

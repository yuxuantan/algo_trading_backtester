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


def _finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _safe_mean(values: pd.Series) -> float:
    v = _finite_series(values)
    if v.empty:
        return float("nan")
    return float(v.mean())


def _safe_sum(values: pd.Series) -> float:
    v = _finite_series(values)
    if v.empty:
        return 0.0
    return float(v.sum())


def _safe_count(values: pd.Series) -> int:
    return int(_finite_series(values).shape[0])


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for m in mask:
        if bool(m):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _drawdown_duration_metrics(equity: pd.Series) -> dict:
    if equity is None or len(equity) < 2:
        return {
            "max_drawdown_duration_bars": 0,
            "avg_drawdown_duration_bars": 0.0,
            "max_drawdown_duration_ratio": float("nan"),
        }

    dd_abs = equity.cummax() - equity
    tol = max(abs(float(equity.iloc[0])) * 1e-9, 1e-9)
    in_dd = (dd_abs > tol).to_numpy(dtype=bool)

    max_run = _longest_true_run(in_dd)
    runs: list[int] = []
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)

    avg_run = float(np.mean(runs)) if runs else 0.0
    denom = max(len(in_dd), 1)
    return {
        "max_drawdown_duration_bars": int(max_run),
        "avg_drawdown_duration_bars": float(avg_run),
        "max_drawdown_duration_ratio": float(max_run / denom),
    }


def _flat_period_metrics(equity: pd.Series, *, initial_equity: float) -> dict:
    if equity is None or len(equity) < 2:
        return {
            "flat_bars": 0,
            "flat_bars_ratio": float("nan"),
            "longest_flat_run_bars": 0,
            "longest_flat_run_ratio": float("nan"),
        }

    tol = max(abs(float(initial_equity)) * 1e-6, 1e-8)
    steps = equity.diff().abs().fillna(0.0)
    # Ignore the first bar (always diff=0 by definition).
    flat_mask = (steps.iloc[1:] <= tol).to_numpy(dtype=bool)
    total = max(len(flat_mask), 1)
    flat_bars = int(flat_mask.sum())
    longest = _longest_true_run(flat_mask)
    return {
        "flat_bars": flat_bars,
        "flat_bars_ratio": float(flat_bars / total),
        "longest_flat_run_bars": int(longest),
        "longest_flat_run_ratio": float(longest / total),
    }


def _tharp_expectancy(trades_df: pd.DataFrame) -> float:
    if trades_df is None or trades_df.empty:
        return float("nan")

    if "r_multiple" in trades_df.columns:
        r = _finite_series(trades_df["r_multiple"])
        if not r.empty:
            return float(r.mean())

    if "pnl" not in trades_df.columns:
        return float("nan")

    pnl = _finite_series(trades_df["pnl"])
    if pnl.empty:
        return float("nan")

    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    p_win = float(len(wins) / len(pnl))
    if losses.empty:
        return float("inf") if not wins.empty else float("nan")

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean())
    if avg_loss <= 0:
        return float("nan")
    return float((p_win * (avg_win / avg_loss)) - (1.0 - p_win))


def trade_concentration_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "gross_profit_abs": float("nan"),
            "largest_winning_trade_abs": float("nan"),
            "top_trade_share": float("nan"),
        }

    pnl = _finite_series(trades_df["pnl"])
    wins = pnl[pnl > 0]
    if wins.empty:
        return {
            "gross_profit_abs": 0.0,
            "largest_winning_trade_abs": float("nan"),
            "top_trade_share": float("nan"),
        }

    gross_profit = float(wins.sum())
    largest_win = float(wins.max())
    top_trade_share = float(largest_win / gross_profit) if gross_profit > 0 else float("nan")
    return {
        "gross_profit_abs": gross_profit,
        "largest_winning_trade_abs": largest_win,
        "top_trade_share": top_trade_share,
    }


def build_initial_review_report(
    *,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    aggregated_summary: dict,
    initial_equity: float,
    commission_per_round_trip: float,
    spread_pips: float,
) -> dict:
    equity = (
        pd.Series(dtype=float)
        if equity_df is None or equity_df.empty or "equity" not in equity_df.columns
        else equity_df["equity"].astype(float)
    )
    trades = pd.DataFrame() if trades_df is None else trades_df

    net_profit = float(aggregated_summary.get("net_profit_abs", float("nan")))
    mdd_abs = float(aggregated_summary.get("max_drawdown_abs", float("nan")))
    mdd_pct = float(aggregated_summary.get("max_drawdown_abs_%", float("nan")))
    slope = float(aggregated_summary.get("equity_slope_per_bar", float("nan")))
    r2 = float(aggregated_summary.get("equity_linearity_r2", float("nan")))

    years = float("nan")
    if equity_df is not None and not equity_df.empty and isinstance(equity_df.index, pd.DatetimeIndex):
        span_days = float((equity_df.index.max() - equity_df.index.min()).days)
        years = span_days / 365.25 if span_days > 0 else float("nan")

    annualized_net = float(net_profit / years) if years and np.isfinite(years) and years > 0 else float("nan")

    trade_count = int(aggregated_summary.get("trades", 0))
    avg_trade_net = float(net_profit / trade_count) if trade_count > 0 else float("nan")

    pnl = _finite_series(trades["pnl"]) if "pnl" in trades.columns else pd.Series(dtype=float)
    gains = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    losses = float((-pnl[pnl < 0]).sum()) if not pnl.empty else 0.0
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))

    dd_profit_ratio = float(mdd_abs / net_profit) if np.isfinite(mdd_abs) and net_profit > 0 else float("inf")
    angle_deg = float(np.degrees(np.arctan(slope))) if np.isfinite(slope) else float("nan")

    flat = _flat_period_metrics(equity, initial_equity=initial_equity)
    dd_dur = _drawdown_duration_metrics(equity)
    tharp = _tharp_expectancy(trades)
    commission_total = _safe_sum(trades["commission"]) if "commission" in trades.columns else 0.0
    fuzziness = float(1.0 - r2) if np.isfinite(r2) else float("nan")
    avg_win = _safe_mean(trades.loc[trades["pnl"] > 0, "pnl"]) if "pnl" in trades.columns else float("nan")
    avg_loss = _safe_mean(trades.loc[trades["pnl"] < 0, "pnl"]) if "pnl" in trades.columns else float("nan")
    concentration = trade_concentration_metrics(trades)

    metrics = {
        "total_net_profit_abs": net_profit,
        "annualized_net_profit_abs": annualized_net,
        "profit_factor": pf,
        "avg_trade_net_profit_abs": avg_trade_net,
        "tharp_expectancy": tharp,
        "commission_total_abs": float(commission_total),
        "commission_per_round_trip": float(commission_per_round_trip),
        "spread_pips": float(spread_pips),
        "max_drawdown_abs": mdd_abs,
        "max_drawdown_abs_%": mdd_pct,
        "drawdown_to_net_profit_ratio": dd_profit_ratio,
        "equity_slope_per_bar": slope,
        "equity_slope_angle_deg": angle_deg,
        "equity_linearity_r2": r2,
        "equity_fuzziness": fuzziness,
        "flat_bars": int(flat["flat_bars"]),
        "flat_bars_ratio": float(flat["flat_bars_ratio"]),
        "longest_flat_run_bars": int(flat["longest_flat_run_bars"]),
        "longest_flat_run_ratio": float(flat["longest_flat_run_ratio"]),
        "max_drawdown_duration_bars": int(dd_dur["max_drawdown_duration_bars"]),
        "avg_drawdown_duration_bars": float(dd_dur["avg_drawdown_duration_bars"]),
        "max_drawdown_duration_ratio": float(dd_dur["max_drawdown_duration_ratio"]),
        "avg_win_pnl": avg_win,
        "avg_loss_pnl": avg_loss,
        "top_trade_share": float(concentration.get("top_trade_share", float("nan"))),
        "largest_winning_trade_abs": float(concentration.get("largest_winning_trade_abs", float("nan"))),
        "trade_count": int(trade_count),
        "winning_trades": _safe_count(trades.loc[trades["pnl"] > 0, "pnl"]) if "pnl" in trades.columns else 0,
        "losing_trades": _safe_count(trades.loc[trades["pnl"] < 0, "pnl"]) if "pnl" in trades.columns else 0,
        "equity_years": years,
    }

    thresholds = {
        "annualized_net_profit_abs_min": 10_000.0,
        "profit_factor_min": 1.0,
        "profit_factor_ideal": 1.5,
        "avg_trade_net_profit_abs_min": 50.0,
        "tharp_expectancy_min": 0.10,
        "drawdown_to_net_profit_ratio_max": 1.0,
        "equity_linearity_r2_min": 0.60,
        "flat_bars_ratio_max": 0.35,
        "max_drawdown_duration_ratio_max": 0.35,
        "equity_fuzziness_max": 0.40,
        "spread_pips_max": 2.0,
    }

    checks = {
        "annualized_net_profit_ok": bool(metrics["annualized_net_profit_abs"] >= thresholds["annualized_net_profit_abs_min"]),
        "profit_factor_ok": bool(metrics["profit_factor"] >= thresholds["profit_factor_min"]),
        "avg_trade_net_profit_ok": bool(metrics["avg_trade_net_profit_abs"] >= thresholds["avg_trade_net_profit_abs_min"]),
        "tharp_expectancy_ok": bool(metrics["tharp_expectancy"] >= thresholds["tharp_expectancy_min"]),
        "commission_model_ok": bool(metrics["commission_per_round_trip"] > 0.0),
        "spread_model_ok": bool(0.0 < metrics["spread_pips"] <= thresholds["spread_pips_max"]),
        "drawdown_vs_profit_ok": bool(metrics["drawdown_to_net_profit_ratio"] < thresholds["drawdown_to_net_profit_ratio_max"]),
        "equity_slope_ok": bool(metrics["equity_slope_per_bar"] > 0.0),
        "equity_linearity_ok": bool(metrics["equity_linearity_r2"] >= thresholds["equity_linearity_r2_min"]),
        "equity_flat_period_ok": bool(metrics["flat_bars_ratio"] <= thresholds["flat_bars_ratio_max"]),
        "equity_drawdown_duration_ok": bool(
            metrics["max_drawdown_duration_ratio"] <= thresholds["max_drawdown_duration_ratio_max"]
        ),
        "equity_fuzziness_ok": bool(metrics["equity_fuzziness"] <= thresholds["equity_fuzziness_max"]),
    }
    checks["all_passed"] = bool(all(checks.values()))

    return {
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "notes": {
            "profit_factor_ideal_hint": f">= {thresholds['profit_factor_ideal']}",
            "slippage_commission_guideline": "Reject if commission/spread are zero; prefer realistic non-zero costs.",
            "equity_slope_guideline": "45-degree ideal is treated as positive slope with acceptable linearity.",
        },
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
    out.update(trade_concentration_metrics(trades_df))

    for k, v in list(out.items()):
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = float("nan")
    return out

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from quantbt.core.performance import build_backtest_summary
from quantbt.core.trades import close_trade_with_costs, resolve_intrabar_bracket_exit

@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    risk_pct: float = 0.01
    spread_pips: float = 0.2
    pip_size: float = 0.0001
    conservative_same_bar: bool = True
    min_stop_dist: float = 1e-9  # avoid insane sizing on tiny stops
    commission_per_round_trip: float = 5.0  # dollars per 100k units (standard lot)
    lot_size: float = 100_000.0

def run_backtest_sma_cross(
    df_sig: pd.DataFrame,
    *,
    # Strategy-specific bracket builder injected from strategy module:
    build_brackets_fn,
    strategy_params,
    cfg: BacktestConfig = BacktestConfig(),
):
    """
    Assumes df_sig has columns:
      open, high, low, close, bull_cross, bear_cross
    Enters next bar open after a cross.
    Exits via intrabar SL/TP checks using OHLC.
    """

    df = df_sig.copy().sort_index()
    idx = df.index.to_list()

    equity = cfg.initial_equity
    equity_curve = []
    trades = []
    pos = None

    for i in range(len(df)):
        t = idx[i]
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # ---- Manage open position ----
        if pos is not None:
            side = pos["side"]
            sl = pos["sl"]
            tp = pos["tp"]
            exit_price, exit_reason = resolve_intrabar_bracket_exit(
                side=str(side),
                bar_high=float(h),
                bar_low=float(l),
                sl=float(sl),
                tp=float(tp),
                conservative_same_bar=bool(cfg.conservative_same_bar),
            )

            if exit_price is not None:
                equity, trade = close_trade_with_costs(
                    pos=pos,
                    exit_price=float(exit_price),
                    exit_time=t,
                    exit_reason=str(exit_reason),
                    equity_now=float(equity),
                    cfg=cfg,
                )
                trades.append(trade)
                pos = None

        equity_curve.append({"time": t, "equity": equity})

        # ---- New entry (next bar open) ----
        if pos is None and i < len(df) - 1:
            bull = bool(df.at[t, "bull_cross"])
            bear = bool(df.at[t, "bear_cross"])
            if not (bull or bear):
                continue

            t_next = idx[i + 1]
            entry_open = float(df.at[t_next, "open"])
            prev_low = float(df.at[t, "low"])
            prev_high = float(df.at[t, "high"])

            side = "long" if bull else "short"
            bracket = build_brackets_fn(
                side=side,
                entry_open=entry_open,
                prev_low=prev_low,
                prev_high=prev_high,
                p=strategy_params
            )
            if bracket is None:
                continue

            sl, tp, stop_dist = bracket
            if stop_dist <= cfg.min_stop_dist:
                continue

            risk_dollars = equity * cfg.risk_pct
            units = risk_dollars / stop_dist
            if not np.isfinite(units) or units <= 0:
                continue

            pos = {
                "side": side,
                "entry": entry_open,
                "sl": float(sl),
                "tp": float(tp),
                "units": float(units),
                "entry_time": t_next,
                "risk_dollars": float(risk_dollars),
            }

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    trades_df = pd.DataFrame(trades)

    summary = build_backtest_summary(
        equity_like=equity_df,
        trades_df=trades_df,
        initial_equity=float(cfg.initial_equity),
        include_common_metrics=True,
    )
    return equity_df, trades_df, summary

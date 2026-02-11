from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from quantbt.core.metrics import max_drawdown, profit_factor

@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    risk_pct: float = 0.01
    spread_pips: float = 0.2
    pip_size: float = 0.0001
    conservative_same_bar: bool = True
    min_stop_dist: float = 1e-9  # avoid insane sizing on tiny stops
    commission_per_round_trip: float = 0.0  # dollars per 100k units (standard lot)
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

    spread = cfg.spread_pips * cfg.pip_size

    for i in range(len(df)):
        t = idx[i]
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])

        # ---- Manage open position ----
        if pos is not None:
            side = pos["side"]
            entry = pos["entry"]
            sl = pos["sl"]
            tp = pos["tp"]
            units = pos["units"]

            if side == "long":
                sl_hit = (l <= sl)
                tp_hit = (h >= tp)
            else:
                sl_hit = (h >= sl)
                tp_hit = (l <= tp)

            exit_price = None
            exit_reason = None

            if sl_hit and tp_hit:
                if cfg.conservative_same_bar:
                    exit_price = sl
                    exit_reason = "SL_and_TP_same_bar_assume_SL"
                else:
                    exit_price = tp
                    exit_reason = "SL_and_TP_same_bar_assume_TP"
            elif sl_hit:
                exit_price = sl
                exit_reason = "SL"
            elif tp_hit:
                exit_price = tp
                exit_reason = "TP"

            if exit_price is not None:
                # simplified spread cost on entry+exit
                commission = 0.0
                if cfg.commission_per_round_trip and cfg.lot_size:
                    commission = (units / cfg.lot_size) * cfg.commission_per_round_trip

                if side == "long":
                    entry_eff = entry + spread / 2
                    exit_eff = exit_price - spread / 2
                    pnl = (exit_eff - entry_eff) * units - commission
                else:
                    entry_eff = entry - spread / 2
                    exit_eff = exit_price + spread / 2
                    pnl = (entry_eff - exit_eff) * units - commission

                equity += pnl

                trades.append({
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "side": side,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "units": units,
                    "exit": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "commission": commission,
                    "equity_after": equity,
                    "r_multiple": pnl / pos["risk_dollars"] if pos["risk_dollars"] > 0 else np.nan
                })
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

    total_return = (equity_df["equity"].iloc[-1] / cfg.initial_equity) - 1.0
    mdd = max_drawdown(equity_df["equity"])
    pf = profit_factor(trades_df)
    win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else np.nan
    avg_r = float(trades_df["r_multiple"].mean()) if not trades_df.empty else np.nan

    summary = {
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return_%": float(total_return * 100),
        "max_drawdown_%": float(mdd * 100),
        "win_rate_%": float(win_rate * 100) if np.isfinite(win_rate) else np.nan,
        "profit_factor": pf,
        "avg_R": avg_r
    }
    return equity_df, trades_df, summary

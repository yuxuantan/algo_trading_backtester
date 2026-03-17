from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
from quantbt.core.metrics import max_drawdown
from quantbt.core.performance import common_performance_metrics
from quantbt.core.trades import close_trade_with_costs, resolve_intrabar_bracket_exit


def run_backtest_limited(
    df: pd.DataFrame,
    *,
    cfg,
    entry_iter_fn,
    build_exit_fn,
    exit_params,
    size_fn=None,
):
    """
    Minimal generic backtest:
      - entries from entry_iter_fn(df): yields dicts with keys:
          entry_i, entry_time, side, entry_open, prev_low, prev_high
      - build_exit_fn(side, entry_open, prev_low, prev_high, exit_params) returns:
          {"sl":..., "tp":..., "stop_dist":...} OR {"hold_bars": N}
    Uses same risk sizing (risk_pct) and spread assumptions from cfg.
    """
    df = df.sort_index()
    idx = df.index.to_list()

    equity = float(cfg.initial_equity)
    equity_curve = []
    trades = []

    pos = None

    entries = list(entry_iter_fn(df))
    entry_ptr = 0
    try:
        supports_entry = "entry" in inspect.signature(build_exit_fn).parameters
    except (TypeError, ValueError):
        supports_entry = False

    def _close_trade(
        *,
        pos: dict,
        side: str,
        entry: float,
        units: float,
        exit_price: float,
        exit_reason: str,
        exit_time,
        i: int,
        equity_now: float,
    ) -> tuple[float, dict]:
        equity_after, trade = close_trade_with_costs(
            pos=pos,
            exit_price=float(exit_price),
            exit_time=exit_time,
            exit_reason=str(exit_reason),
            equity_now=float(equity_now),
            cfg=cfg,
        )

        mfe = float(pos.get("mfe", 0.0))
        mae = float(pos.get("mae", 0.0))
        stop_dist = pos.get("stop_dist")
        mfe_r = (mfe / stop_dist) if stop_dist else np.nan
        mae_r = (mae / stop_dist) if stop_dist else np.nan
        mfe_dollars = mfe * units
        if mfe_dollars > 0:
            realized = max(float(trade["pnl"]), 0.0)
            giveback = (mfe_dollars - realized) / mfe_dollars
        else:
            giveback = np.nan

        trade.update({
            "bars_held": int(i - int(pos.get("entry_i", i))),
            # Persist realized bracket levels from the actual limited-test iteration.
            # For non-bracket exits (e.g. time-based), these remain NaN.
            "stop_dist": float(pos["stop_dist"]) if pos.get("stop_dist") is not None else np.nan,
            "mfe": mfe,
            "mae": mae,
            "mfe_R": mfe_r,
            "mae_R": mae_r,
            "giveback": giveback,
        })
        return float(equity_after), trade

    for i in range(len(df)):
        t = idx[i]
        h = float(df.iloc[i]["high"])
        l = float(df.iloc[i]["low"])
        c = float(df.iloc[i]["close"])

        # exit handling
        if pos is not None:
            side = pos["side"]
            entry = pos["entry"]
            units = pos["units"]

            exit_price = None
            exit_reason = None

            # Track MFE/MAE while position is open (price units)
            if side == "long":
                pos["mfe"] = max(pos["mfe"], h - entry)
                pos["mae"] = min(pos["mae"], l - entry)
            else:
                pos["mfe"] = max(pos["mfe"], entry - l)
                pos["mae"] = min(pos["mae"], entry - h)

            if "hold_bars" in pos:
                if i >= pos["exit_i"]:
                    exit_price = c  # exit at close of exit bar
                    exit_reason = "TIME_EXIT"
            else:
                sl = pos["sl"]
                tp = pos["tp"]
                exit_price, exit_reason = resolve_intrabar_bracket_exit(
                    side=str(side),
                    bar_high=float(h),
                    bar_low=float(l),
                    sl=float(sl),
                    tp=float(tp),
                    conservative_same_bar=bool(cfg.conservative_same_bar),
                    same_bar_sl_reason="SL_and_TP_same_bar",
                    same_bar_tp_reason="SL_and_TP_same_bar",
                )

            if exit_price is not None:
                equity, trade = _close_trade(
                    pos=pos,
                    side=side,
                    entry=entry,
                    units=units,
                    exit_price=exit_price,
                    exit_reason=str(exit_reason),
                    exit_time=t,
                    i=i,
                    equity_now=float(equity),
                )
                trades.append(trade)
                pos = None

        equity_curve.append({"time": t, "equity": equity})

        # open next entry if flat
        if pos is None:
            while entry_ptr < len(entries) and entries[entry_ptr]["entry_i"] < i:
                entry_ptr += 1

            if entry_ptr < len(entries) and entries[entry_ptr]["entry_i"] == i:
                e = entries[entry_ptr]
                entry_ptr += 1

                side = e["side"]
                entry_open = float(e["entry_open"])
                prev_low = float(e["prev_low"])
                prev_high = float(e["prev_high"])

                if supports_entry:
                    exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params, entry=e)
                else:
                    exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params)
                if exit_spec is None:
                    continue

                risk_dollars = equity * float(cfg.risk_pct)

                if "hold_bars" in exit_spec:
                    if size_fn is not None:
                        units = size_fn(
                            cfg=cfg,
                            equity=equity,
                            side=side,
                            entry_open=entry_open,
                            exit_spec=exit_spec,
                            entry=e,
                        )
                    else:
                        # time-exit uses a synthetic stop distance just for sizing (optional)
                        # simplest: fixed notional sizing (or skip sizing entirely)
                        # here: size 1 unit risk placeholder
                        units = risk_dollars / (10 * cfg.pip_size)  # heuristic; replace later if you want
                    if units is None or not np.isfinite(units) or units <= 0:
                        continue
                    pos = {
                        "side": side,
                        "entry": entry_open,
                        "entry_i": i,
                        "units": units,
                        "entry_time": e["entry_time"],
                        "hold_bars": int(exit_spec["hold_bars"]),
                        "exit_i": min(len(df) - 1, i + int(exit_spec["hold_bars"])),
                        "mfe": 0.0,
                        "mae": 0.0,
                        "stop_dist": None,
                    }
                else:
                    stop_dist = float(exit_spec["stop_dist"])
                    if stop_dist <= 0:
                        continue
                    if size_fn is not None:
                        units = size_fn(
                            cfg=cfg,
                            equity=equity,
                            side=side,
                            entry_open=entry_open,
                            exit_spec=exit_spec,
                            entry=e,
                        )
                    else:
                        units = risk_dollars / stop_dist
                    if units is None or not np.isfinite(units) or units <= 0:
                        continue

                    pos = {
                        "side": side,
                        "entry": entry_open,
                        "entry_i": i,
                        "sl": float(exit_spec["sl"]),
                        "tp": float(exit_spec["tp"]),
                        "units": units,
                        "entry_time": e["entry_time"],
                        "mfe": 0.0,
                        "mae": 0.0,
                        "stop_dist": stop_dist,
                    }

                    # Evaluate bracket exits on the entry bar itself using wick logic.
                    # If both SL/TP are hit on the same bar, apply the same pessimistic
                    # rule used elsewhere when cfg.conservative_same_bar is enabled.
                    sl = pos["sl"]
                    tp = pos["tp"]
                    if side == "long":
                        pos["mfe"] = max(pos["mfe"], h - entry_open)
                        pos["mae"] = min(pos["mae"], l - entry_open)
                        sl_hit = (l <= sl)
                        tp_hit = (h >= tp)
                    else:
                        pos["mfe"] = max(pos["mfe"], entry_open - l)
                        pos["mae"] = min(pos["mae"], entry_open - h)

                    exit_price, exit_reason = resolve_intrabar_bracket_exit(
                        side=str(side),
                        bar_high=float(h),
                        bar_low=float(l),
                        sl=float(sl),
                        tp=float(tp),
                        conservative_same_bar=bool(cfg.conservative_same_bar),
                        same_bar_sl_reason="SL_and_TP_same_bar",
                        same_bar_tp_reason="SL_and_TP_same_bar",
                    )

                    if exit_price is not None:
                        equity, trade = _close_trade(
                            pos=pos,
                            side=side,
                            entry=float(entry_open),
                            units=float(pos["units"]),
                            exit_price=float(exit_price),
                            exit_reason=str(exit_reason),
                            exit_time=t,
                            i=i,
                            equity_now=float(equity),
                        )
                        trades.append(trade)
                        pos = None
                        # Keep equity curve aligned for the same bar where entry+exit occurred.
                        if equity_curve:
                            equity_curve[-1]["equity"] = equity

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    trades_df = pd.DataFrame(trades)

    total_return = (equity_df["equity"].iloc[-1] / float(cfg.initial_equity) - 1.0) * 100.0
    mdd = max_drawdown(equity_df["equity"])
    mdd_pct = float(mdd * 100.0)
    mdd_abs_pct = float(abs(mdd) * 100.0)
    commission_sum = float(trades_df["commission"].sum()) if "commission" in trades_df.columns else 0.0
    if trades_df.empty:
        win_rate = np.nan
        avg_profit = np.nan
        avg_mfe = np.nan
        avg_mae = np.nan
        avg_mfe_r = np.nan
        avg_mae_r = np.nan
        avg_giveback = np.nan
        avg_bars_held = np.nan
        long_trade_pct = np.nan
        short_trade_pct = np.nan
    else:
        win_rate = float((trades_df["pnl"] > 0).mean()) * 100.0
        avg_profit = float(trades_df["pnl"].mean())
        avg_mfe = float(trades_df["mfe"].mean()) if "mfe" in trades_df.columns else np.nan
        avg_mae = float(trades_df["mae"].mean()) if "mae" in trades_df.columns else np.nan
        avg_mfe_r = float(trades_df["mfe_R"].mean()) if "mfe_R" in trades_df.columns else np.nan
        avg_mae_r = float(trades_df["mae_R"].mean()) if "mae_R" in trades_df.columns else np.nan
        avg_giveback = float(trades_df["giveback"].mean()) if "giveback" in trades_df.columns else np.nan
        avg_bars_held = float(trades_df["bars_held"].mean()) if "bars_held" in trades_df.columns else np.nan
        long_trade_pct = float((trades_df["side"] == "long").mean() * 100.0) if "side" in trades_df.columns else np.nan
        short_trade_pct = float((trades_df["side"] == "short").mean() * 100.0) if "side" in trades_df.columns else np.nan
    summary = {
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "net_profit_abs": float(equity_df["equity"].iloc[-1] - float(cfg.initial_equity)),
        "total_return_%": float(total_return),
        "max_drawdown_%": mdd_pct,
        "max_drawdown_abs_%": mdd_abs_pct,
        "avg_bars_held": avg_bars_held,
        "long_trade_pct": long_trade_pct,
        "short_trade_pct": short_trade_pct,
        "commission_sum": commission_sum,
        "win_rate_%": win_rate,
        "avg_profit_per_trade": avg_profit,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "avg_mfe_R": avg_mfe_r,
        "avg_mae_R": avg_mae_r,
        "avg_giveback": avg_giveback,
    }
    summary.update(
        common_performance_metrics(
            equity_like=equity_df,
            trades_df=trades_df,
            initial_equity=float(cfg.initial_equity),
        )
    )
    return equity_df, trades_df, summary

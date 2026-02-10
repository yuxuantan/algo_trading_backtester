from __future__ import annotations

import inspect
import numpy as np
import pandas as pd


def run_backtest_limited(
    df: pd.DataFrame,
    *,
    cfg,
    entry_iter_fn,
    build_exit_fn,
    exit_params,
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
    spread = float(cfg.spread_pips) * float(cfg.pip_size)

    equity_curve = []
    trades = []

    pos = None

    entries = list(entry_iter_fn(df))
    entry_ptr = 0
    try:
        supports_entry = "entry" in inspect.signature(build_exit_fn).parameters
    except (TypeError, ValueError):
        supports_entry = False

    for i in range(len(df)):
        t = idx[i]
        o = float(df.iloc[i]["open"])
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

            if "hold_bars" in pos:
                if i >= pos["exit_i"]:
                    exit_price = c  # exit at close of exit bar
                    exit_reason = "TIME_EXIT"
            else:
                sl = pos["sl"]
                tp = pos["tp"]

                sl_hit = (l <= sl) if side == "long" else (h >= sl)
                tp_hit = (h >= tp) if side == "long" else (l <= tp)

                if sl_hit and tp_hit:
                    # conservative assume SL
                    exit_price = sl if cfg.conservative_same_bar else tp
                    exit_reason = "SL_and_TP_same_bar"
                elif sl_hit:
                    exit_price = sl
                    exit_reason = "SL"
                elif tp_hit:
                    exit_price = tp
                    exit_reason = "TP"

            if exit_price is not None:
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
                    "exit": exit_price,
                    "exit_reason": exit_reason,
                    "units": units,
                    "pnl": pnl,
                    "commission": commission,
                    "equity_after": equity,
                })
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
                    # time-exit uses a synthetic stop distance just for sizing (optional)
                    # simplest: fixed notional sizing (or skip sizing entirely)
                    # here: size 1 unit risk placeholder
                    units = risk_dollars / (10 * cfg.pip_size)  # heuristic; replace later if you want
                    pos = {
                        "side": side,
                        "entry": entry_open,
                        "units": units,
                        "entry_time": e["entry_time"],
                        "hold_bars": int(exit_spec["hold_bars"]),
                        "exit_i": min(len(df) - 1, i + int(exit_spec["hold_bars"])),
                    }
                else:
                    stop_dist = float(exit_spec["stop_dist"])
                    if stop_dist <= 0:
                        continue
                    units = risk_dollars / stop_dist
                    if not np.isfinite(units) or units <= 0:
                        continue

                    pos = {
                        "side": side,
                        "entry": entry_open,
                        "sl": float(exit_spec["sl"]),
                        "tp": float(exit_spec["tp"]),
                        "units": units,
                        "entry_time": e["entry_time"],
                    }

    equity_df = pd.DataFrame(equity_curve).set_index("time")
    trades_df = pd.DataFrame(trades)

    total_return = (equity_df["equity"].iloc[-1] / float(cfg.initial_equity) - 1.0) * 100.0
    summary = {
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return_%": float(total_return),
    }
    return equity_df, trades_df, summary

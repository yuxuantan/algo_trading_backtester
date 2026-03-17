from __future__ import annotations

import math
from typing import Any

import numpy as np


def _optional_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def close_trade_with_costs(
    *,
    pos: dict[str, Any],
    exit_price: float,
    exit_time: Any,
    exit_reason: str,
    equity_now: float,
    cfg: Any,
) -> tuple[float, dict[str, Any]]:
    spread = float(cfg.spread_pips) * float(cfg.pip_size)
    units = float(pos["units"])
    entry = float(pos["entry"])
    side = str(pos["side"])

    commission = 0.0
    if getattr(cfg, "commission_per_round_trip", 0.0) and getattr(cfg, "lot_size", 0.0):
        commission = (units / float(cfg.lot_size)) * float(cfg.commission_per_round_trip)

    if side == "long":
        entry_eff = entry + spread / 2.0
        exit_eff = float(exit_price) - spread / 2.0
        pnl = (exit_eff - entry_eff) * units - commission
    elif side == "short":
        entry_eff = entry - spread / 2.0
        exit_eff = float(exit_price) + spread / 2.0
        pnl = (entry_eff - exit_eff) * units - commission
    else:
        raise ValueError("side must be 'long' or 'short'")

    equity_after = float(equity_now) + float(pnl)
    trade = {
        "entry_time": pos["entry_time"],
        "exit_time": exit_time,
        "side": side,
        "entry": entry,
        "sl": _optional_float(pos.get("sl")),
        "tp": _optional_float(pos.get("tp")),
        "units": units,
        "exit": float(exit_price),
        "exit_reason": str(exit_reason),
        "pnl": float(pnl),
        "commission": float(commission),
        "equity_after": float(equity_after),
    }

    risk_dollars = _optional_float(pos.get("risk_dollars"))
    if risk_dollars > 0:
        trade["r_multiple"] = float(pnl / risk_dollars)

    return float(equity_after), trade


def resolve_intrabar_bracket_exit(
    *,
    side: str,
    bar_high: float,
    bar_low: float,
    sl: float,
    tp: float,
    conservative_same_bar: bool,
    same_bar_sl_reason: str = "SL_and_TP_same_bar_assume_SL",
    same_bar_tp_reason: str = "SL_and_TP_same_bar_assume_TP",
    sl_reason: str = "SL",
    tp_reason: str = "TP",
) -> tuple[float | None, str | None]:
    if side == "long":
        sl_hit = float(bar_low) <= float(sl)
        tp_hit = float(bar_high) >= float(tp)
    elif side == "short":
        sl_hit = float(bar_high) >= float(sl)
        tp_hit = float(bar_low) <= float(tp)
    else:
        raise ValueError("side must be 'long' or 'short'")

    if sl_hit and tp_hit:
        if conservative_same_bar:
            return float(sl), str(same_bar_sl_reason)
        return float(tp), str(same_bar_tp_reason)
    if sl_hit:
        return float(sl), str(sl_reason)
    if tp_hit:
        return float(tp), str(tp_reason)
    return None, None

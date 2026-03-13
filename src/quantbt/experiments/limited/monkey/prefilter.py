from __future__ import annotations

import inspect

import pandas as pd

from quantbt.experiments.limited.types import EntryEvent, ScheduleMetrics

from .constants import TIME_EXIT_PREFILTER_PLUGIN_NAMES


def load_monkey_match_prefilter_cfg(spec: dict) -> dict | None:
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    raw = test_cfg.get("monkey_match_prefilter")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    try:
        target_trades = float(raw.get("target_trades"))
    except Exception:
        return None
    if target_trades <= 0:
        return None
    cfg = {
        "target_trades": target_trades,
        "target_long_trade_pct": raw.get("target_long_trade_pct"),
        "target_avg_hold_bars": raw.get("target_avg_hold_bars"),
        "trade_tol_pct": float(raw.get("trade_tol_pct", 10.0)),
        "long_tol_pp": float(raw.get("long_tol_pp", 5.0)),
        "hold_tol_pct": float(raw.get("hold_tol_pct", 5.0)),
    }
    return cfg


def time_exit_prefilter_is_supported(exit_name: str, build_exit_fn) -> tuple[bool, bool]:
    if str(exit_name).strip() not in TIME_EXIT_PREFILTER_PLUGIN_NAMES:
        return False, False
    try:
        supports_entry = "entry" in inspect.signature(build_exit_fn).parameters
    except (TypeError, ValueError):
        supports_entry = False
    return True, supports_entry


def simulate_flat_only_time_exit_schedule(
    *,
    entries: list[EntryEvent],
    build_exit_fn,
    exit_params: dict,
    n_bars: int,
    supports_entry_arg: bool,
) -> ScheduleMetrics | None:
    if n_bars <= 0:
        return {"trades": 0.0, "long_trade_pct": float("nan"), "avg_bars_held": float("nan")}

    active_exit_i: int | None = None
    trade_count = 0
    long_count = 0
    hold_bars_vals: list[int] = []

    for e in entries:
        try:
            entry_i = int(e.get("entry_i", -1))
        except Exception:
            continue
        if entry_i < 0 or entry_i >= n_bars:
            continue
        if active_exit_i is not None and entry_i < active_exit_i:
            # Flat-only engine skips entries while a position is open.
            continue

        side = str(e.get("side", "")).strip().lower()
        if side not in {"long", "short"}:
            continue
        try:
            entry_open = float(e["entry_open"])
            prev_low = float(e["prev_low"])
            prev_high = float(e["prev_high"])
        except Exception:
            continue

        if supports_entry_arg:
            exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params, entry=e)
        else:
            exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params)
        if not isinstance(exit_spec, dict) or "hold_bars" not in exit_spec:
            return None
        try:
            hold_bars = int(exit_spec["hold_bars"])
        except Exception:
            return None
        if hold_bars <= 0:
            continue

        exit_i = min(n_bars - 1, entry_i + hold_bars)
        realized_hold = max(0, int(exit_i - entry_i))

        trade_count += 1
        if side == "long":
            long_count += 1
        hold_bars_vals.append(realized_hold)
        active_exit_i = int(exit_i)

    if trade_count <= 0:
        return {"trades": 0.0, "long_trade_pct": float("nan"), "avg_bars_held": float("nan")}

    long_trade_pct = (100.0 * long_count / trade_count)
    avg_bars_held = sum(hold_bars_vals) / trade_count if hold_bars_vals else float("nan")
    return {
        "trades": float(trade_count),
        "long_trade_pct": float(long_trade_pct),
        "avg_bars_held": float(avg_bars_held),
    }


def prefilter_schedule_matches(schedule_metrics: ScheduleMetrics, cfg: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    trades = float(schedule_metrics.get("trades", float("nan")))
    if not pd.notna(trades):
        return False, ["schedule metrics invalid"]
    target_trades = float(cfg["target_trades"])
    trade_tol = abs(float(cfg.get("trade_tol_pct", 10.0))) / 100.0
    trade_lo = target_trades * (1.0 - trade_tol)
    trade_hi = target_trades * (1.0 + trade_tol)
    if not (trade_lo <= trades <= trade_hi):
        reasons.append(f"trades {trades:.0f} not in [{trade_lo:.1f},{trade_hi:.1f}]")

    target_long = cfg.get("target_long_trade_pct")
    if target_long is not None:
        long_pct = float(schedule_metrics.get("long_trade_pct", float("nan")))
        tol_pp = abs(float(cfg.get("long_tol_pp", 5.0)))
        lo = float(target_long) - tol_pp
        hi = float(target_long) + tol_pp
        if not (pd.notna(long_pct) and lo <= long_pct <= hi):
            reasons.append(f"long% {long_pct:.2f} not in [{lo:.2f},{hi:.2f}]")

    target_hold = cfg.get("target_avg_hold_bars")
    if target_hold is not None:
        avg_hold = float(schedule_metrics.get("avg_bars_held", float("nan")))
        hold_tol = abs(float(cfg.get("hold_tol_pct", 5.0))) / 100.0
        lo = float(target_hold) * (1.0 - hold_tol)
        hi = float(target_hold) * (1.0 + hold_tol)
        if not (pd.notna(avg_hold) and lo <= avg_hold <= hi):
            reasons.append(f"avg_hold {avg_hold:.2f} not in [{lo:.2f},{hi:.2f}]")

    return (len(reasons) == 0), reasons

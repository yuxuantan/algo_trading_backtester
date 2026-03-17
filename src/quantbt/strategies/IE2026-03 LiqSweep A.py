from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.core.metrics import max_drawdown, profit_factor


ST_BLACK = 0
ST_PURPLE = 1
ST_RED = 2


@dataclass(frozen=True)
class InterEquityLiqSweepParams:
    # Core tuning knobs (keep this list small to reduce data-snooping risk).
    # - atr_dist_for_liq_generation: equal-high/low proximity tolerance
    # - liq_move_away_atr: confirmation strictness
    # - max_rr: upper reward:risk gate
    max_rr: float = 10.0
    atr_dist_for_liq_generation: float = 1.0
    liq_move_away_atr: float = 3.0

    # Kept mostly fixed (not part of default optimizer grid).
    min_rr: float = 1.0
    max_rr: float = 10.0
    show_levels: bool = True

    # Structural constants
    pivot_len: int = 3
    atr_len: int = 14
    risk_pct: float = 0.01
    sl_buffer_pips: float = 0.5

    # Instrument config
    pip_size: float = 0.0001
    min_tick: float = 1e-5

    # Runtime caps
    max_high_levels: int = 150
    max_low_levels: int = 150

    # Legacy aliases kept for compatibility with older configs. The unified
    # current-timeframe implementation ignores the old LTF/HTF split.
    htf: str | None = None
    show_ltf: bool | None = None
    show_htf: bool | None = None
    ltf_pivot_len: int | None = None
    htf_pivot_len: int | None = None
    max_ltf_h: int | None = None
    max_ltf_l: int | None = None
    max_htf_h: int | None = None
    max_htf_l: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "min_rr", float(self.min_rr))
        object.__setattr__(self, "max_rr", float(self.max_rr))
        object.__setattr__(self, "atr_dist_for_liq_generation", float(self.atr_dist_for_liq_generation))
        object.__setattr__(self, "show_levels", bool(self.show_levels))
        object.__setattr__(self, "pivot_len", int(self.pivot_len))
        object.__setattr__(self, "atr_len", int(self.atr_len))
        object.__setattr__(self, "risk_pct", float(self.risk_pct))
        object.__setattr__(self, "sl_buffer_pips", float(self.sl_buffer_pips))
        object.__setattr__(self, "liq_move_away_atr", float(self.liq_move_away_atr))

        object.__setattr__(self, "pip_size", float(self.pip_size))
        object.__setattr__(self, "min_tick", float(self.min_tick))
        object.__setattr__(self, "max_high_levels", int(self.max_high_levels))
        object.__setattr__(self, "max_low_levels", int(self.max_low_levels))

        if self.show_ltf is not None or self.show_htf is not None:
            legacy_show_levels = bool(self.show_levels)
            if self.show_ltf is not None:
                legacy_show_levels = legacy_show_levels and bool(self.show_ltf)
            if self.show_htf is not None:
                legacy_show_levels = legacy_show_levels and bool(self.show_htf)
            object.__setattr__(self, "show_levels", legacy_show_levels)

        if self.ltf_pivot_len is not None:
            object.__setattr__(self, "pivot_len", int(self.ltf_pivot_len))
        elif self.htf_pivot_len is not None:
            object.__setattr__(self, "pivot_len", int(self.htf_pivot_len))

        if self.max_ltf_h is not None:
            object.__setattr__(self, "max_high_levels", int(self.max_ltf_h))
        elif self.max_htf_h is not None:
            object.__setattr__(self, "max_high_levels", int(self.max_htf_h))

        if self.max_ltf_l is not None:
            object.__setattr__(self, "max_low_levels", int(self.max_ltf_l))
        elif self.max_htf_l is not None:
            object.__setattr__(self, "max_low_levels", int(self.max_htf_l))

        if self.min_rr < 0 or self.max_rr < 0:
            raise ValueError("min_rr/max_rr must be >= 0")
        if self.max_rr <= self.min_rr:
            raise ValueError(f"max_rr must be > min_rr. Got min_rr={self.min_rr}, max_rr={self.max_rr}")
        if self.atr_dist_for_liq_generation < 0:
            raise ValueError("atr_dist_for_liq_generation must be >= 0")
        if self.liq_move_away_atr <= 0:
            raise ValueError("liq_move_away_atr must be > 0")
        if self.pivot_len <= 0:
            raise ValueError("pivot_len must be > 0")
        if self.atr_len <= 0:
            raise ValueError("atr_len must be > 0")
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be > 0")
        if self.pip_size <= 0 or self.min_tick <= 0:
            raise ValueError("pip_size and min_tick must be > 0")


Params = InterEquityLiqSweepParams


@dataclass
class _LevelPool:
    max_size: int
    lvls: list[float] = field(default_factory=list)
    line_ids: list[int] = field(default_factory=list)
    next_line_id: int = 0
    act: list[bool] = field(default_factory=list)
    breach_time: list[pd.Timestamp | None] = field(default_factory=list)
    drawn_act: list[bool] = field(default_factory=list)
    state: list[int] = field(default_factory=list)

    pend_a: list[int] = field(default_factory=list)
    pend_b: list[int] = field(default_factory=list)
    pend_trigger: list[float] = field(default_factory=list)
    pend_ok: list[bool] = field(default_factory=list)

    def trim(
        self,
        *,
        pool_name: str | None = None,
        event_time: pd.Timestamp | None = None,
        line_events: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(self.lvls) <= self.max_size:
            return

        old_lvl = float(self.lvls[0])
        old_line_id = int(self.line_ids[0])
        old_drawn_act = bool(self.drawn_act[0])

        self.lvls.pop(0)
        self.line_ids.pop(0)
        self.act.pop(0)
        self.breach_time.pop(0)
        self.drawn_act.pop(0)
        self.state.pop(0)

        if (
            old_drawn_act
            and pool_name is not None
            and event_time is not None
            and line_events is not None
        ):
            line_events.append(
                {
                    "type": "line_deactivated",
                    "pool": pool_name,
                    "line_id": old_line_id,
                    "time": pd.Timestamp(event_time),
                    "level": old_lvl,
                }
            )

        for k in range(len(self.pend_a) - 1, -1, -1):
            a = self.pend_a[k]
            b = self.pend_b[k]
            if a == 0 or b == 0:
                self.pend_a.pop(k)
                self.pend_b.pop(k)
                self.pend_trigger.pop(k)
                self.pend_ok.pop(k)
            else:
                self.pend_a[k] = a - 1
                self.pend_b[k] = b - 1


def _true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None or math.isnan(prev_close):
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _pivot_high(values: list[float], center: int, left: int, right: int) -> float | None:
    if center - left < 0 or center + right >= len(values):
        return None
    v = values[center]
    for j in range(center - left, center + right + 1):
        if j == center:
            continue
        if values[j] >= v:
            return None
    return v


def _pivot_low(values: list[float], center: int, left: int, right: int) -> float | None:
    if center - left < 0 or center + right >= len(values):
        return None
    v = values[center]
    for j in range(center - left, center + right + 1):
        if j == center:
            continue
        if values[j] <= v:
            return None
    return v


def _both_active(act_arr: list[bool], a: int, b: int) -> bool:
    return (
        a >= 0
        and b >= 0
        and a < len(act_arr)
        and b < len(act_arr)
        and act_arr[a]
        and act_arr[b]
    )


def _track_breach_high(pool: _LevelPool, high_val: float, breach_t: pd.Timestamp) -> None:
    for i in range(len(pool.lvls)):
        if pool.act[i] and high_val > pool.lvls[i]:
            pool.act[i] = False
            pool.breach_time[i] = breach_t


def _track_breach_low(pool: _LevelPool, low_val: float, breach_t: pd.Timestamp) -> None:
    for i in range(len(pool.lvls)):
        if pool.act[i] and low_val < pool.lvls[i]:
            pool.act[i] = False
            pool.breach_time[i] = breach_t


def _append_high_pivot(
    *,
    pool: _LevelPool,
    pool_name: str,
    pivot_value: float,
    pivot_time: pd.Timestamp,
    event_time: pd.Timestamp,
    pivot_atr: float,
    atr_dist_for_liq_generation: float,
    liq_move_away_atr: float,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    swept_earlier = any(bt is not None and bt == pivot_time for bt in pool.breach_time)

    match: int | None = None
    match_lvl: float | None = None
    for j in range(len(pool.lvls) - 1, -1, -1):
        if not pool.act[j]:
            continue
        prev = pool.lvls[j]
        if abs(pivot_value - prev) <= (atr_dist_for_liq_generation * pivot_atr) and pivot_value <= prev:
            match = j
            match_lvl = prev
            break

    base_state = ST_PURPLE if swept_earlier else ST_BLACK

    line_id = pool.next_line_id
    pool.next_line_id += 1
    curr_idx = len(pool.lvls)
    pool.lvls.append(float(pivot_value))
    pool.line_ids.append(line_id)
    pool.act.append(True)
    pool.breach_time.append(None)
    pool.drawn_act.append(True)
    pool.state.append(base_state)

    if line_events is not None:
        line_events.append(
            {
                "type": "line_created",
                "pool": pool_name,
                "line_id": int(line_id),
                "time": pd.Timestamp(pivot_time),
                "confirm_time": pd.Timestamp(event_time),
                "level": float(pivot_value),
                "state": int(base_state),
            }
        )

    if match is not None and match_lvl is not None:
        pair_high = max(float(pivot_value), float(match_lvl))
        trigger_down = pair_high - (liq_move_away_atr * pivot_atr)
        pool.pend_a.append(match)
        pool.pend_b.append(curr_idx)
        pool.pend_trigger.append(trigger_down)
        pool.pend_ok.append(False)

    pool.trim(
        pool_name=pool_name,
        event_time=event_time,
        line_events=line_events,
    )


def _append_low_pivot(
    *,
    pool: _LevelPool,
    pool_name: str,
    pivot_value: float,
    pivot_time: pd.Timestamp,
    event_time: pd.Timestamp,
    pivot_atr: float,
    atr_dist_for_liq_generation: float,
    liq_move_away_atr: float,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    swept_earlier = any(bt is not None and bt == pivot_time for bt in pool.breach_time)

    match: int | None = None
    match_lvl: float | None = None
    for j in range(len(pool.lvls) - 1, -1, -1):
        if not pool.act[j]:
            continue
        prev = pool.lvls[j]
        if abs(pivot_value - prev) <= (atr_dist_for_liq_generation * pivot_atr) and pivot_value >= prev:
            match = j
            match_lvl = prev
            break

    base_state = ST_PURPLE if swept_earlier else ST_BLACK

    line_id = pool.next_line_id
    pool.next_line_id += 1
    curr_idx = len(pool.lvls)
    pool.lvls.append(float(pivot_value))
    pool.line_ids.append(line_id)
    pool.act.append(True)
    pool.breach_time.append(None)
    pool.drawn_act.append(True)
    pool.state.append(base_state)

    if line_events is not None:
        line_events.append(
            {
                "type": "line_created",
                "pool": pool_name,
                "line_id": int(line_id),
                "time": pd.Timestamp(pivot_time),
                "confirm_time": pd.Timestamp(event_time),
                "level": float(pivot_value),
                "state": int(base_state),
            }
        )

    if match is not None and match_lvl is not None:
        pair_low = min(float(pivot_value), float(match_lvl))
        trigger_up = pair_low + (liq_move_away_atr * pivot_atr)
        pool.pend_a.append(match)
        pool.pend_b.append(curr_idx)
        pool.pend_trigger.append(trigger_up)
        pool.pend_ok.append(False)

    pool.trim(
        pool_name=pool_name,
        event_time=event_time,
        line_events=line_events,
    )


def _confirm_move_away_high(
    pool: _LevelPool,
    close_val: float,
    *,
    pool_name: str,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    for k in range(len(pool.pend_a)):
        if pool.pend_ok[k]:
            continue
        trigger_down = pool.pend_trigger[k]
        if close_val <= trigger_down:
            a = pool.pend_a[k]
            b = pool.pend_b[k]
            if _both_active(pool.act, a, b):
                if pool.state[a] != ST_RED:
                    pool.state[a] = ST_RED
                    if line_events is not None:
                        line_events.append(
                            {
                                "type": "line_promoted_red",
                                "pool": pool_name,
                                "line_id": int(pool.line_ids[a]),
                                "time": pd.Timestamp(event_time),
                                "level": float(pool.lvls[a]),
                            }
                        )
                if pool.state[b] != ST_RED:
                    pool.state[b] = ST_RED
                    if line_events is not None:
                        line_events.append(
                            {
                                "type": "line_promoted_red",
                                "pool": pool_name,
                                "line_id": int(pool.line_ids[b]),
                                "time": pd.Timestamp(event_time),
                                "level": float(pool.lvls[b]),
                            }
                        )
                pool.pend_ok[k] = True
                break


def _confirm_move_away_low(
    pool: _LevelPool,
    close_val: float,
    *,
    pool_name: str,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    for k in range(len(pool.pend_a)):
        if pool.pend_ok[k]:
            continue
        trigger_up = pool.pend_trigger[k]
        if close_val >= trigger_up:
            a = pool.pend_a[k]
            b = pool.pend_b[k]
            if _both_active(pool.act, a, b):
                if pool.state[a] != ST_RED:
                    pool.state[a] = ST_RED
                    if line_events is not None:
                        line_events.append(
                            {
                                "type": "line_promoted_red",
                                "pool": pool_name,
                                "line_id": int(pool.line_ids[a]),
                                "time": pd.Timestamp(event_time),
                                "level": float(pool.lvls[a]),
                            }
                        )
                if pool.state[b] != ST_RED:
                    pool.state[b] = ST_RED
                    if line_events is not None:
                        line_events.append(
                            {
                                "type": "line_promoted_red",
                                "pool": pool_name,
                                "line_id": int(pool.line_ids[b]),
                                "time": pd.Timestamp(event_time),
                                "level": float(pool.lvls[b]),
                            }
                        )
                pool.pend_ok[k] = True
                break


def _stop_extending_high(
    pool: _LevelPool,
    high_val: float,
    *,
    pool_name: str,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    for i in range(len(pool.lvls)):
        if pool.drawn_act[i] and high_val > pool.lvls[i]:
            pool.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": pool_name,
                        "line_id": int(pool.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(pool.lvls[i]),
                    }
                )


def _stop_extending_low(
    pool: _LevelPool,
    low_val: float,
    *,
    pool_name: str,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> None:
    for i in range(len(pool.lvls)):
        if pool.drawn_act[i] and low_val < pool.lvls[i]:
            pool.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": pool_name,
                        "line_id": int(pool.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(pool.lvls[i]),
                    }
                )


def _sweep_triggers(
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    high_val: float,
    low_val: float,
    *,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool]:
    trig_short = False
    trig_long = False

    for i in range(len(high_pool.lvls)):
        if high_pool.drawn_act[i] and high_val > high_pool.lvls[i]:
            if high_pool.state[i] == ST_RED:
                trig_short = True
            high_pool.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": "high",
                        "line_id": int(high_pool.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(high_pool.lvls[i]),
                    }
                )

    for i in range(len(low_pool.lvls)):
        if low_pool.drawn_act[i] and low_val < low_pool.lvls[i]:
            if low_pool.state[i] == ST_RED:
                trig_long = True
            low_pool.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": "low",
                        "line_id": int(low_pool.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(low_pool.lvls[i]),
                    }
                )

    return trig_short, trig_long


def _next_purple_high_above(price: float, high_pool: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(high_pool.lvls):
        if high_pool.drawn_act[i] and high_pool.state[i] == ST_PURPLE and lvl > price:
            best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_purple_low_below(price: float, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(low_pool.lvls):
        if low_pool.drawn_act[i] and low_pool.state[i] == ST_PURPLE and lvl < price:
            best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_red_above(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl > price:
                best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_red_below(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl < price:
                best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_state_above_non_black(
    price: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            st = pool.state[i]
            if pool.drawn_act[i] and st != ST_BLACK and lvl > price:
                if math.isnan(best_lvl) or lvl < best_lvl:
                    best_lvl = lvl
                    best_state = st
    return best_state


def _next_state_below_non_black(
    price: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            st = pool.state[i]
            if pool.drawn_act[i] and st != ST_BLACK and lvl < price:
                if math.isnan(best_lvl) or lvl > best_lvl:
                    best_lvl = lvl
                    best_state = st
    return best_state


def _any_red_between(
    p1: float,
    p2: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
) -> bool:
    lo = min(p1, p2)
    hi = max(p1, p2)
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lo < lvl < hi:
                return True
    return False


def compute_features(df: pd.DataFrame, p: InterEquityLiqSweepParams) -> pd.DataFrame:
    out = df.copy().sort_index()
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"Missing required OHLC column: {c}")
        out[c] = out[c].astype(float)
    return out


def compute_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    # This strategy uses a custom state-machine backtest (run_backtest),
    # but we keep these columns for compatibility with the shared interface.
    out = df_feat.copy()
    out["bull_cross"] = False
    out["bear_cross"] = False
    return out


def build_brackets_from_signal(
    side: str,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    p: InterEquityLiqSweepParams,
):
    # Fallback bracket builder for compatibility with generic engine.
    sl_buffer = float(p.sl_buffer_pips) * float(p.pip_size)
    rr = max(float(p.min_rr), 0.1)

    if side == "long":
        sl = float(prev_low) - sl_buffer
        stop_dist = float(entry_open) - sl
        if stop_dist <= 0:
            return None
        tp = float(entry_open) + rr * stop_dist
        return sl, tp, stop_dist

    if side == "short":
        sl = float(prev_high) + sl_buffer
        stop_dist = sl - float(entry_open)
        if stop_dist <= 0:
            return None
        tp = float(entry_open) - rr * stop_dist
        return sl, tp, stop_dist

    raise ValueError("side must be 'long' or 'short'")


def _close_position(
    *,
    pos: dict,
    exit_price: float,
    exit_time: pd.Timestamp,
    exit_reason: str,
    equity: float,
    cfg: BacktestConfig,
) -> tuple[float, dict]:
    spread = float(cfg.spread_pips) * float(cfg.pip_size)
    commission = 0.0
    units = float(pos["units"])

    if cfg.commission_per_round_trip and cfg.lot_size:
        commission = (units / float(cfg.lot_size)) * float(cfg.commission_per_round_trip)

    entry = float(pos["entry"])
    side = str(pos["side"])

    if side == "long":
        entry_eff = entry + spread / 2.0
        exit_eff = float(exit_price) - spread / 2.0
        pnl = (exit_eff - entry_eff) * units - commission
    else:
        entry_eff = entry - spread / 2.0
        exit_eff = float(exit_price) + spread / 2.0
        pnl = (entry_eff - exit_eff) * units - commission

    equity_after = equity + pnl

    trade = {
        "entry_time": pos["entry_time"],
        "exit_time": exit_time,
        "side": side,
        "entry": entry,
        "sl": float(pos["sl"]),
        "tp": float(pos["tp"]),
        "units": units,
        "exit": float(exit_price),
        "exit_reason": exit_reason,
        "pnl": float(pnl),
        "commission": float(commission),
        "equity_after": float(equity_after),
        "r_multiple": float(pnl / pos["risk_dollars"]) if float(pos["risk_dollars"]) > 0 else np.nan,
    }
    return float(equity_after), trade


def run_backtest(
    df_sig: pd.DataFrame,
    *,
    strategy_params: InterEquityLiqSweepParams,
    cfg: BacktestConfig = BacktestConfig(),
    debug: dict[str, Any] | None = None,
):
    p = strategy_params
    df = df_sig.copy().sort_index()
    idx = df.index.to_list()

    equity = float(cfg.initial_equity)
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    pos: dict[str, Any] | None = None
    pending_entry: dict[str, Any] | None = None
    pending_market_exit: dict[str, Any] | None = None
    line_events: list[dict[str, Any]] = []

    high_pool = _LevelPool(max_size=p.max_high_levels)
    low_pool = _LevelPool(max_size=p.max_low_levels)

    high_hist: list[float] = []
    low_hist: list[float] = []
    time_hist: list[pd.Timestamp] = []
    atr_hist: list[float] = []

    prev_close: float | None = None
    prev_atr: float | None = None

    for i, t in enumerate(idx):
        o = float(df.at[t, "open"])
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])
        c = float(df.at[t, "close"])

        # ---- Execute delayed market close at next bar open ----
        if pos is not None and pending_market_exit is not None and int(pending_market_exit["entry_i"]) == i:
            equity, trade = _close_position(
                pos=pos,
                exit_price=o,
                exit_time=t,
                exit_reason=str(pending_market_exit["reason"]),
                equity=equity,
                cfg=cfg,
            )
            trades.append(trade)
            pos = None
            pending_market_exit = None

        # ---- Fill delayed entry at next bar open ----
        if pos is None and pending_entry is not None and int(pending_entry["entry_i"]) == i:
            pos = {
                "side": pending_entry["side"],
                "entry": o,
                "sl": float(pending_entry["sl"]),
                "tp": float(pending_entry["tp"]),
                "units": float(pending_entry["qty"]),
                "entry_time": t,
                "risk_dollars": float(pending_entry["risk_dollars"]),
            }
            pending_entry = None

        # ---- Manage open position via intrabar SL/TP ----
        if pos is not None:
            side = str(pos["side"])
            sl = float(pos["sl"])
            tp = float(pos["tp"])

            if side == "long":
                sl_hit = l <= sl
                tp_hit = h >= tp
            else:
                sl_hit = h >= sl
                tp_hit = l <= tp

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
                equity, trade = _close_position(
                    pos=pos,
                    exit_price=float(exit_price),
                    exit_time=t,
                    exit_reason=str(exit_reason),
                    equity=equity,
                    cfg=cfg,
                )
                trades.append(trade)
                pos = None
                pending_market_exit = None

        equity_curve.append({"time": t, "equity": equity})

        tr = _true_range(h, l, prev_close)
        atr = tr if prev_atr is None else ((prev_atr * (p.atr_len - 1)) + tr) / p.atr_len
        prev_close = c
        prev_atr = atr

        high_hist.append(h)
        low_hist.append(l)
        time_hist.append(t)
        atr_hist.append(float(atr))

        pivot_high = None
        pivot_low = None
        pivot_time = None
        pivot_atr = None

        center_idx = len(high_hist) - 1 - p.pivot_len
        if center_idx >= p.pivot_len:
            pivot_high = _pivot_high(high_hist, center_idx, p.pivot_len, p.pivot_len)
            pivot_low = _pivot_low(low_hist, center_idx, p.pivot_len, p.pivot_len)
            pivot_time = time_hist[center_idx]
            pivot_atr = float(atr_hist[center_idx])

        trig_short = False
        trig_long = False

        if p.show_levels:
            _track_breach_high(high_pool, h, t)
            _track_breach_low(low_pool, l, t)

            if pivot_high is not None and pivot_time is not None and pivot_atr is not None:
                _append_high_pivot(
                    pool=high_pool,
                    pool_name="high",
                    pivot_value=float(pivot_high),
                    pivot_time=pivot_time,
                    event_time=t,
                    pivot_atr=float(pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                    line_events=line_events,
                )

            if pivot_low is not None and pivot_time is not None and pivot_atr is not None:
                _append_low_pivot(
                    pool=low_pool,
                    pool_name="low",
                    pivot_value=float(pivot_low),
                    pivot_time=pivot_time,
                    event_time=t,
                    pivot_atr=float(pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                    line_events=line_events,
                )

            _confirm_move_away_high(
                high_pool,
                c,
                pool_name="high",
                event_time=t,
                line_events=line_events,
            )
            _confirm_move_away_low(
                low_pool,
                c,
                pool_name="low",
                event_time=t,
                line_events=line_events,
            )

            trig_short, trig_long = _sweep_triggers(
                high_pool,
                low_pool,
                h,
                l,
                event_time=t,
                line_events=line_events,
            )
            _stop_extending_high(
                high_pool,
                h,
                pool_name="high",
                event_time=t,
                line_events=line_events,
            )
            _stop_extending_low(
                low_pool,
                l,
                pool_name="low",
                event_time=t,
                line_events=line_events,
            )

        # ---- Strategy entries (signal bar -> enter next bar open) ----
        flat = pos is None
        entry_px = c
        rr_band_valid = p.max_rr > p.min_rr
        sl_buffer = p.sl_buffer_pips * p.pip_size

        if flat and trig_short and i < len(df) - 1:
            sl_raw = _next_purple_high_above(entry_px, high_pool)
            sl = (sl_raw + sl_buffer) if sl_raw is not None else None
            tp = _next_red_below(entry_px, high_pool, low_pool)

            ok_sl = sl is not None and sl > entry_px
            ok_tp = tp is not None and tp < entry_px

            risk_dist = (sl - entry_px) if ok_sl else math.nan
            reward_dist = (entry_px - tp) if ok_tp and tp is not None else math.nan
            rr = (reward_dist / risk_dist) if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)) else math.nan
            ok_rr = rr_band_valid and math.isfinite(rr) and rr > p.min_rr and rr <= p.max_rr

            has_red_between = True
            if ok_sl and sl is not None:
                has_red_between = _any_red_between(entry_px, sl, high_pool, low_pool)

            if ok_sl and ok_tp and ok_rr and not has_red_between and sl is not None and tp is not None:
                risk_amount = float(cfg.initial_equity) * p.risk_pct
                stop_dist = max(sl - entry_px, p.min_tick)
                qty = risk_amount / stop_dist
                if math.isfinite(qty) and qty > 0:
                    pending_entry = {
                        "entry_i": i + 1,
                        "side": "short",
                        "qty": float(qty),
                        "sl": float(sl),
                        "tp": float(tp),
                        "risk_dollars": float(risk_amount),
                    }

        if flat and trig_long and i < len(df) - 1:
            sl_raw = _next_purple_low_below(entry_px, low_pool)
            sl = (sl_raw - sl_buffer) if sl_raw is not None else None
            tp = _next_red_above(entry_px, high_pool, low_pool)

            ok_sl = sl is not None and sl < entry_px
            ok_tp = tp is not None and tp > entry_px

            risk_dist = (entry_px - sl) if ok_sl else math.nan
            reward_dist = (tp - entry_px) if ok_tp and tp is not None else math.nan
            rr = (reward_dist / risk_dist) if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)) else math.nan
            ok_rr = rr_band_valid and math.isfinite(rr) and rr > p.min_rr and rr <= p.max_rr

            has_red_between = True
            if ok_sl and sl is not None:
                has_red_between = _any_red_between(entry_px, sl, high_pool, low_pool)

            if ok_sl and ok_tp and ok_rr and not has_red_between and sl is not None and tp is not None:
                risk_amount = float(cfg.initial_equity) * p.risk_pct
                stop_dist = max(entry_px - sl, p.min_tick)
                qty = risk_amount / stop_dist
                if math.isfinite(qty) and qty > 0:
                    pending_entry = {
                        "entry_i": i + 1,
                        "side": "long",
                        "qty": float(qty),
                        "sl": float(sl),
                        "tp": float(tp),
                        "risk_dollars": float(risk_amount),
                    }

        # ---- Directional force-exit: ignore black lines ----
        if pos is not None and pending_market_exit is None and i < len(df) - 1:
            if str(pos["side"]) == "long":
                next_state = _next_state_above_non_black(entry_px, high_pool, low_pool)
                if next_state == ST_PURPLE:
                    pending_market_exit = {
                        "entry_i": i + 1,
                        "reason": "Nearest above is purple",
                    }
            else:
                next_state = _next_state_below_non_black(entry_px, high_pool, low_pool)
                if next_state == ST_PURPLE:
                    pending_market_exit = {
                        "entry_i": i + 1,
                        "reason": "Nearest below is purple",
                    }

    equity_df = pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame(columns=["equity"])
    trades_df = pd.DataFrame(trades)

    if equity_df.empty:
        summary = {
            "trades": 0,
            "final_equity": float(cfg.initial_equity),
            "total_return_%": 0.0,
            "max_drawdown_%": 0.0,
            "win_rate_%": np.nan,
            "profit_factor": np.nan,
            "avg_R": np.nan,
        }
        if debug is not None:
            debug["line_events"] = line_events
            debug["line_event_count"] = len(line_events)
        return equity_df, trades_df, summary

    total_return = (float(equity_df["equity"].iloc[-1]) / float(cfg.initial_equity)) - 1.0
    mdd = max_drawdown(equity_df["equity"]) if len(equity_df) else 0.0
    pf = profit_factor(trades_df)
    win_rate = float((trades_df["pnl"] > 0).mean()) if not trades_df.empty else np.nan
    avg_r = float(trades_df["r_multiple"].mean()) if not trades_df.empty else np.nan

    summary = {
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return_%": float(total_return * 100.0),
        "max_drawdown_%": float(mdd * 100.0),
        "win_rate_%": float(win_rate * 100.0) if np.isfinite(win_rate) else np.nan,
        "profit_factor": pf,
        "avg_R": avg_r,
    }
    if debug is not None:
        debug["line_events"] = line_events
        debug["line_event_count"] = len(line_events)
    return equity_df, trades_df, summary


STRATEGY = {
    "name": "IE2026-03 LiqSweep A",
    "entry": {
        "mode": "all",
        "rules": [
            {
                "name": "interequity_liqsweep_entry",
                "params": {
                    "min_rr": 1.0,
                    "max_rr": 10.0,
                    "atr_dist_for_liq_generation": 1.0,
                    "liq_move_away_atr": 3.0,
                },
            }
        ],
    },
    "exit": {
        "name": "interequity_liqsweep_exit",
        "params": {"fallback_rr": 1.5, "sl_buffer_pips": 0.5, "pip_size": 0.0001},
    },
    "sizing": {
        "name": "fixed_risk",
        "params": {"risk_pct": 0.01},
    },
    "limited_test": {
        "entry": {
            "optimizable": {
                "atr_dist_for_liq_generation": {
                    "label": "ATR distance for liquidity generation",
                    "default": 1.0,
                    "start": 0.4,
                    "stop": 2.2,
                    "step": 0.2,
                    "values": [0.8, 1.0, 1.2],
                    "integer": False,
                },
                "liq_move_away_atr": {
                    "label": "Liquidity move-away ATR",
                    "default": 3.0,
                    "start": 1.0,
                    "stop": 5.5,
                    "step": 0.5,
                    "values": [2.5, 3.0],
                    "integer": False,
                },
                "max_rr": {
                    "label": "Maximum RR",
                    "default": 10.0,
                    "start": 4.0,
                    "stop": 8.0,
                    "step": 2.0,
                    "values": [4.0, 6.0, 8.0],
                    "integer": False,
                }
            },
            "non_optimizable": [
                "min_rr"
            ],
        },
        "exit": {
            "optimizable": {},
            "non_optimizable": [
                "fallback_rr",
                "sl_buffer_pips",
                "pip_size",
            ],
        },
    },
}


def _coerce_param_space_value(raw_value: Any, *, integer: bool) -> int | float:
    value = int(raw_value) if integer else float(raw_value)
    if not math.isfinite(float(value)):
        raise ValueError(f"param-space value must be finite; got {raw_value!r}")
    return value


def _build_param_space_values(spec: dict[str, Any], *, default_value: Any = None) -> list[int] | list[float]:
    integer = bool(spec.get("integer", isinstance(default_value, int) and not isinstance(default_value, bool)))
    raw_values = spec.get("values")
    if isinstance(raw_values, list) and raw_values:
        values: list[int] | list[float] = []
        seen: set[int | float] = set()
        for raw_value in raw_values:
            value = _coerce_param_space_value(raw_value, integer=integer)
            dedupe_key = int(value) if integer else round(float(value), 12)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append(value)
        if values:
            return values

    raw_end = spec.get("end", spec.get("stop"))
    if all(name in spec for name in ("start", "step")) and raw_end is not None:
        start = float(spec["start"])
        stop = float(raw_end)
        step = float(spec["step"])
        if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step) or step <= 0:
            raise ValueError(f"invalid numeric param-space range: {spec!r}")
        values = []
        current = start
        tolerance = max(abs(step) / 1000.0, 1e-12)
        while current <= stop + tolerance:
            values.append(int(round(current)) if integer else round(current, 12))
            current += step
        if values:
            return values

    if "default" in spec:
        return [_coerce_param_space_value(spec["default"], integer=integer)]
    if default_value is not None:
        return [_coerce_param_space_value(default_value, integer=integer)]
    return []


def _build_param_space_from_limited_test(strategy_cfg: dict[str, Any]) -> dict[str, list[int] | list[float]]:
    limited_cfg = strategy_cfg.get("limited_test", {}) if isinstance(strategy_cfg, dict) else {}
    entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
    exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}
    entry_rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    entry_defaults = (
        dict(entry_rules[0].get("params", {}) or {})
        if isinstance(entry_rules, list) and entry_rules and isinstance(entry_rules[0], dict)
        else {}
    )
    exit_defaults = dict(exit_cfg.get("params", {}) or {}) if isinstance(exit_cfg, dict) else {}

    param_space: dict[str, list[int] | list[float]] = {}
    for section_name, defaults in (("entry", entry_defaults), ("exit", exit_defaults)):
        section_cfg = limited_cfg.get(section_name, {}) if isinstance(limited_cfg, dict) else {}
        raw_optimizable = section_cfg.get("optimizable", {}) if isinstance(section_cfg, dict) else {}
        if isinstance(raw_optimizable, list):
            raw_optimizable = {str(key): {} for key in raw_optimizable}
        if not isinstance(raw_optimizable, dict):
            continue
        for raw_key, raw_spec in raw_optimizable.items():
            key = str(raw_key or "").strip()
            if not key or key in param_space:
                continue
            spec = raw_spec if isinstance(raw_spec, dict) else {}
            values = _build_param_space_values(spec, default_value=defaults.get(key))
            if values:
                param_space[key] = values
    return param_space


# Derived from the richer limited-test metadata so walkforward and the UI
# share one strategy-owned source of truth.
PARAM_SPACE = _build_param_space_from_limited_test(STRATEGY)


def constraints(params: dict) -> bool:
    if "min_rr" in params and "max_rr" in params:
        return float(params["max_rr"]) > float(params["min_rr"])
    return True

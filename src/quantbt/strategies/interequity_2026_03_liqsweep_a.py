from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import inspect
import math

import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.core.indicators import simple_atr
from quantbt.core.performance import build_backtest_summary
from quantbt.core.trades import close_trade_with_costs, resolve_intrabar_bracket_exit
from quantbt.strategies._contracts.common import build_param_space_from_limited_test, min_max_rr_constraint
from quantbt.strategies._support.compat import (
    inert_cross_signals,
    prepare_ohlc_features,
    prev_bar_rr_brackets,
)


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
                    "start": 0.4,
                    "stop": 2.2,
                    "step": 0.2,
                    "integer": False,
                },
                "liq_move_away_atr": {
                    "label": "Liquidity move-away ATR",
                    "start": 1.0,
                    "stop": 5.5,
                    "step": 0.5,
                    "integer": False,
                },
            },
            "non_optimizable": [
                "min_rr",
                "max_rr",
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


PARAM_SPACE = build_param_space_from_limited_test(STRATEGY)


def constraints(params: dict) -> bool:
    return min_max_rr_constraint(params)


@dataclass(frozen=True)
class InterEquityLiqSweepParams:
    # Core tuning knobs (keep this list small to reduce data-snooping risk).
    # - atr_dist_for_liq_generation: equal-high/low proximity tolerance
    # - liq_move_away_atr: confirmation strictness
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

    def __post_init__(self) -> None:
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


ST_BLACK = 0
ST_PURPLE = 1
ST_RED = 2


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

        if old_drawn_act and pool_name is not None and event_time is not None and line_events is not None:
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


def true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None or math.isnan(prev_close):
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def pivot_high(values: list[float], center: int, left: int, right: int) -> float | None:
    if center - left < 0 or center + right >= len(values):
        return None
    pivot_value = values[center]
    for j in range(center - left, center + right + 1):
        if j == center:
            continue
        if values[j] >= pivot_value:
            return None
    return pivot_value


def pivot_low(values: list[float], center: int, left: int, right: int) -> float | None:
    if center - left < 0 or center + right >= len(values):
        return None
    pivot_value = values[center]
    for j in range(center - left, center + right + 1):
        if j == center:
            continue
        if values[j] <= pivot_value:
            return None
    return pivot_value


def _both_active(act_arr: list[bool], a: int, b: int) -> bool:
    return a >= 0 and b >= 0 and a < len(act_arr) and b < len(act_arr) and act_arr[a] and act_arr[b]


def track_breach_high(pool: _LevelPool, high_val: float, breach_t: pd.Timestamp) -> None:
    for i in range(len(pool.lvls)):
        if pool.act[i] and high_val > pool.lvls[i]:
            pool.act[i] = False
            pool.breach_time[i] = breach_t


def track_breach_low(pool: _LevelPool, low_val: float, breach_t: pd.Timestamp) -> None:
    for i in range(len(pool.lvls)):
        if pool.act[i] and low_val < pool.lvls[i]:
            pool.act[i] = False
            pool.breach_time[i] = breach_t


def append_high_pivot(
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

    pool.trim(pool_name=pool_name, event_time=event_time, line_events=line_events)


def append_low_pivot(
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

    pool.trim(pool_name=pool_name, event_time=event_time, line_events=line_events)


def confirm_move_away_high(
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


def confirm_move_away_low(
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


def stop_extending_high(
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


def stop_extending_low(
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


def sweep_triggers(
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


def next_purple_high_above(price: float, high_pool: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(high_pool.lvls):
        if high_pool.drawn_act[i] and high_pool.state[i] == ST_PURPLE and lvl > price:
            best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def next_purple_low_below(price: float, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(low_pool.lvls):
        if low_pool.drawn_act[i] and low_pool.state[i] == ST_PURPLE and lvl < price:
            best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def next_red_above(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl > price:
                best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def next_red_below(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> float | None:
    best = math.nan
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl < price:
                best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def next_state_above_non_black(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            state = pool.state[i]
            if pool.drawn_act[i] and state != ST_BLACK and lvl > price:
                if math.isnan(best_lvl) or lvl < best_lvl:
                    best_lvl = lvl
                    best_state = state
    return best_state


def next_state_below_non_black(price: float, high_pool: _LevelPool, low_pool: _LevelPool) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            state = pool.state[i]
            if pool.drawn_act[i] and state != ST_BLACK and lvl < price:
                if math.isnan(best_lvl) or lvl > best_lvl:
                    best_lvl = lvl
                    best_state = state
    return best_state


def any_red_between(p1: float, p2: float, high_pool: _LevelPool, low_pool: _LevelPool) -> bool:
    lo = min(p1, p2)
    hi = max(p1, p2)
    for pool in (high_pool, low_pool):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lo < lvl < hi:
                return True
    return False


def compute_features(df: pd.DataFrame, p: InterEquityLiqSweepParams) -> pd.DataFrame:
    del p
    return prepare_ohlc_features(df)


def compute_signals(df_feat: pd.DataFrame) -> pd.DataFrame:
    return inert_cross_signals(df_feat)


def build_brackets_from_signal(
    side: str,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    p: InterEquityLiqSweepParams,
):
    return prev_bar_rr_brackets(
        side=side,
        entry_open=entry_open,
        prev_low=prev_low,
        prev_high=prev_high,
        sl_buffer_pips=p.sl_buffer_pips,
        pip_size=p.pip_size,
        rr=p.min_rr,
    )


def _build_pending_entry_from_exit_override(
    *,
    side: str,
    entry_index: int,
    entry_time: pd.Timestamp,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    entry_atr: float | None,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    strategy_params: InterEquityLiqSweepParams,
    cfg: BacktestConfig,
    equity: float,
    exit_builder,
    exit_params: dict[str, Any],
    exit_supports_entry: bool,
    size_fn=None,
    sizing_params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    entry_ctx: dict[str, Any] = {
        "entry_i": int(entry_index),
        "entry_time": pd.Timestamp(entry_time),
        "side": str(side),
        "entry_open": float(entry_open),
        "prev_low": float(prev_low),
        "prev_high": float(prev_high),
    }
    if entry_atr is not None and math.isfinite(float(entry_atr)) and float(entry_atr) > 0:
        entry_ctx["atr"] = float(entry_atr)

    if exit_supports_entry:
        exit_spec = exit_builder(
            side,
            float(entry_open),
            float(prev_low),
            float(prev_high),
            exit_params,
            entry=entry_ctx,
        )
    else:
        exit_spec = exit_builder(
            side,
            float(entry_open),
            float(prev_low),
            float(prev_high),
            exit_params,
        )
    if not isinstance(exit_spec, dict) or "hold_bars" in exit_spec:
        return None

    try:
        sl = float(exit_spec["sl"])
        tp = float(exit_spec["tp"])
        stop_dist = float(exit_spec["stop_dist"])
    except Exception:
        return None
    if not (math.isfinite(sl) and math.isfinite(tp) and math.isfinite(stop_dist) and stop_dist > 0):
        return None

    if side == "long":
        ok_sl = sl < entry_open
        ok_tp = tp > entry_open
        reward_dist = tp - entry_open if ok_tp else math.nan
        risk_dist = entry_open - sl if ok_sl else math.nan
    elif side == "short":
        ok_sl = sl > entry_open
        ok_tp = tp < entry_open
        reward_dist = entry_open - tp if ok_tp else math.nan
        risk_dist = sl - entry_open if ok_sl else math.nan
    else:
        raise ValueError("side must be 'long' or 'short'")

    rr = (
        reward_dist / risk_dist
        if math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)
        else math.nan
    )
    ok_rr = math.isfinite(rr) and rr > float(strategy_params.min_rr) and rr <= float(strategy_params.max_rr)
    has_red_between = any_red_between(entry_open, sl, high_pool, low_pool) if ok_sl else True
    if not (ok_sl and ok_tp and ok_rr and not has_red_between):
        return None

    sizing_params = dict(sizing_params or {})
    if callable(size_fn):
        qty = size_fn(
            cfg=cfg,
            equity=float(equity),
            side=str(side),
            entry_open=float(entry_open),
            exit_spec={"sl": sl, "tp": tp, "stop_dist": stop_dist},
            entry=entry_ctx,
            params=sizing_params,
        )
    else:
        risk_amount = float(cfg.initial_equity) * float(strategy_params.risk_pct)
        qty = risk_amount / stop_dist
    if qty is None or not math.isfinite(float(qty)) or float(qty) <= 0:
        return None

    risk_dollars = float(qty) * stop_dist
    return {
        "entry_i": int(entry_index),
        "side": str(side),
        "qty": float(qty),
        "sl": float(sl),
        "tp": float(tp),
        "risk_dollars": float(risk_dollars),
    }


def _close_pending_market_exit(
    *,
    pos: dict[str, Any] | None,
    pending_market_exit: dict[str, Any] | None,
    bar_index: int,
    bar_open: float,
    bar_time: pd.Timestamp,
    equity: float,
    cfg: Any,
) -> tuple[float, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if pos is None or pending_market_exit is None or int(pending_market_exit["entry_i"]) != bar_index:
        return equity, pos, pending_market_exit, None

    equity_after, trade = close_trade_with_costs(
        pos=pos,
        exit_price=bar_open,
        exit_time=bar_time,
        exit_reason=str(pending_market_exit["reason"]),
        equity_now=equity,
        cfg=cfg,
    )
    return equity_after, None, None, trade


def _fill_pending_entry(
    *,
    pos: dict[str, Any] | None,
    pending_entry: dict[str, Any] | None,
    bar_index: int,
    bar_open: float,
    bar_time: pd.Timestamp,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if pos is not None or pending_entry is None or int(pending_entry["entry_i"]) != bar_index:
        return pos, pending_entry

    pos = {
        "side": pending_entry["side"],
        "entry": bar_open,
        "sl": float(pending_entry["sl"]),
        "tp": float(pending_entry["tp"]),
        "units": float(pending_entry["qty"]),
        "entry_time": bar_time,
        "risk_dollars": float(pending_entry["risk_dollars"]),
    }
    return pos, None


def _build_pending_entry(
    *,
    side: str,
    entry_index: int,
    entry_price: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    strategy_params: InterEquityLiqSweepParams,
    cfg: BacktestConfig,
) -> dict[str, Any] | None:
    sl_buffer = float(strategy_params.sl_buffer_pips) * float(strategy_params.pip_size)
    rr_band_valid = float(strategy_params.max_rr) > float(strategy_params.min_rr)
    if not rr_band_valid:
        return None

    if side == "short":
        sl_raw = next_purple_high_above(entry_price, high_pool)
        sl = (sl_raw + sl_buffer) if sl_raw is not None else None
        tp = next_red_below(entry_price, high_pool, low_pool)
        ok_sl = sl is not None and sl > entry_price
        ok_tp = tp is not None and tp < entry_price
        risk_dist = (sl - entry_price) if ok_sl else math.nan
        reward_dist = (entry_price - tp) if ok_tp and tp is not None else math.nan
    elif side == "long":
        sl_raw = next_purple_low_below(entry_price, low_pool)
        sl = (sl_raw - sl_buffer) if sl_raw is not None else None
        tp = next_red_above(entry_price, high_pool, low_pool)
        ok_sl = sl is not None and sl < entry_price
        ok_tp = tp is not None and tp > entry_price
        risk_dist = (entry_price - sl) if ok_sl else math.nan
        reward_dist = (tp - entry_price) if ok_tp and tp is not None else math.nan
    else:
        raise ValueError("side must be 'long' or 'short'")

    rr = (
        reward_dist / risk_dist
        if math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)
        else math.nan
    )
    ok_rr = math.isfinite(rr) and rr > float(strategy_params.min_rr) and rr <= float(strategy_params.max_rr)
    has_red_between = any_red_between(entry_price, sl, high_pool, low_pool) if ok_sl and sl is not None else True
    if not (ok_sl and ok_tp and ok_rr and not has_red_between and sl is not None and tp is not None):
        return None

    risk_amount = float(cfg.initial_equity) * float(strategy_params.risk_pct)
    stop_dist = max(abs(float(entry_price) - float(sl)), float(strategy_params.min_tick))
    qty = risk_amount / stop_dist
    if not math.isfinite(qty) or qty <= 0:
        return None

    return {
        "entry_i": entry_index,
        "side": side,
        "qty": float(qty),
        "sl": float(sl),
        "tp": float(tp),
        "risk_dollars": float(risk_amount),
    }


def _schedule_force_exit(
    *,
    pos: dict[str, Any] | None,
    pending_market_exit: dict[str, Any] | None,
    entry_price: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    next_bar_index: int,
) -> dict[str, Any] | None:
    if pos is None or pending_market_exit is not None:
        return pending_market_exit

    if str(pos["side"]) == "long":
        next_state = next_state_above_non_black(entry_price, high_pool, low_pool)
        if next_state == ST_PURPLE:
            return {"entry_i": next_bar_index, "reason": "Nearest above is purple"}
    else:
        next_state = next_state_below_non_black(entry_price, high_pool, low_pool)
        if next_state == ST_PURPLE:
            return {"entry_i": next_bar_index, "reason": "Nearest below is purple"}
    return pending_market_exit


def run_backtest(
    df_sig: pd.DataFrame,
    *,
    strategy_params: InterEquityLiqSweepParams,
    cfg: BacktestConfig = BacktestConfig(),
    debug: dict[str, Any] | None = None,
    override_exit_builder=None,
    override_exit_params: dict[str, Any] | None = None,
    override_size_fn=None,
    override_sizing_params: dict[str, Any] | None = None,
):
    p = strategy_params
    df = df_sig.copy().sort_index()
    idx = df.index.to_list()
    exit_override_active = callable(override_exit_builder)
    exit_override_params = dict(override_exit_params or {})
    sizing_override_params = dict(override_sizing_params or {})
    try:
        exit_override_supports_entry = bool(
            exit_override_active and "entry" in inspect.signature(override_exit_builder).parameters
        )
    except (TypeError, ValueError):
        exit_override_supports_entry = False
    override_atr_series = None
    if exit_override_active and bool(getattr(override_exit_builder, "requires_atr", False)):
        atr_period = int(exit_override_params.get("atr_period", 14))
        override_atr_series = simple_atr(df, atr_period)

    equity = float(cfg.initial_equity)
    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    pos: dict[str, Any] | None = None
    pending_entry: dict[str, Any] | None = None
    pending_market_exit: dict[str, Any] | None = None
    line_events: list[dict[str, Any]] = []

    high_pool = _LevelPool(max_size=int(p.max_high_levels))
    low_pool = _LevelPool(max_size=int(p.max_low_levels))

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

        equity, pos, pending_market_exit, trade = _close_pending_market_exit(
            pos=pos,
            pending_market_exit=pending_market_exit,
            bar_index=i,
            bar_open=o,
            bar_time=t,
            equity=equity,
            cfg=cfg,
        )
        if trade is not None:
            trades.append(trade)

        pos, pending_entry = _fill_pending_entry(
            pos=pos,
            pending_entry=pending_entry,
            bar_index=i,
            bar_open=o,
            bar_time=t,
        )

        if pos is not None:
            exit_price, exit_reason = resolve_intrabar_bracket_exit(
                side=str(pos["side"]),
                bar_high=h,
                bar_low=l,
                sl=float(pos["sl"]),
                tp=float(pos["tp"]),
                conservative_same_bar=bool(cfg.conservative_same_bar),
            )
            if exit_price is not None:
                equity, trade = close_trade_with_costs(
                    pos=pos,
                    exit_price=float(exit_price),
                    exit_time=t,
                    exit_reason=str(exit_reason),
                    equity_now=equity,
                    cfg=cfg,
                )
                trades.append(trade)
                pos = None
                pending_market_exit = None

        equity_curve.append({"time": t, "equity": equity})

        tr = true_range(h, l, prev_close)
        atr = tr if prev_atr is None else ((prev_atr * (int(p.atr_len) - 1)) + tr) / int(p.atr_len)
        prev_close = c
        prev_atr = atr

        high_hist.append(h)
        low_hist.append(l)
        time_hist.append(t)
        atr_hist.append(float(atr))

        pivot_high_value = None
        pivot_low_value = None
        pivot_time = None
        pivot_atr = None

        center_idx = len(high_hist) - 1 - int(p.pivot_len)
        if center_idx >= int(p.pivot_len):
            pivot_high_value = pivot_high(high_hist, center_idx, int(p.pivot_len), int(p.pivot_len))
            pivot_low_value = pivot_low(low_hist, center_idx, int(p.pivot_len), int(p.pivot_len))
            pivot_time = time_hist[center_idx]
            pivot_atr = float(atr_hist[center_idx])

        trig_short = False
        trig_long = False

        if bool(p.show_levels):
            track_breach_high(high_pool, h, t)
            track_breach_low(low_pool, l, t)

            if pivot_high_value is not None and pivot_time is not None and pivot_atr is not None:
                append_high_pivot(
                    pool=high_pool,
                    pool_name="high",
                    pivot_value=float(pivot_high_value),
                    pivot_time=pivot_time,
                    event_time=t,
                    pivot_atr=float(pivot_atr),
                    atr_dist_for_liq_generation=float(p.atr_dist_for_liq_generation),
                    liq_move_away_atr=float(p.liq_move_away_atr),
                    line_events=line_events,
                )

            if pivot_low_value is not None and pivot_time is not None and pivot_atr is not None:
                append_low_pivot(
                    pool=low_pool,
                    pool_name="low",
                    pivot_value=float(pivot_low_value),
                    pivot_time=pivot_time,
                    event_time=t,
                    pivot_atr=float(pivot_atr),
                    atr_dist_for_liq_generation=float(p.atr_dist_for_liq_generation),
                    liq_move_away_atr=float(p.liq_move_away_atr),
                    line_events=line_events,
                )

            confirm_move_away_high(
                high_pool,
                c,
                pool_name="high",
                event_time=t,
                line_events=line_events,
            )
            confirm_move_away_low(
                low_pool,
                c,
                pool_name="low",
                event_time=t,
                line_events=line_events,
            )
            trig_short, trig_long = sweep_triggers(
                high_pool,
                low_pool,
                h,
                l,
                event_time=t,
                line_events=line_events,
            )
            stop_extending_high(
                high_pool,
                h,
                pool_name="high",
                event_time=t,
                line_events=line_events,
            )
            stop_extending_low(
                low_pool,
                l,
                pool_name="low",
                event_time=t,
                line_events=line_events,
            )

        flat = pos is None
        entry_px = c

        if flat and trig_short and i < len(df) - 1:
            if exit_override_active:
                t_next = idx[i + 1]
                entry_atr = None
                if override_atr_series is not None:
                    entry_atr = float(override_atr_series.at[t_next])
                pending_entry = _build_pending_entry_from_exit_override(
                    side="short",
                    entry_index=i + 1,
                    entry_time=t_next,
                    entry_open=float(df.at[t_next, "open"]),
                    prev_low=l,
                    prev_high=h,
                    entry_atr=entry_atr,
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                    equity=equity,
                    exit_builder=override_exit_builder,
                    exit_params=exit_override_params,
                    exit_supports_entry=exit_override_supports_entry,
                    size_fn=override_size_fn,
                    sizing_params=sizing_override_params,
                )
            else:
                pending_entry = _build_pending_entry(
                    side="short",
                    entry_index=i + 1,
                    entry_price=entry_px,
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                )

        if flat and pending_entry is None and trig_long and i < len(df) - 1:
            if exit_override_active:
                t_next = idx[i + 1]
                entry_atr = None
                if override_atr_series is not None:
                    entry_atr = float(override_atr_series.at[t_next])
                pending_entry = _build_pending_entry_from_exit_override(
                    side="long",
                    entry_index=i + 1,
                    entry_time=t_next,
                    entry_open=float(df.at[t_next, "open"]),
                    prev_low=l,
                    prev_high=h,
                    entry_atr=entry_atr,
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                    equity=equity,
                    exit_builder=override_exit_builder,
                    exit_params=exit_override_params,
                    exit_supports_entry=exit_override_supports_entry,
                    size_fn=override_size_fn,
                    sizing_params=sizing_override_params,
                )
            else:
                pending_entry = _build_pending_entry(
                    side="long",
                    entry_index=i + 1,
                    entry_price=entry_px,
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                )

        if not exit_override_active and i < len(df) - 1:
            pending_market_exit = _schedule_force_exit(
                pos=pos,
                pending_market_exit=pending_market_exit,
                entry_price=entry_px,
                high_pool=high_pool,
                low_pool=low_pool,
                next_bar_index=i + 1,
            )

    equity_df = pd.DataFrame(equity_curve).set_index("time") if equity_curve else pd.DataFrame(columns=["equity"])
    trades_df = pd.DataFrame(trades)
    summary = build_backtest_summary(
        equity_like=equity_df,
        trades_df=trades_df,
        initial_equity=float(cfg.initial_equity),
    )
    if debug is not None:
        debug["line_events"] = line_events
        debug["line_event_count"] = len(line_events)
    return equity_df, trades_df, summary


__all__ = [
    "InterEquityLiqSweepParams",
    "PARAM_SPACE",
    "Params",
    "ST_BLACK",
    "ST_PURPLE",
    "ST_RED",
    "STRATEGY",
    "build_brackets_from_signal",
    "compute_features",
    "compute_signals",
    "constraints",
    "run_backtest",
]

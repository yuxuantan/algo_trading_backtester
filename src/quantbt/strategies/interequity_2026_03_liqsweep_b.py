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
class InterEquityLiqSweepBParams:
    # Core tuning knobs (keep this list compact to reduce data-snooping risk).
    max_rr: float = 10.0
    min_rr: float = 1.0
    atr_dist_for_liq_generation: float = 1.0
    liq_move_away_atr: float = 3.0

    # Structure/runtime defaults.
    show_ltf: bool = True
    ltf_pivot_len: int = 7
    atr_len: int = 14
    risk_pct: float = 0.01
    sl_atr_buffer_mult: float = 0.1
    max_stop_atr: float = 10.0

    # Instrument config
    pip_size: float = 0.0001
    min_tick: float = 1e-5

    # Runtime caps
    max_ltf_h: int = 150
    max_ltf_l: int = 150

    def __post_init__(self):
        object.__setattr__(self, "min_rr", float(self.min_rr))
        object.__setattr__(self, "max_rr", float(self.max_rr))
        object.__setattr__(self, "atr_dist_for_liq_generation", float(self.atr_dist_for_liq_generation))
        object.__setattr__(self, "show_ltf", bool(self.show_ltf))

        object.__setattr__(self, "ltf_pivot_len", int(self.ltf_pivot_len))
        object.__setattr__(self, "atr_len", int(self.atr_len))
        object.__setattr__(self, "risk_pct", float(self.risk_pct))
        object.__setattr__(self, "liq_move_away_atr", float(self.liq_move_away_atr))
        object.__setattr__(self, "sl_atr_buffer_mult", float(self.sl_atr_buffer_mult))
        object.__setattr__(self, "max_stop_atr", float(self.max_stop_atr))

        object.__setattr__(self, "pip_size", float(self.pip_size))
        object.__setattr__(self, "min_tick", float(self.min_tick))

        object.__setattr__(self, "max_ltf_h", int(self.max_ltf_h))
        object.__setattr__(self, "max_ltf_l", int(self.max_ltf_l))

        if self.min_rr < 0 or self.max_rr < 0:
            raise ValueError("min_rr/max_rr must be >= 0")
        if self.max_rr <= self.min_rr:
            raise ValueError(f"max_rr must be > min_rr. Got min_rr={self.min_rr}, max_rr={self.max_rr}")
        if self.atr_dist_for_liq_generation < 0:
            raise ValueError("atr_dist_for_liq_generation must be >= 0")
        if self.liq_move_away_atr <= 0:
            raise ValueError("liq_move_away_atr must be > 0")
        if self.ltf_pivot_len <= 0:
            raise ValueError("ltf_pivot_len must be > 0")
        if self.atr_len <= 0:
            raise ValueError("atr_len must be > 0")
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be > 0")
        if self.sl_atr_buffer_mult < 0:
            raise ValueError("sl_atr_buffer_mult must be >= 0")
        if self.max_stop_atr <= 0:
            raise ValueError("max_stop_atr must be > 0")
        if self.pip_size <= 0 or self.min_tick <= 0:
            raise ValueError("pip_size and min_tick must be > 0")


Params = InterEquityLiqSweepBParams


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


def _ltf_sweep_triggers(
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
    high_val: float,
    low_val: float,
    *,
    event_time: pd.Timestamp,
    line_events: list[dict[str, Any]] | None = None,
) -> tuple[bool, bool, float | None, float | None]:
    trig_short = False
    trig_long = False
    short_entry_level = math.nan
    long_entry_level = math.nan

    # Deactivate all breached LTF high lines on the current bar.
    for i in range(len(ltf_h.lvls)):
        if ltf_h.drawn_act[i] and high_val > ltf_h.lvls[i]:
            if ltf_h.state[i] == ST_RED:
                trig_short = True
                lvl = float(ltf_h.lvls[i])
                # If multiple red highs are swept on this bar, enter at the latest
                # one taken out (highest swept level for upside sweep).
                short_entry_level = lvl if math.isnan(short_entry_level) else max(short_entry_level, lvl)
            ltf_h.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": "ltf_h",
                        "line_id": int(ltf_h.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(ltf_h.lvls[i]),
                    }
                )

    # Deactivate all breached LTF low lines on the current bar.
    for i in range(len(ltf_l.lvls)):
        if ltf_l.drawn_act[i] and low_val < ltf_l.lvls[i]:
            if ltf_l.state[i] == ST_RED:
                trig_long = True
                lvl = float(ltf_l.lvls[i])
                # If multiple red lows are swept on this bar, enter at the latest
                # one taken out (lowest swept level for downside sweep).
                long_entry_level = lvl if math.isnan(long_entry_level) else min(long_entry_level, lvl)
            ltf_l.drawn_act[i] = False
            if line_events is not None:
                line_events.append(
                    {
                        "type": "line_deactivated",
                        "pool": "ltf_l",
                        "line_id": int(ltf_l.line_ids[i]),
                        "time": pd.Timestamp(event_time),
                        "level": float(ltf_l.lvls[i]),
                    }
                )

    return (
        trig_short,
        trig_long,
        None if math.isnan(short_entry_level) else float(short_entry_level),
        None if math.isnan(long_entry_level) else float(long_entry_level),
    )


def _next_purple_above(price: float, ltf_h: _LevelPool, ltf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (ltf_h, ltf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_PURPLE and lvl > price:
                best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_purple_below(price: float, ltf_h: _LevelPool, ltf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (ltf_h, ltf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_PURPLE and lvl < price:
                best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_red_above(price: float, ltf_h: _LevelPool, ltf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (ltf_h, ltf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl > price:
                best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_red_below(price: float, ltf_h: _LevelPool, ltf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (ltf_h, ltf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl < price:
                best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _any_red_between(
    p1: float,
    p2: float,
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
) -> bool:
    lo = min(p1, p2)
    hi = max(p1, p2)
    for pool in (ltf_h, ltf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lo < lvl < hi:
                return True
    return False


def compute_features(df: pd.DataFrame, p: InterEquityLiqSweepBParams) -> pd.DataFrame:
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
    p: InterEquityLiqSweepBParams,
):
    # Fallback bracket builder for compatibility with generic engine.
    sl_buffer = 0.0
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
    strategy_params: InterEquityLiqSweepBParams,
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
    line_events: list[dict[str, Any]] = []

    ltf_h = _LevelPool(max_size=p.max_ltf_h)
    ltf_l = _LevelPool(max_size=p.max_ltf_l)

    ltf_high_hist: list[float] = []
    ltf_low_hist: list[float] = []
    ltf_close_hist: list[float] = []
    ltf_time_hist: list[pd.Timestamp] = []
    ltf_atr_hist: list[float] = []

    ltf_prev_close: float | None = None
    ltf_prev_atr: float | None = None

    for i, t in enumerate(idx):
        h = float(df.at[t, "high"])
        l = float(df.at[t, "low"])
        c = float(df.at[t, "close"])

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

        equity_curve.append({"time": t, "equity": equity})

        # ---- LTF ATR + pivot state machine ----
        tr_ltf = _true_range(h, l, ltf_prev_close)
        ltf_atr = tr_ltf if ltf_prev_atr is None else ((ltf_prev_atr * (p.atr_len - 1)) + tr_ltf) / p.atr_len
        ltf_prev_close = c
        ltf_prev_atr = ltf_atr

        ltf_high_hist.append(h)
        ltf_low_hist.append(l)
        ltf_close_hist.append(c)
        ltf_time_hist.append(t)
        ltf_atr_hist.append(float(ltf_atr))

        ltf_ph = None
        ltf_pl = None
        ltf_pivot_time = None
        ltf_pivot_atr = None

        c_idx = len(ltf_high_hist) - 1 - p.ltf_pivot_len
        if c_idx >= p.ltf_pivot_len:
            ltf_ph = _pivot_high(ltf_high_hist, c_idx, p.ltf_pivot_len, p.ltf_pivot_len)
            ltf_pl = _pivot_low(ltf_low_hist, c_idx, p.ltf_pivot_len, p.ltf_pivot_len)
            ltf_pivot_time = ltf_time_hist[c_idx]
            ltf_pivot_atr = float(ltf_atr_hist[c_idx])

        trig_short = False
        trig_long = False
        trig_short_entry_px: float | None = None
        trig_long_entry_px: float | None = None

        if p.show_ltf:
            _track_breach_high(ltf_h, h, t)
            _track_breach_low(ltf_l, l, t)

            if ltf_ph is not None and ltf_pivot_time is not None and ltf_pivot_atr is not None:
                _append_high_pivot(
                    pool=ltf_h,
                    pool_name="ltf_h",
                    pivot_value=float(ltf_ph),
                    pivot_time=ltf_pivot_time,
                    event_time=t,
                    pivot_atr=float(ltf_pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                    line_events=line_events,
                )

            if ltf_pl is not None and ltf_pivot_time is not None and ltf_pivot_atr is not None:
                _append_low_pivot(
                    pool=ltf_l,
                    pool_name="ltf_l",
                    pivot_value=float(ltf_pl),
                    pivot_time=ltf_pivot_time,
                    event_time=t,
                    pivot_atr=float(ltf_pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                    line_events=line_events,
                )

            _confirm_move_away_high(
                ltf_h,
                c,
                pool_name="ltf_h",
                event_time=t,
                line_events=line_events,
            )
            _confirm_move_away_low(
                ltf_l,
                c,
                pool_name="ltf_l",
                event_time=t,
                line_events=line_events,
            )

            trig_short, trig_long, trig_short_entry_px, trig_long_entry_px = _ltf_sweep_triggers(
                ltf_h,
                ltf_l,
                h,
                l,
                event_time=t,
                line_events=line_events,
            )

        # ---- Strategy entries (immediate fill at swept-liquidity level) ----
        if pos is None:
            atr_ref = float(ltf_atr)
            rr_band_valid = p.max_rr >= p.min_rr
            if math.isfinite(atr_ref) and atr_ref > 0 and rr_band_valid:
                sl_buffer = float(p.sl_atr_buffer_mult) * atr_ref
                max_stop_dist = float(p.max_stop_atr) * atr_ref

                if trig_short and trig_short_entry_px is not None:
                    entry_px = float(trig_short_entry_px)
                    sl_raw = _next_purple_above(entry_px, ltf_h, ltf_l)
                    sl = (sl_raw + sl_buffer) if sl_raw is not None else None
                    tp = _next_red_below(entry_px, ltf_h, ltf_l)

                    ok_sl = sl is not None and sl > entry_px
                    ok_tp = tp is not None and tp < entry_px

                    risk_dist = (sl - entry_px) if ok_sl else math.nan
                    reward_dist = (entry_px - tp) if ok_tp and tp is not None else math.nan
                    rr = (
                        reward_dist / risk_dist
                        if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist))
                        else math.nan
                    )
                    ok_rr = math.isfinite(rr) and rr >= p.min_rr and rr <= p.max_rr
                    stop_not_too_wide = math.isfinite(risk_dist) and risk_dist < max_stop_dist
                    has_red_between = bool(ok_sl and _any_red_between(entry_px, float(sl), ltf_h, ltf_l))

                    if ok_sl and ok_tp and ok_rr and stop_not_too_wide and not has_red_between and sl is not None and tp is not None:
                        risk_amount = float(cfg.initial_equity) * p.risk_pct
                        stop_dist = max(sl - entry_px, p.min_tick)
                        qty = risk_amount / stop_dist
                        if math.isfinite(qty) and qty > 0:
                            pos = {
                                "side": "short",
                                "entry": entry_px,
                                "sl": float(sl),
                                "tp": float(tp),
                                "units": float(qty),
                                "entry_time": t,
                                "risk_dollars": float(risk_amount),
                            }

                if pos is None and trig_long and trig_long_entry_px is not None:
                    entry_px = float(trig_long_entry_px)
                    sl_raw = _next_purple_below(entry_px, ltf_h, ltf_l)
                    sl = (sl_raw - sl_buffer) if sl_raw is not None else None
                    tp = _next_red_above(entry_px, ltf_h, ltf_l)

                    ok_sl = sl is not None and sl < entry_px
                    ok_tp = tp is not None and tp > entry_px

                    risk_dist = (entry_px - sl) if ok_sl else math.nan
                    reward_dist = (tp - entry_px) if ok_tp and tp is not None else math.nan
                    rr = (
                        reward_dist / risk_dist
                        if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist))
                        else math.nan
                    )
                    ok_rr = math.isfinite(rr) and rr >= p.min_rr and rr <= p.max_rr
                    stop_not_too_wide = math.isfinite(risk_dist) and risk_dist < max_stop_dist
                    has_red_between = bool(ok_sl and _any_red_between(entry_px, float(sl), ltf_h, ltf_l))

                    if ok_sl and ok_tp and ok_rr and stop_not_too_wide and not has_red_between and sl is not None and tp is not None:
                        risk_amount = float(cfg.initial_equity) * p.risk_pct
                        stop_dist = max(entry_px - sl, p.min_tick)
                        qty = risk_amount / stop_dist
                        if math.isfinite(qty) and qty > 0:
                            pos = {
                                "side": "long",
                                "entry": entry_px,
                                "sl": float(sl),
                                "tp": float(tp),
                                "units": float(qty),
                                "entry_time": t,
                                "risk_dollars": float(risk_amount),
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


PARAM_SPACE = {
    # Compact by design to keep optimization robust.
    "atr_dist_for_liq_generation": [0.8, 1.0, 1.2],
    "liq_move_away_atr": [2.0, 2.5, 3.0],
    "max_rr": [6.0, 8.0, 10.0],
    "sl_atr_buffer_mult": [0.1],
    "max_stop_atr": [10.0],
    "ltf_pivot_len": [7],
}


def constraints(params: dict) -> bool:
    if "min_rr" in params and "max_rr" in params:
        return float(params["max_rr"]) > float(params["min_rr"])
    return True


STRATEGY = {
    "entry": {
        "mode": "all",
        "rules": [
            {
                "name": "interequity_liqsweepb_entry",
                "params": {
                    "min_rr": 1.0,
                    "max_rr": 10.0,
                    "atr_dist_for_liq_generation": 1.0,
                    "liq_move_away_atr": 2.5,
                    "ltf_pivot_len": 7,
                    "sl_atr_buffer_mult": 0.1,
                    "max_stop_atr": 10.0,
                },
            }
        ],
    },
    "exit": {
        "name": "interequity_liqsweepb_exit",
        "params": {"min_rr": 1.0, "pip_size": 0.0001},
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
                    "integer": False,
                },
                "liq_move_away_atr": {
                    "label": "Liquidity move-away ATR",
                    "default": 2.5,
                    "start": 1.0,
                    "stop": 5.5,
                    "step": 0.5,
                    "integer": False,
                },
            },
            "non_optimizable": [
                "min_rr",
                "max_rr",
                "ltf_pivot_len",
                "sl_atr_buffer_mult",
                "max_stop_atr",
            ],
        },
        "exit": {
            "optimizable": {},
            "non_optimizable": [
                "min_rr",
                "pip_size",
            ],
        },
    },
}

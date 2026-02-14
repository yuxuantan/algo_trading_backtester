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
    # Tunable params
    min_rr: float = 2.0
    max_rr: float = 10.0
    atr_dist_for_liq_generation: float = 1.0

    # Config-like params
    htf: str = "60"
    show_ltf: bool = True
    show_htf: bool = True

    # Fixed strategy params (defaulted to Pine values)
    ltf_pivot_len: int = 3
    htf_pivot_len: int = 3
    atr_len: int = 14
    risk_pct: float = 0.01
    sl_buffer_pips: float = 1.0
    liq_move_away_atr: float = 3.0

    # Instrument config
    pip_size: float = 0.0001
    min_tick: float = 1e-5

    # Runtime caps (mirrors Pine line budgets)
    max_ltf_h: int = 150
    max_ltf_l: int = 150
    max_htf_h: int = 100
    max_htf_l: int = 100

    def __post_init__(self):
        object.__setattr__(self, "min_rr", float(self.min_rr))
        object.__setattr__(self, "max_rr", float(self.max_rr))
        object.__setattr__(self, "atr_dist_for_liq_generation", float(self.atr_dist_for_liq_generation))

        object.__setattr__(self, "htf", str(self.htf))
        object.__setattr__(self, "show_ltf", bool(self.show_ltf))
        object.__setattr__(self, "show_htf", bool(self.show_htf))

        object.__setattr__(self, "ltf_pivot_len", int(self.ltf_pivot_len))
        object.__setattr__(self, "htf_pivot_len", int(self.htf_pivot_len))
        object.__setattr__(self, "atr_len", int(self.atr_len))
        object.__setattr__(self, "risk_pct", float(self.risk_pct))
        object.__setattr__(self, "sl_buffer_pips", float(self.sl_buffer_pips))
        object.__setattr__(self, "liq_move_away_atr", float(self.liq_move_away_atr))

        object.__setattr__(self, "pip_size", float(self.pip_size))
        object.__setattr__(self, "min_tick", float(self.min_tick))

        object.__setattr__(self, "max_ltf_h", int(self.max_ltf_h))
        object.__setattr__(self, "max_ltf_l", int(self.max_ltf_l))
        object.__setattr__(self, "max_htf_h", int(self.max_htf_h))
        object.__setattr__(self, "max_htf_l", int(self.max_htf_l))

        if self.min_rr < 0 or self.max_rr < 0:
            raise ValueError("min_rr/max_rr must be >= 0")
        if self.max_rr <= self.min_rr:
            raise ValueError(f"max_rr must be > min_rr. Got min_rr={self.min_rr}, max_rr={self.max_rr}")
        if self.atr_dist_for_liq_generation < 0:
            raise ValueError("atr_dist_for_liq_generation must be >= 0")
        if self.ltf_pivot_len <= 0 or self.htf_pivot_len <= 0:
            raise ValueError("ltf_pivot_len and htf_pivot_len must be > 0")
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
    act: list[bool] = field(default_factory=list)
    breach_time: list[pd.Timestamp | None] = field(default_factory=list)
    drawn_act: list[bool] = field(default_factory=list)
    state: list[int] = field(default_factory=list)

    pend_a: list[int] = field(default_factory=list)
    pend_b: list[int] = field(default_factory=list)
    pend_trigger: list[float] = field(default_factory=list)
    pend_ok: list[bool] = field(default_factory=list)

    def trim(self) -> None:
        if len(self.lvls) <= self.max_size:
            return

        self.lvls.pop(0)
        self.act.pop(0)
        self.breach_time.pop(0)
        self.drawn_act.pop(0)
        self.state.pop(0)

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


def _timeframe_to_minutes(tf: str) -> int:
    s = str(tf).strip().upper()
    if s.isdigit():
        return int(s)
    if s in {"D", "1D"}:
        return 1440
    if s in {"W", "1W"}:
        return 10080
    if s.endswith("H"):
        return int(s[:-1]) * 60
    if s.endswith("MIN"):
        return int(s[:-3])
    raise ValueError(f"Unsupported HTF format: {tf!r}")


def _bucket_floor(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return ts.floor(f"{int(minutes)}min")


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
    pivot_value: float,
    pivot_time: pd.Timestamp,
    pivot_atr: float,
    atr_dist_for_liq_generation: float,
    liq_move_away_atr: float,
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

    curr_idx = len(pool.lvls)
    pool.lvls.append(float(pivot_value))
    pool.act.append(True)
    pool.breach_time.append(None)
    pool.drawn_act.append(True)
    pool.state.append(base_state)

    if match is not None and match_lvl is not None:
        pair_high = max(float(pivot_value), float(match_lvl))
        trigger_down = pair_high - (liq_move_away_atr * pivot_atr)
        pool.pend_a.append(match)
        pool.pend_b.append(curr_idx)
        pool.pend_trigger.append(trigger_down)
        pool.pend_ok.append(False)

    pool.trim()


def _append_low_pivot(
    *,
    pool: _LevelPool,
    pivot_value: float,
    pivot_time: pd.Timestamp,
    pivot_atr: float,
    atr_dist_for_liq_generation: float,
    liq_move_away_atr: float,
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

    curr_idx = len(pool.lvls)
    pool.lvls.append(float(pivot_value))
    pool.act.append(True)
    pool.breach_time.append(None)
    pool.drawn_act.append(True)
    pool.state.append(base_state)

    if match is not None and match_lvl is not None:
        pair_low = min(float(pivot_value), float(match_lvl))
        trigger_up = pair_low + (liq_move_away_atr * pivot_atr)
        pool.pend_a.append(match)
        pool.pend_b.append(curr_idx)
        pool.pend_trigger.append(trigger_up)
        pool.pend_ok.append(False)

    pool.trim()


def _confirm_move_away_high(pool: _LevelPool, close_val: float) -> None:
    for k in range(len(pool.pend_a)):
        if pool.pend_ok[k]:
            continue
        trigger_down = pool.pend_trigger[k]
        if close_val <= trigger_down:
            a = pool.pend_a[k]
            b = pool.pend_b[k]
            if _both_active(pool.act, a, b):
                pool.state[a] = ST_RED
                pool.state[b] = ST_RED
                pool.pend_ok[k] = True
                break


def _confirm_move_away_low(pool: _LevelPool, close_val: float) -> None:
    for k in range(len(pool.pend_a)):
        if pool.pend_ok[k]:
            continue
        trigger_up = pool.pend_trigger[k]
        if close_val >= trigger_up:
            a = pool.pend_a[k]
            b = pool.pend_b[k]
            if _both_active(pool.act, a, b):
                pool.state[a] = ST_RED
                pool.state[b] = ST_RED
                pool.pend_ok[k] = True
                break


def _stop_extending_high(pool: _LevelPool, high_val: float) -> None:
    for i in range(len(pool.lvls)):
        if pool.drawn_act[i] and high_val > pool.lvls[i]:
            pool.drawn_act[i] = False


def _stop_extending_low(pool: _LevelPool, low_val: float) -> None:
    for i in range(len(pool.lvls)):
        if pool.drawn_act[i] and low_val < pool.lvls[i]:
            pool.drawn_act[i] = False


def _ltf_sweep_triggers(
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
    high_val: float,
    low_val: float,
) -> tuple[bool, bool]:
    trig_short = False
    trig_long = False

    for i in range(len(ltf_h.lvls)):
        if ltf_h.drawn_act[i] and high_val > ltf_h.lvls[i]:
            if ltf_h.state[i] == ST_RED:
                trig_short = True
            ltf_h.drawn_act[i] = False
            if trig_short:
                break

    if not trig_short:
        for i in range(len(ltf_l.lvls)):
            if ltf_l.drawn_act[i] and low_val < ltf_l.lvls[i]:
                if ltf_l.state[i] == ST_RED:
                    trig_long = True
                ltf_l.drawn_act[i] = False
                if trig_long:
                    break

    return trig_short, trig_long


def _next_htf_purple_high_above(price: float, htf_h: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(htf_h.lvls):
        if htf_h.drawn_act[i] and htf_h.state[i] == ST_PURPLE and lvl > price:
            best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_htf_purple_low_below(price: float, htf_l: _LevelPool) -> float | None:
    best = math.nan
    for i, lvl in enumerate(htf_l.lvls):
        if htf_l.drawn_act[i] and htf_l.state[i] == ST_PURPLE and lvl < price:
            best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_htf_red_above(price: float, htf_h: _LevelPool, htf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (htf_h, htf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl > price:
                best = lvl if math.isnan(best) else min(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_htf_red_below(price: float, htf_h: _LevelPool, htf_l: _LevelPool) -> float | None:
    best = math.nan
    for pool in (htf_h, htf_l):
        for i, lvl in enumerate(pool.lvls):
            if pool.drawn_act[i] and pool.state[i] == ST_RED and lvl < price:
                best = lvl if math.isnan(best) else max(best, lvl)
    return None if math.isnan(best) else float(best)


def _next_state_above_non_black(
    price: float,
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
    htf_h: _LevelPool,
    htf_l: _LevelPool,
) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (ltf_h, ltf_l, htf_h, htf_l):
        for i, lvl in enumerate(pool.lvls):
            st = pool.state[i]
            if pool.drawn_act[i] and st != ST_BLACK and lvl > price:
                if math.isnan(best_lvl) or lvl < best_lvl:
                    best_lvl = lvl
                    best_state = st
    return best_state


def _next_state_below_non_black(
    price: float,
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
    htf_h: _LevelPool,
    htf_l: _LevelPool,
) -> int:
    best_lvl = math.nan
    best_state = -1
    for pool in (ltf_h, ltf_l, htf_h, htf_l):
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
    ltf_h: _LevelPool,
    ltf_l: _LevelPool,
    htf_h: _LevelPool,
    htf_l: _LevelPool,
) -> bool:
    lo = min(p1, p2)
    hi = max(p1, p2)
    for pool in (ltf_h, ltf_l, htf_h, htf_l):
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

    ltf_h = _LevelPool(max_size=p.max_ltf_h)
    ltf_l = _LevelPool(max_size=p.max_ltf_l)
    htf_h = _LevelPool(max_size=p.max_htf_h)
    htf_l = _LevelPool(max_size=p.max_htf_l)

    ltf_high_hist: list[float] = []
    ltf_low_hist: list[float] = []
    ltf_close_hist: list[float] = []
    ltf_time_hist: list[pd.Timestamp] = []
    ltf_atr_hist: list[float] = []

    htf_high_hist: list[float] = []
    htf_low_hist: list[float] = []
    htf_close_hist: list[float] = []
    htf_time_hist: list[pd.Timestamp] = []
    htf_atr_hist: list[float] = []

    ltf_prev_close: float | None = None
    ltf_prev_atr: float | None = None

    htf_prev_close: float | None = None
    htf_prev_atr: float | None = None

    htf_minutes = _timeframe_to_minutes(p.htf)
    curr_bucket: pd.Timestamp | None = None
    htf_open = math.nan
    htf_high = math.nan
    htf_low = math.nan
    htf_close = math.nan

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

        if p.show_ltf:
            _track_breach_high(ltf_h, h, t)
            _track_breach_low(ltf_l, l, t)

            if ltf_ph is not None and ltf_pivot_time is not None and ltf_pivot_atr is not None:
                _append_high_pivot(
                    pool=ltf_h,
                    pivot_value=float(ltf_ph),
                    pivot_time=ltf_pivot_time,
                    pivot_atr=float(ltf_pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                )

            if ltf_pl is not None and ltf_pivot_time is not None and ltf_pivot_atr is not None:
                _append_low_pivot(
                    pool=ltf_l,
                    pivot_value=float(ltf_pl),
                    pivot_time=ltf_pivot_time,
                    pivot_atr=float(ltf_pivot_atr),
                    atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                    liq_move_away_atr=p.liq_move_away_atr,
                )

            _confirm_move_away_high(ltf_h, c)
            _confirm_move_away_low(ltf_l, c)

            trig_short, trig_long = _ltf_sweep_triggers(ltf_h, ltf_l, h, l)

        # ---- HTF aggregation + state machine ----
        bucket = _bucket_floor(t, htf_minutes)
        if curr_bucket is None:
            curr_bucket = bucket
            htf_open = o
            htf_high = h
            htf_low = l
            htf_close = c
        elif bucket == curr_bucket:
            htf_high = max(htf_high, h)
            htf_low = min(htf_low, l)
            htf_close = c
        else:
            fin_time = curr_bucket
            fin_high = float(htf_high)
            fin_low = float(htf_low)
            fin_close = float(htf_close)

            tr_htf = _true_range(fin_high, fin_low, htf_prev_close)
            htf_atr = tr_htf if htf_prev_atr is None else ((htf_prev_atr * (p.atr_len - 1)) + tr_htf) / p.atr_len
            htf_prev_close = fin_close
            htf_prev_atr = htf_atr

            htf_high_hist.append(fin_high)
            htf_low_hist.append(fin_low)
            htf_close_hist.append(fin_close)
            htf_time_hist.append(fin_time)
            htf_atr_hist.append(float(htf_atr))

            htf_ph = None
            htf_pl = None
            htf_pivot_time = None
            htf_pivot_atr = None

            hc_idx = len(htf_high_hist) - 1 - p.htf_pivot_len
            if hc_idx >= p.htf_pivot_len:
                htf_ph = _pivot_high(htf_high_hist, hc_idx, p.htf_pivot_len, p.htf_pivot_len)
                htf_pl = _pivot_low(htf_low_hist, hc_idx, p.htf_pivot_len, p.htf_pivot_len)
                htf_pivot_time = htf_time_hist[hc_idx]
                htf_pivot_atr = float(htf_atr_hist[hc_idx])

            if p.show_htf:
                _track_breach_high(htf_h, fin_high, fin_time)
                _track_breach_low(htf_l, fin_low, fin_time)

                if htf_ph is not None and htf_pivot_time is not None and htf_pivot_atr is not None:
                    _append_high_pivot(
                        pool=htf_h,
                        pivot_value=float(htf_ph),
                        pivot_time=htf_pivot_time,
                        pivot_atr=float(htf_pivot_atr),
                        atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                        liq_move_away_atr=p.liq_move_away_atr,
                    )

                if htf_pl is not None and htf_pivot_time is not None and htf_pivot_atr is not None:
                    _append_low_pivot(
                        pool=htf_l,
                        pivot_value=float(htf_pl),
                        pivot_time=htf_pivot_time,
                        pivot_atr=float(htf_pivot_atr),
                        atr_dist_for_liq_generation=p.atr_dist_for_liq_generation,
                        liq_move_away_atr=p.liq_move_away_atr,
                    )

                _confirm_move_away_high(htf_h, fin_close)
                _confirm_move_away_low(htf_l, fin_close)

            curr_bucket = bucket
            htf_open = o
            htf_high = h
            htf_low = l
            htf_close = c

        if p.show_htf:
            _stop_extending_high(htf_h, h)
            _stop_extending_low(htf_l, l)

        # ---- Strategy entries (signal bar -> enter next bar open) ----
        flat = pos is None
        entry_px = c
        rr_band_valid = p.max_rr > p.min_rr
        sl_buffer = p.sl_buffer_pips * p.pip_size

        if flat and trig_short and i < len(df) - 1:
            sl_raw = _next_htf_purple_high_above(entry_px, htf_h)
            sl = (sl_raw + sl_buffer) if sl_raw is not None else None
            tp = _next_htf_red_below(entry_px, htf_h, htf_l)

            ok_sl = sl is not None and sl > entry_px
            ok_tp = tp is not None and tp < entry_px

            risk_dist = (sl - entry_px) if ok_sl else math.nan
            reward_dist = (entry_px - tp) if ok_tp and tp is not None else math.nan
            rr = (reward_dist / risk_dist) if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)) else math.nan
            ok_rr = rr_band_valid and math.isfinite(rr) and rr > p.min_rr and rr <= p.max_rr

            has_red_between = True
            if ok_sl and sl is not None:
                has_red_between = _any_red_between(entry_px, sl, ltf_h, ltf_l, htf_h, htf_l)

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
            sl_raw = _next_htf_purple_low_below(entry_px, htf_l)
            sl = (sl_raw - sl_buffer) if sl_raw is not None else None
            tp = _next_htf_red_above(entry_px, htf_h, htf_l)

            ok_sl = sl is not None and sl < entry_px
            ok_tp = tp is not None and tp > entry_px

            risk_dist = (entry_px - sl) if ok_sl else math.nan
            reward_dist = (tp - entry_px) if ok_tp and tp is not None else math.nan
            rr = (reward_dist / risk_dist) if (math.isfinite(risk_dist) and risk_dist > 0 and math.isfinite(reward_dist)) else math.nan
            ok_rr = rr_band_valid and math.isfinite(rr) and rr > p.min_rr and rr <= p.max_rr

            has_red_between = True
            if ok_sl and sl is not None:
                has_red_between = _any_red_between(entry_px, sl, ltf_h, ltf_l, htf_h, htf_l)

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
                next_state = _next_state_above_non_black(entry_px, ltf_h, ltf_l, htf_h, htf_l)
                if next_state == ST_PURPLE:
                    pending_market_exit = {
                        "entry_i": i + 1,
                        "reason": "Nearest above is purple",
                    }
            else:
                next_state = _next_state_below_non_black(entry_px, ltf_h, ltf_l, htf_h, htf_l)
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
    return equity_df, trades_df, summary


PARAM_SPACE = {
    "min_rr": [2.0, 2.5, 3.0],
    "max_rr": [8.0, 10.0, 12.0],
    "atr_dist_for_liq_generation": [0.8, 1.0, 1.2],
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
                "name": "interequity_liqsweep_entry",
                "params": {
                    "min_rr": 2.0,
                    "max_rr": 10.0,
                    "atr_dist_for_liq_generation": 1.0,
                    "htf": "60",
                },
            }
        ],
    },
    "exit": {
        "name": "interequity_liqsweep_exit",
        "params": {"min_rr": 2.0, "sl_buffer_pips": 1.0, "pip_size": 0.0001},
    },
    "sizing": {
        "name": "fixed_risk",
        "params": {"risk_pct": 0.01},
    },
}

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import inspect
import math

import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.core.indicators import simple_atr
from quantbt.core.performance import build_backtest_summary
from quantbt.core.trades import close_trade_with_costs, resolve_intrabar_bracket_exit
from quantbt.strategies._contracts.common import build_param_space_from_limited_test, min_max_rr_constraint
from quantbt.strategies.interequity_2026_03_liqsweep_a import (
    InterEquityLiqSweepParams,
    ST_BLACK,
    ST_PURPLE,
    ST_RED,
    _LevelPool,
    _close_pending_market_exit,
    _open_position_from_entry_spec,
    _schedule_force_exit,
    _structural_entry_gate as _base_structural_entry_gate,
    append_high_pivot,
    append_low_pivot,
    any_red_between,
    build_brackets_from_signal,
    compute_features,
    compute_signals,
    confirm_move_away_high,
    confirm_move_away_low,
    pivot_high,
    pivot_low,
    stop_extending_high,
    stop_extending_low,
    sweep_triggers,
    track_breach_high,
    track_breach_low,
    true_range,
)


STRATEGY = deepcopy(
    {
        "name": "IE2026-03 LiqSweep C",
        "entry": {
            "mode": "all",
            "rules": [
                {
                    "name": "interequity_liqsweepc_entry",
                    "params": {
                        "min_rr": 1.5,
                        "max_rr": 10.0,
                        "atr_dist_for_liq_generation": 1.0,
                        "liq_move_away_atr": 3.0,
                    },
                }
            ],
        },
        "exit": {
            "name": "interequity_liqsweepc_exit",
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
)


PARAM_SPACE = build_param_space_from_limited_test(STRATEGY)


def constraints(params: dict) -> bool:
    return min_max_rr_constraint(params)


@dataclass(frozen=True)
class InterEquityLiqSweepCParams(InterEquityLiqSweepParams):
    min_rr: float = 1.5
    max_sl_atr_mult: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "max_sl_atr_mult", float(self.max_sl_atr_mult))
        if self.max_sl_atr_mult <= 0:
            raise ValueError("max_sl_atr_mult must be > 0")


Params = InterEquityLiqSweepCParams


def _structural_entry_gate(
    *,
    side: str,
    entry_price: float,
    structural_atr: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    strategy_params: InterEquityLiqSweepCParams,
) -> dict[str, float] | None:
    gate = _base_structural_entry_gate(
        side=side,
        entry_price=float(entry_price),
        high_pool=high_pool,
        low_pool=low_pool,
        strategy_params=strategy_params,
    )
    if gate is None:
        return None
    if not math.isfinite(float(structural_atr)) or float(structural_atr) <= 0:
        return None

    max_stop_dist = float(strategy_params.max_sl_atr_mult) * float(structural_atr)
    if not math.isfinite(max_stop_dist) or max_stop_dist <= 0:
        return None
    if float(gate["risk_dist"]) > max_stop_dist:
        return None
    return gate


def _build_entry_spec_from_exit_override(
    *,
    side: str,
    entry_index: int,
    entry_time: pd.Timestamp,
    signal_price: float,
    entry_open: float,
    prev_low: float,
    prev_high: float,
    structural_atr: float,
    entry_atr: float | None,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    strategy_params: InterEquityLiqSweepCParams,
    cfg: BacktestConfig,
    equity: float,
    exit_builder,
    exit_params: dict[str, Any],
    exit_supports_entry: bool,
    size_fn=None,
    sizing_params: dict[str, Any] | None = None,
    max_exit_index: int | None = None,
) -> dict[str, Any] | None:
    del signal_price
    structural_gate = _structural_entry_gate(
        side=side,
        entry_price=float(entry_open),
        structural_atr=float(structural_atr),
        high_pool=high_pool,
        low_pool=low_pool,
        strategy_params=strategy_params,
    )
    if structural_gate is None:
        return None

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
    if not isinstance(exit_spec, dict):
        return None

    if "hold_bars" in exit_spec:
        try:
            hold_bars = int(exit_spec["hold_bars"])
        except Exception:
            return None
        if hold_bars < 0:
            return None

        sizing_params = dict(sizing_params or {})
        structural_stop_dist = float(structural_gate["risk_dist"])
        if not (math.isfinite(structural_stop_dist) and structural_stop_dist > 0):
            return None
        use_structural_stop_sizing = bool(exit_params.get("use_structural_stop_sizing"))
        if use_structural_stop_sizing:
            risk_amount = float(cfg.initial_equity) * float(strategy_params.risk_pct)
            qty = risk_amount / structural_stop_dist
            risk_dollars = float(risk_amount)
        elif callable(size_fn):
            qty = size_fn(
                cfg=cfg,
                equity=float(equity),
                side=str(side),
                entry_open=float(entry_open),
                exit_spec={
                    "hold_bars": hold_bars,
                    "sl": float(structural_gate["sl"]),
                    "tp": float(structural_gate["tp"]),
                    "stop_dist": float(structural_stop_dist),
                },
                entry=entry_ctx,
                params=sizing_params,
            )
            risk_dollars = float(qty) * structural_stop_dist if qty is not None and math.isfinite(float(qty)) else float("nan")
        else:
            risk_amount = float(cfg.initial_equity) * float(strategy_params.risk_pct)
            qty = risk_amount / structural_stop_dist
            risk_dollars = float(risk_amount)
        if qty is None or not math.isfinite(float(qty)) or float(qty) <= 0:
            return None

        time_exit_i = int(entry_index + hold_bars)
        if max_exit_index is not None:
            time_exit_i = min(time_exit_i, int(max_exit_index))

        return {
            "entry_i": int(entry_index),
            "side": str(side),
            "qty": float(qty),
            "sl": float("nan"),
            "tp": float("nan"),
            "risk_dollars": float(risk_dollars) if math.isfinite(float(risk_dollars)) else float("nan"),
            "time_exit_i": int(time_exit_i),
        }

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
    elif side == "short":
        ok_sl = sl > entry_open
        ok_tp = tp < entry_open
    else:
        raise ValueError("side must be 'long' or 'short'")
    if not (ok_sl and ok_tp):
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


def _build_entry_spec(
    *,
    side: str,
    entry_index: int,
    entry_price: float,
    structural_atr: float,
    high_pool: _LevelPool,
    low_pool: _LevelPool,
    strategy_params: InterEquityLiqSweepCParams,
    cfg: BacktestConfig,
) -> dict[str, Any] | None:
    structural_gate = _structural_entry_gate(
        side=side,
        entry_price=float(entry_price),
        structural_atr=float(structural_atr),
        high_pool=high_pool,
        low_pool=low_pool,
        strategy_params=strategy_params,
    )
    if structural_gate is None:
        return None

    risk_amount = float(cfg.initial_equity) * float(strategy_params.risk_pct)
    sl = float(structural_gate["sl"])
    tp = float(structural_gate["tp"])
    stop_dist = max(abs(float(entry_price) - sl), float(strategy_params.min_tick))
    qty = risk_amount / stop_dist
    if not math.isfinite(qty) or qty <= 0:
        return None

    return {
        "entry_i": int(entry_index),
        "side": str(side),
        "qty": float(qty),
        "sl": float(sl),
        "tp": float(tp),
        "risk_dollars": float(risk_amount),
    }


def run_backtest(
    df_sig: pd.DataFrame,
    *,
    strategy_params: InterEquityLiqSweepCParams,
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

        if pos is not None and math.isfinite(float(pos.get("sl", float("nan")))) and math.isfinite(float(pos.get("tp", float("nan")))):
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

        short_entry_price: float | None = None
        long_entry_price: float | None = None

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
            short_entry_price, long_entry_price = sweep_triggers(
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

        opened_this_bar = False

        if pos is None and short_entry_price is not None:
            if exit_override_active:
                entry_atr = None
                if override_atr_series is not None:
                    entry_atr = float(override_atr_series.at[t])
                entry_spec = _build_entry_spec_from_exit_override(
                    side="short",
                    entry_index=i,
                    entry_time=t,
                    signal_price=float(short_entry_price),
                    entry_open=float(short_entry_price),
                    prev_low=l,
                    prev_high=h,
                    structural_atr=float(atr),
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
                    max_exit_index=len(df) - 1,
                )
            else:
                entry_spec = _build_entry_spec(
                    side="short",
                    entry_index=i,
                    entry_price=float(short_entry_price),
                    structural_atr=float(atr),
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                )
            if entry_spec is not None:
                pos = _open_position_from_entry_spec(
                    entry_spec=entry_spec,
                    entry_price=float(short_entry_price),
                    entry_time=t,
                )
                opened_this_bar = True

        if pos is None and long_entry_price is not None:
            if exit_override_active:
                entry_atr = None
                if override_atr_series is not None:
                    entry_atr = float(override_atr_series.at[t])
                entry_spec = _build_entry_spec_from_exit_override(
                    side="long",
                    entry_index=i,
                    entry_time=t,
                    signal_price=float(long_entry_price),
                    entry_open=float(long_entry_price),
                    prev_low=l,
                    prev_high=h,
                    structural_atr=float(atr),
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
                    max_exit_index=len(df) - 1,
                )
            else:
                entry_spec = _build_entry_spec(
                    side="long",
                    entry_index=i,
                    entry_price=float(long_entry_price),
                    structural_atr=float(atr),
                    high_pool=high_pool,
                    low_pool=low_pool,
                    strategy_params=p,
                    cfg=cfg,
                )
            if entry_spec is not None:
                pos = _open_position_from_entry_spec(
                    entry_spec=entry_spec,
                    entry_price=float(long_entry_price),
                    entry_time=t,
                )
                opened_this_bar = True

        if opened_this_bar and pos is not None and math.isfinite(float(pos.get("sl", float("nan")))) and math.isfinite(float(pos.get("tp", float("nan")))):
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

        if pos is not None and "time_exit_i" in pos and int(pos["time_exit_i"]) <= i:
            equity, trade = close_trade_with_costs(
                pos=pos,
                exit_price=float(c),
                exit_time=t,
                exit_reason="TIME_EXIT",
                equity_now=equity,
                cfg=cfg,
            )
            trades.append(trade)
            pos = None
            pending_market_exit = None

        if not exit_override_active and i < len(df) - 1:
            pending_market_exit = _schedule_force_exit(
                pos=pos,
                pending_market_exit=pending_market_exit,
                entry_price=c,
                high_pool=high_pool,
                low_pool=low_pool,
                next_bar_index=i + 1,
            )

        equity_curve.append({"time": t, "equity": equity})

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
    "InterEquityLiqSweepCParams",
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

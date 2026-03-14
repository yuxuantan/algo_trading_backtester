from __future__ import annotations

import importlib
import math
import pandas as pd

from quantbt.core.engine import BacktestConfig
from quantbt.plugins.registry import register_entry


_STRAT_MODULE = "quantbt.strategies.interequity_2026_03_liqsweep_b"
_CACHED_BRACKETS: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}


def _load_strategy_module():
    return importlib.import_module(_STRAT_MODULE)


@register_entry("interequity_liqsweepb_entry")
def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Entry plugin backed by the full InterEquity LiqSweepB state-machine.

    It runs the strategy's own custom backtest once and maps executed entry times
    back into plugin long/short entry signals (signal bar = same bar as strategy entry).
    """
    mod = _load_strategy_module()
    p = mod.Params(**params)

    df_feat = mod.compute_features(df, p)
    df_sig = mod.compute_signals(df_feat)

    # Use strategy-native simulation to identify canonical entry times.
    _, trades_df, _ = mod.run_backtest(df_sig, strategy_params=p, cfg=BacktestConfig())

    out = pd.DataFrame(index=df_sig.index)
    out["long_entry"] = False
    out["short_entry"] = False

    if trades_df is None or trades_df.empty:
        return out

    idx = out.index
    idx_pos = {ts: i for i, ts in enumerate(idx)}
    _cache_trade_brackets(trades_df, idx=idx, idx_pos=idx_pos)

    for _, tr in trades_df.iterrows():
        et = pd.Timestamp(tr["entry_time"])
        side = str(tr["side"])
        pos = idx_pos.get(et)
        # LiqSweepB enters on signal-bar close. Plugin engine can only enter on next bar open,
        # so we emit signal on the same bar and map cached brackets to the next timestamp.
        if pos is None or pos >= len(idx) - 1:
            continue
        signal_t = idx[pos]
        if side == "long":
            out.at[signal_t, "long_entry"] = True
        elif side == "short":
            out.at[signal_t, "short_entry"] = True

    return out


def _cache_trade_brackets(trades_df: pd.DataFrame | None, *, idx: pd.Index, idx_pos: dict[pd.Timestamp, int]) -> None:
    global _CACHED_BRACKETS
    _CACHED_BRACKETS = {}

    if trades_df is None or trades_df.empty:
        return

    for _, tr in trades_df.iterrows():
        strategy_entry_time = pd.Timestamp(tr["entry_time"])
        side = str(tr["side"])
        entry_px = float(tr.get("entry", float("nan")))
        sl = float(tr.get("sl", float("nan")))
        tp = float(tr.get("tp", float("nan")))

        pos = idx_pos.get(strategy_entry_time)
        if pos is None or pos >= len(idx) - 1:
            continue
        engine_entry_time = pd.Timestamp(idx[pos + 1])

        if not (math.isfinite(entry_px) and math.isfinite(sl) and math.isfinite(tp)):
            continue

        if side == "long":
            stop_dist = entry_px - sl
        elif side == "short":
            stop_dist = sl - entry_px
        else:
            continue

        if not math.isfinite(stop_dist) or stop_dist <= 0:
            continue

        _CACHED_BRACKETS[(engine_entry_time, side)] = {
            "sl": sl,
            "tp": tp,
            "stop_dist": stop_dist,
        }


def get_cached_bracket(entry_time, side: str) -> dict[str, float] | None:
    key = (pd.Timestamp(entry_time), str(side))
    data = _CACHED_BRACKETS.get(key)
    if data is None:
        return None
    return dict(data)


def _validate(params: dict) -> bool:
    try:
        mod = _load_strategy_module()
        mod.Params(**params)
    except Exception:
        return False
    return True


signals.validate = _validate

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FixedBracketExitParams:
    rr: float
    sl_buffer_pips: float
    pip_size: float = 0.0001


def build_fixed_brackets(side: str, entry_open: float, prev_low: float, prev_high: float, p: FixedBracketExitParams):
    sl_buffer = float(p.sl_buffer_pips) * float(p.pip_size)

    if side == "long":
        sl = prev_low - sl_buffer
        stop_dist = entry_open - sl
        if stop_dist <= 0:
            return None
        tp = entry_open + float(p.rr) * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    if side == "short":
        sl = prev_high + sl_buffer
        stop_dist = sl - entry_open
        if stop_dist <= 0:
            return None
        tp = entry_open - float(p.rr) * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    raise ValueError("side must be long/short")


@dataclass(frozen=True)
class TimeExitParams:
    hold_bars: int  # exit N bars after entry (at bar close, for simplicity)


def build_time_exit(side: str, entry_open: float, prev_low: float, prev_high: float, p: TimeExitParams):
    # no SL/TP; engine will close after N bars
    hb = int(p.hold_bars)
    if hb <= 0:
        return None
    return {"hold_bars": hb}

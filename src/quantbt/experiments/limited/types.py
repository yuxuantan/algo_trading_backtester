from __future__ import annotations

from typing import Any, Literal, TypedDict


class EntryEvent(TypedDict, total=False):
    """Canonical entry payload passed between signal builders, monkey schedulers, and engines."""

    entry_i: int
    entry_time: Any
    side: Literal["long", "short"]
    entry_open: float
    prev_low: float
    prev_high: float
    atr: float
    monkey_hold_bars: int


class ExitSpec(TypedDict, total=False):
    """Canonical exit payload emitted by exit plugins."""

    hold_bars: int
    sl: float
    tp: float
    stop_dist: float


class ScheduleMetrics(TypedDict):
    """Flat-only schedule metrics used for monkey prefiltering."""

    trades: float
    long_trade_pct: float
    avg_bars_held: float

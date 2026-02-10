from __future__ import annotations

from quantbt.plugins.registry import register_exit


@register_exit("time_exit")
def build_exit(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
    hb = int(params.get("hold_bars", 0))
    if hb <= 0:
        return None
    return {"hold_bars": hb}

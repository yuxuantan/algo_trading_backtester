from __future__ import annotations

import math
import random

import pandas as pd

from quantbt.experiments.limited.types import EntryEvent, ScheduleMetrics


def _maybe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _build_monkey_scheduler_seed(*, entry_params: dict, exit_params: dict) -> int:
    entry_seed = _maybe_int(entry_params.get("seed", 7), 7)
    exit_seed = _maybe_int(exit_params.get("seed", 7), 7)
    return ((entry_seed * 1_000_003) ^ (exit_seed * 9_176)) & 0xFFFFFFFF


def _random_composition(total: int, parts: int, *, rng: random.Random) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be > 0")
    if total < 0:
        raise ValueError("total must be >= 0")
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for c in cuts:
        out.append(int(c - prev - 1))
        prev = c
    out.append(int((total + parts - 1) - prev - 1))
    return out


def _sample_trade_count_within_tolerance(*, target_trades: float, trade_tol_pct: float, rng: random.Random) -> int:
    tol = abs(float(trade_tol_pct)) / 100.0
    lo = max(1, int(math.ceil(float(target_trades) * (1.0 - tol))))
    hi = max(lo, int(math.floor(float(target_trades) * (1.0 + tol))))
    return int(rng.randint(lo, hi))


def _sample_random_holds_with_target_mean(
    *,
    n_trades: int,
    target_avg_hold: float,
    rng: random.Random,
    min_hold_bars: int = 1,
    max_hold_bars: int | None = None,
) -> list[int]:
    if n_trades <= 0:
        return []
    if not math.isfinite(float(target_avg_hold)) or float(target_avg_hold) <= 0:
        raise ValueError("target_avg_hold must be > 0")

    min_hb = max(1, int(min_hold_bars))
    max_hb = None if max_hold_bars is None else max(min_hb, int(max_hold_bars))

    target = float(target_avg_hold)
    if max_hb is not None:
        target = min(target, float(max_hb))
    target = max(target, float(min_hb))

    total_f = target * float(n_trades)
    total_lo = int(math.floor(total_f))
    total = total_lo + (1 if rng.random() < (total_f - total_lo) else 0)
    min_total = n_trades * min_hb
    max_total = None if max_hb is None else (n_trades * max_hb)
    total = max(total, min_total)
    if max_total is not None:
        total = min(total, max_total)

    base = int(total // n_trades)
    base = max(base, min_hb)
    if max_hb is not None:
        base = min(base, max_hb)
    holds = [int(base)] * n_trades
    delta = int(total - sum(holds))

    if delta > 0:
        idxs = list(range(n_trades))
        rng.shuffle(idxs)
        ptr = 0
        while delta > 0:
            if ptr >= len(idxs):
                ptr = 0
                rng.shuffle(idxs)
            j = idxs[ptr]
            ptr += 1
            if max_hb is not None and holds[j] >= max_hb:
                continue
            holds[j] += 1
            delta -= 1
    elif delta < 0:
        idxs = list(range(n_trades))
        rng.shuffle(idxs)
        ptr = 0
        while delta < 0:
            if ptr >= len(idxs):
                ptr = 0
                rng.shuffle(idxs)
            j = idxs[ptr]
            ptr += 1
            if holds[j] <= min_hb:
                continue
            holds[j] -= 1
            delta += 1

    # Add zero-sum jitter so integer target means don't collapse into all-equal holds.
    inc_candidates = [i for i, v in enumerate(holds) if max_hb is None or v < max_hb]
    dec_candidates = [i for i, v in enumerate(holds) if v > min_hb]
    if len(holds) >= 2 and inc_candidates and dec_candidates:
        max_pairs = min(len(inc_candidates), len(dec_candidates), max(1, len(holds) // 4))
        n_pairs = int(rng.randint(0, max_pairs))
        if n_pairs > 0:
            rng.shuffle(inc_candidates)
            rng.shuffle(dec_candidates)
            used_inc: set[int] = set()
            used_dec: set[int] = set()
            pairs_done = 0
            for inc_i in inc_candidates:
                if pairs_done >= n_pairs:
                    break
                if inc_i in used_inc:
                    continue
                dec_i = next((j for j in dec_candidates if j != inc_i and j not in used_dec), None)
                if dec_i is None:
                    continue
                holds[inc_i] += 1
                holds[dec_i] -= 1
                used_inc.add(inc_i)
                used_dec.add(dec_i)
                pairs_done += 1

    rng.shuffle(holds)
    return [int(v) for v in holds]


def _assign_monkey_sides_exact(
    *,
    n_trades: int,
    entry_params: dict,
    cfg: dict,
    rng: random.Random,
) -> list[str]:
    side_mode = str(entry_params.get("side", "both")).strip().lower()
    if side_mode == "long":
        return ["long"] * n_trades
    if side_mode == "short":
        return ["short"] * n_trades
    if side_mode != "both":
        raise ValueError("side must be long/short/both")

    target_long_pct = cfg.get("target_long_trade_pct")
    if target_long_pct is None:
        target_long_pct = 100.0 * float(entry_params.get("long_ratio", entry_params.get("long_pct", 0.5)))
    target_long_pct = max(0.0, min(100.0, float(target_long_pct)))
    tol_pp = abs(float(cfg.get("long_tol_pp", 5.0)))
    lo_pct = max(0.0, target_long_pct - tol_pp)
    hi_pct = min(100.0, target_long_pct + tol_pp)
    lo_n = max(0, int(math.ceil(n_trades * lo_pct / 100.0)))
    hi_n = min(n_trades, int(math.floor(n_trades * hi_pct / 100.0)))
    if lo_n > hi_n:
        n_long = int(round(n_trades * target_long_pct / 100.0))
        n_long = max(0, min(n_trades, n_long))
    else:
        n_long = int(rng.randint(lo_n, hi_n))
    sides = (["long"] * n_long) + (["short"] * (n_trades - n_long))
    rng.shuffle(sides)
    return sides


def build_exact_monkey_entries_for_time_exit(
    *,
    df_sig: pd.DataFrame,
    entry_params: dict,
    exit_params: dict,
    monkey_cfg: dict,
) -> tuple[list[EntryEvent], ScheduleMetrics]:
    n_bars = int(len(df_sig))
    if n_bars < 3:
        raise ValueError("not enough bars to build exact monkey schedule")

    rng = random.Random(_build_monkey_scheduler_seed(entry_params=entry_params, exit_params=exit_params))
    n_trades = _sample_trade_count_within_tolerance(
        target_trades=float(monkey_cfg["target_trades"]),
        trade_tol_pct=float(monkey_cfg.get("trade_tol_pct", 5.0)),
        rng=rng,
    )

    target_avg_hold = monkey_cfg.get("target_avg_hold_bars")
    if target_avg_hold is None:
        if exit_params.get("avg_hold_bars") is not None:
            target_avg_hold = float(exit_params["avg_hold_bars"])
        elif exit_params.get("hold_bars") is not None:
            target_avg_hold = float(exit_params["hold_bars"])
        else:
            raise ValueError("target avg hold missing for exact monkey scheduler")

    min_hb = int(exit_params.get("min_hold_bars", 1))
    max_hb_raw = exit_params.get("max_hold_bars")
    max_hb = None if max_hb_raw is None else int(max_hb_raw)
    holds = _sample_random_holds_with_target_mean(
        n_trades=n_trades,
        target_avg_hold=float(target_avg_hold),
        rng=rng,
        min_hold_bars=min_hb,
        max_hold_bars=max_hb,
    )

    # entry_i starts at >=1 (needs a previous bar), and we require no truncation of holds
    # so the realized mean stays aligned with the target.
    slack = (n_bars - 2) - int(sum(holds))
    if slack < 0:
        raise ValueError(
            f"insufficient bars for exact monkey schedule (need {sum(holds)+2}, have {n_bars})"
        )
    gaps = _random_composition(int(slack), int(n_trades) + 1, rng=rng)
    sides = _assign_monkey_sides_exact(n_trades=n_trades, entry_params=entry_params, cfg=monkey_cfg, rng=rng)

    idx = df_sig.index.to_list()
    starts: list[int] = []
    cur = 1 + int(gaps[0])
    for k in range(n_trades):
        starts.append(int(cur))
        cur = int(cur + int(holds[k]))
        if k < (n_trades - 1):
            cur = int(cur + int(gaps[k + 1]))

    entries: list[EntryEvent] = []
    long_count = 0
    for start_i, side, hb in zip(starts, sides, holds):
        if not (1 <= int(start_i) < n_bars):
            raise ValueError(f"generated entry index out of range: {start_i}")
        prev_i = int(start_i) - 1
        if side == "long":
            long_count += 1
        entries.append({
            "entry_i": int(start_i),
            "entry_time": idx[int(start_i)],
            "side": str(side),
            "entry_open": float(df_sig.iloc[int(start_i)]["open"]),
            "prev_low": float(df_sig.iloc[prev_i]["low"]),
            "prev_high": float(df_sig.iloc[prev_i]["high"]),
            "monkey_hold_bars": int(hb),
        })

    metrics: ScheduleMetrics = {
        "trades": float(n_trades),
        "long_trade_pct": float(100.0 * long_count / n_trades) if n_trades else float("nan"),
        "avg_bars_held": float(sum(int(v) for v in holds) / n_trades) if n_trades else float("nan"),
    }
    return entries, metrics


def build_precomputed_time_exit_wrapper(base_exit_fn):
    def _wrapped(side: str, entry_open: float, prev_low: float, prev_high: float, params: dict, entry=None):
        if isinstance(entry, dict) and entry.get("monkey_hold_bars") is not None:
            hb = int(entry["monkey_hold_bars"])
            if hb > 0:
                return {"hold_bars": hb}
        return base_exit_fn(side, entry_open, prev_low, prev_high, params, entry=entry)

    return _wrapped

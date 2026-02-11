from __future__ import annotations

import random
import pandas as pd

from quantbt.plugins.registry import register_entry


def _draw_entry_indices(
    *,
    n_bars: int,
    target_entries: int,
    min_bars_between: int,
    rng: random.Random,
) -> list[int]:
    n_candidates = max(0, n_bars - 1)
    if target_entries < 0:
        raise ValueError("target_entries must be >= 0")
    if min_bars_between < 0:
        raise ValueError("min_bars_between must be >= 0")
    if target_entries == 0:
        return []
    if target_entries > n_candidates:
        raise ValueError(
            f"target_entries={target_entries} exceeds available bars={n_candidates}. "
            "Reduce target_entries or use more data."
        )

    if min_bars_between == 0:
        return sorted(rng.sample(range(n_candidates), target_entries))

    # Max entries possible with spacing constraint.
    max_possible = (n_candidates + min_bars_between) // (min_bars_between + 1)
    if target_entries > max_possible:
        raise ValueError(
            f"target_entries={target_entries} is impossible with min_bars_between={min_bars_between} "
            f"on {n_candidates} candidate bars (max {max_possible})."
        )

    # Fast path: repeatedly sample exact-size sets and accept when spacing constraint is met.
    attempts = 300
    for _ in range(attempts):
        picks = sorted(rng.sample(range(n_candidates), target_entries))
        if all((picks[k] - picks[k - 1]) > min_bars_between for k in range(1, len(picks))):
            return picks

    # Fallback: randomized greedy fill.
    candidates = list(range(n_candidates))
    rng.shuffle(candidates)
    picked: list[int] = []
    for i in candidates:
        if all(abs(i - j) > min_bars_between for j in picked):
            picked.append(i)
            if len(picked) == target_entries:
                return sorted(picked)

    raise ValueError(
        "Could not generate target_entries with the current min_bars_between/seed. "
        "Try lowering min_bars_between or changing seed."
    )


def _assign_long_short_flags(
    *,
    n_entries: int,
    side: str,
    long_ratio: float,
    rng: random.Random,
) -> list[bool]:
    if side == "long":
        return [True] * n_entries
    if side == "short":
        return [False] * n_entries
    if side != "both":
        raise ValueError("side must be long/short/both")

    if not 0.0 <= long_ratio <= 1.0:
        raise ValueError("long_ratio must be between 0 and 1")

    n_long = int(round(n_entries * long_ratio))
    n_long = max(0, min(n_entries, n_long))
    flags = [True] * n_long + [False] * (n_entries - n_long)
    rng.shuffle(flags)
    return flags


@register_entry("monkey_entry")
@register_entry("random")
def signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    seed = int(params.get("seed", 7))
    min_bars_between = int(params.get("min_bars_between", 0))
    side = str(params.get("side", "both")).strip().lower()
    prob = float(params.get("prob", 0.01))
    target_entries = params.get("target_entries")
    long_ratio = float(params.get("long_ratio", params.get("long_pct", 0.5)))

    rng = random.Random(seed)
    idx = df.index.to_list()

    long_sig = pd.Series(False, index=idx)
    short_sig = pd.Series(False, index=idx)
    n_candidates = max(0, len(idx) - 1)

    if target_entries is not None:
        picks = _draw_entry_indices(
            n_bars=len(idx),
            target_entries=int(target_entries),
            min_bars_between=min_bars_between,
            rng=rng,
        )
        flags = _assign_long_short_flags(
            n_entries=len(picks),
            side=side,
            long_ratio=long_ratio,
            rng=rng,
        )
        for i, is_long in zip(picks, flags):
            if is_long:
                long_sig.iat[i] = True
            else:
                short_sig.iat[i] = True
    else:
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")
        last_i = -10**9
        for i in range(n_candidates):
            if i - last_i <= min_bars_between:
                continue
            if rng.random() >= prob:
                continue

            if side == "both":
                is_long = rng.random() < long_ratio
            elif side == "long":
                is_long = True
            elif side == "short":
                is_long = False
            else:
                raise ValueError("side must be long/short/both")

            if is_long:
                long_sig.iat[i] = True
            else:
                short_sig.iat[i] = True
            last_i = i

    out = pd.DataFrame(index=idx)
    out["long_entry"] = long_sig
    out["short_entry"] = short_sig
    return out


def _validate(params: dict) -> bool:
    try:
        seed = int(params.get("seed", 7))
        _ = random.Random(seed)
        min_bars_between = int(params.get("min_bars_between", 0))
        if min_bars_between < 0:
            return False

        side = str(params.get("side", "both")).strip().lower()
        if side not in {"long", "short", "both"}:
            return False

        if "target_entries" in params and params.get("target_entries") is not None:
            target_entries = int(params["target_entries"])
            if target_entries < 0:
                return False
        else:
            prob = float(params.get("prob", 0.01))
            if not 0.0 <= prob <= 1.0:
                return False

        long_ratio = float(params.get("long_ratio", params.get("long_pct", 0.5)))
        if not 0.0 <= long_ratio <= 1.0:
            return False
    except Exception:
        return False
    return True


signals.validate = _validate

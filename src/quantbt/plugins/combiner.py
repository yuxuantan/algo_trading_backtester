from __future__ import annotations

import pandas as pd


def combine_signals(signals_list: list[pd.DataFrame], mode: str = "all", vote_k: int | None = None) -> pd.DataFrame:
    if not signals_list:
        raise ValueError("signals_list is empty")

    # align on intersection of indices
    idx = signals_list[0].index
    for s in signals_list[1:]:
        idx = idx.intersection(s.index)

    longs = []
    shorts = []
    for s in signals_list:
        s = s.loc[idx]
        longs.append(s["long_entry"].astype(bool))
        shorts.append(s["short_entry"].astype(bool))

    if mode == "all":
        long_comb = longs[0]
        short_comb = shorts[0]
        for l in longs[1:]:
            long_comb = long_comb & l
        for s in shorts[1:]:
            short_comb = short_comb & s
    elif mode == "any":
        long_comb = longs[0]
        short_comb = shorts[0]
        for l in longs[1:]:
            long_comb = long_comb | l
        for s in shorts[1:]:
            short_comb = short_comb | s
    elif mode == "vote":
        if vote_k is None:
            raise ValueError("vote_k is required for vote mode")
        long_votes = sum(l.astype(int) for l in longs)
        short_votes = sum(s.astype(int) for s in shorts)
        long_comb = long_votes >= int(vote_k)
        short_comb = short_votes >= int(vote_k)
    else:
        raise ValueError("mode must be all/any/vote")

    # suppress conflicting signals
    conflict = long_comb & short_comb
    if conflict.any():
        long_comb = long_comb & ~conflict
        short_comb = short_comb & ~conflict

    out = pd.DataFrame(index=idx)
    out["long_entry"] = long_comb
    out["short_entry"] = short_comb
    return out

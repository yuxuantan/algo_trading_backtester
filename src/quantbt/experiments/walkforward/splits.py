from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int


def validate_oos_ratio(is_bars: int, oos_bars: int, *, min_ratio: float = 0.10, max_ratio: float = 0.50) -> float:
    if is_bars <= 0:
        raise ValueError("is_bars must be > 0")
    if oos_bars <= 0:
        raise ValueError("oos_bars must be > 0")
    ratio = float(oos_bars) / float(is_bars)
    if ratio < min_ratio or ratio > max_ratio:
        raise ValueError(
            f"oos/is ratio must be between {min_ratio:.2f} and {max_ratio:.2f}. "
            f"Got is_bars={is_bars}, oos_bars={oos_bars}, ratio={ratio:.3f}."
        )
    return ratio


def build_walkforward_splits(
    *,
    total_bars: int,
    is_bars: int,
    oos_bars: int,
    step_bars: int | None = None,
    anchored: bool = True,
    start_bar: int = 0,
    end_bar: int | None = None,
) -> list[WalkForwardFold]:
    if total_bars <= 0:
        raise ValueError("total_bars must be > 0")
    if is_bars <= 0:
        raise ValueError("is_bars must be > 0")
    if oos_bars <= 0:
        raise ValueError("oos_bars must be > 0")
    if start_bar < 0:
        raise ValueError("start_bar must be >= 0")

    effective_end = int(total_bars if end_bar is None else min(end_bar, total_bars))
    if effective_end <= start_bar:
        raise ValueError("end_bar must be greater than start_bar")

    step = int(oos_bars if step_bars is None else step_bars)
    if step <= 0:
        raise ValueError("step_bars must be > 0")

    folds: list[WalkForwardFold] = []
    cursor = int(start_bar) + int(is_bars)
    fold_id = 1
    while cursor + oos_bars <= effective_end:
        if anchored:
            is_start = int(start_bar)
            is_end = int(cursor)
        else:
            is_start = int(cursor - is_bars)
            is_end = int(cursor)
        oos_start = int(cursor)
        oos_end = int(cursor + oos_bars)

        folds.append(
            WalkForwardFold(
                fold=fold_id,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
        fold_id += 1
        cursor += step

    if not folds:
        raise ValueError(
            "No walk-forward folds were generated. "
            "Check is_bars/oos_bars/start_bar/end_bar."
        )
    return folds

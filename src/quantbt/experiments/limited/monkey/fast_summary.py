from __future__ import annotations

import math

import pandas as pd

from quantbt.experiments.limited.types import EntryEvent


def run_backtest_limited_time_exit_fast_summary(
    df_sig: pd.DataFrame,
    *,
    cfg,
    entries: list[EntryEvent],
    build_exit_fn,
    exit_params: dict,
    size_fn=None,
    supports_entry_arg: bool,
) -> tuple[None, None, dict]:
    n_bars = int(len(df_sig))
    if n_bars <= 0:
        return None, None, {
            "trades": 0,
            "final_equity": float(cfg.initial_equity),
            "net_profit_abs": 0.0,
            "total_return_%": 0.0,
            "max_drawdown_%": 0.0,
            "max_drawdown_abs_%": 0.0,
            "avg_bars_held": float("nan"),
            "long_trade_pct": float("nan"),
            "short_trade_pct": float("nan"),
            "commission_sum": 0.0,
            "win_rate_%": float("nan"),
            "avg_profit_per_trade": float("nan"),
            "avg_mfe": float("nan"),
            "avg_mae": float("nan"),
            "avg_mfe_R": float("nan"),
            "avg_mae_R": float("nan"),
            "avg_giveback": float("nan"),
            "wins": 0,
            "losses": 0,
            "equity_years": float("nan"),
            "cagr_%": float("nan"),
            "sortino": float("nan"),
            "mar": float("nan"),
        }

    closes = df_sig["close"].astype(float).to_numpy()
    spread = float(cfg.spread_pips) * float(cfg.pip_size)
    equity = float(cfg.initial_equity)
    peak_equity = float(equity)
    mdd_frac = 0.0

    active_exit_i: int | None = None
    trade_count = 0
    long_count = 0
    short_count = 0
    win_count = 0
    loss_count = 0
    pnl_sum = 0.0
    commission_sum = 0.0
    bars_held_sum = 0.0

    for e in entries:
        try:
            entry_i = int(e.get("entry_i", -1))
        except Exception:
            continue
        if not (0 <= entry_i < n_bars):
            continue
        if active_exit_i is not None and entry_i < active_exit_i:
            continue

        side = str(e.get("side", "")).strip().lower()
        if side not in {"long", "short"}:
            continue
        try:
            entry_open = float(e["entry_open"])
            prev_low = float(e["prev_low"])
            prev_high = float(e["prev_high"])
        except Exception:
            continue

        if supports_entry_arg:
            exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params, entry=e)
        else:
            exit_spec = build_exit_fn(side, entry_open, prev_low, prev_high, exit_params)
        if not isinstance(exit_spec, dict) or "hold_bars" not in exit_spec:
            continue
        try:
            hold_bars = int(exit_spec["hold_bars"])
        except Exception:
            continue
        if hold_bars < 0:
            continue

        risk_dollars = equity * float(cfg.risk_pct)
        if size_fn is not None:
            units = size_fn(
                cfg=cfg,
                equity=equity,
                side=side,
                entry_open=entry_open,
                exit_spec=exit_spec,
                entry=e,
            )
        else:
            units = risk_dollars / (10 * cfg.pip_size)
        try:
            units = float(units)
        except Exception:
            continue
        if not math.isfinite(units) or units <= 0:
            continue

        exit_i = min(n_bars - 1, entry_i + hold_bars)
        exit_price = float(closes[exit_i])  # TIME_EXIT closes at close of exit bar

        commission = 0.0
        if cfg.commission_per_round_trip and cfg.lot_size:
            commission = (units / cfg.lot_size) * cfg.commission_per_round_trip
        if side == "long":
            entry_eff = entry_open + spread / 2
            exit_eff = exit_price - spread / 2
            pnl = (exit_eff - entry_eff) * units - commission
            long_count += 1
        else:
            entry_eff = entry_open - spread / 2
            exit_eff = exit_price + spread / 2
            pnl = (entry_eff - exit_eff) * units - commission
            short_count += 1

        equity = float(equity + pnl)
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            dd = (equity / peak_equity) - 1.0
            if dd < mdd_frac:
                mdd_frac = float(dd)

        trade_count += 1
        pnl_sum += float(pnl)
        commission_sum += float(commission)
        bars_held_sum += float(max(0, exit_i - entry_i))
        if pnl > 0:
            win_count += 1
        elif pnl < 0:
            loss_count += 1

        active_exit_i = int(exit_i)

    total_return_pct = ((equity / float(cfg.initial_equity)) - 1.0) * 100.0 if float(cfg.initial_equity) else float("nan")
    avg_profit = (pnl_sum / trade_count) if trade_count > 0 else float("nan")
    avg_bars_held = (bars_held_sum / trade_count) if trade_count > 0 else float("nan")
    long_trade_pct = (100.0 * long_count / trade_count) if trade_count > 0 else float("nan")
    short_trade_pct = (100.0 * short_count / trade_count) if trade_count > 0 else float("nan")
    win_rate_pct = (100.0 * win_count / trade_count) if trade_count > 0 else float("nan")
    mdd_abs_pct = abs(mdd_frac) * 100.0
    mdd_pct = mdd_frac * 100.0

    summary = {
        "trades": int(trade_count),
        "final_equity": float(equity),
        "net_profit_abs": float(equity - float(cfg.initial_equity)),
        "total_return_%": float(total_return_pct),
        "max_drawdown_%": float(mdd_pct),
        "max_drawdown_abs_%": float(mdd_abs_pct),
        "avg_bars_held": float(avg_bars_held) if math.isfinite(avg_bars_held) else float("nan"),
        "long_trade_pct": float(long_trade_pct) if math.isfinite(long_trade_pct) else float("nan"),
        "short_trade_pct": float(short_trade_pct) if math.isfinite(short_trade_pct) else float("nan"),
        "commission_sum": float(commission_sum),
        "win_rate_%": float(win_rate_pct) if math.isfinite(win_rate_pct) else float("nan"),
        "avg_profit_per_trade": float(avg_profit) if math.isfinite(avg_profit) else float("nan"),
        "avg_mfe": float("nan"),
        "avg_mae": float("nan"),
        "avg_mfe_R": float("nan"),
        "avg_mae_R": float("nan"),
        "avg_giveback": float("nan"),
        "wins": int(win_count),
        "losses": int(loss_count),
        "equity_years": float("nan"),
        "cagr_%": float("nan"),
        "sortino": float("nan"),
        "mar": float("nan"),
    }
    return None, None, summary

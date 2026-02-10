# Example:
# python scripts/run_entry_limited_fixed.py \
#   --data data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv \
#   --strategy sma_cross \
#   --entry-params '{"fast":50,"slow":200}' \
#   --fast-values 20,50,100 \
#   --slow-values 100,200,300 \
#   --rr 2.0 \
#   --sldist-atr-mult 1.5 \
#   --atr-period 14
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import time

import pandas as pd
from quantbt.io.dataio import load_ohlc_csv
from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.experiments.limited.base import limited_test_pass_rate
from quantbt.experiments.limited.runlog import make_limited_run_dir, write_json


def limited_progress_printer(i, total, elapsed, last_summary, favourable_so_far):
    pct = 100 * i / total
    rate = elapsed / i
    eta = rate * (total - i)
    fav_pct = 100 * favourable_so_far / i if i > 0 else 0.0

    print(
        f"[{i:>3}/{total}] "
        f"{pct:6.2f}% | "
        f"elapsed {elapsed:6.1f}s | "
        f"ETA {eta:6.1f}s | "
        f"favourable {fav_pct:5.1f}% | "
        f"last_ret {last_summary.get('total_return_%', float('nan')):6.2f}%",
        flush=True,
    )


def _to_dict(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    return dict(x)


def parse_list(value: str, cast):
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        items = [s for s in raw.split(",") if s.strip() != ""]
    return [cast(v) for v in items]


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


@dataclass(frozen=True)
class AtrBracketExitParams:
    rr: float
    sldist_atr: float
    atr_period: int = 14


def build_atr_brackets(side: str, entry_open: float, prev_low: float, prev_high: float, p: AtrBracketExitParams, entry=None):
    if entry is None:
        return None
    atr = float(entry.get("atr", float("nan")))
    if not math.isfinite(atr) or atr <= 0:
        return None
    stop_dist = float(p.sldist_atr) * atr
    if stop_dist <= 0:
        return None

    if side == "long":
        sl = entry_open - stop_dist
        tp = entry_open + float(p.rr) * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    if side == "short":
        sl = entry_open + stop_dist
        tp = entry_open - float(p.rr) * stop_dist
        return {"sl": sl, "tp": tp, "stop_dist": stop_dist}

    raise ValueError("side must be long/short")


def load_strategy(strategy_name: str):
    module_name = strategy_name if "." in strategy_name else f"quantbt.strategies.{strategy_name}"
    mod = importlib.import_module(module_name)
    for attr in ("Params", "compute_features", "compute_signals", "iter_entries"):
        if not hasattr(mod, attr):
            raise AttributeError(f"{module_name} missing required attribute: {attr}")
    return mod, module_name


def load_entry_params(mod, entry_params_arg: str | None, cfg: BacktestConfig):
    params_cls = mod.Params
    if entry_params_arg is None:
        params = params_cls()
        if hasattr(params, "pip_size"):
            try:
                params = params_cls(**{**_to_dict(params), "pip_size": cfg.pip_size})
            except TypeError:
                pass
        return params

    path = Path(entry_params_arg)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(entry_params_arg)
    if "pip_size" in getattr(params_cls, "__annotations__", {}) and "pip_size" not in data:
        data["pip_size"] = cfg.pip_size
    return params_cls(**data)


def build_cfg(args) -> BacktestConfig:
    kwargs = {}
    if args.initial_equity is not None:
        kwargs["initial_equity"] = args.initial_equity
    if args.risk_pct is not None:
        kwargs["risk_pct"] = args.risk_pct
    if args.spread_pips is not None:
        kwargs["spread_pips"] = args.spread_pips
    if args.pip_size is not None:
        kwargs["pip_size"] = args.pip_size
    if args.conservative_same_bar is not None:
        kwargs["conservative_same_bar"] = args.conservative_same_bar
    if args.min_stop_dist is not None:
        kwargs["min_stop_dist"] = args.min_stop_dist
    if args.commission_rt is not None:
        kwargs["commission_per_round_trip"] = args.commission_rt
    if args.lot_size is not None:
        kwargs["lot_size"] = args.lot_size
    return BacktestConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Entry robustness test: vary SMA, ATR-based SL/TP.")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV.")
    parser.add_argument("--ts-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--strategy", required=True, help="Strategy module (e.g., sma_cross).")
    parser.add_argument("--strategy-tag", default=None, help="Run folder tag; defaults to strategy module name.")
    parser.add_argument("--entry-params", default=None, help="JSON string or path to JSON file for entry params.")
    parser.add_argument("--run-base", default="runs/limited", help="Base directory for run logs.")
    parser.add_argument("--test-name", default="entry_test_fixed_brackets", help="Run test name tag.")

    parser.add_argument("--fast-values", required=True, help="Fast SMA values list (CSV or JSON list).")
    parser.add_argument("--slow-values", required=True, help="Slow SMA values list (CSV or JSON list).")
    parser.add_argument("--rr", type=float, required=True, help="Fixed RR for all iterations.")
    parser.add_argument("--sldist-atr-mult", type=float, required=True, help="SL distance in ATR multiples.")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period for SL distance.")

    parser.add_argument("--initial-equity", type=float, default=None)
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--spread-pips", type=float, default=None)
    parser.add_argument("--pip-size", type=float, default=None)
    parser.add_argument("--min-stop-dist", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commission-rt", type=float, default=5.0, help="Commission per round trip per 100k units.")
    parser.add_argument("--lot-size", type=float, default=100000.0, help="Units per standard lot for commission calc.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--conservative-same-bar", dest="conservative_same_bar", action="store_true")
    group.add_argument("--no-conservative-same-bar", dest="conservative_same_bar", action="store_false")
    parser.set_defaults(conservative_same_bar=None)

    args = parser.parse_args()

    cfg = build_cfg(args)
    strategy_mod, module_name = load_strategy(args.strategy)
    base_entry_params = load_entry_params(strategy_mod, args.entry_params, cfg)

    data_path = Path(args.data)
    df = load_ohlc_csv(str(data_path), ts_col=args.ts_col)
    atr_series = compute_atr(df, args.atr_period)

    fast_values = parse_list(args.fast_values, int)
    slow_values = parse_list(args.slow_values, int)
    if not fast_values or not slow_values:
        raise ValueError("fast-values and slow-values must be non-empty.")

    params_cls = strategy_mod.Params
    annotations = getattr(params_cls, "__annotations__", {})
    if "fast" not in annotations or "slow" not in annotations:
        raise ValueError(f"{module_name}.Params must define fast and slow to vary SMA values.")

    base_entry_dict = _to_dict(base_entry_params)
    entry_param_space = []
    for fast, slow in itertools.product(fast_values, slow_values):
        if slow <= fast:
            continue
        data = dict(base_entry_dict)
        data["fast"] = int(fast)
        data["slow"] = int(slow)
        if "pip_size" in annotations:
            data["pip_size"] = cfg.pip_size
        entry_param_space.append(params_cls(**data))
    if not entry_param_space:
        raise ValueError("No valid (fast, slow) pairs after filtering slow > fast.")

    fixed_exit_params = AtrBracketExitParams(rr=args.rr, sldist_atr=args.sldist_atr_mult, atr_period=args.atr_period)
    favourable_fn = lambda s: float(s.get("total_return_%", -999)) > 0

    strategy_tag = args.strategy_tag or module_name.split(".")[-1]
    run_dir = make_limited_run_dir(
        base=args.run_base,
        strategy=strategy_tag,
        dataset_tag=data_path.name,
        test_name=args.test_name,
    )

    pass_threshold = 70.0
    run_meta = {
        "strategy_module": module_name,
        "strategy_tag": strategy_tag,
        "dataset": str(data_path),
        "dataset_tag": data_path.name,
        "test_name": args.test_name,
        "entry_params_base": base_entry_dict,
        "entry_param_space": [asdict(p) for p in entry_param_space],
        "config": asdict(cfg),
        "atr_exit_params": asdict(fixed_exit_params),
        "favourable_rule": "total_return_% > 0",
        "pass_threshold_%": pass_threshold,
        "min_trades": args.min_trades,
        "seed": args.seed,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    rows = []
    start_ts = time.time()
    total = len(entry_param_space)

    def iter_entries_with_atr(df_in):
        for e in strategy_mod.iter_entries(df_in):
            atr = float(df_in.at[e["entry_time"], "atr"])
            if not math.isfinite(atr) or atr <= 0:
                continue
            e = dict(e)
            e["atr"] = atr
            yield e

    for i, entry_params in enumerate(entry_param_space, start=1):
        df_feat = strategy_mod.compute_features(df, entry_params)
        df_feat["atr"] = atr_series.reindex(df_feat.index)
        df_sig = strategy_mod.compute_signals(df_feat)

        _equity, _trades, summary = run_backtest_limited(
            df_sig,
            cfg=cfg,
            entry_iter_fn=iter_entries_with_atr,
            build_exit_fn=build_atr_brackets,
            exit_params=fixed_exit_params,
        )

        if summary.get("trades", 0) < args.min_trades:
            ok = False
        else:
            ok = bool(favourable_fn(summary))

        rows.append({
            "iter": i,
            **_to_dict(entry_params),
            **summary,
            "favourable": ok,
        })

        elapsed = time.time() - start_ts
        limited_progress_printer(
            i=i,
            total=total,
            elapsed=elapsed,
            last_summary=summary,
            favourable_so_far=sum(r["favourable"] for r in rows),
        )

    res_fixed = pd.DataFrame(rows)

    pass_fixed = limited_test_pass_rate(res_fixed)
    pass_summary = {
        "favourable_pct": pass_fixed,
        "pass_threshold_%": pass_threshold,
        "passed": pass_fixed >= pass_threshold,
        "total_iters": int(len(res_fixed)),
        "min_trades": args.min_trades,
    }
    write_json(run_dir / "pass_summary.json", pass_summary)
    res_fixed.to_csv(run_dir / "entry_fixed_brackets.csv", index=False)

    print(f"ATR SL/TP favourable%: {pass_fixed:.1f}%  -> {'PASS' if pass_fixed >= pass_threshold else 'FAIL'}")
    print(f"Saved: {run_dir}/entry_fixed_brackets.csv")
    print(f"Run meta: {run_dir}/run_meta.json")
    print(f"Pass summary: {run_dir}/pass_summary.json")


if __name__ == "__main__":
    main()

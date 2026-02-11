"""
Walk-forward CLI examples.

Proper anchored walk-forward test (OOS = 20% of IS, baseline enabled):
python3 scripts/run_walkforward.py \
  --strategy sma_cross_test_strat \
  --dataset data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv \
  --optimizer grid \
  --objective return_on_account \
  --direction maximize \
  --is-bars 18000 \
  --oos-bars 6000 \
  --step-bars 1200 \
  --min-trades 30 \
  --progress-every 10 \
  --commission-rt 5

Proper unanchored walk-forward test (same window lengths):
python3 scripts/run_walkforward.py \
  --strategy sma_cross_test_strat \
  --dataset data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv \
  --unanchored \
  --optimizer grid \
  --objective return_on_account \
  --direction maximize \
  --is-bars 18000 \
  --oos-bars 6000 \
  --step-bars 6000 \
  --min-trades 30 \
  --progress-every 10 \
  --commission-rt 5
"""

from __future__ import annotations

import argparse

from quantbt.core.engine import BacktestConfig
from quantbt.experiments.walkforward.runner import run_walkforward


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Walk-forward strategy tester")

    ap.add_argument("--strategy", required=True, help="Strategy module short name or full module path.")
    ap.add_argument("--dataset", required=True, help="Path to OHLC CSV.")
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--run-base", default="runs")

    ap.add_argument("--optimizer", default="grid", choices=["grid", "optuna"])
    ap.add_argument("--objective", default="return_on_account")
    ap.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    ap.add_argument("--min-trades", type=int, default=30)

    ap.add_argument("--param-space", default=None, help="JSON dict (or path) key->list. If omitted, uses strategy PARAM_SPACE.")

    ap.add_argument("--is-bars", type=int, required=True)
    ap.add_argument("--oos-bars", type=int, required=True)
    ap.add_argument("--step-bars", type=int, default=None, help="Default: oos-bars")
    ap.add_argument("--start-bar", type=int, default=0)
    ap.add_argument("--end-bar", type=int, default=None)
    ap.add_argument("--warmup-bars", type=int, default=None)

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--anchored", action="store_true")
    g.add_argument("--unanchored", action="store_true")

    ap.add_argument("--margin-rate", type=float, default=0.0, help="Required-margin proxy from max notional * margin_rate.")
    ap.add_argument("--required-margin-abs", type=float, default=None, help="Fixed required margin override for ROA denominator.")

    ap.add_argument("--no-baseline-full-data", action="store_true", help="Disable full-data optimize baseline comparison.")
    ap.add_argument("--no-compound-oos", action="store_true", help="Disable equity compounding across OOS folds.")

    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--timeout-s", type=int, default=0)
    ap.add_argument("--sampler", default="tpe", choices=["tpe", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=20, help="Print optimizer progress every N iterations/trials.")

    ap.add_argument("--initial-equity", type=float, default=100_000.0)
    ap.add_argument("--risk-pct", type=float, default=0.01)
    ap.add_argument("--spread-pips", type=float, default=0.2)
    ap.add_argument("--pip-size", type=float, default=0.0001)
    ap.add_argument("--conservative-same-bar", action="store_true")
    ap.add_argument("--commission-rt", type=float, default=0.0, help="Commission per round trip (USD per standard lot).")
    ap.add_argument("--lot-size", type=float, default=100_000.0, help="Lot size in units for commission scaling.")
    return ap


def main():
    args = build_parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")

    anchored = not args.unanchored
    if args.anchored:
        anchored = True

    cfg = BacktestConfig(
        initial_equity=args.initial_equity,
        risk_pct=args.risk_pct,
        spread_pips=args.spread_pips,
        pip_size=args.pip_size,
        conservative_same_bar=args.conservative_same_bar,
        commission_per_round_trip=args.commission_rt,
        lot_size=args.lot_size,
    )

    run_dir = run_walkforward(
        strategy=args.strategy,
        dataset=args.dataset,
        ts_col=args.ts_col,
        run_base=args.run_base,
        optimizer=args.optimizer,
        objective=args.objective,
        direction=args.direction,
        min_trades=args.min_trades,
        is_bars=args.is_bars,
        oos_bars=args.oos_bars,
        step_bars=args.step_bars,
        anchored=anchored,
        start_bar=args.start_bar,
        end_bar=args.end_bar,
        warmup_bars=args.warmup_bars,
        margin_rate=args.margin_rate,
        required_margin_abs=args.required_margin_abs,
        baseline_full_data=not args.no_baseline_full_data,
        compound_oos=not args.no_compound_oos,
        n_trials=args.n_trials,
        timeout_s=args.timeout_s,
        sampler=args.sampler,
        seed=args.seed,
        param_space_arg=args.param_space,
        progress_every=args.progress_every,
        cfg=cfg,
    )
    print(f"Saved walk-forward run: {run_dir}")


if __name__ == "__main__":
    main()

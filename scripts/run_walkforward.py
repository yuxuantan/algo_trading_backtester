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

Stability-first walk-forward optimization (plateau + WFE filters):
python3 scripts/run_walkforward.py \
  --strategy sma_cross_test_strat \
  --dataset data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv \
  --optimizer grid \
  --optimization-mode stability_robustness \
  --selection-mode plateau \
  --objective return_on_account \
  --wfe-metric total_return_% \
  --wfe-min-pct 50 \
  --max-top-trade-share 0.30 \
  --is-bars 18000 \
  --oos-bars 6000 \
  --step-bars 6000
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
    ap.add_argument(
        "--optimization-mode",
        default="peak",
        choices=["peak", "stability_robustness"],
        help="`peak` picks best objective; `stability_robustness` favors plateau stability + stricter filters.",
    )
    ap.add_argument("--selection-mode", default=None, choices=["peak", "plateau"])
    ap.add_argument("--objective", default="return_on_account")
    ap.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-is-trades", type=int, default=None)
    ap.add_argument("--min-oos-trades", type=int, default=None)
    ap.add_argument("--max-top-trade-share", type=float, default=None, help="Reject IS candidates where top winning trade > this share of gross profit.")
    ap.add_argument("--wfe-metric", default="total_return_%")
    ap.add_argument("--wfe-min-pct", type=float, default=None, help="Walk-forward efficiency pass threshold in percent.")
    ap.add_argument("--plateau-min-neighbors", type=int, default=3)
    ap.add_argument("--plateau-stability-penalty", type=float, default=0.5)

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


def _resolve_optimization_controls(args: argparse.Namespace) -> dict[str, float | int | str]:
    mode = str(args.optimization_mode)
    if mode == "stability_robustness":
        selection_mode = args.selection_mode or "plateau"
        min_is_base = int(args.min_is_trades) if args.min_is_trades is not None else int(args.min_trades)
        min_oos_base = int(args.min_oos_trades) if args.min_oos_trades is not None else int(args.min_trades)
        min_is_trades = max(min_is_base, 50)
        min_oos_trades = max(min_oos_base, 50)
        max_top_base = float(args.max_top_trade_share) if args.max_top_trade_share is not None else 0.30
        max_top_trade_share = (
            min(max_top_base, 0.30)
        )
        wfe_base = float(args.wfe_min_pct) if args.wfe_min_pct is not None else 50.0
        wfe_min_pct = max(wfe_base, 50.0)
    else:
        selection_mode = args.selection_mode or "peak"
        min_is_trades = int(args.min_is_trades) if args.min_is_trades is not None else int(args.min_trades)
        min_oos_trades = int(args.min_oos_trades) if args.min_oos_trades is not None else int(args.min_trades)
        max_top_trade_share = float(args.max_top_trade_share) if args.max_top_trade_share is not None else 1.0
        wfe_min_pct = float(args.wfe_min_pct) if args.wfe_min_pct is not None else 0.0

    if min_is_trades < 0 or min_oos_trades < 0:
        raise ValueError("min trades values must be >= 0")
    if max_top_trade_share <= 0:
        raise ValueError("--max-top-trade-share must be > 0")
    if args.plateau_min_neighbors < 0:
        raise ValueError("--plateau-min-neighbors must be >= 0")
    if args.plateau_stability_penalty < 0:
        raise ValueError("--plateau-stability-penalty must be >= 0")
    if wfe_min_pct < 0:
        raise ValueError("--wfe-min-pct must be >= 0")

    return {
        "optimization_mode": mode,
        "selection_mode": str(selection_mode),
        "min_is_trades": int(min_is_trades),
        "min_oos_trades": int(min_oos_trades),
        "max_top_trade_share": float(max_top_trade_share),
        "wfe_metric": str(args.wfe_metric),
        "wfe_min_pct": float(wfe_min_pct),
        "plateau_min_neighbors": int(args.plateau_min_neighbors),
        "plateau_stability_penalty": float(args.plateau_stability_penalty),
    }


def main():
    args = build_parser().parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    controls = _resolve_optimization_controls(args)

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
        optimization_mode=str(controls["optimization_mode"]),
        selection_mode=str(controls["selection_mode"]),
        objective=args.objective,
        direction=args.direction,
        min_trades=args.min_trades,
        min_is_trades=int(controls["min_is_trades"]),
        min_oos_trades=int(controls["min_oos_trades"]),
        max_top_trade_share=float(controls["max_top_trade_share"]),
        wfe_metric=str(controls["wfe_metric"]),
        wfe_min_pct=float(controls["wfe_min_pct"]),
        plateau_min_neighbors=int(controls["plateau_min_neighbors"]),
        plateau_stability_penalty=float(controls["plateau_stability_penalty"]),
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

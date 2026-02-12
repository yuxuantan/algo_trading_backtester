"""
Monte Carlo simulation over walk-forward OOS trades.

Example:
python3 scripts/run_monte_carlo.py \
  --run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/run_ddmmyy_hhmmss \
  --n-sims 8000 \
  --replace \
  --ruin-equity 70000 \
  --stop-at-ruin \
  --pnl-mode actual \
  --progress-every 200 \
  --save-sample-paths-count 120
"""

from __future__ import annotations

import argparse

from quantbt.experiments.montecarlo.runner import run_monte_carlo


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Run Monte Carlo simulations on walk-forward OOS trades.")
    ap.add_argument("--run-dir", required=True, help="Walk-forward run directory containing oos_trades.csv.")
    ap.add_argument("--n-sims", type=int, default=8000)
    ap.add_argument("--n-trades", type=int, default=None, help="Trades per simulation. Default: size of trade pool.")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--replace", action="store_true", help="Sample trades with replacement (default).")
    g.add_argument("--without-replacement", action="store_true", help="Sample trades without replacement.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ruin-equity", type=float, required=True, help="Equity level below which a simulation is considered ruined.")
    ap.add_argument("--stop-at-ruin", action="store_true", help="Stop the simulation path immediately after ruin is hit.")

    ap.add_argument("--pnl-mode", default="actual", choices=["actual", "fixed_risk"])
    ap.add_argument("--fixed-risk-dollars", type=float, default=None, help="Required when pnl-mode=fixed_risk.")

    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--save-sample-paths-count", type=int, default=120, help="How many individual equity paths to store for plotting.")
    ap.add_argument("--no-save-quantile-paths", action="store_true", help="Disable quantile path output (mc_paths_quantiles.csv).")

    ap.add_argument("--target-risk-of-ruin-pct-max", type=float, default=10.0)
    ap.add_argument("--target-median-max-dd-pct-max", type=float, default=40.0)
    ap.add_argument("--target-median-return-pct-min", type=float, default=40.0)
    ap.add_argument("--target-return-dd-ratio-min", type=float, default=2.0)
    return ap


def main():
    args = build_parser().parse_args()
    replace = not args.without_replacement
    if args.replace:
        replace = True

    mc_run_dir = run_monte_carlo(
        run_dir=args.run_dir,
        n_sims=args.n_sims,
        n_trades=args.n_trades,
        replace=replace,
        seed=args.seed,
        ruin_equity=args.ruin_equity,
        stop_at_ruin=args.stop_at_ruin,
        pnl_mode=args.pnl_mode,
        fixed_risk_dollars=args.fixed_risk_dollars,
        progress_every=args.progress_every,
        save_sample_paths_count=args.save_sample_paths_count,
        save_quantile_paths=not args.no_save_quantile_paths,
        target_risk_of_ruin_pct_max=args.target_risk_of_ruin_pct_max,
        target_median_max_dd_pct_max=args.target_median_max_dd_pct_max,
        target_median_return_pct_min=args.target_median_return_pct_min,
        target_return_dd_ratio_min=args.target_return_dd_ratio_min,
    )
    print(f"Saved Monte Carlo run: {mc_run_dir}")


if __name__ == "__main__":
    main()

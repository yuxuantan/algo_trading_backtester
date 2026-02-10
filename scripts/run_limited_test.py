from dataclasses import asdict
import itertools
from pathlib import Path
import random

from quantbt.io.dataio import load_ohlc_csv
from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.experiments.limited.base import limited_test, limited_test_pass_rate
from quantbt.experiments.limited.runlog import make_limited_run_dir, write_json
from quantbt.experiments.limited.exits import (
    FixedBracketExitParams, build_fixed_brackets,
    TimeExitParams, build_time_exit
)
from quantbt.strategies import sma_cross

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

def main():
    data_path = Path("data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv")
    df = load_ohlc_csv(str(data_path), ts_col="timestamp")

    cfg = BacktestConfig(
        initial_equity=100_000,
        risk_pct=0.01,
        spread_pips=0.2,
        pip_size=0.0001,
        conservative_same_bar=True
    )

    # ---- Fix entry (fast/slow fixed) ----
    sp = sma_cross.SMACrossParams(fast=50, slow=200, rr=2.0, sl_buffer_pips=1.0, pip_size=cfg.pip_size)
    df_feat = sma_cross.compute_features(df, sp)      # or cached version if you have it
    df_sig = sma_cross.compute_signals(df_feat)       # generates bull_cross/bear_cross

    # favourable definition (edit as you like)
    favourable_fn = lambda s: float(s.get("total_return_%", -999)) > 0

    strategy_name = "sma_cross"
    run_dir = make_limited_run_dir(
        strategy=strategy_name,
        dataset_tag=data_path.name,
        test_name="limited_tests",
    )

    def make_param_sampler(param_list):
        if not param_list:
            raise ValueError("param_list is empty")
        it = iter(param_list)

        def _sample(_rng: random.Random):
            return next(it)

        return _sample

    # ========== TEST 1: Fixed SL/TP (vary rr + sl_buffer) ==========
    rr_values = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    slb_values = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    fixed_param_space = [
        FixedBracketExitParams(rr=rr, sl_buffer_pips=slb, pip_size=cfg.pip_size)
        for rr, slb in itertools.product(rr_values, slb_values)
    ]
    sample_fixed = make_param_sampler(fixed_param_space)

    res_fixed = limited_test(
        df_sig=df_sig,
        cfg=cfg,
        entry_iter_fn=sma_cross.iter_entries,
        run_backtest_fn=run_backtest_limited,
        build_exit_fn=build_fixed_brackets,
        exit_param_sampler=sample_fixed,
        n_iter=len(fixed_param_space),
        favourable_fn=favourable_fn,
        seed=42,
        min_trades=30,
        progress_fn=limited_progress_printer,   # 👈 HERE

    )
    pass_fixed = limited_test_pass_rate(res_fixed)

    # ========== TEST 2: Time Exit (vary hold_bars 1..50) ==========
    hold_bars_values = list(range(1, 51))
    time_param_space = [TimeExitParams(hold_bars=hb) for hb in hold_bars_values]
    sample_time = make_param_sampler(time_param_space)

    run_meta = {
        "strategy": strategy_name,
        "dataset": str(data_path),
        "dataset_tag": data_path.name,
        "test_name": "limited_tests",
        "entry_params": asdict(sp),
        "config": asdict(cfg),
        "fixed_param_space": [asdict(p) for p in fixed_param_space],
        "time_param_space": [asdict(p) for p in time_param_space],
        "favourable_rule": "total_return_% > 0",
        "min_trades": 30,
        "seed_fixed": 42,
        "seed_time": 43,
    }
    write_json(run_dir / "run_meta.json", run_meta)

    res_time = limited_test(
        df_sig=df_sig,
        cfg=cfg,
        entry_iter_fn=sma_cross.iter_entries,
        run_backtest_fn=run_backtest_limited,
        build_exit_fn=build_time_exit,
        exit_param_sampler=sample_time,
        n_iter=len(time_param_space),
        favourable_fn=favourable_fn,
        seed=43,
        min_trades=30,
        progress_fn=limited_progress_printer,   # 👈 HERE
    )

    pass_time = limited_test_pass_rate(res_time)

    print(f"Fixed SL/TP favourable%: {pass_fixed:.1f}%  -> {'PASS' if pass_fixed >= 70 else 'FAIL'}")
    print(f"Time-exit favourable%:  {pass_time:.1f}%  -> {'PASS' if pass_time >= 70 else 'FAIL'}")

    res_fixed.to_csv(run_dir / "limited_fixed_brackets.csv", index=False)
    res_time.to_csv(run_dir / "limited_time_exit.csv", index=False)
    print(f"Saved: {run_dir}/limited_fixed_brackets.csv, {run_dir}/limited_time_exit.csv")
    print(f"Run meta: {run_dir}/run_meta.json")


if __name__ == "__main__":
    main()

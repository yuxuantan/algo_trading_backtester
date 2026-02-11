# quantbt

Modular backtesting and limited-testing framework for strategy validation.

Detailed codebase documentation lives in `docs/CODEBASE.md`.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you do not install editable, run scripts with `PYTHONPATH=src`.

## 2. Download Data (Step by Step)

The project includes a Dukascopy downloader CLI.

### 2.1 Download EURUSD 1H data for 2010-01-01 to 2013-01-01

```bash
python3 scripts/download_data.py \
  --symbol EURUSD \
  --timeframe 1H \
  --start 2010-01-01 \
  --end 2013-01-01 \
  --save-dir data/processed \
  --file-ext csv
```

### 2.2 Expected outputs

- `data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv`
- `data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv.meta.json`

## 3. Limited Testing Workflow

All limited tests run through:

- `scripts/run_limited_tests.py`

Default strategy in this repo:

- `quantbt.strategies.sma_cross_test_strat`

Shared dataset used in examples:

- `data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv`

### How pass/fail works

- Each iteration is marked `favourable` when its summary matches `--favourable-criteria`.
- Final pass uses:
  - `favourable_pct >= --pass-threshold`
- Summary is written to `pass_summary.json`.

## 4. Core System Test

Purpose:

- Validate the strategy’s own entry + exit interaction.

Command:

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --commission-rt 5
```

Run folder prefix:

- `core_system_test__...`

## 5. Entry Test (Change Exit, Keep Entry)

Purpose:

- Validate entry logic robustness while altering exit behavior.

Command:

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin sma_cross \
  --entry-params '{"fast":[20,30,40,50,60,70,80],"slow":[125,150,175,200,225,250,275,300,325,350]}' \
  --exit-plugin time_exit \
  --exit-params '{"hold_bars":[1]}' \
  --commission-rt 5
```

Run folder prefix:

- `entry_test__...`

## 6. Exit Test (Change Entry, Keep Exit)

Purpose:

- Validate exit logic robustness while altering entry behavior.

Command:

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin donchian_breakout \
  --entry-params '{"lookback":[20]}' \
  --exit-plugin atr_brackets \
  --exit-params '{"rr":[1.0,1.5,2.0,2.5,3.0],"sldist_atr_mult":[0.5,1.0,1.5,2.0,2.5,3.0],"atr_period":14}' \
  --commission-rt 5
```

Run folder prefix:

- `exit_test__...`

## 7. Monkey Entry Test

Purpose:

- Replace entry with random/monkey entry while keeping strategy exit.
- Check that strategy outperforms monkey entry in most runs.

### 7.1 First get baseline core metrics

Run core test, then read latest core result row:

```bash
python3 - <<'PY'
import glob, os, pandas as pd
root = "runs/limited/sma_cross_test_strat/eurusd_1h_20100101_20130101/core_system_test__sma_cross__fixed_atr_exit"
latest = max(glob.glob(os.path.join(root, "run_*")), key=os.path.getmtime)
row = pd.read_csv(os.path.join(latest, "limited_results.csv")).iloc[0]
print("core_path:", latest)
print("trades:", row["trades"])
print("long_trade_pct:", row["long_trade_pct"])
print("avg_bars_held:", row["avg_bars_held"])
print("total_return_%:", row["total_return_%"])
print("max_drawdown_abs_%:", row["max_drawdown_abs_%"])
PY
```

### 7.2 Run monkey entry iterations

Use:

- `target_entries` = core `trades`
- `long_ratio` = core `long_trade_pct / 100`
- criteria describing “monkey is worse than core”

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin monkey_entry \
  --entry-params '{"target_entries":132,"side":"both","long_ratio":0.5}' \
  --seed-count 8000 \
  --seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5
```

Interpretation:

- `favourable_pct` in `pass_summary.json` = percent of monkey runs worse than core under your criteria.

## 8. Monkey Exit Test

Purpose:

- Keep strategy entry and replace exit with random/monkey time exit.
- Match average bars held to core system.

Command:

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --exit-plugin monkey_exit \
  --exit-params '{"avg_hold_bars":15.75}' \
  --exit-seed-count 8000 \
  --exit-seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5
```

## 9. Monkey Entry + Monkey Exit Test

Purpose:

- Replace both entry and exit with monkey logic.
- Validate strategy edge is stronger than random entry+exit interaction.

Command:

```bash
python3 scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin monkey_entry \
  --entry-params '{"target_entries":132,"side":"both","long_ratio":0.5}' \
  --seed-count 8000 \
  --seed-start 1 \
  --exit-plugin monkey_exit \
  --exit-params '{"avg_hold_bars":15.75}' \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5
```

Run folder prefix:

- `monkey_entry_exit_test__...`

Note:

- If you set both `--seed-count` and `--exit-seed-count`, iterations are Cartesian (`entry seeds x exit seeds`).

## 10. Limited Test Outputs

Each run writes:

- `run_meta.json`
- `limited_results.csv`
- `pass_summary.json`

`limited_results.csv` includes:

- return and drawdown: `total_return_%`, `max_drawdown_%`, `max_drawdown_abs_%`
- trade structure: `trades`, `avg_bars_held`, `long_trade_pct`, `short_trade_pct`
- verdict: `favourable`

## 11. Optimization (Optional)

```bash
python3 scripts/run_optimize.py \
  --strategy sma_cross_test_strat \
  --optimizer grid \
  --dataset data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --objective total_return_% \
  --min-trades 30
```

## 12. Walk-Forward Analysis

Purpose:

- Optimize on in-sample (IS), test on next out-of-sample (OOS), and roll forward.
- Compare aggregated OOS performance against a full-data optimization baseline.

Key defaults aligned with the book workflow:

- OOS/IS ratio is enforced to be between 10% and 50%.
- Full-data baseline optimization runs first (disable with `--no-baseline-full-data`).
- Objective defaults to `return_on_account`:
  - `net_profit_abs / (max_drawdown_abs + required_margin_abs)`
- If `--param-space` is empty (or all params are singletons), no optimization is run and fixed params are tested fold-by-fold.

Example (anchored WFA with grid optimization):

```bash
python3 scripts/run_walkforward.py \
  --strategy sma_cross_test_strat \
  --dataset data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --optimizer grid \
  --objective return_on_account \
  --is-bars 2000 \
  --oos-bars 400 \
  --step-bars 400 \
  --min-trades 30 \
  --commission-rt 5
```

Example (unanchored WFA):

```bash
python3 scripts/run_walkforward.py \
  --strategy sma_cross_test_strat \
  --dataset data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --unanchored \
  --is-bars 2000 \
  --oos-bars 400 \
  --step-bars 400
```

Key outputs:

- `runs/walkforward/.../config.json`
- `runs/walkforward/.../baseline_results.csv` (if baseline enabled)
- `runs/walkforward/.../folds.csv`
- `runs/walkforward/.../oos_equity_curve.csv`
- `runs/walkforward/.../oos_trades.csv`
- `runs/walkforward/.../summary.json`

Interactive OOS equity visualization:

```bash
python3 scripts/plot_oos_equity.py \
  --run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/20260211-191824_d32fc95e
```

This writes:

- `.../oos_equity_interactive.html`

Hover includes trade-level details (side, entry/exit, PnL, commission, R multiple) and fold window parameter tooltips.

## 13. Monte Carlo Simulation

Purpose:

- Randomize the order of OOS trades to estimate distribution of outcomes under sequencing uncertainty.
- Measure:
  - risk of ruin
  - median max drawdown
  - median return
  - return/drawdown ratio

Example:

```bash
python3 scripts/run_monte_carlo.py \
  --run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/20260211-191824_d32fc95e \
  --n-sims 8000 \
  --replace \
  --ruin-equity 70000 \
  --stop-at-ruin \
  --pnl-mode actual \
  --progress-every 200 \
  --save-sample-paths-count 120
```

Default threshold checks in `mc_summary.json`:

- `risk_of_ruin_pct < 10`
- `median_max_drawdown_% < 40`
- `median_return_% > 40`
- `return_drawdown_ratio_ratio_of_medians > 2.0`

Interactive Monte Carlo visualization:

```bash
python3 scripts/plot_monte_carlo.py \
  --mc-run-dir runs/walkforward/sma_cross_test_strat/eurusd_1h_20100101_20260209/grid_unanchored/20260211-191824_d32fc95e/monte_carlo/run_XXXXXXXX
```

This writes:

- `.../mc_interactive.html`

Includes:

- Equity path envelope (5/25/50/75/95 quantiles)
- Sample equity paths
- Return and drawdown distributions
- Return vs drawdown scatter with ruin-hit highlighting

## 14. Notes

- The project currently wires `EURUSD` in downloader.
- Limited tests intentionally prioritize robustness over finding one best parameter set.
- For internals, extension points, and module-level docs, see `docs/CODEBASE.md`.

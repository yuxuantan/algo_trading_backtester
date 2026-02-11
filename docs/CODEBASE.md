# QuantBT Codebase Documentation

## Overview
This repository implements a modular backtesting and experimentation framework focused on:

- Reusable entry/exit/sizing plugins
- Strategy-level backtesting and optimization
- Limited testing workflows for robustness checks
- Monkey tests (randomized entry and/or exit)

The code is organized under `src/quantbt` with runnable scripts under `scripts/`.

## Directory Structure

| Path | Purpose |
|---|---|
| `src/quantbt/core` | Backtest engines and performance metrics |
| `src/quantbt/plugins` | Plugin registry + entry/exit/sizing plugin implementations |
| `src/quantbt/strategies` | Strategy modules (currently `sma_cross_test_strat`) |
| `src/quantbt/io` | Data loading, dataset naming/version metadata, downloader integration |
| `src/quantbt/optimisers` | Grid and Optuna optimization drivers |
| `src/quantbt/experiments` | Run directory/index artifact helpers |
| `src/quantbt/experiments/limited` | Limited-test criteria and logging helpers |
| `scripts/download_data.py` | CLI downloader for Dukascopy data |
| `scripts/run_optimize.py` | CLI strategy optimizer |
| `scripts/run_limited_tests.py` | CLI limited test runner (core/entry/exit/monkey variants) |
| `runs/` | Output artifacts from optimization and limited tests |
| `data/processed/` | Local generated datasets and metadata |

## Core Layer (`src/quantbt/core`)

### `engine.py`
Main strategy-style backtest engine used by optimization workflows.

- `BacktestConfig`: shared configuration for equity, risk, spread, and commissions.
- `run_backtest_sma_cross(...)`: executes entries/exits for a strategy that emits bull/bear cross signals.

### `engine_limited.py`
Generic limited-test backtest engine used by plugin-based test sweeps.

- `run_backtest_limited(...)`: consumes generic entry iterator and exit builder.
- Supports bracket exits (`sl/tp`) and time exits (`hold_bars`).
- Tracks per-iteration summary metrics used by limited tests, including:
  - `total_return_%`
  - `max_drawdown_%`
  - `max_drawdown_abs_%`
  - `avg_bars_held`
  - `long_trade_pct` / `short_trade_pct`
  - trade and quality diagnostics (`win_rate_%`, `avg_profit_per_trade`, MFE/MAE metrics, etc.)

### `metrics.py`
- `max_drawdown(equity_series)`
- `profit_factor(trades_df)`

## Plugin System (`src/quantbt/plugins`)

### `registry.py`
Central plugin registry.

- Decorators:
  - `@register_entry(name)`
  - `@register_exit(name)`
  - `@register_sizing(name)`
- Accessors:
  - `get_entry`, `get_exit`, `get_sizing`
- `load_default_plugins()` imports and registers built-ins.

### Entry plugins (`plugins/entries`)

- `sma_cross.py`: delegates signal generation to `strategies/sma_cross_test_strat.py`.
- `donchian_breakout.py`: Donchian channel breakout entries.
- `random_entry.py`:
  - `random` and `monkey_entry` plugin names.
  - Supports either `prob`-based entry generation or exact `target_entries`.
  - Supports `side` and `long_ratio` controls.

### Exit plugins (`plugins/exits`)

- `atr_brackets.py`: ATR-based stop/target brackets.
- `time_exit.py`: fixed `hold_bars` time exit.
- `random_time_exit.py`:
  - `random_time_exit` and `monkey_exit` plugin names.
  - Supports:
    - `hold_bars_values`
    - `avg_hold_bars` (samples floor/ceil to preserve mean)
    - fixed `hold_bars`
  - Seeded random behavior for reproducibility.

### Sizing plugins (`plugins/sizing`)

- `fixed_risk.py`: risk-based position sizing.
- `fixed_units.py`: fixed-unit sizing.

## Strategy Layer (`src/quantbt/strategies`)

### `sma_cross_test_strat.py`
Built-in SMA crossover strategy module.

- `SMACrossTestStratParams` / `Params`
- `compute_features(df, params)`
- `compute_signals(df_feat)`
- `build_brackets_from_signal(...)`
- `STRATEGY` default plugin spec used by limited tests.

## IO Layer (`src/quantbt/io`)

### `dataio.py`
- `load_ohlc_csv(...)` with schema validation for OHLC columns.

### `downloader.py`
- `download_dukascopy_fx(...)`
- Fetches Dukascopy data, normalizes schema, writes dataset + metadata file.

### `datasets.py`
- Content hashing and metadata model (`DatasetMeta`)
- Metadata read/write helpers
- `dataset_tag_for_runs(...)` for stable run folder naming

### `naming.py`
- `dataset_filename(...)` standard file naming convention

## Optimization Layer (`src/quantbt/optimisers`)

### `grid.py`
- `grid_search(...)`: Cartesian parameter sweep.

### `optuna_opt.py`
- `optuna_search(...)`: Optuna-based optimization with progress callback hooks.

## Experiment Artifact Layer (`src/quantbt/experiments`)

### `runners.py`
- `make_run_dir(...)`
- `save_artifacts(...)`
- `append_run_index(...)` maintains cumulative run index CSV.

### `limited/*`
- `criteria.py`: parse and evaluate favourable criteria.
- `runlog.py`: limited test run folder and JSON writing.
- `base.py`: generic helper to compute limited test pass-rate.

## Scripts

### `scripts/download_data.py`
CLI downloader.

### `scripts/run_optimize.py`
Runs parameter optimization for a strategy module.

### `scripts/run_limited_tests.py`
Main limited testing orchestrator.

Capabilities:

- Core, entry-only, exit-only test classification
- Monkey entry / monkey exit / monkey entry+exit support
- Seed-grid generation for repeatable large iteration runs:
  - `--seed-count`, `--seed-start` for entry params
  - `--exit-seed-count`, `--exit-seed-start` for exit params
- Criteria-based pass/fail and run artifact outputs.

## Limited Test Output Contract

Each limited test run writes:

- `run_meta.json`: spec + criteria + metadata snapshot
- `limited_results.csv`: one row per iteration/combination
- `pass_summary.json`: aggregate pass metrics

Key fields in `limited_results.csv`:

- `iter`
- `entry_params`
- `exit_params`
- `trades`
- `total_return_%`
- `max_drawdown_%`
- `max_drawdown_abs_%`
- `avg_bars_held`
- `long_trade_pct`
- `short_trade_pct`
- `favourable`

## Test Focus Naming

`run_limited_tests.py` uses test-name prefixes:

- `core_system_test__...`: entry and exit plugins match strategy defaults
- `entry_test__...`: entry plugin(s) match, exit differs
- `exit_test__...`: exit matches, entry differs
- `monkey_entry_exit_test__...`: explicit exception when both are monkey plugins

Any other “both entry and exit changed” combination is blocked.

## Reproducibility Notes

- Use fixed seed ranges for monkey tests.
- `seed_count` and `exit_seed_count` are Cartesian when both are set.
- Keep run metadata and dataset metadata together for auditability.

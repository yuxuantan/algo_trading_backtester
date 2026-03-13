# Monkey Testing README

This document explains, at code level, how monkey testing is implemented for:

1. Monkey entry (random entries, strategy exit unchanged)
2. Monkey exit (strategy entry unchanged, random time exit)
3. Monkey entry + monkey exit

It describes the exact control flow from CLI args -> parameter expansion -> backtest loop -> scoring.

## 1) Source map (where monkey logic lives)

- CLI/spec building: `src/quantbt/experiments/limited/spec_building.py`
- Test-focus classification: `src/quantbt/experiments/limited/naming.py`
- Param expansion and Cartesian product: `src/quantbt/experiments/limited/param_grid.py`
- Runner/orchestration and monkey runtime features: `src/quantbt/experiments/limited/runner.py`
- Monkey entry plugin: `src/quantbt/plugins/entries/random_entry.py`
- Monkey exit plugin: `src/quantbt/plugins/exits/random_time_exit.py`
- Entry event construction from signals: `src/quantbt/experiments/limited/data_prep.py`
- Limited backtest engine: `src/quantbt/core/engine_limited.py`
- CLI wrapper: `scripts/run_limited_tests.py`

## 2) How the run type is classified

The run is classified by comparing current strategy entry/exit plugins vs base strategy plugins (`classify_test_focus`):

- `core_system_test`: entry same, exit same
- `entry_test`: entry same, exit changed
- `exit_test`: exit same, entry changed
- `monkey_entry_exit_test`: special exception when entry is subset of `{monkey_entry, random}` AND exit is one of `{monkey_exit, random_time_exit}`

If both entry and exit are changed but not this monkey exception, the run is blocked.

## 3) End-to-end execution pipeline

### 3.1 CLI args -> spec

`build_spec_from_args`:

- Loads strategy module and copies `STRATEGY`.
- Applies entry/exit plugin overrides.
- Applies optional seed grids:
  - `--seed-count`, `--seed-start` -> entry params `seed=[start..start+count-1]`
  - `--exit-seed-count`, `--exit-seed-start` -> exit params `seed=[start..start+count-1]`
- Optional monkey runtime settings:
  - `--monkey-match-prefilter ...`
  - `--monkey-fast-summary`
  - `--monkey-davey-style`
  - `--monkey-seq-stop ...`

Important validation:

- If params already contain `seed`, using `--seed-count` / `--exit-seed-count` raises an error.
- If both seed grids are set, runner uses Cartesian combinations.

### 3.2 Parameter expansion (how many iterations are created)

`expand_params` in `param_grid.py` treats list-valued params as grid dimensions, except keys ending in `_values`.

That means:

- `seed: [1,2,3]` expands into 3 variants.
- `hold_bars_values: [5,10,15]` does **not** expand; it is passed as a single param value for random sampling in the exit plugin.

Total iterations = product(entry variant counts) * product(exit param variants).

### 3.3 Main loop in `run_spec`

For each entry param combo:

1. Build/lookup signals from entry plugin(s).
2. Combine signals (`all` / `any` / `vote`).
3. Build `df_sig` with OHLC (+ ATR if needed).
4. Build entry iterator (`iter_entries_from_signals`): signal at bar `i` enters at bar `i+1` open.

For each exit param combo:

1. (Optional) monkey prefilter or exact scheduler.
2. Run full backtest (`run_backtest_limited`) or fast summary (`_run_backtest_limited_time_exit_fast_summary`).
3. Evaluate iteration as favourable/not favourable.
4. Save per-iteration row.

## 4) Monkey Entry test (entry randomized, exit preserved)

### 4.1 Signal generation: `random_entry.py`

Registered aliases:

- `monkey_entry`
- `random`

Supported modes:

1. `target_entries` mode:
   - Draw exactly `target_entries` candidate bars from `[0 .. n_bars-2]`.
   - Enforce optional spacing via `min_bars_between`.
   - If spacing is non-zero:
     - First try 300 exact-size random draws and accept one that satisfies spacing.
     - Fallback to randomized greedy fill.
2. `prob` mode:
   - Walk bars and enter with Bernoulli(`prob`) while respecting `min_bars_between`.

Side assignment:

- `side=long`: all long
- `side=short`: all short
- `side=both`: assign count by `long_ratio` (or `long_pct`) then shuffle.

Output is a DataFrame with boolean `long_entry` and `short_entry`.

### 4.2 From signals to trades

`iter_entries_from_signals` converts `long_entry` / `short_entry` into events:

- `entry_i = i+1`
- `entry_open = open[i+1]`
- `prev_low/high` from bar `i`

Key behavior:

- Engine is flat-only (one position at a time).
- If another entry signal appears before current position exits, it is skipped.
- So `target_entries` is target signal count, not guaranteed executed trade count.

### 4.3 Exit path during monkey-entry runs

Exit is the strategy/default (or CLI-selected non-monkey exit):

- Bracket exits (`sl`/`tp`) or
- Time exits (`hold_bars`)

Engine computes PnL with spread + optional commission and writes summary metrics.

## 5) Monkey Exit test (entry preserved, exit randomized by hold time)

### 5.1 Exit generation: `random_time_exit.py`

Registered aliases:

- `monkey_exit`
- `random_time_exit`

Accepted param styles:

1. `hold_bars_values`: choose one positive integer uniformly at random each trade.
2. `avg_hold_bars`: sample floor/ceil stochastically so expected hold equals the average.
3. `hold_bars`: fixed constant hold.

`avg_hold_bars` path supports clamps:

- `min_hold_bars` (default 1)
- `max_hold_bars` (optional)

### 5.2 RNG model for reproducibility

Each trade gets an entry-specific RNG:

- `mixed_seed = (seed * 1_000_003) ^ entry_i`
- `random.Random(mixed_seed)`

This makes hold-bar sampling deterministic per `(seed, entry_i)`.

### 5.3 How exit is applied in engine

For time exit:

- Open at entry bar open.
- Set `exit_i = min(last_bar, entry_i + hold_bars)`.
- Close at **close price of exit bar** with reason `TIME_EXIT`.

## 6) Monkey Entry + Monkey Exit test

This is the special allowed case where both sides change:

- Entry in `{monkey_entry, random}`
- Exit in `{monkey_exit, random_time_exit}`

### 6.1 Seed combination behavior

If both seed grids are provided:

- Entry seed variants x exit seed variants are expanded by Cartesian product.
- Each pair is a separate attempted iteration.

### 6.2 Optional exact-match prefilter scheduler

When `--monkey-match-prefilter` is enabled and exit is monkey time-exit, runner can build an exact flat-only schedule before full backtest:

1. Sample trade count within tolerance around target trades.
2. Build random hold sequence with target mean hold and optional min/max bounds.
3. Allocate slack bars as random composition to place entries without overlap.
4. Assign long/short counts inside tolerance around target long%.
5. Create synthetic entries carrying `monkey_hold_bars`.
6. Wrap exit plugin so it returns precomputed `hold_bars` from each entry.

If schedule cannot be built (for example insufficient bars), attempt is rejected before backtest and counted as prefilter reject.

## 7) Optional monkey runtime features

### 7.1 Monkey match prefilter

`_simulate_flat_only_time_exit_schedule` estimates:

- `trades`
- `long_trade_pct`
- `avg_bars_held`

Then `_prefilter_schedule_matches` compares those against target +/- tolerances.

If it fails, iteration is rejected before full backtest.

### 7.2 Monkey fast summary

For time-based exits, `--monkey-fast-summary` switches to a lightweight evaluator:

- No per-trade dataframe export
- No full equity curve dataframe
- Computes only summary metrics needed for scoring

This is faster for large monkey sweeps.

### 7.3 Sequential stopping

`--monkey-seq-stop` checks Wilson confidence interval on favourable percentage every N accepted runs:

- Early PASS if lower bound > pass threshold
- Early FAIL if upper bound < fail threshold

This stops runs early when outcome is already statistically clear.

### 7.4 Strict Davey-style scoring

`--monkey-davey-style` changes pass logic:

- Track separately:
  - `% runs with monkey return worse than baseline`
  - `% runs with monkey max drawdown worse than baseline`
- Final PASS requires both percentages >= threshold.

Baseline thresholds are read from criteria rules when possible (`total_return_% < X`, `max_drawdown_abs_% > Y`) or explicit config.

## 8) Scoring and pass/fail

Default (non-Davey):

- `trades >= min_trades`
- `criteria_pass(summary, criteria)` is true

Run-level pass:

- `favourable_pct >= pass_threshold_%`

Davey mode:

- Iteration favourable when both return-worse and drawdown-worse conditions hold (and min trades passes).
- Run-level pass when both aggregated percentages exceed threshold.

## 9) Outputs and monkey-specific fields

Every run writes:

- `run_meta.json`
- `limited_results.csv`
- `pass_summary.json`
- `limited_trades.csv` (unless disabled)

Monkey-related fields can include:

- `attempt` vs `iter` (attempted vs accepted)
- `prefilter_trades`
- `prefilter_long_trade_pct`
- `prefilter_avg_bars_held`
- `davey_return_worse`
- `davey_maxdd_worse`
- `davey_both_worse`
- `prefilter_rejects` in `pass_summary.json`
- `sequential_stop` block in `pass_summary.json` when enabled

## 10) Practical reproducibility notes

- Fix seeds to reproduce the same monkey signals and time exits.
- Because exit RNG mixes with `entry_i`, changing entry schedule changes exit sampling even with same exit seed.
- With flat-only execution, signal count and executed-trade count can differ if signals overlap while a trade is active.
- If both entry and exit seed grids are set, total workload grows multiplicatively.

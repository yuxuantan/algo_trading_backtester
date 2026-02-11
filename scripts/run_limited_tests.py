"""
CLI wrapper for spec-driven limited tests.

Example (entry-only: SMA grid + fixed ATR exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin sma_cross \
  --entry-params '{"fast":[10,20,30,40,50,60,70,80,90,100],"slow":[100,125,150,175,200,225,250,275,300,325]}' \
  --exit-plugin atr_brackets \
  --exit-params '{"rr":2.0,"sldist_atr_mult":1.5,"atr_period":14}' \
  --commission-rt 5

Example (entry-only: SMA grid + time stop exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin sma_cross \
  --entry-params '{"fast":[20,30,40,50,60,70,80],"slow":[125,150,175,200,225,250,275,300,325,350]}' \
  --exit-plugin time_exit \
  --exit-params '{"hold_bars":[1]}' \
  --commission-rt 5

Example (exit-only: similar-approach Donchian entry, use strategy exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin donchian_breakout \
  --entry-params '{"lookback":[20]}' \
  --favourable-criteria '{"total_return_%":{">":0}}' \
  --exit-plugin atr_brackets \
  --exit-params '{"rr":[1.0,1.5,2.0,2.5,3.0],"sldist_atr_mult":[0.5,1.0,1.5,2.0,2.5,3.0],"atr_period":14}' \
  --commission-rt 5

Example (monkey entry: random 132 entry signals, keep strategy exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin monkey_entry \
  --entry-params '{"target_entries":132,"side":"both","long_ratio":0.5}' \
  --seed-count 100 \
  --seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5

Example (monkey exit: random exit timing around core avg bars held):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --exit-plugin monkey_exit \
  --exit-params '{"avg_hold_bars":15.75}' \
  --exit-seed-count 100 \
  --exit-seed-start 1 \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5

Example (monkey entry + monkey exit):
python scripts/run_limited_tests.py \
  --strategy quantbt.strategies.sma_cross_test_strat \
  --data data/processed/eurusd_1h_20100101_20130101_dukascopy_python.csv \
  --entry-plugin monkey_entry \
  --entry-params '{"target_entries":132,"side":"both","long_ratio":0.5}' \
  --seed-count 100 \
  --seed-start 1 \
  --exit-plugin monkey_exit \
  --exit-params '{"avg_hold_bars":15.75}' \
  --favourable-criteria '{"mode":"all","rules":[{"metric":"total_return_%","op":"<","value":16.3},{"metric":"max_drawdown_abs_%","op":">","value":11.4}]}' \
  --pass-threshold 90 \
  --commission-rt 5
"""

from __future__ import annotations

from quantbt.experiments.limited.runner import run_spec
from quantbt.experiments.limited.spec_building import build_cli_parser, build_spec_from_args


def main():
    parser = build_cli_parser()
    args = parser.parse_args()
    spec = build_spec_from_args(args)
    run_spec(spec, progress_every=args.progress_every)


if __name__ == "__main__":
    main()

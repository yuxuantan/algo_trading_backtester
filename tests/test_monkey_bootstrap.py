from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.artifacts import limited_trades_path
from quantbt.experiments.limited.monkey.bootstrap import (
    load_baseline_fixed_units,
    load_baseline_hold_bars_values,
)


class MonkeyBootstrapTests(unittest.TestCase):
    def test_load_baseline_hold_bars_values_filters_selected_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_trades_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 1, "bars_held": 3},
                    {"iter": 1, "bars_held": 5},
                    {"iter": 2, "bars_held": 7},
                    {"iter": 2, "bars_held": 9},
                    {"iter": 2, "bars_held": 0},
                ]
            ).to_csv(limited_trades_path(run_dir), index=False)

            values = load_baseline_hold_bars_values(run_dir, iter_id=2)

            self.assertEqual(values, [7, 9])

    def test_load_baseline_hold_bars_values_requires_positive_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_trades_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 3, "bars_held": 0},
                    {"iter": 3, "bars_held": -2},
                ]
            ).to_csv(limited_trades_path(run_dir), index=False)

            with self.assertRaises(ValueError):
                load_baseline_hold_bars_values(run_dir, iter_id=3)

    def test_load_baseline_fixed_units_uses_median_absolute_units_for_selected_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_trades_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 1, "units": 10_000},
                    {"iter": 1, "units": -20_000},
                    {"iter": 1, "units": 30_000},
                    {"iter": 2, "units": 50_000},
                ]
            ).to_csv(limited_trades_path(run_dir), index=False)

            units = load_baseline_fixed_units(run_dir, iter_id=1)

            self.assertEqual(units, 20_000.0)

    def test_load_baseline_fixed_units_requires_positive_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_trades_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 2, "units": 0},
                    {"iter": 2, "units": float("nan")},
                ]
            ).to_csv(limited_trades_path(run_dir), index=False)

            with self.assertRaises(ValueError):
                load_baseline_fixed_units(run_dir, iter_id=2)


if __name__ == "__main__":
    unittest.main()

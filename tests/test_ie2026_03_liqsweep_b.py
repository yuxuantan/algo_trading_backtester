from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

liq_sweep_b = importlib.import_module("quantbt.strategies.interequity_2026_03_liqsweep_b")

from quantbt.plugins.entries import interequity_liqsweepb_entry
from quantbt.plugins.exits.time_exit import build_exit as build_time_exit


def _sample_ohlc(rows: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    base = 1.10 + np.sin(np.linspace(0, 2, rows)) * 0.002
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


class Ie202603LiqSweepBTests(unittest.TestCase):
    def test_strategy_metadata_points_to_b_plugins(self) -> None:
        self.assertEqual(liq_sweep_b.STRATEGY["name"], "IE2026-03 LiqSweep B")
        self.assertEqual(liq_sweep_b.STRATEGY["entry"]["rules"][0]["name"], "interequity_liqsweepb_entry")
        self.assertEqual(liq_sweep_b.STRATEGY["exit"]["name"], "interequity_liqsweepb_exit")
        self.assertEqual(liq_sweep_b.RECLAIM_WINDOW_BARS, 3)

    def test_run_backtest_enters_on_reclaim_bar_at_swept_red_level(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.1002, 1.1003, 1.0995],
                "high": [1.1008, 1.1004, 1.0998],
                "low": [1.0998, 1.0992, 1.0990],
                "close": [1.1004, 1.0994, 1.0993],
            },
            index=idx,
        )
        params = liq_sweep_b.Params()
        captured: dict[str, float | int] = {}

        def fake_build_entry_spec_from_exit_override(**kwargs):
            captured["entry_open"] = float(kwargs["entry_open"])
            captured["entry_index"] = int(kwargs["entry_index"])
            return {
                "side": "short",
                "qty": 1.0,
                "sl": float("nan"),
                "tp": float("nan"),
                "risk_dollars": 100.0,
                "time_exit_i": int(kwargs["entry_index"]) + 1,
            }

        sweep_seq = [(1.1000, None), (None, None), (None, None)]
        with mock.patch.multiple(
            liq_sweep_b,
            track_breach_high=lambda *args, **kwargs: None,
            track_breach_low=lambda *args, **kwargs: None,
            append_high_pivot=lambda *args, **kwargs: None,
            append_low_pivot=lambda *args, **kwargs: None,
            confirm_move_away_high=lambda *args, **kwargs: None,
            confirm_move_away_low=lambda *args, **kwargs: None,
            stop_extending_high=lambda *args, **kwargs: None,
            stop_extending_low=lambda *args, **kwargs: None,
            sweep_triggers=mock.Mock(side_effect=sweep_seq),
            _build_entry_spec_from_exit_override=fake_build_entry_spec_from_exit_override,
        ):
            _equity_df, trades_df, summary = liq_sweep_b.run_backtest(
                df,
                strategy_params=params,
                override_exit_builder=build_time_exit,
                override_exit_params={"hold_bars": 1},
            )

        self.assertEqual(captured, {"entry_open": 1.1, "entry_index": 1})
        self.assertEqual(int(summary["trades"]), 1)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(pd.Timestamp(trades_df.iloc[0]["entry_time"]), idx[1])
        self.assertAlmostEqual(float(trades_df.iloc[0]["entry"]), 1.1000, places=8)
        self.assertEqual(str(trades_df.iloc[0]["exit_reason"]), "TIME_EXIT")

    def test_run_backtest_allows_same_bar_sweep_and_reclaim(self) -> None:
        idx = pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.1002, 1.1001],
                "high": [1.1005, 1.1002],
                "low": [1.0992, 1.0994],
                "close": [1.1003, 1.1000],
            },
            index=idx,
        )
        params = liq_sweep_b.Params()
        captured: dict[str, float | int] = {}

        def fake_build_entry_spec_from_exit_override(**kwargs):
            captured["entry_open"] = float(kwargs["entry_open"])
            captured["entry_index"] = int(kwargs["entry_index"])
            return {
                "side": "long",
                "qty": 1.0,
                "sl": float("nan"),
                "tp": float("nan"),
                "risk_dollars": 100.0,
                "time_exit_i": int(kwargs["entry_index"]) + 1,
            }

        sweep_seq = [(None, 1.1000), (None, None)]
        with mock.patch.multiple(
            liq_sweep_b,
            track_breach_high=lambda *args, **kwargs: None,
            track_breach_low=lambda *args, **kwargs: None,
            append_high_pivot=lambda *args, **kwargs: None,
            append_low_pivot=lambda *args, **kwargs: None,
            confirm_move_away_high=lambda *args, **kwargs: None,
            confirm_move_away_low=lambda *args, **kwargs: None,
            stop_extending_high=lambda *args, **kwargs: None,
            stop_extending_low=lambda *args, **kwargs: None,
            sweep_triggers=mock.Mock(side_effect=sweep_seq),
            _build_entry_spec_from_exit_override=fake_build_entry_spec_from_exit_override,
        ):
            _equity_df, trades_df, summary = liq_sweep_b.run_backtest(
                df,
                strategy_params=params,
                override_exit_builder=build_time_exit,
                override_exit_params={"hold_bars": 1},
            )

        self.assertEqual(captured, {"entry_open": 1.1, "entry_index": 0})
        self.assertEqual(int(summary["trades"]), 1)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(pd.Timestamp(trades_df.iloc[0]["entry_time"]), idx[0])
        self.assertAlmostEqual(float(trades_df.iloc[0]["entry"]), 1.1000, places=8)
        self.assertEqual(str(trades_df.iloc[0]["side"]), "long")

    def test_run_backtest_reclaim_must_happen_within_next_three_bars(self) -> None:
        idx = pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1.1002, 1.1003, 1.1002, 1.1001, 1.0996],
                "high": [1.1008, 1.1005, 1.1004, 1.1003, 1.0998],
                "low": [1.0998, 1.1000, 1.0999, 1.0998, 1.0992],
                "close": [1.1004, 1.1002, 1.1001, 1.10005, 1.0994],
            },
            index=idx,
        )
        params = liq_sweep_b.Params()

        sweep_seq = [(1.1000, None)] + [(None, None)] * (len(df) - 1)
        with mock.patch.multiple(
            liq_sweep_b,
            track_breach_high=lambda *args, **kwargs: None,
            track_breach_low=lambda *args, **kwargs: None,
            append_high_pivot=lambda *args, **kwargs: None,
            append_low_pivot=lambda *args, **kwargs: None,
            confirm_move_away_high=lambda *args, **kwargs: None,
            confirm_move_away_low=lambda *args, **kwargs: None,
            stop_extending_high=lambda *args, **kwargs: None,
            stop_extending_low=lambda *args, **kwargs: None,
            sweep_triggers=mock.Mock(side_effect=sweep_seq),
            _build_entry_spec_from_exit_override=mock.Mock(side_effect=AssertionError("should not enter outside reclaim window")),
        ):
            _equity_df, trades_df, summary = liq_sweep_b.run_backtest(
                df,
                strategy_params=params,
                override_exit_builder=build_time_exit,
                override_exit_params={"hold_bars": 1},
            )

        self.assertEqual(int(summary["trades"]), 0)
        self.assertTrue(trades_df.empty)

    def test_entry_plugin_exposes_full_system_rerun_bridge(self) -> None:
        captured: dict[str, object] = {}

        class FakeParams:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

        class FakeModule:
            Params = FakeParams

            @staticmethod
            def compute_features(df, p):
                captured["params"] = p.kwargs
                return df

            @staticmethod
            def compute_signals(df):
                return df

            @staticmethod
            def run_backtest(df_sig, **kwargs):
                captured["run_backtest_kwargs"] = dict(kwargs)
                return pd.DataFrame(), pd.DataFrame(), {"trades": 0}

        def fake_exit_plugin(*args, **kwargs):
            return {"hold_bars": 1}

        def fake_sizing_plugin(**kwargs):
            return 1.0

        with mock.patch.object(interequity_liqsweepb_entry, "_load_strategy_module", return_value=FakeModule):
            interequity_liqsweepb_entry.run_full_system(
                _sample_ohlc(10),
                entry_params={"min_rr": 1.0},
                exit_plugin=fake_exit_plugin,
                exit_params={"hold_bars": 1},
                cfg=liq_sweep_b.BacktestConfig(),
                sizing_plugin=fake_sizing_plugin,
                sizing_params={"risk_pct": 0.01},
            )

        self.assertEqual(captured["params"], {"min_rr": 1.0})
        kwargs = captured["run_backtest_kwargs"]
        self.assertIs(kwargs["override_exit_builder"], fake_exit_plugin)
        self.assertEqual(kwargs["override_exit_params"], {"hold_bars": 1})
        self.assertIs(kwargs["override_size_fn"], fake_sizing_plugin)
        self.assertEqual(kwargs["override_sizing_params"], {"risk_pct": 0.01})


if __name__ == "__main__":
    unittest.main()

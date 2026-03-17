from __future__ import annotations

import importlib
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

liq_sweep_a = importlib.import_module("quantbt.strategies.interequity_2026_03_liqsweep_a")

from quantbt.plugins.exits.interequity_liqsweep_exit import build_exit as build_liqsweep_exit
from quantbt.plugins.registry import ENTRY_PLUGINS, load_default_plugins


def _sample_ohlc(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    base = 1.10 + np.sin(np.linspace(0, 12, rows)) * 0.01
    close = pd.Series(base, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + 0.0008
    low = np.minimum(open_, close) - 0.0008
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


class Ie202603LiqSweepATests(unittest.TestCase):
    def test_default_plugin_loading_skips_missing_legacy_modules_without_crashing(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_default_plugins()
        self.assertIn("interequity_liqsweep_entry", ENTRY_PLUGINS)
        self.assertIn("random", ENTRY_PLUGINS)
        self.assertIn("donchian_breakout", ENTRY_PLUGINS)
        self.assertTrue(any("quantbt.plugins.entries.sma_cross" in str(w.message) for w in caught))

    def test_liqsweep_entry_plugin_targets_current_strategy_alias(self) -> None:
        from quantbt.plugins.entries import interequity_liqsweep_entry

        mod = interequity_liqsweep_entry._load_strategy_module()  # pylint: disable=protected-access
        self.assertEqual(mod.STRATEGY["name"], "IE2026-03 LiqSweep A")

    def test_param_space_is_derived_from_richer_limited_test_metadata(self) -> None:
        entry_opt = liq_sweep_a.STRATEGY["limited_test"]["entry"]["optimizable"]
        self.assertNotIn("max_rr", entry_opt)
        self.assertIn("max_rr", liq_sweep_a.STRATEGY["limited_test"]["entry"]["non_optimizable"])
        self.assertNotIn("values", entry_opt["atr_dist_for_liq_generation"])
        self.assertNotIn("default", entry_opt["atr_dist_for_liq_generation"])
        self.assertNotIn("values", entry_opt["liq_move_away_atr"])
        self.assertNotIn("default", entry_opt["liq_move_away_atr"])
        self.assertEqual(
            liq_sweep_a.PARAM_SPACE,
            {
                "atr_dist_for_liq_generation": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],
                "liq_move_away_atr": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            },
        )

    def test_legacy_aliases_map_to_unified_current_tf_params(self) -> None:
        params = liq_sweep_a.Params(  # pylint: disable=not-callable
            ltf_pivot_len=5,
            max_ltf_h=90,
            max_ltf_l=80,
            htf="60",
        )
        self.assertEqual(params.pivot_len, 5)
        self.assertEqual(params.max_high_levels, 90)
        self.assertEqual(params.max_low_levels, 80)

    def test_run_backtest_smoke_with_current_tf_only_logic(self) -> None:
        df = _sample_ohlc()
        params = liq_sweep_a.Params()  # pylint: disable=not-callable
        feat = liq_sweep_a.compute_features(df, params)
        sig = liq_sweep_a.compute_signals(feat)
        equity_df, trades_df, summary = liq_sweep_a.run_backtest(sig, strategy_params=params)

        self.assertEqual(len(equity_df), len(df))
        self.assertIsInstance(trades_df, pd.DataFrame)
        self.assertIn("trades", summary)
        self.assertNotIn("htf", liq_sweep_a.PARAM_SPACE)
        self.assertNotIn("htf", liq_sweep_a.STRATEGY["entry"]["rules"][0]["params"])
        self.assertIn("fallback_rr", liq_sweep_a.STRATEGY["exit"]["params"])
        self.assertNotIn("min_rr", liq_sweep_a.STRATEGY["exit"]["params"])

    def test_exit_plugin_prefers_fallback_rr_and_keeps_legacy_min_rr_compat(self) -> None:
        from_fallback_rr = build_liqsweep_exit(
            side="long",
            entry_open=1.1000,
            prev_low=1.0990,
            prev_high=1.1010,
            params={"fallback_rr": 2.0, "sl_buffer_pips": 0.2, "pip_size": 0.0001},
            entry=None,
        )
        self.assertIsNotNone(from_fallback_rr)
        self.assertAlmostEqual(from_fallback_rr["tp"], 1.10204, places=8)

        from_legacy_min_rr = build_liqsweep_exit(
            side="long",
            entry_open=1.1000,
            prev_low=1.0990,
            prev_high=1.1010,
            params={"min_rr": 2.0, "sl_buffer_pips": 0.2, "pip_size": 0.0001},
            entry=None,
        )
        self.assertIsNotNone(from_legacy_min_rr)
        self.assertAlmostEqual(from_legacy_min_rr["tp"], 1.10204, places=8)

    def test_exit_override_entry_builder_uses_swapped_bracket_and_sizes_from_it(self) -> None:
        params = liq_sweep_a.Params(min_rr=1.0, max_rr=10.0, risk_pct=0.01)
        cfg = liq_sweep_a.BacktestConfig(initial_equity=100_000.0)
        high_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access

        def fake_exit_builder(side, entry_open, prev_low, prev_high, params_dict, entry=None):
            self.assertEqual(side, "long")
            self.assertAlmostEqual(float(entry_open), 1.1005, places=8)
            self.assertAlmostEqual(float(prev_low), 1.0990, places=8)
            self.assertAlmostEqual(float(prev_high), 1.1010, places=8)
            self.assertEqual(params_dict, {"rr": 1.5})
            self.assertIsNotNone(entry)
            self.assertAlmostEqual(float(entry["atr"]), 0.001, places=8)
            return {"sl": 1.0995, "tp": 1.1020, "stop_dist": 0.001}

        pending = liq_sweep_a._build_pending_entry_from_exit_override(  # pylint: disable=protected-access
            side="long",
            entry_index=5,
            entry_time=pd.Timestamp("2025-01-01T00:25:00Z"),
            entry_open=1.1005,
            prev_low=1.0990,
            prev_high=1.1010,
            entry_atr=0.001,
            high_pool=high_pool,
            low_pool=low_pool,
            strategy_params=params,
            cfg=cfg,
            equity=100_000.0,
            exit_builder=fake_exit_builder,
            exit_params={"rr": 1.5},
            exit_supports_entry=True,
            size_fn=None,
            sizing_params={},
        )

        self.assertIsNotNone(pending)
        self.assertEqual(pending["side"], "long")
        self.assertAlmostEqual(float(pending["sl"]), 1.0995, places=8)
        self.assertAlmostEqual(float(pending["tp"]), 1.1020, places=8)
        self.assertAlmostEqual(float(pending["qty"]), 1_000_000.0, places=8)
        self.assertAlmostEqual(float(pending["risk_dollars"]), 1_000.0, places=8)

    def test_entry_plugin_exposes_full_system_rerun_bridge(self) -> None:
        from quantbt.plugins.entries import interequity_liqsweep_entry

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
            return {"sl": 1.0, "tp": 2.0, "stop_dist": 1.0}

        def fake_sizing_plugin(**kwargs):
            return 1.0

        with mock.patch.object(interequity_liqsweep_entry, "_load_strategy_module", return_value=FakeModule):
            interequity_liqsweep_entry.run_full_system(
                _sample_ohlc(10),
                entry_params={"min_rr": 1.0},
                exit_plugin=fake_exit_plugin,
                exit_params={"rr": 1.5},
                cfg=liq_sweep_a.BacktestConfig(),
                sizing_plugin=fake_sizing_plugin,
                sizing_params={"risk_pct": 0.01},
            )

        self.assertEqual(captured["params"], {"min_rr": 1.0})
        kwargs = captured["run_backtest_kwargs"]
        self.assertIs(kwargs["override_exit_builder"], fake_exit_plugin)
        self.assertEqual(kwargs["override_exit_params"], {"rr": 1.5})
        self.assertIs(kwargs["override_size_fn"], fake_sizing_plugin)
        self.assertEqual(kwargs["override_sizing_params"], {"risk_pct": 0.01})


if __name__ == "__main__":
    unittest.main()

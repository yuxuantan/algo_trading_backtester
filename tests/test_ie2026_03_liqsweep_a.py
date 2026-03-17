from __future__ import annotations

import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_ROOT / "src" / "quantbt" / "strategies" / "IE2026-03 LiqSweep A.py"
MODULE_SPEC = importlib.util.spec_from_file_location("ie2026_03_liqsweep_a", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
liq_sweep_a = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = liq_sweep_a
MODULE_SPEC.loader.exec_module(liq_sweep_a)

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
        self.assertIn("max_rr", entry_opt)
        self.assertNotIn("max_rr", liq_sweep_a.STRATEGY["limited_test"]["entry"]["non_optimizable"])
        self.assertEqual(
            liq_sweep_a.PARAM_SPACE,
            {
                "atr_dist_for_liq_generation": [0.8, 1.0, 1.2],
                "liq_move_away_atr": [2.5, 3.0],
                "max_rr": [4.0, 6.0, 8.0],
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


if __name__ == "__main__":
    unittest.main()

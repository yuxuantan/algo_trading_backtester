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
from quantbt.plugins.exits.time_exit import build_exit as build_time_exit
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
        high_pool.lvls = [1.10210]
        high_pool.drawn_act = [True]
        high_pool.state = [liq_sweep_a.ST_RED]
        low_pool.lvls = [1.09900]
        low_pool.drawn_act = [True]
        low_pool.state = [liq_sweep_a.ST_PURPLE]

        def fake_exit_builder(side, entry_open, prev_low, prev_high, params_dict, entry=None):
            self.assertEqual(side, "long")
            self.assertAlmostEqual(float(entry_open), 1.1005, places=8)
            self.assertAlmostEqual(float(prev_low), 1.0990, places=8)
            self.assertAlmostEqual(float(prev_high), 1.1010, places=8)
            self.assertEqual(params_dict, {"rr": 1.5})
            self.assertIsNotNone(entry)
            self.assertAlmostEqual(float(entry["atr"]), 0.001, places=8)
            return {"sl": 1.0995, "tp": 1.1020, "stop_dist": 0.001}

        entry_spec = liq_sweep_a._build_entry_spec_from_exit_override(  # pylint: disable=protected-access
            side="long",
            entry_index=5,
            entry_time=pd.Timestamp("2025-01-01T00:25:00Z"),
            signal_price=1.1000,
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

        self.assertIsNotNone(entry_spec)
        self.assertEqual(entry_spec["side"], "long")
        self.assertAlmostEqual(float(entry_spec["sl"]), 1.0995, places=8)
        self.assertAlmostEqual(float(entry_spec["tp"]), 1.1020, places=8)
        self.assertAlmostEqual(float(entry_spec["qty"]), 1_000_000.0, places=8)
        self.assertAlmostEqual(float(entry_spec["risk_dollars"]), 1_000.0, places=8)

    def test_exit_override_entry_builder_rejects_when_native_structural_rr_is_outside_band(self) -> None:
        params = liq_sweep_a.Params(min_rr=1.0, max_rr=10.0, risk_pct=0.01)
        cfg = liq_sweep_a.BacktestConfig(initial_equity=100_000.0)
        high_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        high_pool.lvls = [1.10090]
        high_pool.drawn_act = [True]
        high_pool.state = [liq_sweep_a.ST_RED]
        low_pool.lvls = [1.09900]
        low_pool.drawn_act = [True]
        low_pool.state = [liq_sweep_a.ST_PURPLE]

        def fake_exit_builder(side, entry_open, prev_low, prev_high, params_dict, entry=None):
            del side, entry_open, prev_low, prev_high, params_dict, entry
            return {"sl": 1.0995, "tp": 1.1020, "stop_dist": 0.001}

        entry_spec = liq_sweep_a._build_entry_spec_from_exit_override(  # pylint: disable=protected-access
            side="long",
            entry_index=5,
            entry_time=pd.Timestamp("2025-01-01T00:25:00Z"),
            signal_price=1.1000,
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

        self.assertIsNone(entry_spec)

    def test_time_exit_override_can_use_native_structural_sizing(self) -> None:
        params = liq_sweep_a.Params(min_rr=1.0, max_rr=10.0, risk_pct=0.01)
        cfg = liq_sweep_a.BacktestConfig(initial_equity=100_000.0)
        high_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        high_pool.lvls = [1.10210]
        high_pool.drawn_act = [True]
        high_pool.state = [liq_sweep_a.ST_RED]
        low_pool.lvls = [1.09900]
        low_pool.drawn_act = [True]
        low_pool.state = [liq_sweep_a.ST_PURPLE]

        def fake_time_exit(*args, **kwargs):
            del args, kwargs
            return {"hold_bars": 0}

        entry_spec = liq_sweep_a._build_entry_spec_from_exit_override(  # pylint: disable=protected-access
            side="long",
            entry_index=5,
            entry_time=pd.Timestamp("2025-01-01T00:25:00Z"),
            signal_price=1.1000,
            entry_open=1.1005,
            prev_low=1.0990,
            prev_high=1.1010,
            entry_atr=0.001,
            high_pool=high_pool,
            low_pool=low_pool,
            strategy_params=params,
            cfg=cfg,
            equity=150_000.0,
            exit_builder=fake_time_exit,
            exit_params={"use_structural_stop_sizing": True},
            exit_supports_entry=True,
            size_fn=lambda **kwargs: self.fail("native structural sizing should bypass override size_fn"),
            sizing_params={},
            max_exit_index=10,
        )

        self.assertIsNotNone(entry_spec)
        self.assertEqual(int(entry_spec["time_exit_i"]), 5)
        self.assertAlmostEqual(float(entry_spec["qty"]), 645_161.2903225806, places=6)
        self.assertAlmostEqual(float(entry_spec["risk_dollars"]), 1_000.0, places=8)

    def test_exit_override_entry_builder_uses_actual_entry_open_for_structural_rr_gate(self) -> None:
        params = liq_sweep_a.Params(min_rr=1.0, max_rr=10.0, risk_pct=0.01)
        cfg = liq_sweep_a.BacktestConfig(initial_equity=100_000.0)
        high_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        high_pool.lvls = [1.10110]
        high_pool.drawn_act = [True]
        high_pool.state = [liq_sweep_a.ST_RED]
        low_pool.lvls = [1.09900]
        low_pool.drawn_act = [True]
        low_pool.state = [liq_sweep_a.ST_PURPLE]

        def fake_exit_builder(side, entry_open, prev_low, prev_high, params_dict, entry=None):
            del side, entry_open, prev_low, prev_high, params_dict, entry
            return {"sl": 1.0995, "tp": 1.1020, "stop_dist": 0.001}

        entry_spec = liq_sweep_a._build_entry_spec_from_exit_override(  # pylint: disable=protected-access
            side="long",
            entry_index=5,
            entry_time=pd.Timestamp("2025-01-01T00:25:00Z"),
            signal_price=1.1000,
            entry_open=1.1001,
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

        self.assertIsNone(entry_spec)

    def test_run_backtest_supports_time_exit_override(self) -> None:
        df = _sample_ohlc(rows=6)
        params = liq_sweep_a.Params(min_rr=1.0, max_rr=10.0, risk_pct=0.01)  # pylint: disable=not-callable
        feat = liq_sweep_a.compute_features(df, params)
        sig = liq_sweep_a.compute_signals(feat)

        trigger_seq = [(None, 1.1000)] + [(None, None)] * (len(sig) - 1)
        with mock.patch.object(liq_sweep_a, "sweep_triggers", side_effect=trigger_seq), mock.patch.object(
            liq_sweep_a,
            "_structural_entry_gate",
            return_value={"sl": 1.0990, "tp": 1.1020, "risk_dist": 0.001, "reward_dist": 0.002, "rr": 2.0},
        ):
            _eq, trades_df, summary = liq_sweep_a.run_backtest(
                sig,
                strategy_params=params,
                override_exit_builder=build_time_exit,
                override_exit_params={"hold_bars": 1},
            )

        self.assertEqual(int(summary["trades"]), 1)
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(str(trades_df.iloc[0]["exit_reason"]), "TIME_EXIT")

    def test_sweep_triggers_return_last_breached_red_levels(self) -> None:
        high_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        low_pool = liq_sweep_a._LevelPool(max_size=10)  # pylint: disable=protected-access
        high_pool.lvls = [1.1010, 1.1020]
        high_pool.drawn_act = [True, True]
        high_pool.state = [liq_sweep_a.ST_RED, liq_sweep_a.ST_RED]
        low_pool.lvls = [1.0990, 1.0980]
        low_pool.drawn_act = [True, True]
        low_pool.state = [liq_sweep_a.ST_RED, liq_sweep_a.ST_RED]

        short_entry_price, long_entry_price = liq_sweep_a.sweep_triggers(
            high_pool,
            low_pool,
            high_val=1.1030,
            low_val=1.0970,
            event_time=pd.Timestamp("2025-01-01T00:25:00Z"),
        )

        self.assertAlmostEqual(float(short_entry_price), 1.1020, places=8)
        self.assertAlmostEqual(float(long_entry_price), 1.0980, places=8)
        self.assertEqual(high_pool.drawn_act, [False, False])
        self.assertEqual(low_pool.drawn_act, [False, False])

    def test_run_backtest_enters_immediately_on_breach_bar_at_red_line_price(self) -> None:
        idx = pd.DatetimeIndex([pd.Timestamp("2025-01-01T00:00:00Z")])
        df = pd.DataFrame(
            {
                "open": [1.1002],
                "high": [1.1004],
                "low": [1.0988],
                "close": [1.0996],
            },
            index=idx,
        )
        params = liq_sweep_a.Params()
        captured: dict[str, float | int] = {}

        def fake_build_entry_spec(**kwargs):
            captured["entry_price"] = float(kwargs["entry_price"])
            captured["entry_index"] = int(kwargs["entry_index"])
            return {
                "side": "short",
                "qty": 1.0,
                "sl": 1.1010,
                "tp": 1.0990,
                "risk_dollars": 100.0,
            }

        with mock.patch.multiple(
            liq_sweep_a,
            track_breach_high=lambda *args, **kwargs: None,
            track_breach_low=lambda *args, **kwargs: None,
            append_high_pivot=lambda *args, **kwargs: None,
            append_low_pivot=lambda *args, **kwargs: None,
            confirm_move_away_high=lambda *args, **kwargs: None,
            confirm_move_away_low=lambda *args, **kwargs: None,
            stop_extending_high=lambda *args, **kwargs: None,
            stop_extending_low=lambda *args, **kwargs: None,
            sweep_triggers=lambda *args, **kwargs: (1.1000, None),
            _build_entry_spec=fake_build_entry_spec,
        ):
            _equity_df, trades_df, summary = liq_sweep_a.run_backtest(df, strategy_params=params)

        self.assertEqual(captured, {"entry_price": 1.1, "entry_index": 0})
        self.assertEqual(len(trades_df), 1)
        self.assertEqual(pd.Timestamp(trades_df.iloc[0]["entry_time"]), idx[0])
        self.assertAlmostEqual(float(trades_df.iloc[0]["entry"]), 1.1000, places=8)
        self.assertAlmostEqual(float(trades_df.iloc[0]["exit"]), 1.0990, places=8)
        self.assertEqual(str(trades_df.iloc[0]["side"]), "short")
        self.assertEqual(int(summary["trades"]), 1)

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

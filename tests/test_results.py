from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.artifacts import limited_iterations_path, limited_trades_path, spec_path, summary_path
from quantbt.results import enrich_limited_results, load_limited_summary


class ResultsTests(unittest.TestCase):
    def test_enrich_limited_results_backfills_avg_profit_and_recomputes_favourable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_iterations_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 1, "trades": 40, "total_return_%": 5.0, "favourable": False},
                    {"iter": 2, "trades": 35, "total_return_%": -3.0, "favourable": False},
                    {"iter": 3, "trades": 20, "total_return_%": 4.0, "favourable": False},
                ]
            ).to_csv(limited_iterations_path(run_dir), index=False)
            pd.DataFrame(
                [
                    {"iter": 1, "pnl": 100.0},
                    {"iter": 1, "pnl": 50.0},
                    {"iter": 2, "pnl": -25.0},
                    {"iter": 2, "pnl": -10.0},
                    {"iter": 3, "pnl": 40.0},
                ]
            ).to_csv(limited_trades_path(run_dir), index=False)
            run_meta = {
                "criteria": {"mode": "all", "rules": [{"metric": "avg_profit_per_trade", "op": ">", "value": 0.0}]},
                "spec": {"initial_equity": 100000.0},
            }
            pass_summary = {"pass_threshold_%": 50.0, "min_trades": 30}

            enriched, updated_summary = enrich_limited_results(
                run_dir,
                pd.read_csv(limited_iterations_path(run_dir)),
                run_meta,
                pass_summary,
            )

            self.assertAlmostEqual(float(enriched.loc[0, "avg_profit_per_trade"]), 75.0)
            self.assertAlmostEqual(float(enriched.loc[1, "avg_profit_per_trade"]), -17.5)
            self.assertAlmostEqual(float(enriched.loc[2, "avg_profit_per_trade"]), 40.0)
            self.assertEqual(enriched["favourable"].tolist(), [True, False, False])
            self.assertAlmostEqual(float(updated_summary["favourable_pct"]), 50.0)
            self.assertFalse(bool(updated_summary["passed"]))

    def test_load_limited_summary_recomputes_favourable_pct_from_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            limited_iterations_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"iter": 1, "favourable": True, "trades": 40, "net_profit_abs": 100.0},
                    {"iter": 2, "favourable": False, "trades": 25, "net_profit_abs": -50.0},
                    {"iter": 3, "favourable": True, "trades": 35, "net_profit_abs": 75.0},
                    {"iter": 4, "favourable": False, "trades": 45, "net_profit_abs": -25.0},
                ]
            ).to_csv(limited_iterations_path(run_dir), index=False)
            summary_path(run_dir).write_text(
                json.dumps(
                    {
                        "favourable_pct": 0.0,
                        "pass_threshold_%": 40.0,
                        "passed": False,
                        "total_iters": 4,
                        "required_valid_iters": 3,
                        "min_trades": 30,
                    }
                ),
                encoding="utf-8",
            )
            spec_path(run_dir).write_text(
                json.dumps(
                    {
                        "criteria": {"mode": "all", "rules": [{"metric": "avg_profit_per_trade", "op": ">", "value": 0.0}]},
                        "spec": {"strategy": {"name": "Test Strategy"}},
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_limited_summary(run_dir)

            self.assertAlmostEqual(float(loaded["pass_summary"]["favourable_pct"]), 66.66666666666666)
            self.assertEqual(loaded["pass_summary"]["valid_iters"], 3)
            self.assertEqual(loaded["pass_summary"]["required_valid_iters"], 3)
            self.assertTrue(loaded["pass_summary"]["passed"])
            self.assertEqual(loaded["decision"], "PASS")


if __name__ == "__main__":
    unittest.main()

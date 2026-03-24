from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.plugins.exits.random_time_exit import build_exit


class RandomTimeExitTests(unittest.TestCase):
    def test_avg_hold_bars_sampling_is_uniform_from_zero_to_double_mean(self) -> None:
        exit_spec = build_exit(
            side="long",
            entry_open=1.1,
            prev_low=1.0,
            prev_high=1.2,
            params={"avg_hold_bars": 5.0, "seed": 7},
            entry={"entry_i": 4},
        )

        self.assertEqual(exit_spec, {"hold_bars": 5})

    def test_avg_hold_bars_sampling_can_return_zero(self) -> None:
        seen = {
            build_exit(
                side="long",
                entry_open=1.1,
                prev_low=1.0,
                prev_high=1.2,
                params={"avg_hold_bars": 0.5, "seed": seed},
                entry={"entry_i": 1},
            )["hold_bars"]
            for seed in range(1, 20)
        }

        self.assertIn(0, seen)
        self.assertTrue(seen.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

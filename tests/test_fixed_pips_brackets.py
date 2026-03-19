from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.plugins.exits.fixed_pips_brackets import build_exit


class FixedPipsBracketsTests(unittest.TestCase):
    def test_build_exit_long_uses_constant_pip_distance(self) -> None:
        out = build_exit(
            side="long",
            entry_open=1.1000,
            prev_low=1.0990,
            prev_high=1.1010,
            params={"rr": 1.5, "stop_pips": 20.0, "pip_size": 0.0001},
        )

        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["sl"], 1.0980, places=8)
        self.assertAlmostEqual(out["tp"], 1.1030, places=8)
        self.assertAlmostEqual(out["stop_dist"], 0.0020, places=8)

    def test_build_exit_short_uses_constant_pip_distance(self) -> None:
        out = build_exit(
            side="short",
            entry_open=1.1000,
            prev_low=1.0990,
            prev_high=1.1010,
            params={"rr": 2.0, "stop_pips": 15.0, "pip_size": 0.0001},
        )

        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["sl"], 1.1015, places=8)
        self.assertAlmostEqual(out["tp"], 1.0970, places=8)
        self.assertAlmostEqual(out["stop_dist"], 0.0015, places=8)


if __name__ == "__main__":
    unittest.main()

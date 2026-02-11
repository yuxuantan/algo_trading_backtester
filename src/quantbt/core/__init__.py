"""Core backtest engines and metrics."""

from .engine import BacktestConfig, run_backtest_sma_cross
from .engine_limited import run_backtest_limited
from .metrics import max_drawdown, profit_factor

__all__ = [
    "BacktestConfig",
    "run_backtest_sma_cross",
    "run_backtest_limited",
    "max_drawdown",
    "profit_factor",
]

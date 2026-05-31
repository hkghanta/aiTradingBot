"""Backtest stage: walk-forward validation and performance metrics.

The :class:`WalkForwardBacktester` rolls a train/test window across history,
fitting the Brain on each in-sample window and evaluating the regime-driven
allocation strategy out-of-sample with causal (no look-ahead) signals.
"""

from trading_bot.backtest.engine import (
    BacktestResult,
    WalkForwardBacktester,
    WindowResult,
)
from trading_bot.backtest.performance import (
    PerformanceReport,
    compute_performance,
    max_drawdown,
    sharpe_ratio,
)

__all__ = [
    "WalkForwardBacktester",
    "BacktestResult",
    "WindowResult",
    "PerformanceReport",
    "compute_performance",
    "max_drawdown",
    "sharpe_ratio",
]

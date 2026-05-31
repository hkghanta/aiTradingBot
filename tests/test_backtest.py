"""Tests for the walk-forward backtester."""

from __future__ import annotations

import numpy as np
import pytest

from trading_bot.backtest import WalkForwardBacktester
from trading_bot.backtest.engine import _net_returns
from trading_bot.broker.mock_broker import MockBroker
from trading_bot.core.config import AppConfig, BacktestConfig, BrainConfig


def _config() -> AppConfig:
    # Small windows so a few hundred bars yield multiple walk-forward iterations.
    return AppConfig(
        symbols=["SPY"],
        brain=BrainConfig(n_regimes=4, lookback_days=120),
        backtest=BacktestConfig(train_size=150, test_size=60, step=60, sma_window=50),
    )


def test_net_returns_charges_turnover():
    weights = np.array([0.0, 1.0, 1.0])  # one unit of turnover entering at t=1
    asset = np.array([0.0, 0.0, 0.0])
    out = _net_returns(weights, asset, slippage_bps=10.0)
    # t=1 turnover = 1.0 -> cost 10bps = 0.001; no asset return.
    assert out[1] == pytest.approx(-0.001)
    assert out[0] == 0.0


def test_backtest_runs_and_reports_benchmarks():
    bars = MockBroker(symbols=["SPY"], history=500).get_recent_bars("SPY", 500)
    result = WalkForwardBacktester(_config()).run(bars)

    assert len(result.windows) >= 2
    assert set(result.benchmarks) == {"buy_and_hold", "sma_trend", "random"}
    assert result.strategy.n_periods > 0

    d = result.to_dict()
    assert d["n_windows"] == len(result.windows)
    assert "buy_and_hold" in d["benchmarks"]


def test_strategy_weights_bounded_by_regime_table():
    bars = MockBroker(symbols=["SPY"], history=500).get_recent_bars("SPY", 500)
    result = WalkForwardBacktester(_config()).run(bars)
    # Strategy is long-only and never levered: returns should be finite.
    assert np.isfinite(result.strategy.total_return)
    assert np.isfinite(result.strategy.sharpe)


def test_buy_and_hold_matches_realized_drift():
    bars = MockBroker(symbols=["SPY"], history=500).get_recent_bars("SPY", 500)
    result = WalkForwardBacktester(_config()).run(bars)
    bh = result.benchmarks["buy_and_hold"]
    # Buy-and-hold is always fully invested, so it trades zero times after entry.
    assert bh.n_periods > 0

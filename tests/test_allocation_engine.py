"""Tests for allocation and end-to-end engine wiring against the mock broker."""

from __future__ import annotations

from datetime import datetime

import pytest

from trading_bot.allocation import Allocator
from trading_bot.broker.mock_broker import MockBroker
from trading_bot.core.config import AppConfig, BrainConfig
from trading_bot.core.types import Regime, RegimeSignal
from trading_bot.engine import Engine


def _signal(regime: Regime, confidence: float) -> RegimeSignal:
    return RegimeSignal(
        timestamp=datetime(2024, 1, 1),
        regime=regime,
        probabilities={regime: confidence},
        confidence=confidence,
    )


def test_calmer_regime_allocates_more():
    alloc = Allocator(["SPY"])
    bull = alloc.allocate(_signal(Regime.BULL, 1.0)).invested_fraction
    bear = alloc.allocate(_signal(Regime.BEAR, 1.0)).invested_fraction
    crash = alloc.allocate(_signal(Regime.CRASH, 1.0)).invested_fraction
    assert crash <= bear <= bull


def test_confidence_scales_exposure():
    alloc = Allocator(["SPY"])
    high = alloc.allocate(_signal(Regime.BULL, 1.0)).invested_fraction
    low = alloc.allocate(_signal(Regime.BULL, 0.3)).invested_fraction
    assert low < high


def test_weights_split_across_symbols_and_sum_to_invested():
    alloc = Allocator(["SPY", "QQQ"])
    target = alloc.allocate(_signal(Regime.BULL, 1.0))
    assert set(target.weights) == {"SPY", "QQQ"}
    assert sum(target.weights.values()) == pytest.approx(target.invested_fraction)


def test_allocator_requires_symbols():
    with pytest.raises(ValueError):
        Allocator([])


def test_engine_runs_end_to_end():
    config = AppConfig(symbols=["SPY"], brain=BrainConfig(n_regimes=4, lookback_days=150))
    broker = MockBroker(symbols=["SPY"], history=500)
    engine = Engine(config, broker)

    state = engine.run_cycle()
    assert engine.brain.is_fitted
    assert state.signal is not None
    assert state.account is not None

    d = engine.state_dict()
    assert d["regime"] is not None
    assert d["account"]["equity"] > 0
    assert "tripped" in d["safety"]


def test_engine_multiple_cycles_stay_consistent():
    config = AppConfig(symbols=["SPY"], brain=BrainConfig(n_regimes=4, lookback_days=150))
    broker = MockBroker(symbols=["SPY"], history=500)
    engine = Engine(config, broker)
    for _ in range(4):
        state = engine.run_cycle()
    # Cash + position value should equal reported equity (no money created).
    acct = state.account
    pos_value = sum(p.market_value for p in acct.positions)
    assert acct.cash + pos_value == pytest.approx(acct.equity, rel=1e-6)

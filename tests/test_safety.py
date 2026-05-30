"""Tests for the safety circuit breakers."""

from __future__ import annotations

from datetime import datetime

from trading_bot.core.config import SafetyConfig
from trading_bot.core.types import AccountSnapshot, Position
from trading_bot.safety import CircuitBreaker


def _account(equity: float, positions=()) -> AccountSnapshot:
    return AccountSnapshot(
        timestamp=datetime(2024, 1, 1), equity=equity, cash=equity, positions=positions
    )


def test_allows_trading_within_limits():
    cb = CircuitBreaker(SafetyConfig(max_daily_loss_pct=0.05))
    cb.start_day(100_000)
    assert cb.check(_account(99_000), []).allowed is True


def test_daily_loss_trips_breaker():
    cb = CircuitBreaker(SafetyConfig(max_daily_loss_pct=0.05))
    cb.start_day(100_000)
    decision = cb.check(_account(94_000), [])  # -6% on the day
    assert decision.allowed is False
    assert any("daily loss" in r for r in decision.reasons)


def test_breaker_latches_until_reset():
    cb = CircuitBreaker(SafetyConfig(max_daily_loss_pct=0.05))
    cb.start_day(100_000)
    cb.check(_account(94_000), [])
    # Even after equity recovers, the breaker stays tripped.
    assert cb.check(_account(100_500), []).allowed is False
    cb.reset()
    assert cb.check(_account(100_500), []).allowed is True


def test_drawdown_trips_breaker():
    cb = CircuitBreaker(SafetyConfig(max_drawdown_pct=0.20))
    cb.start_day(100_000)
    cb.check(_account(120_000), [])  # new high-water mark
    decision = cb.check(_account(90_000), [])  # -25% from peak
    assert decision.allowed is False
    assert any("drawdown" in r for r in decision.reasons)


def test_position_concentration_trips_breaker():
    cb = CircuitBreaker(SafetyConfig(max_position_pct=0.25))
    cb.start_day(100_000)
    pos = Position(symbol="SPY", quantity=10, avg_entry_price=100, market_value=40_000)
    decision = cb.check(_account(100_000, positions=(pos,)), [])
    assert decision.allowed is False
    assert any("concentration" in r for r in decision.reasons)

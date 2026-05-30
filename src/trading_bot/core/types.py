"""Shared domain types that flow between pipeline stages.

These are deliberately small, immutable, and dependency-free so every stage
(Brain, Allocation, Safety, Broker, Dashboard) can speak the same language
without importing each other's implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping


class Regime(Enum):
    """Market regime emitted by the Brain (HMM).

    Ordered from most risk-off to most risk-on. The integer values are used as
    a coarse risk axis by the Allocation stage.
    """

    CRASH = 0
    BEAR = 1
    NEUTRAL = 2
    BULL = 3
    EUPHORIA = 4


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Bar:
    """A single OHLCV price bar for one symbol."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class RegimeSignal:
    """Output of the Brain stage for a point in time."""

    timestamp: datetime
    regime: Regime
    # Per-regime posterior probabilities, e.g. {Regime.BULL: 0.7, ...}.
    probabilities: Mapping[Regime, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class TargetAllocation:
    """Output of the Allocation stage.

    ``weights`` maps symbol -> target portfolio weight in [0, 1].
    ``invested_fraction`` is how much of the portfolio should be deployed
    (1.0 - cash). Calm regimes lean higher; turbulent regimes lean lower.
    """

    timestamp: datetime
    invested_fraction: float
    weights: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    """A trade instruction handed to the Broker."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    client_order_id: str | None = None


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    market_value: float


@dataclass(frozen=True)
class AccountSnapshot:
    """Point-in-time account state, used by Safety and the Dashboard."""

    timestamp: datetime
    equity: float
    cash: float
    positions: tuple[Position, ...] = ()

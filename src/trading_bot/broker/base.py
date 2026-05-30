"""Broker interface.

Every broker adapter implements this so the rest of the pipeline never depends
on a specific brokerage. Keep this surface minimal: account state, market data,
and order submission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trading_bot.core.types import AccountSnapshot, Bar, Order


class BrokerBase(ABC):
    """Abstract base for broker adapters."""

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """Return the current account snapshot (equity, cash, positions)."""

    @abstractmethod
    def get_recent_bars(self, symbol: str, lookback: int) -> Sequence[Bar]:
        """Return the most recent ``lookback`` price bars for ``symbol``."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order; return a broker-assigned order id."""

    @abstractmethod
    def cancel_all(self) -> None:
        """Cancel all open orders (used by Safety when a breaker trips)."""

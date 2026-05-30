"""In-memory mock broker for tests and dry runs.

Implements :class:`BrokerBase` without any network access so the pipeline can be
exercised end-to-end offline.

NOTE: scaffolding only -- methods raise NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.broker.base import BrokerBase
from trading_bot.core.types import AccountSnapshot, Bar, Order


class MockBroker(BrokerBase):
    """Simulated broker holding state in memory."""

    def __init__(self, starting_cash: float = 100_000.0) -> None:
        self.starting_cash = starting_cash

    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError("MockBroker.get_account not yet implemented.")

    def get_recent_bars(self, symbol: str, lookback: int) -> Sequence[Bar]:
        raise NotImplementedError("MockBroker.get_recent_bars not yet implemented.")

    def submit_order(self, order: Order) -> str:
        raise NotImplementedError("MockBroker.submit_order not yet implemented.")

    def cancel_all(self) -> None:
        raise NotImplementedError("MockBroker.cancel_all not yet implemented.")

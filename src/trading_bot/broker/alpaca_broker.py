"""Alpaca broker adapter (paper trading first).

Wraps ``alpaca-py``. Defaults to the paper-trading endpoint; flipping to live
requires changing ``ALPACA_BASE_URL`` and is intentionally not the default.

NOTE: scaffolding only -- methods raise NotImplementedError. Credentials are
read from :class:`~trading_bot.core.config.AlpacaSettings`.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.broker.base import BrokerBase
from trading_bot.core.config import AlpacaSettings
from trading_bot.core.types import AccountSnapshot, Bar, Order


class AlpacaBroker(BrokerBase):
    """Broker adapter backed by Alpaca's REST API."""

    def __init__(self, settings: AlpacaSettings | None = None) -> None:
        self.settings = settings or AlpacaSettings()
        self._is_paper = "paper" in self.settings.base_url
        self._trading_client = None  # alpaca TradingClient, set in connect()
        self._data_client = None  # alpaca StockHistoricalDataClient

    def connect(self) -> None:
        """Instantiate Alpaca clients from settings."""

        raise NotImplementedError("AlpacaBroker.connect not yet implemented.")

    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError("AlpacaBroker.get_account not yet implemented.")

    def get_recent_bars(self, symbol: str, lookback: int) -> Sequence[Bar]:
        raise NotImplementedError("AlpacaBroker.get_recent_bars not yet implemented.")

    def submit_order(self, order: Order) -> str:
        raise NotImplementedError("AlpacaBroker.submit_order not yet implemented.")

    def cancel_all(self) -> None:
        raise NotImplementedError("AlpacaBroker.cancel_all not yet implemented.")

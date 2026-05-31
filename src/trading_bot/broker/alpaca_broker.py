"""Alpaca broker adapter (paper trading first).

Wraps ``alpaca-py``. Defaults to the paper-trading endpoint; flipping to live
requires pointing ``ALPACA_BASE_URL`` at the live API and is intentionally not
the default. Credentials come from :class:`~trading_bot.core.config.AlpacaSettings`
(i.e. the environment / ``.env``) and are never hardcoded.

The Alpaca SDK is imported lazily inside :meth:`connect` so the rest of the
package (and the test suite) does not require network/credentials just to import
this module. Methods translate between Alpaca's models and the bot's own
dependency-free :mod:`trading_bot.core.types`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from trading_bot.broker.base import BrokerBase
from trading_bot.core.config import AlpacaSettings
from trading_bot.core.logging import get_logger
from trading_bot.core.types import (
    AccountSnapshot,
    Bar,
    Order,
    OrderSide,
    Position,
)

logger = get_logger(__name__)


class AlpacaBroker(BrokerBase):
    """Broker adapter backed by Alpaca's REST API."""

    def __init__(self, settings: AlpacaSettings | None = None) -> None:
        self.settings = settings or AlpacaSettings()
        self._is_paper = "paper" in self.settings.base_url
        self._trading_client = None  # alpaca TradingClient, set in connect()
        self._data_client = None  # alpaca StockHistoricalDataClient

    def connect(self) -> None:
        """Instantiate Alpaca clients from settings and verify credentials."""

        if not self.settings.api_key or not self.settings.secret_key:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your environment / .env."
            )

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        self._trading_client = TradingClient(
            api_key=self.settings.api_key,
            secret_key=self.settings.secret_key,
            paper=self._is_paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self.settings.api_key,
            secret_key=self.settings.secret_key,
        )
        account = self._trading_client.get_account()
        logger.info(
            "Connected to Alpaca (%s). Account status=%s equity=%s",
            "paper" if self._is_paper else "LIVE",
            getattr(account, "status", "?"),
            getattr(account, "equity", "?"),
        )

    def _require_trading(self):
        if self._trading_client is None:
            raise RuntimeError("AlpacaBroker.connect() must be called first.")
        return self._trading_client

    def _require_data(self):
        if self._data_client is None:
            raise RuntimeError("AlpacaBroker.connect() must be called first.")
        return self._data_client

    def get_account(self) -> AccountSnapshot:
        client = self._require_trading()
        account = client.get_account()
        positions = tuple(
            Position(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
            )
            for p in client.get_all_positions()
        )
        return AccountSnapshot(
            timestamp=datetime.now(UTC),
            equity=float(account.equity),
            cash=float(account.cash),
            positions=positions,
        )

    def get_recent_bars(self, symbol: str, lookback: int) -> Sequence[Bar]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._require_data()
        # Pad the calendar window generously; markets are closed ~30% of days.
        start = datetime.now(UTC) - timedelta(days=int(lookback * 2) + 10)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
        )
        barset = client.get_stock_bars(request)
        rows = barset.data.get(symbol, [])
        bars = [
            Bar(
                symbol=symbol,
                timestamp=b.timestamp,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in rows
        ]
        return bars[-lookback:]

    def submit_order(self, order: Order) -> str:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        client = self._require_trading()
        side = AlpacaSide.BUY if order.side is OrderSide.BUY else AlpacaSide.SELL

        if order.limit_price is not None:
            request = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        else:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=order.client_order_id,
            )

        submitted = client.submit_order(request)
        logger.info(
            "Submitted %s %s x%s -> id=%s",
            order.side.value,
            order.symbol,
            order.quantity,
            submitted.id,
        )
        return str(submitted.id)

    def cancel_all(self) -> None:
        client = self._require_trading()
        client.cancel_orders()
        logger.warning("Cancelled all open Alpaca orders.")

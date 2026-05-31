"""Tests for the Alpaca adapter using in-process fakes (no network, no keys).

We exercise the translation between Alpaca's models and the bot's core types,
and the request mapping, by injecting fake trading/data clients. The real SDK is
imported lazily inside connect()/submit_order(); here we bypass connect() and
set the clients directly, then assert on what the adapter does with them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from trading_bot.broker.alpaca_broker import AlpacaBroker
from trading_bot.core.config import AlpacaSettings
from trading_bot.core.types import Order, OrderSide


class _FakeTradingClient:
    def __init__(self):
        self.submitted = []
        self.cancelled = False

    def get_account(self):
        return SimpleNamespace(equity="100000", cash="25000", status="ACTIVE")

    def get_all_positions(self):
        return [
            SimpleNamespace(
                symbol="SPY", qty="10", avg_entry_price="400", market_value="4200"
            )
        ]

    def submit_order(self, request):
        self.submitted.append(request)
        return SimpleNamespace(id="order-123")

    def cancel_orders(self):
        self.cancelled = True


class _FakeBar:
    def __init__(self, close):
        self.symbol = "SPY"
        self.timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        self.open = close - 1
        self.high = close + 1
        self.low = close - 2
        self.close = close
        self.volume = 1_000_000


class _FakeDataClient:
    def get_stock_bars(self, request):
        bars = [_FakeBar(100 + i) for i in range(5)]
        return SimpleNamespace(data={"SPY": bars})


def _broker_with_fakes():
    broker = AlpacaBroker(AlpacaSettings(api_key="k", secret_key="s"))
    broker._trading_client = _FakeTradingClient()
    broker._data_client = _FakeDataClient()
    return broker


def test_connect_requires_credentials():
    broker = AlpacaBroker(AlpacaSettings(api_key="", secret_key=""))
    with pytest.raises(RuntimeError, match="credentials missing"):
        broker.connect()


def test_defaults_to_paper_endpoint():
    broker = AlpacaBroker(AlpacaSettings())
    assert broker._is_paper is True


def test_get_account_translates_models():
    broker = _broker_with_fakes()
    snap = broker.get_account()
    assert snap.equity == 100000.0
    assert snap.cash == 25000.0
    assert len(snap.positions) == 1
    pos = snap.positions[0]
    assert pos.symbol == "SPY"
    assert pos.quantity == 10.0
    assert pos.market_value == 4200.0


def test_get_recent_bars_translates_and_truncates():
    broker = _broker_with_fakes()
    bars = broker.get_recent_bars("SPY", lookback=3)
    assert len(bars) == 3  # truncated to lookback
    assert bars[-1].close == 104.0
    assert all(b.symbol == "SPY" for b in bars)


def test_submit_market_order_maps_side():
    broker = _broker_with_fakes()
    order_id = broker.submit_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=5))
    assert order_id == "order-123"
    req = broker._trading_client.submitted[0]
    assert req.symbol == "SPY"
    assert float(req.qty) == 5.0


def test_submit_limit_order_when_limit_price_set():
    broker = _broker_with_fakes()
    broker.submit_order(
        Order(symbol="SPY", side=OrderSide.SELL, quantity=2, limit_price=410.0)
    )
    req = broker._trading_client.submitted[0]
    assert float(req.limit_price) == 410.0


def test_methods_require_connect():
    broker = AlpacaBroker(AlpacaSettings(api_key="k", secret_key="s"))
    with pytest.raises(RuntimeError, match="connect"):
        broker.get_account()


def test_cancel_all_delegates():
    broker = _broker_with_fakes()
    broker.cancel_all()
    assert broker._trading_client.cancelled is True

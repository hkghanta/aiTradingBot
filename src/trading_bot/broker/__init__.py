"""Broker stage: execute orders against a brokerage.

``BrokerBase`` defines the interface; concrete adapters (Alpaca paper trading
first, IBKR later) implement it.
"""

from trading_bot.broker.alpaca_broker import AlpacaBroker
from trading_bot.broker.base import BrokerBase

__all__ = ["BrokerBase", "AlpacaBroker"]

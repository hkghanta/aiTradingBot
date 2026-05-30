"""Pipeline orchestration: Brain -> Allocation -> Safety -> Broker.

The :class:`Engine` runs one trading cycle by pulling data from the broker,
asking the Brain for a regime, the Allocator for targets, converting targets to
orders, screening them through Safety, and finally submitting survivors to the
broker. It records state for the Dashboard to read.

NOTE: scaffolding only -- ``run_cycle`` outlines the flow but raises
NotImplementedError where real logic is required.
"""

from __future__ import annotations

from trading_bot.allocation import Allocator
from trading_bot.brain import RegimeModel
from trading_bot.broker.base import BrokerBase
from trading_bot.core.config import AppConfig
from trading_bot.core.logging import get_logger
from trading_bot.safety import CircuitBreaker

logger = get_logger(__name__)


class Engine:
    """Coordinates the five pipeline stages for a single account."""

    def __init__(
        self,
        config: AppConfig,
        broker: BrokerBase,
        brain: RegimeModel | None = None,
        allocator: Allocator | None = None,
        safety: CircuitBreaker | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.brain = brain or RegimeModel(config.brain)
        self.allocator = allocator or Allocator(config.symbols, config.allocation)
        self.safety = safety or CircuitBreaker(config.safety)

    def run_cycle(self) -> None:
        """Execute one full Brain -> Broker cycle.

        Outline:
          1. account  = broker.get_account()
          2. bars     = broker.get_recent_bars(...) per symbol
          3. signal   = brain.predict(bars)
          4. targets  = allocator.allocate(signal)
          5. orders   = self._targets_to_orders(targets, account)
          6. decision = safety.check(account, orders)
          7. if decision.allowed: submit orders; else broker.cancel_all()
        """

        raise NotImplementedError("Engine.run_cycle not yet implemented.")

    def _targets_to_orders(self, targets, account):
        """Diff target weights against current positions into orders."""

        raise NotImplementedError("Engine._targets_to_orders not yet implemented.")

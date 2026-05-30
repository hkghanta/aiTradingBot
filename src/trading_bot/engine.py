"""Pipeline orchestration: Brain -> Allocation -> Safety -> Broker.

The :class:`Engine` runs one trading cycle: pull data and account state from the
broker, ask the Brain for a regime, the Allocator for targets, diff targets into
orders, screen them through Safety, and submit survivors. It records the latest
state so the Dashboard can read it.

The Brain is trained lazily on the first cycle (or via :meth:`warmup`) using the
first configured symbol's history, then reused for inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trading_bot.allocation import Allocator
from trading_bot.brain import RegimeModel
from trading_bot.broker.base import BrokerBase
from trading_bot.core.config import AppConfig
from trading_bot.core.logging import get_logger
from trading_bot.core.types import (
    AccountSnapshot,
    Order,
    OrderSide,
    RegimeSignal,
    TargetAllocation,
)
from trading_bot.safety import CircuitBreaker, SafetyDecision

logger = get_logger(__name__)

# Don't bother trading dust: ignore order notionals below this fraction of equity.
_MIN_ORDER_FRACTION = 0.005


@dataclass
class EngineState:
    """Snapshot of the most recent cycle, surfaced to the dashboard."""

    signal: RegimeSignal | None = None
    allocation: TargetAllocation | None = None
    account: AccountSnapshot | None = None
    safety: SafetyDecision | None = None
    recent_orders: list[Order] = field(default_factory=list)


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
        self.state = EngineState()

    def warmup(self) -> None:
        """Train the Brain on the primary symbol's history."""

        primary = self.config.symbols[0]
        bars = self.broker.get_recent_bars(primary, self.config.brain.lookback_days * 2)
        self.brain.fit(bars)

    def run_cycle(self) -> EngineState:
        """Execute one full Brain -> Broker cycle and return the new state."""

        if not self.brain.is_fitted:
            self.warmup()

        account = self.broker.get_account()
        primary = self.config.symbols[0]
        bars = self.broker.get_recent_bars(primary, self.config.brain.lookback_days)

        signal = self.brain.predict(bars)
        targets = self.allocator.allocate(signal)
        orders = self._targets_to_orders(targets, account)
        decision = self.safety.check(account, orders)

        submitted: list[Order] = []
        if decision.allowed:
            for order in orders:
                self.broker.submit_order(order)
                submitted.append(order)
        else:
            logger.warning("Safety halted trading: %s", "; ".join(decision.reasons))
            self.broker.cancel_all()

        self.state = EngineState(
            signal=signal,
            allocation=targets,
            account=account,
            safety=decision,
            recent_orders=submitted,
        )
        return self.state

    def _targets_to_orders(
        self, targets: TargetAllocation, account: AccountSnapshot
    ) -> list[Order]:
        """Diff target weights against current positions into market orders."""

        equity = account.equity
        if equity <= 0:
            return []

        current_value = {p.symbol: p.market_value for p in account.positions}
        min_notional = equity * _MIN_ORDER_FRACTION
        orders: list[Order] = []

        for symbol, weight in targets.weights.items():
            target_value = equity * weight
            delta = target_value - current_value.get(symbol, 0.0)
            if abs(delta) < min_notional:
                continue
            price = self._price_for(symbol, account)
            if price <= 0:
                continue
            qty = abs(delta) / price
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(symbol=symbol, side=side, quantity=qty))

        return orders

    def _price_for(self, symbol: str, account: AccountSnapshot) -> float:
        bars = self.broker.get_recent_bars(symbol, 1)
        return bars[-1].close if bars else 0.0

    def state_dict(self) -> dict[str, Any]:
        """Serialize the latest state for the dashboard API."""

        s = self.state
        return {
            "regime": s.signal.regime.name if s.signal else None,
            "confidence": round(s.signal.confidence, 4) if s.signal else None,
            "probabilities": (
                {r.name: round(p, 4) for r, p in s.signal.probabilities.items()}
                if s.signal
                else {}
            ),
            "invested_fraction": (
                round(s.allocation.invested_fraction, 4) if s.allocation else None
            ),
            "account": (
                {
                    "equity": round(s.account.equity, 2),
                    "cash": round(s.account.cash, 2),
                    "positions": [
                        {"symbol": p.symbol, "qty": round(p.quantity, 4),
                         "market_value": round(p.market_value, 2)}
                        for p in s.account.positions
                    ],
                }
                if s.account
                else None
            ),
            "safety": {
                "tripped": (not s.safety.allowed) if s.safety else False,
                "reasons": list(s.safety.reasons) if s.safety else [],
            },
            "recent_orders": [
                {"symbol": o.symbol, "side": o.side.value, "qty": round(o.quantity, 4)}
                for o in s.recent_orders
            ],
        }

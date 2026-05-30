"""Circuit breakers.

Hard, model-independent risk limits: daily loss, total drawdown, and per-position
concentration. If any limit trips, trading is halted regardless of what the
Brain or Allocator want.

NOTE: scaffolding only -- evaluation logic raises NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_bot.core.config import SafetyConfig
from trading_bot.core.types import AccountSnapshot, Order


@dataclass(frozen=True)
class SafetyDecision:
    """Result of a safety check.

    ``allowed`` is False when a breaker has tripped; ``reasons`` lists every
    triggered limit for logging and the dashboard.
    """

    allowed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


class CircuitBreaker:
    """Stateful guard that tracks equity highs and enforces risk limits."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._equity_high_water: float | None = None
        self._day_start_equity: float | None = None
        self._tripped = False

    def check(
        self, account: AccountSnapshot, proposed_orders: list[Order]
    ) -> SafetyDecision:
        """Evaluate all breakers against the account and proposed orders.

        Planned checks: daily loss vs ``max_daily_loss_pct``, drawdown from
        the equity high-water mark vs ``max_drawdown_pct``, and per-position
        concentration vs ``max_position_pct``. Once tripped, stays tripped
        until :meth:`reset`.
        """

        raise NotImplementedError("CircuitBreaker.check not yet implemented.")

    def reset(self) -> None:
        """Clear a tripped breaker (e.g. at the start of a new trading day)."""

        self._tripped = False

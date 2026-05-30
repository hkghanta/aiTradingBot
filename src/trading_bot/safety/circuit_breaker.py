"""Circuit breakers.

Hard, model-independent risk limits: daily loss, total drawdown, and per-
position concentration. If any limit trips, trading is halted regardless of what
the Brain or Allocator want. Once tripped, the breaker stays tripped until
:meth:`reset` (e.g. at the start of a new trading day) -- a deliberate "go look
at what happened" gate rather than something the model can silently clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_bot.core.config import SafetyConfig
from trading_bot.core.logging import get_logger
from trading_bot.core.types import AccountSnapshot, Order

logger = get_logger(__name__)


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

    @property
    def tripped(self) -> bool:
        return self._tripped

    def start_day(self, equity: float) -> None:
        """Mark the start-of-day equity baseline and clear daily trips."""

        self._day_start_equity = equity
        self._tripped = False

    def check(self, account: AccountSnapshot, proposed_orders: list[Order]) -> SafetyDecision:
        """Evaluate all breakers against the account and proposed orders."""

        equity = account.equity
        if self._day_start_equity is None:
            self._day_start_equity = equity
        if self._equity_high_water is None or equity > self._equity_high_water:
            self._equity_high_water = equity

        reasons: list[str] = []

        if self._day_start_equity > 0:
            daily_change = (equity - self._day_start_equity) / self._day_start_equity
            if daily_change <= -self.config.max_daily_loss_pct:
                reasons.append(
                    f"daily loss {daily_change:.2%} <= -{self.config.max_daily_loss_pct:.2%}"
                )

        if self._equity_high_water > 0:
            drawdown = (equity - self._equity_high_water) / self._equity_high_water
            if drawdown <= -self.config.max_drawdown_pct:
                reasons.append(
                    f"drawdown {drawdown:.2%} <= -{self.config.max_drawdown_pct:.2%}"
                )

        # Position concentration on existing holdings.
        if equity > 0:
            for pos in account.positions:
                frac = pos.market_value / equity
                if frac > self.config.max_position_pct:
                    reasons.append(
                        f"{pos.symbol} concentration {frac:.2%} > "
                        f"{self.config.max_position_pct:.2%}"
                    )

        if reasons:
            if not self._tripped:
                logger.warning("Circuit breaker TRIPPED: %s", "; ".join(reasons))
            self._tripped = True

        if self._tripped:
            return SafetyDecision(allowed=False, reasons=tuple(reasons) or ("breaker latched",))
        return SafetyDecision(allowed=True)

    def reset(self) -> None:
        """Clear a tripped breaker (manual intervention / new trading day)."""

        self._tripped = False

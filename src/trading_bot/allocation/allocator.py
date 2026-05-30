"""Regime-aware allocation.

Maps the Brain's regime signal to how much of the portfolio should be invested
and how it is split across symbols. Calm/bullish regimes deploy more capital;
turbulent/bearish regimes raise cash. Confidence scales exposure: a low-
confidence signal invests less than the regime's nominal target.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.core.config import AllocationConfig
from trading_bot.core.types import RegimeSignal, TargetAllocation


class Allocator:
    """Produces a :class:`TargetAllocation` from a :class:`RegimeSignal`."""

    def __init__(self, symbols: Sequence[str], config: AllocationConfig | None = None) -> None:
        if not symbols:
            raise ValueError("Allocator requires at least one symbol.")
        self.symbols = list(symbols)
        self.config = config or AllocationConfig()

    def allocate(self, signal: RegimeSignal) -> TargetAllocation:
        """Compute target invested fraction and equal-weight per-symbol split."""

        table = self.config.invested_fraction_by_regime
        nominal = table.get(signal.regime.name, 0.0)

        # Scale exposure by confidence; clamp into [0, 1].
        confidence = min(max(signal.confidence, 0.0), 1.0)
        invested = max(0.0, min(1.0, nominal * confidence))

        per_symbol = invested / len(self.symbols)
        weights = {sym: per_symbol for sym in self.symbols}

        return TargetAllocation(
            timestamp=signal.timestamp,
            invested_fraction=invested,
            weights=weights,
        )

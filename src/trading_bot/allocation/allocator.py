"""Regime-aware allocation.

Maps the Brain's regime signal to how much of the portfolio should be invested
and how it should be split across symbols. Calm/bullish regimes deploy more
capital; turbulent/bearish regimes raise cash.

NOTE: scaffolding only -- methods raise NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.core.config import AllocationConfig
from trading_bot.core.types import RegimeSignal, TargetAllocation


class Allocator:
    """Produces a :class:`TargetAllocation` from a :class:`RegimeSignal`."""

    def __init__(self, symbols: Sequence[str], config: AllocationConfig | None = None) -> None:
        self.symbols = list(symbols)
        self.config = config or AllocationConfig()

    def allocate(self, signal: RegimeSignal) -> TargetAllocation:
        """Compute target invested fraction and per-symbol weights.

        Planned behavior: look up ``invested_fraction_by_regime`` for the
        signaled regime, optionally scale by ``signal.confidence``, then split
        the invested fraction across ``self.symbols``.
        """

        raise NotImplementedError("Allocator.allocate not yet implemented.")

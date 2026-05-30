"""Hidden Markov Model that classifies the market into regimes.

Regimes: Crash, Bear, Neutral, Bull, Euphoria.

The model is trained on engineered features (see ``features.py``) and, at
inference time, emits a :class:`RegimeSignal` with the most likely regime and
the full posterior distribution.

NOTE: scaffolding only -- methods raise NotImplementedError. The interface is
stable so downstream stages can be wired and tested against it.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.core.config import BrainConfig
from trading_bot.core.types import Bar, RegimeSignal


class RegimeModel:
    """Wraps an HMM and maps its hidden states to named market regimes.

    A trained HMM produces anonymous hidden states; we sort/label them by
    expected return so state -> :class:`~trading_bot.core.types.Regime` mapping
    is stable across retrains.
    """

    def __init__(self, config: BrainConfig | None = None) -> None:
        self.config = config or BrainConfig()
        self._model = None  # hmmlearn GaussianHMM, set in fit()
        self._state_to_regime: dict[int, int] = {}

    def fit(self, bars: Sequence[Bar]) -> "RegimeModel":
        """Train the HMM on historical bars and label its hidden states."""

        raise NotImplementedError("RegimeModel.fit not yet implemented.")

    def predict(self, bars: Sequence[Bar]) -> RegimeSignal:
        """Return the current regime signal for the latest bar."""

        raise NotImplementedError("RegimeModel.predict not yet implemented.")

    def save(self, path: str) -> None:
        raise NotImplementedError("RegimeModel.save not yet implemented.")

    @classmethod
    def load(cls, path: str) -> "RegimeModel":
        raise NotImplementedError("RegimeModel.load not yet implemented.")

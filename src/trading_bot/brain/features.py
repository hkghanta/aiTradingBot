"""Feature engineering for the regime model.

The HMM observes a small set of return/volatility features rather than raw
prices. Implementations live here so the model code stays focused on the HMM
itself.

NOTE: scaffolding only -- functions raise NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Sequence

from trading_bot.core.types import Bar


def build_features(bars: Sequence[Bar]):
    """Turn a sequence of price bars into a feature matrix for the HMM.

    Intended features (to be implemented): log returns, rolling realized
    volatility, and a volume/turnover signal. Returns an (n_samples, n_features)
    array aligned to ``bars``.
    """

    raise NotImplementedError("Brain feature engineering not yet implemented.")

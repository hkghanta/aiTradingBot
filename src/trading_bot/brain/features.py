"""Feature engineering for the regime model.

The HMM observes a small set of return/volatility/volume features rather than
raw prices. Keeping this here lets the model code focus on the HMM itself.

All features are causal: the value at bar ``t`` uses only bars ``<= t``. The
first ``warmup`` bars cannot be computed (they need history) and are dropped;
:func:`build_features` returns both the matrix and the bars it aligns to.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trading_bot.core.types import Bar

# Number of feature columns produced, in order:
#   0: log return
#   1: rolling realized volatility (std of log returns)
#   2: volume z-score (turnover signal)
N_FEATURES = 3


def build_features(
    bars: Sequence[Bar], vol_window: int = 20
) -> tuple[np.ndarray, list[Bar]]:
    """Turn price bars into a causal feature matrix for the HMM.

    Returns ``(features, aligned_bars)`` where ``features`` has shape
    ``(n, N_FEATURES)`` and ``aligned_bars`` are the ``n`` bars each row
    corresponds to. Raises ``ValueError`` if there are too few bars.
    """

    warmup = vol_window  # need vol_window returns before the first usable row
    if len(bars) <= warmup + 1:
        raise ValueError(
            f"Need more than {warmup + 1} bars to build features, got {len(bars)}."
        )

    closes = np.array([b.close for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)

    # Log returns; r[i] corresponds to bars[i] (return from i-1 to i). r[0]=0.
    log_ret = np.zeros_like(closes)
    log_ret[1:] = np.log(closes[1:] / np.clip(closes[:-1], 1e-12, None))

    n = len(bars)
    feats = np.empty((n, N_FEATURES))
    feats[:, 0] = log_ret

    # Trailing realized vol and volume stats, strictly using past+current bars.
    for i in range(n):
        lo = max(0, i - vol_window + 1)
        window_ret = log_ret[lo : i + 1]
        window_vol = volumes[lo : i + 1]
        feats[i, 1] = window_ret.std() if window_ret.size > 1 else 0.0
        v_std = window_vol.std()
        feats[i, 2] = (volumes[i] - window_vol.mean()) / v_std if v_std > 0 else 0.0

    # Drop the warmup region where statistics are not yet meaningful.
    return feats[warmup:], list(bars[warmup:])

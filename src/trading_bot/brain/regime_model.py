"""Hidden Markov Model that classifies the market into regimes.

Regimes (most risk-off to most risk-on): Crash, Bear, Neutral, Bull, Euphoria.

Two details matter and are implemented deliberately:

  * **No look-ahead.** Inference uses the *forward* algorithm only (filtering):
    the regime at the latest bar is computed from bars ``<= t`` exclusively.
    hmmlearn's ``predict`` / ``predict_proba`` run forward-backward (smoothing),
    which leaks future information, so we do not use them for live signals.
  * **Stability filter.** A newly-detected regime must persist for
    ``min_persist`` consecutive bars before it is committed. Frequent flipping
    is treated as instability and lowers confidence.
"""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from trading_bot.brain.features import build_features
from trading_bot.core.config import BrainConfig
from trading_bot.core.logging import get_logger
from trading_bot.core.types import Bar, Regime, RegimeSignal

logger = get_logger(__name__)

# Regimes ordered from most risk-off to most risk-on. Subsets are taken from the
# ends inward so a 3-regime model is Bear/Neutral/Bull, not Crash/Bear/Neutral.
_REGIME_ORDER = [Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA]


def regimes_for_count(n: int) -> list[Regime]:
    """Return ``n`` regime labels, ordered risk-off -> risk-on."""

    if not 1 <= n <= len(_REGIME_ORDER):
        raise ValueError(f"n_regimes must be in 1..{len(_REGIME_ORDER)}, got {n}.")
    if n == len(_REGIME_ORDER):
        return list(_REGIME_ORDER)
    # Center the chosen labels (e.g. n=3 -> BEAR, NEUTRAL, BULL).
    start = (len(_REGIME_ORDER) - n) // 2
    return _REGIME_ORDER[start : start + n]


class RegimeModel:
    """Wraps a Gaussian HMM and maps its hidden states to named regimes.

    A trained HMM produces anonymous states; we label them by mean log-return so
    the state -> :class:`Regime` mapping is stable across retrains.
    """

    def __init__(self, config: BrainConfig | None = None, min_persist: int = 3) -> None:
        self.config = config or BrainConfig()
        self.min_persist = min_persist
        self._model: GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._state_to_regime: dict[int, Regime] = {}
        # Stability-filter state (live inference).
        self._recent_raw: deque[Regime] = deque(maxlen=20)
        self._committed: Regime | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, bars: Sequence[Bar]) -> RegimeModel:
        """Train the HMM on historical bars and label its hidden states."""

        feats, _ = build_features(bars)
        self._scaler = StandardScaler().fit(feats)
        X = self._scaler.transform(feats)

        n = self.config.n_regimes
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        self._model = model

        # Label states by mean (unscaled) log-return: lowest -> most risk-off.
        states = model.predict(X)  # training-time labeling only; not a live signal
        labels = regimes_for_count(n)
        mean_ret = {
            s: feats[states == s, 0].mean() if np.any(states == s) else 0.0
            for s in range(n)
        }
        ordered_states = sorted(range(n), key=lambda s: mean_ret[s])
        self._state_to_regime = {state: labels[i] for i, state in enumerate(ordered_states)}

        self._recent_raw.clear()
        self._committed = None
        logger.info("RegimeModel fitted: %d regimes, %d training samples", n, len(X))
        return self

    def filtered_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Per-bar filtered state posteriors via the forward algorithm.

        Returns an ``(n_samples, n_states)`` array where row ``t`` is the
        posterior over states given bars ``0..t`` only (filtering, not
        smoothing) -- so it never uses future bars. Covariances are jittered and
        non-finite emissions are floored to keep the recursion numerically safe.
        """

        assert self._model is not None
        m = self._model
        K = m.n_components
        n = len(X)
        dim = X.shape[1]

        log_b = np.empty((n, K))
        for k in range(K):
            cov = np.asarray(m.covars_[k]) + np.eye(dim) * 1e-6
            log_b[:, k] = multivariate_normal.logpdf(X, m.means_[k], cov, allow_singular=True)
        log_b = np.nan_to_num(log_b, nan=-1e10, neginf=-1e10, posinf=1e10)

        log_t = np.log(m.transmat_ + 1e-300)
        out = np.empty((n, K))
        log_alpha = np.log(m.startprob_ + 1e-300) + log_b[0]
        out[0] = self._normalize(log_alpha)
        for t in range(1, n):
            log_alpha = logsumexp(log_alpha[:, None] + log_t, axis=0) + log_b[t]
            out[t] = self._normalize(log_alpha)
        return out

    @staticmethod
    def _normalize(log_alpha: np.ndarray) -> np.ndarray:
        post = np.exp(log_alpha - logsumexp(log_alpha))
        total = post.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full_like(post, 1.0 / len(post))
        return post / total

    def _filtered_state_posterior(self, X: np.ndarray) -> np.ndarray:
        """Filtering posterior over states at the latest bar (no look-ahead)."""

        return self.filtered_posteriors(X)[-1]

    def _signal_from_state_post(
        self, state_post: np.ndarray, timestamp
    ) -> RegimeSignal:
        """Build a stability-filtered RegimeSignal from a state posterior."""

        probs: dict[Regime, float] = {r: 0.0 for r in regimes_for_count(self.config.n_regimes)}
        for state, p in enumerate(state_post):
            probs[self._state_to_regime[state]] += float(p)

        raw = max(probs, key=probs.get)
        committed = self._apply_stability(raw)
        confidence = probs[committed] * self._stability_factor()

        return RegimeSignal(
            timestamp=timestamp,
            regime=committed,
            probabilities=probs,
            confidence=confidence,
        )

    def predict(self, bars: Sequence[Bar]) -> RegimeSignal:
        """Return the current regime signal for the latest bar."""

        if self._model is None or self._scaler is None:
            raise RuntimeError("RegimeModel must be fitted before predict().")

        feats, aligned = build_features(bars)
        X = self._scaler.transform(feats)
        state_post = self._filtered_state_posterior(X)
        return self._signal_from_state_post(state_post, aligned[-1].timestamp)

    def predict_path(self, bars: Sequence[Bar]) -> list[RegimeSignal]:
        """Return a causal regime signal for every usable bar.

        Each signal at index ``t`` uses only bars ``<= t`` (filtering) and the
        stability filter is advanced bar-by-bar, exactly as it would be live.
        This is what the backtester consumes so there is no look-ahead. The
        returned list aligns with ``build_features(bars)[1]`` (the warmup region
        is dropped).
        """

        if self._model is None or self._scaler is None:
            raise RuntimeError("RegimeModel must be fitted before predict_path().")

        feats, aligned = build_features(bars)
        X = self._scaler.transform(feats)
        posteriors = self.filtered_posteriors(X)

        # Reset stability state so the walk starts clean.
        self._recent_raw.clear()
        self._committed = None
        return [
            self._signal_from_state_post(posteriors[t], aligned[t].timestamp)
            for t in range(len(aligned))
        ]

    def _apply_stability(self, raw: Regime) -> Regime:
        """Only switch the committed regime after ``min_persist`` agreeing bars."""

        self._recent_raw.append(raw)
        if self._committed is None:
            self._committed = raw
            return raw

        recent = list(self._recent_raw)[-self.min_persist :]
        if len(recent) >= self.min_persist and all(r == raw for r in recent):
            if raw != self._committed:
                logger.warning("Regime change: %s -> %s", self._committed.name, raw.name)
            self._committed = raw
        return self._committed

    def _stability_factor(self) -> float:
        """Scale confidence down when the raw regime has been flickering."""

        if len(self._recent_raw) < 2:
            return 1.0
        recent = list(self._recent_raw)
        changes = sum(1 for a, b in zip(recent, recent[1:], strict=False) if a != b)
        # >4 changes in the recent window is treated as unstable.
        return 0.5 if changes > 4 else 1.0

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save an unfitted RegimeModel.")
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "config": self.config,
                    "min_persist": self.min_persist,
                    "model": self._model,
                    "scaler": self._scaler,
                    "state_to_regime": self._state_to_regime,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> RegimeModel:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls(config=data["config"], min_persist=data["min_persist"])
        obj._model = data["model"]
        obj._scaler = data["scaler"]
        obj._state_to_regime = data["state_to_regime"]
        return obj

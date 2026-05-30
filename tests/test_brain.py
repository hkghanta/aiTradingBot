"""Tests for the Brain: feature engineering, HMM fit, and no-look-ahead inference."""

from __future__ import annotations

import numpy as np
import pytest

from trading_bot.brain.features import N_FEATURES, build_features
from trading_bot.brain.regime_model import RegimeModel, regimes_for_count
from trading_bot.broker.mock_broker import MockBroker
from trading_bot.core.config import BrainConfig
from trading_bot.core.types import Regime


@pytest.fixture
def bars():
    return list(MockBroker(symbols=["SPY"], history=500).get_recent_bars("SPY", 500))


def test_build_features_shape_and_alignment(bars):
    feats, aligned = build_features(bars, vol_window=20)
    assert feats.shape[1] == N_FEATURES
    assert feats.shape[0] == len(aligned)
    assert len(aligned) == len(bars) - 20
    assert np.isfinite(feats).all()


def test_build_features_requires_enough_bars():
    with pytest.raises(ValueError):
        build_features(MockBroker(history=500).get_recent_bars("SPY", 5))


def test_regimes_for_count_is_ordered_subset():
    assert regimes_for_count(5) == [
        Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA,
    ]
    assert regimes_for_count(3) == [Regime.BEAR, Regime.NEUTRAL, Regime.BULL]
    with pytest.raises(ValueError):
        regimes_for_count(9)


def test_fit_and_predict(bars):
    model = RegimeModel(BrainConfig(n_regimes=4))
    assert not model.is_fitted
    model.fit(bars)
    assert model.is_fitted

    signal = model.predict(bars)
    assert isinstance(signal.regime, Regime)
    assert 0.0 <= signal.confidence <= 1.0
    total = sum(signal.probabilities.values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_inference_has_no_look_ahead(bars):
    """The filtered posterior at bar T must not depend on bars after T.

    We compute per-bar filtered posteriors on the full feature matrix, then on a
    matrix truncated at bar T. A forward (filtering) algorithm yields the same
    posterior for bar T either way; a smoothing algorithm (look-ahead) would not,
    because it would have folded in the future bars.
    """

    model = RegimeModel(BrainConfig(n_regimes=4)).fit(bars)
    feats, _ = build_features(bars)
    X = model._scaler.transform(feats)

    full = model.filtered_posteriors(X)
    cut = 80
    truncated = model.filtered_posteriors(X[: len(X) - cut])

    target_bar = len(X) - cut - 1
    np.testing.assert_allclose(full[target_bar], truncated[-1], rtol=1e-9, atol=1e-12)


def test_filtering_differs_from_smoothing(bars):
    """Sanity check: filtering (ours) genuinely differs from smoothing.

    If they were identical we might be leaking future data; they should differ on
    at least some bars because smoothing incorporates the whole sequence.
    """

    model = RegimeModel(BrainConfig(n_regimes=4)).fit(bars)
    feats, _ = build_features(bars)
    X = model._scaler.transform(feats)

    filt = model.filtered_posteriors(X)
    _, smoothed = model._model.score_samples(X)  # forward-backward (look-ahead)
    # Early/middle bars should differ; the last bar coincides by definition.
    assert not np.allclose(filt[: len(X) // 2], smoothed[: len(X) // 2], atol=1e-3)


def test_save_load_roundtrip(tmp_path, bars):
    model = RegimeModel(BrainConfig(n_regimes=4)).fit(bars)
    path = tmp_path / "model.pkl"
    model.save(str(path))
    loaded = RegimeModel.load(str(path))
    assert loaded.is_fitted
    sig = loaded.predict(bars)
    assert isinstance(sig.regime, Regime)


def test_predict_before_fit_raises(bars):
    with pytest.raises(RuntimeError):
        RegimeModel().predict(bars)

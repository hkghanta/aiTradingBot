"""Tests for the regime stability filter (must persist before committing)."""

from __future__ import annotations

from trading_bot.brain.regime_model import RegimeModel
from trading_bot.core.types import Regime


def test_regime_must_persist_before_committing():
    model = RegimeModel(min_persist=3)

    # First observation commits immediately (no prior state).
    assert model._apply_stability(Regime.NEUTRAL) is Regime.NEUTRAL

    # A single divergent reading does not flip the committed regime.
    assert model._apply_stability(Regime.BULL) is Regime.NEUTRAL
    assert model._apply_stability(Regime.BULL) is Regime.NEUTRAL
    # Third consecutive BULL meets min_persist -> commit.
    assert model._apply_stability(Regime.BULL) is Regime.BULL


def test_flickering_lowers_confidence_factor():
    model = RegimeModel(min_persist=3)
    seq = [Regime.BULL, Regime.BEAR] * 4  # 8 readings, 7 changes
    for r in seq:
        model._apply_stability(r)
    assert model._stability_factor() == 0.5


def test_stable_sequence_full_confidence_factor():
    model = RegimeModel(min_persist=3)
    for _ in range(6):
        model._apply_stability(Regime.BULL)
    assert model._stability_factor() == 1.0

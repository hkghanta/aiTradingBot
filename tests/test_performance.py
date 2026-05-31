"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pytest

from trading_bot.backtest.performance import (
    compute_performance,
    equity_curve,
    max_drawdown,
    sharpe_ratio,
)


def test_equity_curve_compounds():
    curve = equity_curve(np.array([0.1, -0.5, 2.0]))
    # 1 -> 1.1 -> 0.55 -> 1.65
    np.testing.assert_allclose(curve, [1.0, 1.1, 0.55, 1.65])


def test_max_drawdown_is_negative_and_correct():
    curve = np.array([1.0, 1.2, 0.6, 0.9])
    # Peak 1.2 -> trough 0.6 = -50%.
    assert max_drawdown(curve) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(np.array([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_sharpe_zero_variance_is_zero():
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, size=500)
    assert sharpe_ratio(r) > 0


def test_compute_performance_breakdowns():
    r = np.array([0.02, -0.01, 0.03, -0.02])
    regimes = ["BULL", "BEAR", "BULL", "BEAR"]
    conf = np.array([0.9, 0.4, 0.85, 0.3])
    report = compute_performance(r, regimes=regimes, confidences=conf, n_trades=3)

    assert report.n_periods == 4
    assert report.n_trades == 3
    assert 0.0 <= report.win_rate <= 1.0
    assert report.regime_returns["BULL"] == pytest.approx((0.02 + 0.03) / 2)
    assert report.regime_returns["BEAR"] == pytest.approx((-0.01 - 0.02) / 2)
    assert "high(>=0.8)" in report.confidence_returns
    assert "low(<0.5)" in report.confidence_returns


def test_compute_performance_empty():
    report = compute_performance(np.array([]))
    assert report.n_periods == 0
    assert report.total_return == 0.0


def test_report_to_dict_is_json_friendly():
    report = compute_performance(np.array([0.01, 0.02]))
    d = report.to_dict()
    assert set(d) >= {"total_return", "sharpe", "max_drawdown", "win_rate"}

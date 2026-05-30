"""Smoke tests for the scaffolding.

These verify the package wires together (imports, types, config, dashboard
factory) without asserting any not-yet-implemented trading logic.
"""

from __future__ import annotations

import pytest

from trading_bot.core.config import AppConfig, load_config
from trading_bot.core.types import OrderSide, Regime


def test_package_imports():
    import trading_bot

    assert trading_bot.__version__


def test_regime_is_ordered_risk_axis():
    assert Regime.CRASH.value < Regime.NEUTRAL.value < Regime.EUPHORIA.value


def test_order_side_values():
    assert OrderSide.BUY.value == "buy"
    assert OrderSide.SELL.value == "sell"


def test_load_default_config():
    config = load_config("config")
    assert isinstance(config, AppConfig)
    assert config.symbols
    assert config.brain.n_regimes == 5
    # Calmer regimes should deploy at least as much capital as turbulent ones.
    fractions = config.allocation.invested_fraction_by_regime
    assert fractions["CRASH"] <= fractions["NEUTRAL"] <= fractions["EUPHORIA"]


def test_dashboard_health_endpoint():
    from fastapi.testclient import TestClient

    from trading_bot.dashboard import create_app

    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: __import__("trading_bot.brain", fromlist=["RegimeModel"]).RegimeModel(),
        lambda: __import__(
            "trading_bot.safety", fromlist=["CircuitBreaker"]
        ).CircuitBreaker(),
    ],
)
def test_stages_construct(factory):
    # Stages should construct cleanly even though their logic is unimplemented.
    assert factory() is not None

"""FastAPI dashboard.

Exposes the bot's live state: current regime, target allocation, recent orders,
account equity, and whether any circuit breaker has tripped.

The factory wires a health endpoint and ``/api/state``; pass a ``state_provider``
(e.g. ``Engine.state_dict``) to surface live bot state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from trading_bot.core.config import DashboardConfig

# A state provider returns the current bot state as a JSON-serializable dict
# (see Engine.state_dict). When omitted, the dashboard serves an empty snapshot.
StateProvider = Callable[[], dict[str, Any]]

_EMPTY_STATE: dict[str, Any] = {
    "regime": None,
    "confidence": None,
    "probabilities": {},
    "invested_fraction": None,
    "account": None,
    "safety": {"tripped": False, "reasons": []},
    "recent_orders": [],
}


def create_app(
    config: DashboardConfig | None = None,
    state_provider: StateProvider | None = None,
) -> FastAPI:
    """Build the dashboard FastAPI application.

    Pass ``state_provider`` (e.g. ``engine.state_dict``) to surface live bot
    state on ``/api/state``; without it the endpoint returns an empty snapshot.
    """

    config = config or DashboardConfig()
    app = FastAPI(title="aiTradingBot Dashboard", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        """Current bot state (regime, allocation, account, safety, orders)."""

        return state_provider() if state_provider else dict(_EMPTY_STATE)

    return app

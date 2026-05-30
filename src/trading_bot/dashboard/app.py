"""FastAPI dashboard.

Exposes the bot's live state: current regime, target allocation, recent orders,
account equity, and whether any circuit breaker has tripped.

The factory below returns a configured app with a health endpoint wired up;
the data endpoints are stubs to be filled in as the pipeline produces state.
"""

from __future__ import annotations

from fastapi import FastAPI

from trading_bot.core.config import DashboardConfig


def create_app(config: DashboardConfig | None = None) -> FastAPI:
    """Build the dashboard FastAPI application."""

    config = config or DashboardConfig()
    app = FastAPI(title="aiTradingBot Dashboard", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def state() -> dict[str, object]:
        """Current bot state. Stub -- returns an empty snapshot for now."""

        return {
            "regime": None,
            "allocation": None,
            "account": None,
            "safety": {"tripped": False, "reasons": []},
            "recent_orders": [],
        }

    return app

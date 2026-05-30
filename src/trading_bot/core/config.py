"""Configuration loading.

Settings come from two layers, merged at startup:

  1. ``config/default.yaml`` (and an optional ``config/local.yaml`` override)
     for non-secret tunables.
  2. Environment variables / ``.env`` for secrets (API keys).

This module only defines the schema and loader; the actual values live in the
YAML files and the environment so nothing secret is committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlpacaSettings(BaseSettings):
    """Broker secrets, read from the environment / .env."""

    model_config = SettingsConfigDict(env_prefix="ALPACA_", env_file=".env", extra="ignore")

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"


class BrainConfig(BaseModel):
    n_regimes: int = 5
    lookback_days: int = 252
    retrain_interval_days: int = 5


class AllocationConfig(BaseModel):
    # Target invested fraction per regime name (calm -> more invested).
    invested_fraction_by_regime: dict[str, float] = Field(
        default_factory=lambda: {
            "CRASH": 0.0,
            "BEAR": 0.25,
            "NEUTRAL": 0.5,
            "BULL": 0.85,
            "EUPHORIA": 1.0,
        }
    )


class SafetyConfig(BaseModel):
    # Halt trading if intraday loss exceeds this fraction of equity.
    max_daily_loss_pct: float = 0.05
    # Halt if peak-to-trough drawdown exceeds this fraction.
    max_drawdown_pct: float = 0.20
    # Largest fraction of equity allowed in a single position.
    max_position_pct: float = 0.25


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY"])
    brain: BrainConfig = Field(default_factory=BrainConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Load and merge configuration.

    ``config/default.yaml`` provides defaults; ``config/local.yaml`` (gitignored)
    overrides them; secrets are injected from the environment via pydantic.
    """

    config_dir = Path(config_dir)
    merged: dict[str, Any] = {}
    merged.update(_load_yaml(config_dir / "default.yaml"))
    for key, value in _load_yaml(config_dir / "local.yaml").items():
        merged[key] = value

    config = AppConfig(**merged)
    # Secrets always come from the environment, regardless of YAML.
    config.alpaca = AlpacaSettings()
    return config

"""Command-line entry point.

Subcommands:
  * ``run``       -- run the trading engine loop (not yet implemented).
  * ``dashboard`` -- serve the FastAPI dashboard.
"""

from __future__ import annotations

import argparse

from trading_bot.core.config import load_config
from trading_bot.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    logger.info("Loaded config for symbols: %s", config.symbols)
    logger.warning("Engine run loop is not yet implemented (scaffolding).")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    from trading_bot.dashboard import create_app

    config = load_config(args.config_dir)
    app = create_app(config.dashboard)
    uvicorn.run(app, host=config.dashboard.host, port=config.dashboard.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading-bot", description="Automatic trading bot.")
    parser.add_argument("--config-dir", default="config", help="Path to config directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the trading engine loop.")
    run_p.set_defaults(func=_cmd_run)

    dash_p = sub.add_parser("dashboard", help="Serve the dashboard.")
    dash_p.set_defaults(func=_cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

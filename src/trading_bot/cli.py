"""Command-line entry point.

Subcommands:
  * ``run``       -- run the trading engine loop.
  * ``backtest``  -- walk-forward backtest against synthetic/mock data.
  * ``dashboard`` -- serve the FastAPI dashboard.
"""

from __future__ import annotations

import argparse

from trading_bot.core.config import load_config
from trading_bot.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _cmd_run(args: argparse.Namespace) -> int:
    import json

    from trading_bot.broker.mock_broker import MockBroker
    from trading_bot.engine import Engine

    config = load_config(args.config_dir)
    logger.info("Loaded config for symbols: %s", config.symbols)

    if args.broker == "mock":
        broker = MockBroker(symbols=config.symbols, history=config.brain.lookback_days * 3)
    else:  # alpaca
        from trading_bot.broker.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker(config.alpaca)
        broker.connect()

    engine = Engine(config, broker)
    for i in range(args.cycles):
        state = engine.run_cycle()
        regime = state.signal.regime.name if state.signal else "?"
        logger.info("cycle %d: regime=%s orders=%d", i + 1, regime, len(state.recent_orders))

    print(json.dumps(engine.state_dict(), indent=2))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    import json

    from trading_bot.backtest import WalkForwardBacktester
    from trading_bot.broker.mock_broker import MockBroker

    config = load_config(args.config_dir)
    broker = MockBroker(symbols=config.symbols, history=args.history)
    bars = broker.get_recent_bars(config.symbols[0], args.history)

    result = WalkForwardBacktester(config).run(bars)
    print(json.dumps(result.to_dict(), indent=2))
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
    run_p.add_argument(
        "--broker", choices=["mock", "alpaca"], default="mock", help="Broker backend."
    )
    run_p.add_argument("--cycles", type=int, default=5, help="Number of cycles to run.")
    run_p.set_defaults(func=_cmd_run)

    bt_p = sub.add_parser("backtest", help="Walk-forward backtest on synthetic data.")
    bt_p.add_argument("--history", type=int, default=1200, help="Number of bars to simulate.")
    bt_p.set_defaults(func=_cmd_backtest)

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

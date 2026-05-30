# aiTradingBot

An automatic trading bot built as a five-stage pipeline:

```
Brain ──> Allocation ──> Safety ──> Broker ──> Dashboard
```

| Stage          | Responsibility                                                                 |
| -------------- | ------------------------------------------------------------------------------ |
| **Brain**      | Classifies the market into regimes with Hidden Markov Models: Crash, Bear, Neutral, Bull, Euphoria. |
| **Allocation** | Adjusts how much of the portfolio is invested (calm vs. turbulent markets).     |
| **Safety**     | Circuit breakers that halt trading if losses hit thresholds — runs independently of the AI model. |
| **Broker**     | Executes trades. Alpaca (free API) first; IBKR (paid, high volume) later.       |
| **Dashboard**  | Real-time view of trades and insights so you can understand everything.         |

> **Status: scaffolding.** This repository currently contains the project
> structure, interfaces, configuration, and tests. Stage logic is stubbed and
> raises `NotImplementedError`. The skeleton imports, the config loads, and the
> dashboard serves.

## Layout

```
src/trading_bot/
  core/         shared types, config, logging
  brain/        HMM regime model + feature engineering
  allocation/   regime -> portfolio target mapping
  safety/       circuit breakers (model-independent)
  broker/       broker interface + Alpaca / mock adapters
  dashboard/    FastAPI real-time dashboard
  engine.py     orchestrates one Brain -> Broker cycle
  cli.py        `trading-bot run` / `trading-bot dashboard`
config/         default.yaml (tunables; secrets stay in .env)
tests/          smoke tests for the scaffolding
```

## Getting started

```bash
# 1. Install (editable, with dev tools)
python -m pip install -e ".[dev]"

# 2. Configure secrets (Alpaca paper-trading keys)
cp .env.example .env
# edit .env and add ALPACA_API_KEY / ALPACA_SECRET_KEY

# 3. Run the checks
pytest
ruff check .

# 4. Serve the dashboard
trading-bot dashboard      # http://127.0.0.1:8000/health
```

Get free paper-trading API keys from [Alpaca](https://alpaca.markets/). The
default endpoint is the **paper** endpoint — no real money is at risk.

## Safety first

The Safety stage is deliberately independent of the Brain. Even if the model
misbehaves, circuit breakers (daily loss, drawdown, position concentration)
can halt trading and cancel open orders. Limits live in `config/default.yaml`.

## Roadmap

- [ ] Brain: feature engineering + HMM training/inference
- [ ] Allocation: regime → invested fraction → per-symbol weights
- [ ] Safety: implement and unit-test each circuit breaker
- [ ] Broker: Alpaca paper-trading adapter (account, data, orders)
- [ ] Engine: target-to-order diffing and the run loop
- [ ] Dashboard: live state endpoints + UI
- [ ] Backtesting harness with the mock broker
- [ ] IBKR adapter for high-volume / live trading
```

"""Automatic trading bot.

Pipeline: Brain -> Allocation -> Safety -> Broker -> Dashboard.

  * brain      regime classification via Hidden Markov Models
  * allocation regime-aware portfolio sizing
  * safety     circuit breakers independent of the AI model
  * broker     order execution (Alpaca paper trading first)
  * dashboard  real-time view of trades and insights
"""

__version__ = "0.1.0"

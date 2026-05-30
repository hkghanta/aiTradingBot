"""Safety stage: circuit breakers that can halt trading.

This stage runs independently of the AI model so a bad model cannot bypass it.
"""

from trading_bot.safety.circuit_breaker import CircuitBreaker, SafetyDecision

__all__ = ["CircuitBreaker", "SafetyDecision"]

"""Performance metrics for an equity curve and its trades.

Pure functions over numpy arrays so the backtester (and later the live monitor)
can share one definition of "how did we do". No look-ahead here -- callers pass
realized period returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PerformanceReport:
    """Summary statistics for a strategy run."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_periods: int
    n_trades: int
    # Optional per-regime breakdown: regime name -> mean period return.
    regime_returns: dict[str, float] = field(default_factory=dict)
    # Optional confidence-bucket breakdown: bucket label -> mean period return.
    confidence_returns: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_return": round(self.total_return, 6),
            "annualized_return": round(self.annualized_return, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "win_rate": round(self.win_rate, 4),
            "n_periods": self.n_periods,
            "n_trades": self.n_trades,
            "regime_returns": {k: round(v, 6) for k, v in self.regime_returns.items()},
            "confidence_returns": {
                k: round(v, 6) for k, v in self.confidence_returns.items()
            },
        }


def equity_curve(period_returns: np.ndarray, starting_equity: float = 1.0) -> np.ndarray:
    """Compound period returns into an equity curve (length n+1)."""

    curve = starting_equity * np.cumprod(1.0 + np.asarray(period_returns, dtype=float))
    return np.concatenate([[starting_equity], curve])


def max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative fraction."""

    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(curve)
    drawdowns = curve / running_peak - 1.0
    return float(drawdowns.min())


def sharpe_ratio(period_returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed zero)."""

    r = np.asarray(period_returns, dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def compute_performance(
    period_returns: np.ndarray,
    *,
    periods_per_year: int = 252,
    n_trades: int = 0,
    regimes: list[str] | None = None,
    confidences: np.ndarray | None = None,
) -> PerformanceReport:
    """Compute a full :class:`PerformanceReport` from realized period returns.

    ``regimes`` / ``confidences``, when given, must align element-wise with
    ``period_returns`` and drive the optional breakdowns.
    """

    r = np.asarray(period_returns, dtype=float)
    n = r.size
    if n == 0:
        return PerformanceReport(0, 0, 0, 0, 0, 0, 0, n_trades)

    curve = equity_curve(r)
    total_return = float(curve[-1] / curve[0] - 1.0)
    ann_return = float((1.0 + r.mean()) ** periods_per_year - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    win_rate = float((r > 0).mean())

    regime_returns: dict[str, float] = {}
    if regimes is not None:
        regimes_arr = np.asarray(regimes, dtype=object)
        for name in dict.fromkeys(regimes):  # preserve first-seen order
            mask = regimes_arr == name
            if mask.any():
                regime_returns[name] = float(r[mask].mean())

    confidence_returns: dict[str, float] = {}
    if confidences is not None:
        c = np.asarray(confidences, dtype=float)
        buckets = {"low(<0.5)": c < 0.5, "med(0.5-0.8)": (c >= 0.5) & (c < 0.8),
                   "high(>=0.8)": c >= 0.8}
        for label, mask in buckets.items():
            if mask.any():
                confidence_returns[label] = float(r[mask].mean())

    return PerformanceReport(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe_ratio(r, periods_per_year),
        max_drawdown=max_drawdown(curve),
        win_rate=win_rate,
        n_periods=n,
        n_trades=n_trades,
        regime_returns=regime_returns,
        confidence_returns=confidence_returns,
    )

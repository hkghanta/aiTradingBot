"""Walk-forward backtesting engine.

This is *not* a single in-sample optimization. The data is split into rolling
windows: the HMM is fit on an in-sample training window, then the strategy is
evaluated on the immediately-following out-of-sample window using only causal
(filtered) regime signals. The window then rolls forward and the process
repeats, so every evaluated bar was scored by a model that never saw it.

Strategy under test: allocate to the single asset according to the Allocator's
regime-driven ``invested_fraction``, decided on bar ``t`` and earned over the
return from ``t`` to ``t+1`` (no look-ahead). Turnover is charged ``slippage_bps``.

Benchmarks for comparison: buy-and-hold, 200-day SMA trend following, and a
random-allocation control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from trading_bot.allocation import Allocator
from trading_bot.backtest.performance import PerformanceReport, compute_performance
from trading_bot.brain import RegimeModel
from trading_bot.core.config import AllocationConfig, AppConfig, BrainConfig
from trading_bot.core.logging import get_logger
from trading_bot.core.types import Bar

logger = get_logger(__name__)


@dataclass(frozen=True)
class WindowResult:
    """Out-of-sample result for one walk-forward window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    performance: PerformanceReport


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate walk-forward result plus benchmark comparisons."""

    strategy: PerformanceReport
    benchmarks: dict[str, PerformanceReport]
    windows: list[WindowResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.to_dict(),
            "benchmarks": {k: v.to_dict() for k, v in self.benchmarks.items()},
            "n_windows": len(self.windows),
        }


def _net_returns(weights: np.ndarray, asset_returns: np.ndarray, slippage_bps: float):
    """Strategy period returns net of turnover cost.

    ``weights[t]`` is decided from information up to bar ``t`` and earns
    ``asset_returns[t]`` (the return from ``t`` to ``t+1``). Turnover is the
    absolute change in weight versus the prior bar.
    """

    cost_rate = slippage_bps / 10_000.0
    prev = np.concatenate([[0.0], weights[:-1]])
    turnover = np.abs(weights - prev)
    gross = weights * asset_returns
    return gross - turnover * cost_rate


class WalkForwardBacktester:
    """Runs the rolling train/test loop and computes benchmark comparisons."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.bt = self.config.backtest

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        bars = list(bars)
        closes = np.array([b.close for b in bars], dtype=float)
        # asset_returns[t] = simple return from bar t to t+1; last bar has none.
        asset_returns = np.zeros(len(bars))
        asset_returns[:-1] = closes[1:] / closes[:-1] - 1.0

        weights = np.full(len(bars), np.nan)
        regime_labels: list[str | None] = [None] * len(bars)
        confidences = np.full(len(bars), np.nan)
        windows: list[WindowResult] = []

        train, test, step = self.bt.train_size, self.bt.test_size, self.bt.step
        start = 0
        while start + train + 1 < len(bars):
            tr_lo, tr_hi = start, start + train
            te_lo, te_hi = tr_hi, min(tr_hi + test, len(bars))
            if te_lo >= te_hi:
                break

            window_weights = self._evaluate_window(
                bars, tr_lo, tr_hi, te_lo, te_hi, regime_labels, confidences
            )
            for i, idx in enumerate(range(te_lo, te_hi)):
                weights[idx] = window_weights[i]

            w_perf = compute_performance(
                _net_returns(
                    window_weights,
                    asset_returns[te_lo:te_hi],
                    self.bt.slippage_bps,
                ),
                periods_per_year=self.bt.periods_per_year,
            )
            windows.append(WindowResult(tr_lo, tr_hi, te_lo, te_hi, w_perf))
            start += step

        # Stitch all out-of-sample bars into one strategy track record.
        oos = ~np.isnan(weights)
        strat_returns = _net_returns(
            weights[oos], asset_returns[oos], self.bt.slippage_bps
        )
        strat_regimes = [regime_labels[i] for i in np.flatnonzero(oos)]
        strategy = compute_performance(
            strat_returns,
            periods_per_year=self.bt.periods_per_year,
            n_trades=int(np.count_nonzero(np.diff(np.concatenate([[0.0], weights[oos]])))),
            regimes=[r for r in strat_regimes if r is not None] or None,
            confidences=confidences[oos] if np.isfinite(confidences[oos]).any() else None,
        )

        benchmarks = self._benchmarks(closes, asset_returns, oos)
        logger.info(
            "Backtest: %d windows, strategy Sharpe %.2f vs buy&hold %.2f",
            len(windows),
            strategy.sharpe,
            benchmarks["buy_and_hold"].sharpe,
        )
        return BacktestResult(strategy=strategy, benchmarks=benchmarks, windows=windows)

    def _evaluate_window(
        self,
        bars: list[Bar],
        tr_lo: int,
        tr_hi: int,
        te_lo: int,
        te_hi: int,
        regime_labels: list[str | None],
        confidences: np.ndarray,
    ) -> np.ndarray:
        """Fit on the training window, emit causal weights for the test window."""

        brain = RegimeModel(BrainConfig(**self.config.brain.model_dump()))
        brain.fit(bars[tr_lo:tr_hi])
        allocator = Allocator(
            [bars[0].symbol], AllocationConfig(**self.config.allocation.model_dump())
        )

        # Score the whole [tr_lo, te_hi) span causally in one pass. predict_path
        # drops the feature warmup from the front (the training region), so its
        # last (te_hi - te_lo) signals align exactly to bars [te_lo, te_hi).
        signals = brain.predict_path(bars[tr_lo:te_hi])
        tail = signals[-(te_hi - te_lo):]
        weights = np.empty(te_hi - te_lo)
        for i, sig in enumerate(tail):
            target = allocator.allocate(sig)
            weights[i] = target.invested_fraction
            regime_labels[te_lo + i] = sig.regime.name
            confidences[te_lo + i] = sig.confidence
        return weights

    def _benchmarks(
        self, closes: np.ndarray, asset_returns: np.ndarray, oos: np.ndarray
    ) -> dict[str, PerformanceReport]:
        ppy = self.bt.periods_per_year

        # Buy and hold: fully invested every OOS bar.
        bh = compute_performance(asset_returns[oos], periods_per_year=ppy)

        # 200-day SMA trend following: invest only when price > its SMA.
        sma = self._sma(closes, self.bt.sma_window)
        sma_w = (closes > sma).astype(float)
        sma_ret = _net_returns(sma_w[oos], asset_returns[oos], self.bt.slippage_bps)
        sma_perf = compute_performance(sma_ret, periods_per_year=ppy)

        # Random allocation control (seeded for reproducibility).
        rng = np.random.default_rng(0)
        rand_w = rng.uniform(0, 1, size=len(closes))
        rand_ret = _net_returns(rand_w[oos], asset_returns[oos], self.bt.slippage_bps)
        rand_perf = compute_performance(rand_ret, periods_per_year=ppy)

        return {
            "buy_and_hold": bh,
            "sma_trend": sma_perf,
            "random": rand_perf,
        }

    @staticmethod
    def _sma(closes: np.ndarray, window: int) -> np.ndarray:
        """Causal simple moving average; early bars use the expanding mean."""

        out = np.empty_like(closes)
        cumsum = np.cumsum(closes)
        for i in range(len(closes)):
            if i + 1 < window:
                out[i] = cumsum[i] / (i + 1)
            else:
                out[i] = (cumsum[i] - cumsum[i - window]) / window
        return out

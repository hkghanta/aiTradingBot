"""In-memory mock broker for tests and offline dry runs.

Implements :class:`BrokerBase` with a synthetic, regime-switching price series so
the whole pipeline can be exercised end-to-end without any network access or API
keys. Prices are generated deterministically from a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from trading_bot.broker.base import BrokerBase
from trading_bot.core.types import AccountSnapshot, Bar, Order, OrderSide, Position


class MockBroker(BrokerBase):
    """Simulated broker holding cash, positions, and synthetic prices in memory."""

    def __init__(
        self,
        symbols: Sequence[str] = ("SPY",),
        starting_cash: float = 100_000.0,
        history: int = 600,
        seed: int = 7,
    ) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.symbols = list(symbols)
        self._positions: dict[str, Position] = {}
        self._order_seq = 0
        self._bars: dict[str, list[Bar]] = {
            sym: self._generate_series(sym, history, seed + i)
            for i, sym in enumerate(self.symbols)
        }

    # ------------------------------------------------------------------ data
    @staticmethod
    def _generate_series(symbol: str, n: int, seed: int) -> list[Bar]:
        """Generate a regime-switching geometric random walk."""

        rng = np.random.default_rng(seed)
        # Alternate calm-bull and turbulent-bear blocks so the HMM has structure.
        drifts = [0.0008, -0.0012, 0.0002, 0.0015, -0.0020]
        vols = [0.006, 0.020, 0.010, 0.008, 0.025]
        rets = np.empty(n)
        i = 0
        while i < n:
            block = rng.integers(30, 70)
            k = rng.integers(0, len(drifts))
            seg = rng.normal(drifts[k], vols[k], size=min(block, n - i))
            rets[i : i + len(seg)] = seg
            i += len(seg)
        price = 100.0 * np.exp(np.cumsum(rets))

        start = datetime(2023, 1, 2)
        bars: list[Bar] = []
        for d in range(n):
            c = float(price[d])
            o = float(price[d - 1]) if d > 0 else c
            hi = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
            vol = float(rng.integers(1_000_000, 5_000_000))
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=start + timedelta(days=d),
                    open=o,
                    high=hi,
                    low=lo,
                    close=c,
                    volume=vol,
                )
            )
        return bars

    def _last_price(self, symbol: str) -> float:
        return self._bars[symbol][-1].close

    # --------------------------------------------------------------- account
    def get_account(self) -> AccountSnapshot:
        positions = []
        equity = self.cash
        for sym, pos in self._positions.items():
            price = self._last_price(sym)
            mv = pos.quantity * price
            equity += mv
            positions.append(
                Position(
                    symbol=sym,
                    quantity=pos.quantity,
                    avg_entry_price=pos.avg_entry_price,
                    market_value=mv,
                )
            )
        return AccountSnapshot(
            timestamp=self._bars[self.symbols[0]][-1].timestamp,
            equity=equity,
            cash=self.cash,
            positions=tuple(positions),
        )

    def get_recent_bars(self, symbol: str, lookback: int) -> Sequence[Bar]:
        return self._bars[symbol][-lookback:]

    def submit_order(self, order: Order) -> str:
        price = self._last_price(order.symbol)
        signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
        existing = self._positions.get(order.symbol)
        prev_qty = existing.quantity if existing else 0.0
        new_qty = prev_qty + signed

        self.cash -= signed * price

        if abs(new_qty) < 1e-9:
            self._positions.pop(order.symbol, None)
        else:
            # Update average entry only when increasing exposure in one direction.
            if existing and (prev_qty > 0) == (signed > 0):
                total_cost = existing.avg_entry_price * prev_qty + price * signed
                avg = total_cost / new_qty
            else:
                avg = price
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=new_qty,
                avg_entry_price=avg,
                market_value=new_qty * price,
            )

        self._order_seq += 1
        return f"mock-{self._order_seq}"

    def cancel_all(self) -> None:
        # Mock fills are immediate, so there are no resting orders to cancel.
        return None

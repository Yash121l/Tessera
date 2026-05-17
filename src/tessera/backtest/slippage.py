"""Slippage models for OHLCV-only and L2-book backtest simulation.

For OHLCV-only bars (the common case):
    impact_bps = k * sqrt(order_notional / adv_notional)

k defaults to 1.0 (conservative). It can be fit per-symbol from historical
fill data via `OHLCVSlippageModel.fit_k()`.

Latency model note:
    Nautilus LatencyModel applies a fixed insert latency to order commands.
    For sub-bar latency (< bar period), the effective market impact is captured
    here via a half-spread baseline penalty on taker fills. For supra-bar latency,
    the strategy-level signal_delay_bars parameter controls bar-level staleness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OHLCVSlippageModel:
    """Square-root market impact model for OHLCV-only bars.

    Parameters:
        k: Market impact coefficient. Default 1.0 (conservative).
           Fit from fill data via fit_k() when available.
        half_spread_bps: Half-spread baseline penalty added to taker fills.
                         Represents the cost of crossing the spread even when
                         order_notional → 0.
    """

    k: float = 1.0
    half_spread_bps: float = 2.5

    _k_by_symbol: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def fit_k(self, symbol: str, fills_df: pd.DataFrame) -> None:  # type: ignore[name-defined]  # noqa: F821
        """Fit k for a symbol using OLS on historical fill slippage.

        fills_df columns: order_notional (USD), adv_notional (USD), actual_slippage_bps.
        """
        if fills_df.empty or len(fills_df) < 5:
            return

        x = np.sqrt(fills_df["order_notional"] / fills_df["adv_notional"].clip(lower=1.0))
        y = fills_df["actual_slippage_bps"].values
        denom = float(np.dot(x, x))
        if denom < 1e-12:
            return
        k_hat = float(np.dot(x, y) / denom)
        self._k_by_symbol[symbol] = max(0.1, k_hat)

    def impact_bps(
        self,
        order_notional: float,
        adv_notional: float,
        symbol: str | None = None,
        is_taker: bool = True,
    ) -> float:
        """Slippage in basis points (always ≥ 0 for adverse impact).

        Args:
            order_notional: Order size in USD.
            adv_notional: Average daily volume in USD (fallback: 1M).
            symbol: Symbol key for per-symbol k lookup.
            is_taker: If True, adds half-spread penalty.
        """
        if adv_notional <= 0:
            adv_notional = 1_000_000.0

        k = self._k_by_symbol.get(symbol or "", self.k) if symbol else self.k
        impact = k * math.sqrt(order_notional / adv_notional) * 1e4  # → bps
        if is_taker:
            impact += self.half_spread_bps
        return max(0.0, impact)

    def adjust_price(
        self,
        price: float,
        side: str,
        order_notional: float,
        adv_notional: float,
        symbol: str | None = None,
        is_taker: bool = True,
    ) -> float:
        """Slippage-adjusted fill price (adverse direction).

        Buys get a higher fill price; sells get a lower fill price.
        """
        bps = self.impact_bps(order_notional, adv_notional, symbol, is_taker)
        adj = bps / 1e4
        if side.lower() == "buy":
            return price * (1.0 + adj)
        return price * (1.0 - adj)

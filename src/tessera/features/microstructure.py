"""Order-book microstructure features.

References:
- Cont et al. (2014) - Order Flow Imbalance
- Stoikov (2018) - Microprice
- Easley, Lopez de Prado & O'Hara (2012) - VPIN
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.features.base import Feature


class OrderFlowImbalance(Feature):
    """OFI per Cont et al. (2014). Requires L2 bid/ask price and size columns."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, depth: int = 1) -> None:
        self.depth = depth
        self.name = f"ofi_{depth}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        bid_col = "bid_price" if self.depth == 1 else f"bid_price_{self.depth}"
        ask_col = "ask_price" if self.depth == 1 else f"ask_price_{self.depth}"
        bid_size_col = "bid_size" if self.depth == 1 else f"bid_size_{self.depth}"
        ask_size_col = "ask_size" if self.depth == 1 else f"ask_size_{self.depth}"

        for col in [bid_col, ask_col, bid_size_col, ask_size_col]:
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index, name=self.name)

        bid_p = df[bid_col]
        ask_p = df[ask_col]
        bid_s = df[bid_size_col]
        ask_s = df[ask_size_col]

        prev_bid_p = bid_p.shift(1)
        prev_ask_p = ask_p.shift(1)
        prev_bid_s = bid_s.shift(1)
        prev_ask_s = ask_s.shift(1)

        # Bid side contribution
        delta_bid = np.where(
            bid_p > prev_bid_p,
            bid_s,
            np.where(bid_p == prev_bid_p, bid_s - prev_bid_s, -prev_bid_s),
        )

        # Ask side contribution
        delta_ask = np.where(
            ask_p < prev_ask_p,
            ask_s,
            np.where(ask_p == prev_ask_p, ask_s - prev_ask_s, -prev_ask_s),
        )

        ofi = pd.Series(delta_bid - delta_ask, index=df.index, name=self.name, dtype=float)
        ofi.iloc[0] = np.nan
        return ofi


class MicroPrice(Feature):
    """Stoikov microprice: size-weighted midprice."""

    name = "microprice"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        required = ["bid_price", "ask_price", "bid_size", "ask_size"]
        for col in required:
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index, name=self.name)

        bid = df["bid_price"]
        ask = df["ask_price"]
        bid_size = df["bid_size"]
        ask_size = df["ask_size"]

        total_size = bid_size + ask_size
        microprice = np.where(
            total_size > 0,
            (bid_size * ask + ask_size * bid) / total_size,
            (bid + ask) / 2,
        )
        return pd.Series(microprice, index=df.index, name=self.name, dtype=float)


class VPIN(Feature):
    """Volume-synchronized PIN (Easley, Lopez de Prado & O'Hara 2012)."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, bucket_size: float = 1000.0, window: int = 50) -> None:
        self.bucket_size = bucket_size
        self.window = window
        self.name = f"vpin_{int(bucket_size)}_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "volume" not in df.columns or "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        prices = df["close"].values
        volumes = df["volume"].values

        # Classify volume using tick rule
        price_diff = np.diff(np.asarray(prices, dtype=float), prepend=prices[0])
        buy_volume = np.where(price_diff >= 0, volumes, 0.0)
        sell_volume = np.where(price_diff < 0, volumes, 0.0)

        # Aggregate into volume buckets
        cum_vol = np.cumsum(volumes)
        bucket_ids = (cum_vol / self.bucket_size).astype(int)

        n = len(df)
        vpin_values = np.full(n, np.nan)

        # Compute VPIN per bar using rolling bucket approach
        max_bucket = bucket_ids[-1]
        if max_bucket < self.window:
            return pd.Series(vpin_values, index=df.index, name=self.name)

        bucket_buy = np.zeros(max_bucket + 1)
        bucket_sell = np.zeros(max_bucket + 1)
        bucket_total = np.zeros(max_bucket + 1)

        for i in range(n):
            b = bucket_ids[i]
            bucket_buy[b] += buy_volume[i]
            bucket_sell[b] += sell_volume[i]
            bucket_total[b] += volumes[i]

        # Compute cumulative imbalance per bucket
        bucket_imbalance = np.abs(bucket_buy - bucket_sell)

        # Assign VPIN to each bar based on its bucket
        for i in range(n):
            b = bucket_ids[i]
            if b >= self.window - 1:
                start_b = b - self.window + 1
                total_vol = bucket_total[start_b : b + 1].sum()
                if total_vol > 0:
                    vpin_values[i] = bucket_imbalance[start_b : b + 1].sum() / total_vol

        return pd.Series(vpin_values, index=df.index, name=self.name)


class SpreadBps(Feature):
    """Bid-ask spread in basis points."""

    name = "spread_bps"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "bid_price" not in df.columns or "ask_price" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        bid = df["bid_price"]
        ask = df["ask_price"]
        mid = (bid + ask) / 2

        spread = np.where(mid > 0, (ask - bid) / mid * 10_000, np.nan)
        return pd.Series(spread, index=df.index, name=self.name, dtype=float)


class DepthWeightedSlippage(Feature):
    """Simulated slippage to fill a given notional by walking the order book."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, notional: float = 100_000.0) -> None:
        self.notional = notional
        self.name = f"slippage_{int(notional / 1000)}k"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        # Requires multi-level book data: ask_price_1..N, ask_size_1..N
        ask_price_cols = sorted(
            [c for c in df.columns if c.startswith("ask_price")], key=_level_key
        )
        ask_size_cols = sorted([c for c in df.columns if c.startswith("ask_size")], key=_level_key)

        if not ask_price_cols or not ask_size_cols:
            if "ask_price" in df.columns and "ask_size" in df.columns:
                ask_price_cols = ["ask_price"]
                ask_size_cols = ["ask_size"]
            else:
                return pd.Series(np.nan, index=df.index, name=self.name)

        mid_col = None
        if "bid_price" in df.columns and "ask_price" in df.columns:
            mid_col = (df["bid_price"] + df["ask_price"]) / 2
        elif "close" in df.columns:
            mid_col = df["close"]

        if mid_col is None:
            return pd.Series(np.nan, index=df.index, name=self.name)

        n = len(df)
        slippage = np.full(n, np.nan)

        for i in range(n):
            remaining = self.notional
            cost = 0.0
            for pc, sc in zip(ask_price_cols, ask_size_cols, strict=False):
                price = df[pc].iloc[i]
                size = df[sc].iloc[i]
                if pd.isna(price) or pd.isna(size) or price <= 0:
                    break
                fill_notional = min(remaining, price * size)
                cost += fill_notional
                remaining -= fill_notional
                if remaining <= 0:
                    break

            filled = self.notional - remaining
            if filled > 0:
                avg_price = cost / (filled / mid_col.iloc[i]) if mid_col.iloc[i] > 0 else np.nan
                if not np.isnan(avg_price):
                    slippage[i] = (avg_price / mid_col.iloc[i] - 1) * 10_000

        return pd.Series(slippage, index=df.index, name=self.name)


def _level_key(col: str) -> int:
    """Extract level number from column name like 'ask_price_5'."""
    parts = col.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return 0

"""Funding rate features for perpetual futures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from tessera.features.base import Feature

logger = structlog.get_logger(__name__)


class FundingRate(Feature):
    """Current funding rate, forward-filled between funding events."""

    name = "funding_rate"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "funding_rate" in df.columns:
            return df["funding_rate"].ffill().rename(self.name)

        return pd.Series(np.nan, index=df.index, name=self.name)


class FundingZScore(Feature):
    """Z-score of funding rate over a rolling window."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = ["funding_rate"]

    def __init__(self, window: int = 30 * 24) -> None:
        self.window = window
        self.name = f"funding_zscore_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "funding_rate" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        fr = df["funding_rate"].ffill()
        mean = fr.rolling(window=self.window, min_periods=self.window).mean()
        std = fr.rolling(window=self.window, min_periods=self.window).std()

        zscore = np.where(std > 0, (fr - mean) / std, 0.0)
        return pd.Series(zscore, index=df.index, name=self.name, dtype=float)


class SpotPerpBasis(Feature):
    """Basis between spot and perp price: (perp - spot) / spot * 10_000 bps."""

    name = "spot_perp_basis"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "spot_price" not in df.columns:
            logger.info("spot_perp_basis_deferred", reason="spot_price data not available")
            return pd.Series(np.nan, index=df.index, name=self.name)

        perp = df["close"] if "close" in df.columns else None
        spot = df["spot_price"]

        if perp is None:
            return pd.Series(np.nan, index=df.index, name=self.name)

        basis = np.where(spot > 0, (perp - spot) / spot * 10_000, np.nan)
        return pd.Series(basis, index=df.index, name=self.name, dtype=float)

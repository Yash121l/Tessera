"""Cross-sectional features: universe rank, beta, idiosyncratic residual."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.features.base import Feature


class UniverseRank(Feature):
    """Cross-sectional rank of a metric across the active universe at each bar.

    Expects the DataFrame to contain a '{metric}' column with values for
    multiple symbols stacked (with a 'symbol' column) OR to be pre-filtered
    for a single symbol (in which case returns constant 0.5).
    """

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, metric: str = "log_return_1") -> None:
        self.metric = metric
        self.name = f"rank_{metric}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if self.metric not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        if "symbol" in df.columns:
            # Cross-sectional rank within each timestamp
            if "event_time" in df.columns:
                ranks = df.groupby("event_time")[self.metric].rank(pct=True)
            else:
                ranks = df.groupby(df.index)[self.metric].rank(pct=True)
            return ranks.rename(self.name)

        # Single symbol: expanding percentile rank (point-in-time safe)
        values = df[self.metric].values
        n = len(values)
        ranks = np.full(n, np.nan)
        for i in range(n):
            if pd.isna(values[i]):
                continue
            window = values[: i + 1]
            valid = window[~pd.isna(window)]
            if len(valid) > 0:
                ranks[i] = (valid < values[i]).sum() / len(valid)
        return pd.Series(ranks, index=df.index, name=self.name)


class BetaToBTC(Feature):
    """Rolling beta to BTCUSDT returns.

    Expects 'btc_return' column to be present (pre-joined).
    """

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, window: int = 1440) -> None:
        self.window = window
        self.name = f"beta_btc_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "close" not in df.columns or "btc_return" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        asset_ret = np.log(df["close"] / df["close"].shift(1))
        btc_ret = df["btc_return"]

        cov = asset_ret.rolling(self.window, min_periods=self.window).cov(btc_ret)
        var_btc = btc_ret.rolling(self.window, min_periods=self.window).var()

        beta = np.where(var_btc > 0, cov / var_btc, np.nan)
        return pd.Series(beta, index=df.index, name=self.name, dtype=float)


class IdiosyncraticResidual(Feature):
    """Residual return after regressing on BTC (rolling beta)."""

    point_in_time_safe = True
    version = "0.1.0"

    def __init__(self, window: int = 1440) -> None:
        self.window = window
        self.name = f"idio_residual_{window}"
        self.dependencies = [f"beta_btc_{window}"]

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        beta_col = f"beta_btc_{self.window}"
        if "close" not in df.columns or "btc_return" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        asset_ret = np.log(df["close"] / df["close"].shift(1))
        btc_ret = df["btc_return"]

        if beta_col in df.columns:
            beta = df[beta_col]
        else:
            cov = asset_ret.rolling(self.window, min_periods=self.window).cov(btc_ret)
            var_btc = btc_ret.rolling(self.window, min_periods=self.window).var()
            beta = np.where(var_btc > 0, cov / var_btc, 0.0)

        residual = asset_ret - beta * btc_ret
        return residual.rename(self.name)

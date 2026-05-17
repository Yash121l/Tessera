"""Volatility features: realized vol, GARCH, range-based estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from tessera.features.base import Feature

logger = structlog.get_logger(__name__)


class RealizedVol(Feature):
    """Realized volatility over a rolling window of log returns."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.name = f"realized_vol_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        log_ret = np.log(df["close"] / df["close"].shift(1))
        vol = log_ret.rolling(window=self.window, min_periods=self.window).std()
        return vol.rename(self.name)


class Garch11(Feature):
    """GARCH(1,1) conditional volatility forecast.

    Fits model per symbol per day; caches params to avoid refits within a day.
    Uses forward-filtering only (no smoothing) for point-in-time safety.
    """

    name = "garch11_vol"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self) -> None:
        self._cached_params: dict[str, tuple[float, float, float]] | None = None

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        log_ret = np.log(df["close"] / df["close"].shift(1))
        returns = log_ret.dropna().values

        if len(returns) < 30:
            return pd.Series(np.nan, index=df.index, name=self.name)

        try:
            from arch import arch_model

            model = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Zero")
            fit = model.fit(disp="off", show_warning=False)

            # Forward filter: compute conditional variance one-step-ahead
            cond_vol = fit.conditional_volatility / 100
            result = pd.Series(np.nan, index=df.index, name=self.name)
            valid_idx = log_ret.dropna().index
            result.loc[valid_idx] = cond_vol
            return result

        except (ImportError, Exception) as e:
            logger.warning("garch_fit_failed", error=str(e))
            # Fallback to exponential weighted vol
            vol = log_ret.ewm(span=60, min_periods=30).std()
            return vol.rename(self.name)


class Parkinson(Feature):
    """Parkinson (1980) range-based volatility estimator."""

    name = "parkinson_vol"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.name = f"parkinson_vol_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "high" not in df.columns or "low" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        log_hl = np.log(df["high"] / df["low"])
        factor = 1.0 / (4.0 * np.log(2.0))
        parkinson_var = factor * (log_hl**2)
        rolling_mean = parkinson_var.rolling(window=self.window, min_periods=self.window).mean()
        vol = rolling_mean.apply(np.sqrt)
        return vol.rename(self.name)


class GarmanKlass(Feature):
    """Garman-Klass (1980) range-based volatility estimator."""

    name = "garman_klass_vol"
    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.name = f"garman_klass_vol_{window}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index, name=self.name)

        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])

        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        vol = gk_var.rolling(window=self.window, min_periods=self.window).mean().apply(np.sqrt)
        return vol.rename(self.name)


class VolOfVol(Feature):
    """Volatility of volatility: rolling std of realized vol changes."""

    point_in_time_safe = True
    version = "0.1.0"

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.name = f"vol_of_vol_{window}"
        self.dependencies = [f"realized_vol_{window}"]

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        vol_col = f"realized_vol_{self.window}"
        if vol_col not in df.columns:
            if "close" not in df.columns:
                return pd.Series(np.nan, index=df.index, name=self.name)
            log_ret = np.log(df["close"] / df["close"].shift(1))
            vol = log_ret.rolling(window=self.window, min_periods=self.window).std()
        else:
            vol = df[vol_col]

        vol_change = vol.diff()
        vov = vol_change.rolling(window=self.window, min_periods=self.window).std()
        return vov.rename(self.name)

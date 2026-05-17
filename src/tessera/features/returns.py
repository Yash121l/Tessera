"""Return features at multiple horizons.

All returns use event_time, never ingest_time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.features.base import Feature


class LogReturn(Feature):
    """Log return over a fixed horizon (in bars)."""

    point_in_time_safe = True
    version = "0.1.0"
    dependencies: list[str] = []

    def __init__(self, horizon: int = 1) -> None:
        self.horizon = horizon
        self.name = f"log_return_{horizon}"

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index, name=self.name)

        log_ret = np.log(df["close"] / df["close"].shift(self.horizon))
        return log_ret.rename(self.name)

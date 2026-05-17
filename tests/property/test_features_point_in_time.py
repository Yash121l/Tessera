"""Point-in-time safety property test for ALL features.

THE MOST IMPORTANT TEST IN THE PROJECT.

For any synthetic price series, all features must be point-in-time safe:
    feature.compute(df[:t]).iloc[-1] == feature.compute(df).iloc[t-1]

This guarantees no feature leaks future information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from tessera.features.base import Feature

from tessera.features.cross_sectional import BetaToBTC, IdiosyncraticResidual, UniverseRank
from tessera.features.funding import FundingRate, FundingZScore, SpotPerpBasis
from tessera.features.microstructure import (
    VPIN,
    MicroPrice,
    OrderFlowImbalance,
    SpreadBps,
)
from tessera.features.returns import LogReturn
from tessera.features.volatility import (
    GarmanKlass,
    Parkinson,
    RealizedVol,
    VolOfVol,
)


def _make_synthetic_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV + L2 DataFrame."""
    rng = np.random.default_rng(seed)

    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)

    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(100, 10000, n)

    bid_price = close - rng.uniform(0.01, 0.1, n)
    ask_price = close + rng.uniform(0.01, 0.1, n)
    bid_size = rng.uniform(1, 100, n)
    ask_size = rng.uniform(1, 100, n)

    funding = rng.normal(0.0001, 0.0005, n)
    btc_ret = rng.normal(0, 0.01, n)
    spot_price = close - rng.uniform(0, 0.5, n)

    event_time = pd.date_range("2023-01-01", periods=n, freq="1min")

    return pd.DataFrame(
        {
            "event_time": event_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "funding_rate": funding,
            "btc_return": btc_ret,
            "spot_price": spot_price,
        }
    )


ALL_FEATURES: list[Feature] = [
    LogReturn(horizon=1),
    LogReturn(horizon=5),
    LogReturn(horizon=15),
    RealizedVol(window=20),
    Parkinson(window=20),
    GarmanKlass(window=20),
    VolOfVol(window=20),
    OrderFlowImbalance(depth=1),
    MicroPrice(),
    SpreadBps(),
    VPIN(bucket_size=500, window=5),
    FundingRate(),
    FundingZScore(window=30),
    SpotPerpBasis(),
    BetaToBTC(window=30),
    IdiosyncraticResidual(window=30),
    UniverseRank(metric="close"),
]


@pytest.mark.parametrize("feature", ALL_FEATURES, ids=lambda f: f.name)
@given(t_offset=st.integers(min_value=0, max_value=49))
@settings(max_examples=20, deadline=None)
def test_point_in_time_safety(feature: Feature, t_offset: int) -> None:
    """Assert feature.compute(df[:t]).iloc[-1] == feature.compute(df).iloc[t-1]."""
    n = 200
    df = _make_synthetic_df(n, seed=42)

    # Pick t in the second half to ensure enough lookback
    t = 100 + t_offset
    assert t <= n

    # Compute on full data
    full_result = feature.compute(df)

    # Compute on truncated data (only up to t)
    truncated_result = feature.compute(df.iloc[:t].copy())

    val_full = full_result.iloc[t - 1]
    val_trunc = truncated_result.iloc[-1]

    # Both NaN is fine (feature hasn't warmed up)
    if pd.isna(val_full) and pd.isna(val_trunc):
        return

    # If one is NaN and the other isn't, that's a failure
    if pd.isna(val_full) or pd.isna(val_trunc):
        pytest.fail(f"Feature '{feature.name}' at t={t}: full={val_full}, truncated={val_trunc}")

    # Values must match
    np.testing.assert_allclose(
        val_full,
        val_trunc,
        rtol=1e-10,
        atol=1e-14,
        err_msg=f"Feature '{feature.name}' is NOT point-in-time safe at t={t}",
    )

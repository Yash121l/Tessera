"""Property-based tests for OHLCV validation using Hypothesis."""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.data.validate import validate_ohlcv


@st.composite
def ohlcv_rows(draw: st.DrawFn) -> pd.DataFrame:
    """Generate random OHLCV DataFrames with varying validity."""
    n = draw(st.integers(min_value=1, max_value=50))
    rows = []
    base_time = pd.Timestamp("2024-01-01", tz="UTC")

    price_st = st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
    hl_st = st.floats(min_value=0.1, max_value=200000.0, allow_nan=False, allow_infinity=False)
    vol_st = st.floats(min_value=-100.0, max_value=1e9, allow_nan=False, allow_infinity=False)

    for i in range(n):
        o = draw(price_st)
        c = draw(price_st)
        h = draw(hl_st)
        low = draw(hl_st)
        vol = draw(vol_st)

        rows.append(
            {
                "event_time": base_time + pd.Timedelta(minutes=i),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": vol,
                "exchange": "binance",
                "symbol": "BTCUSDT",
            }
        )

    return pd.DataFrame(rows)


@given(df=ohlcv_rows())
@settings(max_examples=200, deadline=None)
def test_validate_never_raises(df: pd.DataFrame) -> None:
    """Validator must never raise on well-formed DataFrames."""
    clean, quarantine = validate_ohlcv(df)
    # Should not raise
    assert clean is not None
    assert quarantine is not None


@given(df=ohlcv_rows())
@settings(max_examples=200, deadline=None)
def test_validate_partitions_all_rows(df: pd.DataFrame) -> None:
    """Clean + quarantine must cover all input rows without duplication."""
    clean, quarantine = validate_ohlcv(df)
    assert len(clean) + len(quarantine) == len(df)


@given(df=ohlcv_rows())
@settings(max_examples=100, deadline=None)
def test_clean_rows_have_valid_ohlc(df: pd.DataFrame) -> None:
    """All rows in the clean set must satisfy OHLC constraints."""
    clean, _ = validate_ohlcv(df)
    if clean.empty:
        return

    # high >= max(open, close)
    max_oc = clean[["open", "close"]].max(axis=1)
    assert (clean["high"] >= max_oc).all()

    # low <= min(open, close)
    min_oc = clean[["open", "close"]].min(axis=1)
    assert (clean["low"] <= min_oc).all()

    # No negative volumes
    assert (clean["volume"] >= 0).all()

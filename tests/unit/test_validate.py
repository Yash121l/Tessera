"""Unit tests for OHLCV validation."""

from __future__ import annotations

import pandas as pd
import pytest

from tessera.data.validate import validate_ohlcv


@pytest.fixture
def good_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [102.0] * 10,
            "low": [98.0] * 10,
            "close": [101.0] * 10,
            "volume": [500.0] * 10,
            "exchange": ["binance"] * 10,
            "symbol": ["BTCUSDT"] * 10,
        }
    )


class TestValidateOHLCV:
    def test_valid_data_passes(self, good_ohlcv: pd.DataFrame) -> None:
        clean, quarantine = validate_ohlcv(good_ohlcv)
        assert len(clean) == 10
        assert len(quarantine) == 0

    def test_negative_volume_quarantined(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[3, "volume"] = -100.0
        clean, quarantine = validate_ohlcv(df)
        assert len(quarantine) >= 1
        assert 3 in quarantine.index

    def test_high_less_than_open_quarantined(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[5, "high"] = 99.0  # less than open=100
        clean, quarantine = validate_ohlcv(df)
        assert 5 in quarantine.index

    def test_high_less_than_close_quarantined(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[2, "high"] = 100.5  # less than close=101
        clean, quarantine = validate_ohlcv(df)
        assert 2 in quarantine.index

    def test_low_greater_than_open_quarantined(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[7, "low"] = 100.5  # greater than open=100
        clean, quarantine = validate_ohlcv(df)
        assert 7 in quarantine.index

    def test_low_greater_than_close_quarantined(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[1, "low"] = 101.5  # greater than close=101
        clean, quarantine = validate_ohlcv(df)
        assert 1 in quarantine.index

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["event_time", "open", "high", "low", "close", "volume"])
        clean, quarantine = validate_ohlcv(df)
        assert clean.empty
        assert quarantine.empty

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"event_time": [1], "open": [100]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_clean_and_quarantine_partition_all_rows(self, good_ohlcv: pd.DataFrame) -> None:
        df = good_ohlcv.copy()
        df.loc[0, "volume"] = -1.0
        df.loc[4, "high"] = 50.0
        clean, quarantine = validate_ohlcv(df)
        assert len(clean) + len(quarantine) == len(df)

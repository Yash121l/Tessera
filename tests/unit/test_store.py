"""Unit tests for the Parquet store module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tessera.data.store import duckdb_connect, read_parquet, write_parquet


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "raw"


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
            "exchange": ["binance"] * 10,
            "symbol": ["BTCUSDT"] * 10,
        }
    )


class TestWriteParquet:
    def test_roundtrip_no_partitions(self, tmp_data_root: Path, sample_ohlcv: pd.DataFrame) -> None:
        write_parquet(sample_ohlcv, "test_table", data_root=tmp_data_root)
        result = read_parquet("test_table", data_root=tmp_data_root)
        assert len(result) == 10
        assert set(result.columns) >= {"event_time", "open", "high", "low", "close", "volume"}

    def test_roundtrip_with_partitions(
        self, tmp_data_root: Path, sample_ohlcv: pd.DataFrame
    ) -> None:
        write_parquet(
            sample_ohlcv, "ohlcv", partition_cols=["exchange", "symbol"], data_root=tmp_data_root
        )
        result = read_parquet("ohlcv", data_root=tmp_data_root)
        assert len(result) == 10

    def test_atomic_write_no_partial_files(
        self, tmp_data_root: Path, sample_ohlcv: pd.DataFrame
    ) -> None:
        write_parquet(sample_ohlcv, "atomic_test", data_root=tmp_data_root)
        table_dir = tmp_data_root / "atomic_test"
        # No .tmp_ directories should remain
        for item in table_dir.rglob("*"):
            assert ".tmp_" not in str(item)

    def test_read_nonexistent_returns_empty(self, tmp_data_root: Path) -> None:
        result = read_parquet("nonexistent", data_root=tmp_data_root)
        assert result.empty

    def test_overwrite_existing(self, tmp_data_root: Path, sample_ohlcv: pd.DataFrame) -> None:
        write_parquet(sample_ohlcv, "overwrite_test", data_root=tmp_data_root)
        # Write again with different data
        new_df = sample_ohlcv.head(5).copy()
        write_parquet(new_df, "overwrite_test", data_root=tmp_data_root)
        result = read_parquet("overwrite_test", data_root=tmp_data_root)
        assert len(result) == 5


class TestDuckDBConnect:
    def test_connect_empty_root(self, tmp_data_root: Path) -> None:
        tmp_data_root.mkdir(parents=True, exist_ok=True)
        conn = duckdb_connect(data_root=tmp_data_root)
        assert conn is not None

    def test_connect_with_ohlcv(self, tmp_data_root: Path, sample_ohlcv: pd.DataFrame) -> None:
        write_parquet(
            sample_ohlcv, "ohlcv", partition_cols=["exchange", "symbol"], data_root=tmp_data_root
        )
        conn = duckdb_connect(data_root=tmp_data_root)
        result = conn.execute("SELECT count(*) FROM ohlcv_1m").fetchone()
        assert result is not None
        assert result[0] == 10

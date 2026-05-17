"""Integration tests for OHLCV ingestion with mocked CCXT."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tessera.data.ingest_ohlcv import backfill_ohlcv


def _generate_mock_ohlcv(start_ms: int, count: int, interval_ms: int = 60_000) -> list[list]:
    """Generate mock CCXT OHLCV response data."""
    rows = []
    for i in range(count):
        ts = start_ms + (i * interval_ms)
        rows.append([ts, 50000.0 + i, 50100.0 + i, 49900.0 + i, 50050.0 + i, 100.0 + i])
    return rows


@pytest.fixture
def mock_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("TESSERA_DATA_ROOT", str(data_dir))
    return data_dir / "raw"


class TestBackfillOHLCV:
    def test_backfill_writes_correct_partition_structure(self, mock_data_root: Path) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)  # 1 hour = 60 bars
        start_ms = int(start.timestamp() * 1000)

        # Mock returns 60 bars in one batch (< 1000, so single batch)
        mock_data = _generate_mock_ohlcv(start_ms, 60)

        mock_exchange = AsyncMock()
        mock_exchange.id = "binance"
        mock_exchange.close = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_data)

        with patch("tessera.data.ccxt_client._create_exchange", return_value=mock_exchange):
            rows = backfill_ohlcv("binance", "BTCUSDT", "1m", start, end)

        assert rows == 60

        # Verify partition structure exists
        ohlcv_dir = mock_data_root / "ohlcv"
        assert ohlcv_dir.exists()
        parquet_files = list(ohlcv_dir.rglob("*.parquet"))
        assert len(parquet_files) > 0

    def test_backfill_5000_bars_pages_correctly(self, mock_data_root: Path) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 4, 11, 20, tzinfo=UTC)  # ~5000 mins
        start_ms = int(start.timestamp() * 1000)

        # Build 5 batches of 1000 bars
        batches = []
        for i in range(5):
            batch_start = start_ms + (i * 1000 * 60_000)
            batches.append(_generate_mock_ohlcv(batch_start, 1000))

        mock_exchange = AsyncMock()
        mock_exchange.id = "binance"
        mock_exchange.close = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=batches + [[]])

        with patch("tessera.data.ccxt_client._create_exchange", return_value=mock_exchange):
            rows = backfill_ohlcv("binance", "BTCUSDT", "1m", start, end)

        assert rows == 5000

    def test_backfill_idempotent(self, mock_data_root: Path) -> None:
        """Re-running backfill produces same row count (deduplication)."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 30, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_data = _generate_mock_ohlcv(start_ms, 30)

        mock_exchange = AsyncMock()
        mock_exchange.id = "binance"
        mock_exchange.close = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=mock_data)

        with patch("tessera.data.ccxt_client._create_exchange", return_value=mock_exchange):
            rows_first = backfill_ohlcv("binance", "BTCUSDT", "1m", start, end)
            rows_second = backfill_ohlcv("binance", "BTCUSDT", "1m", start, end)

        assert rows_first == rows_second == 30

    def test_backfill_deduplicates_overlapping_data(self, mock_data_root: Path) -> None:
        """Overlapping backfills don't create duplicate rows."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 1, 0, 20, tzinfo=UTC)
        start_ms = int(start.timestamp() * 1000)

        mock_data_20 = _generate_mock_ohlcv(start_ms, 20)
        mock_data_10 = _generate_mock_ohlcv(start_ms, 10)  # Overlapping first 10

        mock_exchange = AsyncMock()
        mock_exchange.id = "binance"
        mock_exchange.close = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=[mock_data_20, mock_data_10])

        with patch("tessera.data.ccxt_client._create_exchange", return_value=mock_exchange):
            backfill_ohlcv("binance", "BTCUSDT", "1m", start, end)

            # Second backfill with overlapping data
            end2 = datetime(2024, 1, 1, 0, 10, tzinfo=UTC)
            rows = backfill_ohlcv("binance", "BTCUSDT", "1m", start, end2)

        # Should still have exactly 20 unique rows
        assert rows == 20

"""Unit tests for the Universe class."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from tessera.data.universe import Universe


@pytest.fixture
def universe_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data" / "raw"
    root.mkdir(parents=True)
    monkeypatch.setenv("TESSERA_DATA_ROOT", str(tmp_path / "data"))
    return root


@pytest.fixture
def sample_universe_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "exchange": ["binance", "binance", "binance"],
            "base": ["BTC", "ETH", "SOL"],
            "quote": ["USDT", "USDT", "USDT"],
            "contract_type": ["perp", "perp", "perp"],
            "listed_at": [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-06-01", tz="UTC"),
                pd.Timestamp("2021-03-01", tz="UTC"),
            ],
            "delisted_at": [None, None, pd.Timestamp("2024-06-01", tz="UTC")],
            "tick_size": [0.01, 0.01, 0.01],
            "min_qty": [0.001, 0.001, 0.01],
            "is_active": [True, True, False],
        }
    )


class TestActiveAt:
    def test_all_active_before_delist(
        self, universe_root: Path, sample_universe_df: pd.DataFrame
    ) -> None:
        u = Universe(data_root=universe_root)
        u.save(sample_universe_df)

        ts = datetime(2023, 1, 1, tzinfo=UTC)
        active = u.active_at(ts)
        assert set(active) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

    def test_delisted_excluded_after_delist_date(
        self, universe_root: Path, sample_universe_df: pd.DataFrame
    ) -> None:
        u = Universe(data_root=universe_root)
        u.save(sample_universe_df)

        ts = datetime(2024, 7, 1, tzinfo=UTC)
        active = u.active_at(ts)
        assert "SOLUSDT" not in active
        assert "BTCUSDT" in active

    def test_before_listing_date_excluded(
        self, universe_root: Path, sample_universe_df: pd.DataFrame
    ) -> None:
        u = Universe(data_root=universe_root)
        u.save(sample_universe_df)

        ts = datetime(2020, 3, 1, tzinfo=UTC)
        active = u.active_at(ts)
        assert "BTCUSDT" in active
        assert "ETHUSDT" not in active
        assert "SOLUSDT" not in active

    def test_empty_universe(self, universe_root: Path) -> None:
        u = Universe(data_root=universe_root)
        active = u.active_at(datetime(2024, 1, 1, tzinfo=UTC))
        assert active == []

    def test_exact_listing_boundary(
        self, universe_root: Path, sample_universe_df: pd.DataFrame
    ) -> None:
        u = Universe(data_root=universe_root)
        u.save(sample_universe_df)

        ts = datetime(2020, 6, 1, tzinfo=UTC)
        active = u.active_at(ts)
        assert "ETHUSDT" in active


class TestLoadSave:
    def test_save_and_load(self, universe_root: Path, sample_universe_df: pd.DataFrame) -> None:
        u = Universe(data_root=universe_root)
        u.save(sample_universe_df)

        loaded = u.load()
        assert len(loaded) == 3
        assert set(loaded["symbol"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

    def test_load_nonexistent_returns_empty(self, universe_root: Path) -> None:
        u = Universe(data_root=universe_root)
        loaded = u.load()
        assert loaded.empty

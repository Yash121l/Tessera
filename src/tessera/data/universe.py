"""Tradeable symbol universe management.

Maintains a Parquet-backed registry of all symbols across exchanges,
tracking listing/delisting dates for survivorship-bias-free backtesting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog
from pydantic import BaseModel, field_validator

from tessera.config import TesseraSettings

logger = structlog.get_logger(__name__)

SEED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "FILUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
]


class UniverseRecord(BaseModel):
    """Schema for a single universe entry."""

    symbol: str
    exchange: str
    base: str
    quote: str
    contract_type: str = "perp"
    listed_at: datetime
    delisted_at: datetime | None = None
    tick_size: float
    min_qty: float
    is_active: bool = True

    @field_validator("contract_type")
    @classmethod
    def _validate_contract_type(cls, v: str) -> str:
        if v != "perp":
            msg = f"Only 'perp' contract type supported, got '{v}'"
            raise ValueError(msg)
        return v


class Universe:
    """Manages the tradeable symbol universe backed by Parquet."""

    def __init__(self, data_root: Path | None = None) -> None:
        settings = TesseraSettings()
        root = data_root or settings.data_root / "raw"
        root.mkdir(parents=True, exist_ok=True)
        self._path = root / "universe.parquet"
        self._df: pd.DataFrame | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> pd.DataFrame:
        """Load the universe from Parquet."""
        if self._path.exists():
            self._df = pd.read_parquet(self._path)
        else:
            self._df = pd.DataFrame(
                columns=[
                    "symbol",
                    "exchange",
                    "base",
                    "quote",
                    "contract_type",
                    "listed_at",
                    "delisted_at",
                    "tick_size",
                    "min_qty",
                    "is_active",
                ]
            )
        return self._df

    def save(self, df: pd.DataFrame) -> None:
        """Save the universe DataFrame to Parquet."""
        df.to_parquet(self._path, index=False)
        self._df = df
        logger.info("universe_saved", rows=len(df), path=str(self._path))

    def active_at(self, timestamp: datetime) -> list[str]:
        """Return symbols that were active at a given moment.

        A symbol is active if listed_at <= timestamp and
        (delisted_at is None or delisted_at > timestamp).
        """
        df = self.load()
        if df.empty:
            return []

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        listed = df["listed_at"] <= ts
        not_delisted = df["delisted_at"].isna() | (df["delisted_at"] > ts)
        active = df[listed & not_delisted]
        return active["symbol"].tolist()

    def refresh(self) -> pd.DataFrame:
        """Pull current symbols from exchanges and merge with existing universe."""
        from tessera.data.ccxt_client import fetch_markets

        existing = self.load()
        exchanges = ["binance", "bybit"]
        all_new: list[pd.DataFrame] = []

        for exchange_id in exchanges:
            try:
                markets_df = asyncio.run(fetch_markets(exchange_id))
                if not markets_df.empty:
                    markets_df["exchange"] = exchange_id
                    all_new.append(markets_df)
                    logger.info(
                        "universe_fetched",
                        exchange=exchange_id,
                        count=len(markets_df),
                    )
            except Exception as exc:
                logger.error("universe_fetch_failed", exchange=exchange_id, error=str(exc))

        if not all_new:
            logger.warning("universe_no_new_data")
            return existing

        new_df = pd.concat(all_new, ignore_index=True)
        merged = self._merge(existing, new_df)
        self.save(merged)
        return merged

    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """Merge new market data with existing universe, preserving history."""
        if existing.empty:
            return new

        merged = existing.copy()
        existing_keys = set(zip(merged["symbol"], merged["exchange"], strict=False))

        now = pd.Timestamp.now(tz="UTC")

        for _, row in new.iterrows():
            key = (row["symbol"], row["exchange"])
            if key not in existing_keys:
                merged = pd.concat([merged, pd.DataFrame([row])], ignore_index=True)
                existing_keys.add(key)
            else:
                mask = (merged["symbol"] == row["symbol"]) & (merged["exchange"] == row["exchange"])
                if not row.get("is_active", True):
                    merged.loc[mask, "delisted_at"] = now
                    merged.loc[mask, "is_active"] = False
                else:
                    merged.loc[mask, "is_active"] = True
                    merged.loc[mask, "tick_size"] = row["tick_size"]
                    merged.loc[mask, "min_qty"] = row["min_qty"]

        # Mark symbols no longer in new data as delisted
        new_keys = set(zip(new["symbol"], new["exchange"], strict=False))
        for idx, erow in merged.iterrows():
            key = (erow["symbol"], erow["exchange"])
            if key not in new_keys and erow["is_active"]:
                merged.at[idx, "delisted_at"] = now
                merged.at[idx, "is_active"] = False

        return merged

    def seed(self) -> pd.DataFrame:
        """Bootstrap the universe with hardcoded seed symbols from Binance."""
        from tessera.data.ccxt_client import fetch_markets

        markets_df = asyncio.run(fetch_markets("binance"))
        if markets_df.empty:
            logger.error("universe_seed_no_markets")
            return pd.DataFrame()

        seed_set = set(SEED_SYMBOLS)
        seeded = markets_df[markets_df["symbol"].isin(seed_set)].copy()
        seeded["exchange"] = "binance"

        # Fill missing listed_at with a reasonable default
        default_listed = pd.Timestamp("2020-01-01", tz="UTC")
        seeded["listed_at"] = seeded["listed_at"].fillna(default_listed)

        self.save(seeded)
        logger.info("universe_seeded", count=len(seeded))
        return seeded

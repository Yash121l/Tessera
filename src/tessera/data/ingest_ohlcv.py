"""OHLCV bar ingestion: backfill and incremental modes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import structlog

from tessera.data.ccxt_client import fetch_ohlcv
from tessera.data.store import read_parquet, write_parquet
from tessera.data.validate import quarantine_rows, validate_ohlcv

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 1000


def _to_ccxt_symbol(symbol: str) -> str:
    """Convert BTCUSDT -> BTC/USDT:USDT for CCXT perpetual futures."""
    for quote in ("USDT", "BUSD"):
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}:{quote}"
    return symbol


def _timeframe_ms(timeframe: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping.get(timeframe, 60_000)


async def _fetch_batch(
    exchange_id: str, symbol: str, timeframe: str, since_ms: int
) -> pd.DataFrame:
    ccxt_symbol = _to_ccxt_symbol(symbol)
    return await fetch_ohlcv(exchange_id, ccxt_symbol, timeframe, since=since_ms, limit=_BATCH_SIZE)


def backfill_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Backfill OHLCV history by paging through batches.

    Deduplicates on (exchange, symbol, timeframe, event_time).
    Returns the total number of rows written.
    """
    start = start or datetime(2021, 1, 1, tzinfo=UTC)
    end = end or datetime.now(UTC)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    tf_ms = _timeframe_ms(timeframe)

    all_bars: list[pd.DataFrame] = []
    cursor_ms = start_ms

    logger.info(
        "backfill_start",
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    while cursor_ms < end_ms:
        batch = asyncio.run(_fetch_batch(exchange, symbol, timeframe, cursor_ms))

        if batch.empty:
            break

        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        batch = batch[batch["event_time"] < end_ts]
        if batch.empty:
            break

        all_bars.append(batch)
        last_ts = batch["event_time"].max()
        cursor_ms = int(last_ts.timestamp() * 1000) + tf_ms

        if len(batch) < _BATCH_SIZE:
            break

    if not all_bars:
        logger.warning("backfill_no_data", exchange=exchange, symbol=symbol)
        return 0

    df = pd.concat(all_bars, ignore_index=True)
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["ingest_time"] = pd.Timestamp.now(tz="UTC")

    # Deduplicate
    df = df.drop_duplicates(subset=["exchange", "symbol", "timeframe", "event_time"])

    # Merge with existing data for idempotency
    existing = read_parquet("ohlcv")
    if not existing.empty:
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["exchange", "symbol", "timeframe", "event_time"], keep="last"
        )
        df = combined

    # Validate
    clean_df, quarantine_df = validate_ohlcv(df, timeframe)
    quarantine_rows(quarantine_df)

    # Write
    write_parquet(clean_df, "ohlcv", partition_cols=["exchange", "symbol"])

    logger.info(
        "backfill_complete",
        exchange=exchange,
        symbol=symbol,
        rows_written=len(clean_df),
        quarantined=len(quarantine_df),
    )
    return len(clean_df)


def incremental_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
) -> int:
    """Fetch OHLCV bars from max(event_time) forward to now.

    Returns the number of new rows written.
    """
    existing = read_parquet("ohlcv")

    if not existing.empty:
        mask = (existing["exchange"] == exchange) & (existing["symbol"] == symbol)
        symbol_data = existing[mask]
        if not symbol_data.empty:
            last_time = symbol_data["event_time"].max()
            start = last_time.to_pydatetime()
        else:
            start = datetime(2021, 1, 1, tzinfo=UTC)
    else:
        start = datetime(2021, 1, 1, tzinfo=UTC)

    end = datetime.now(UTC)

    logger.info(
        "incremental_start",
        exchange=exchange,
        symbol=symbol,
        from_time=start.isoformat(),
    )

    return backfill_ohlcv(exchange, symbol, timeframe, start, end)

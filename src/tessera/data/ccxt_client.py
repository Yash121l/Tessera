"""Async CCXT wrapper with retry, rate-limit respect, and observability."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd
import structlog
from prometheus_client import Histogram

CCXT_LATENCY = Histogram(
    "tessera_ccxt_latency_seconds",
    "Latency of CCXT API calls.",
    ["exchange", "method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 0.5


def _get_exchange_class(exchange_id: str) -> Any:
    import ccxt.async_support as ccxt_async

    cls = getattr(ccxt_async, exchange_id, None)
    if cls is None:
        msg = f"Unknown exchange: {exchange_id}"
        raise ValueError(msg)
    return cls


async def _create_exchange(exchange_id: str, **kwargs: Any) -> Any:
    cls = _get_exchange_class(exchange_id)
    return cls({"enableRateLimit": True, **kwargs})


async def _retry_call(exchange: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    func = getattr(exchange, method)
    last_exc: BaseException | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            CCXT_LATENCY.labels(exchange=exchange.id, method=method).observe(elapsed)
            logger.debug(
                "ccxt_call_success",
                exchange=exchange.id,
                method=method,
                attempt=attempt + 1,
                elapsed_s=round(elapsed, 3),
            )
            return result
        except Exception as exc:
            last_exc = exc
            delay = _BASE_DELAY * (2**attempt)
            logger.warning(
                "ccxt_call_retry",
                exchange=exchange.id,
                method=method,
                attempt=attempt + 1,
                error=str(exc),
                next_delay_s=delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"CCXT call {method} failed after {_MAX_RETRIES} retries") from last_exc


async def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1m",
    since: int | None = None,
    limit: int = 1000,
    **exchange_kwargs: Any,
) -> pd.DataFrame:
    """Fetch OHLCV bars from an exchange.

    Returns a DataFrame with columns:
        event_time, open, high, low, close, volume
    """
    exchange = await _create_exchange(exchange_id, **exchange_kwargs)
    try:
        await _retry_call(exchange, "load_markets")
        data = await _retry_call(exchange, "fetch_ohlcv", symbol, timeframe, since, limit)
    finally:
        await exchange.close()

    if not data:
        return pd.DataFrame(columns=["event_time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data, columns=["event_time", "open", "high", "low", "close", "volume"])
    df["event_time"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)
    return df


async def fetch_funding_rate_history(
    exchange_id: str,
    symbol: str,
    since: int | None = None,
    limit: int = 1000,
    **exchange_kwargs: Any,
) -> pd.DataFrame:
    """Fetch historical funding rates."""
    exchange = await _create_exchange(exchange_id, **exchange_kwargs)
    try:
        await _retry_call(exchange, "load_markets")
        data = await _retry_call(exchange, "fetch_funding_rate_history", symbol, since, limit)
    finally:
        await exchange.close()

    if not data:
        return pd.DataFrame(columns=["event_time", "symbol", "funding_rate"])

    rows = [
        {
            "event_time": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
            "symbol": r["symbol"],
            "funding_rate": r["fundingRate"],
        }
        for r in data
    ]
    return pd.DataFrame(rows)


async def fetch_markets(exchange_id: str, **exchange_kwargs: Any) -> pd.DataFrame:
    """Fetch all markets from an exchange. Returns perpetual futures only."""
    exchange = await _create_exchange(exchange_id, **exchange_kwargs)
    try:
        await _retry_call(exchange, "load_markets")
        markets = exchange.markets
    finally:
        await exchange.close()

    rows = []
    for mkt in markets.values():
        if mkt.get("swap") and mkt.get("linear") and mkt.get("quote") == "USDT":
            rows.append(
                {
                    "symbol": mkt["id"],
                    "base": mkt["base"],
                    "quote": mkt["quote"],
                    "contract_type": "perp",
                    "tick_size": mkt["precision"].get("price", 0.01),
                    "min_qty": mkt["precision"].get("amount", 0.001),
                    "listed_at": (
                        pd.Timestamp(mkt.get("created"), unit="ms", tz="UTC")
                        if mkt.get("created")
                        else None
                    ),
                    "is_active": mkt.get("active", True),
                }
            )

    return pd.DataFrame(rows)

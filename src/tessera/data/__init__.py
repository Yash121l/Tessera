"""Data ingestion and storage layer.

Modules:
- ccxt_client: Async exchange API wrapper with retries and rate limiting.
- store: Partitioned Parquet I/O with DuckDB integration.
- universe: Tradeable symbol universe management.
- ingest_ohlcv: OHLCV bar ingestion (backfill and incremental).
- validate: Data quality checks and quarantine.
"""

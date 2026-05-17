# Data Layer

## Storage Layout

All raw data lives under `data/raw/` using Hive-style partitioned Parquet:

```
data/raw/
├── universe.parquet                    # Symbol universe (single file)
├── ohlcv/
│   └── exchange=binance/
│       └── symbol=BTCUSDT/
│           └── <uuid>.parquet          # OHLCV bars
├── funding_rates/
│   └── exchange=binance/
│       └── symbol=BTCUSDT/
│           └── <uuid>.parquet
└── quarantine/
    └── ohlcv/
        └── exchange=binance/
            └── symbol=BTCUSDT/
                └── <uuid>.parquet      # Rows that failed validation
```

## Point-in-Time Contract

Every row contains two timestamps:

- **`event_time`**: The exchange-provided timestamp (bar open time for OHLCV, 
  funding settlement time for rates). All joins downstream MUST use this field.
- **`ingest_time`**: Wall clock when the row was fetched. Used for debugging 
  data freshness, never for joins.

The universe table tracks `listed_at` and `delisted_at` per symbol. The
`Universe.active_at(timestamp)` method returns only symbols that were tradeable
at a given moment, preventing survivorship bias in backtests.

## DuckDB Integration

`store.duckdb_connect()` returns an in-memory DuckDB connection with views 
registered over the Parquet files:

- `ohlcv_1m` — all OHLCV data
- `funding_rates` — funding rate history
- `universe` — the symbol universe

These support SQL queries with predicate pushdown into the Parquet partitions:

```sql
SELECT symbol, count(*) as bars
FROM ohlcv_1m
WHERE exchange = 'binance'
GROUP BY symbol
ORDER BY bars DESC;
```

## Adding a New Data Source

1. Create `src/tessera/data/ingest_<source>.py` with `backfill_*` and 
   `incremental_*` functions following the OHLCV pattern.
2. Add a validation function in `validate.py`.
3. Register the Parquet glob in `store.duckdb_connect()`.
4. Add a CLI subcommand in `cli.py` under the `ingest_app` typer group.
5. Write tests with mocked network calls.

## CLI Commands

```bash
# Refresh the universe from Binance + Bybit
tessera ingest universe

# Backfill OHLCV for a specific symbol
tessera ingest ohlcv --exchange binance --symbol BTCUSDT --timeframe 1m \
    --start 2021-01-01 --end 2025-01-01

# Incremental ingest for all active symbols
tessera ingest ohlcv --incremental
```

## Validation & Quarantine

Before writing, all OHLCV data passes through `validate_ohlcv()` which checks:

- Schema completeness
- Monotonic timestamps per (exchange, symbol)
- No time gaps > 5× the bar interval
- No negative volumes
- `high >= max(open, close)`
- `low <= min(open, close)`

Rows failing any check are written to `data/raw/quarantine/ohlcv/` for manual
review. Clean rows proceed to the main partitions.

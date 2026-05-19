"""Build deterministic synthetic OHLCV fixture for backtest smoke tests.

Generates 5 symbols × 200 bars × 1-minute frequency using geometric Brownian
motion seeded at 42. Writes to tests/fixtures/synthetic_ohlcv.parquet.

Usage (idempotent — safe to re-run):
    uv run python tests/fixtures/build_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
N_BARS = 200
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INITIAL_PRICES = {
    "BTCUSDT": 50_000.0,
    "ETHUSDT": 3_000.0,
    "SOLUSDT": 150.0,
    "BNBUSDT": 400.0,
    "XRPUSDT": 0.60,
}
SIGMA = 0.001  # per-bar log-return std (≈0.1% per minute, ~14% annualised)

OUT = Path(__file__).parent / "synthetic_ohlcv.parquet"


def _gbm_ohlcv(
    symbol: str,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s0 = INITIAL_PRICES[symbol]

    log_ret = rng.normal(0.0, SIGMA, n)
    close = s0 * np.exp(np.cumsum(log_ret))

    noise_h = rng.uniform(0.0, 2 * SIGMA, n)
    noise_l = rng.uniform(0.0, 2 * SIGMA, n)
    high = close * (1.0 + noise_h)
    low = close * (1.0 - noise_l)

    # open = previous close (first open = s0)
    open_ = np.empty(n)
    open_[0] = s0
    open_[1:] = close[:-1]

    # Ensure OHLC invariant
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(50.0, 500.0, n)
    ts = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1min", tz="UTC")

    return pd.DataFrame(
        {
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=ts,
    )


def build(out: Path = OUT) -> None:
    rng = np.random.default_rng(SEED)
    seeds = rng.integers(0, 2**31, size=len(SYMBOLS))

    frames = [_gbm_ohlcv(sym, N_BARS, int(s)) for sym, s in zip(SYMBOLS, seeds, strict=True)]
    df = pd.concat(frames).reset_index().rename(columns={"index": "timestamp"})
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, out, compression="snappy")
    print(f"Written {out}  rows={len(df)}  symbols={SYMBOLS}")


if __name__ == "__main__":
    build()

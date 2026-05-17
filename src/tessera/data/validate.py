"""OHLCV data quality validation and quarantine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from tessera.data.store import write_parquet

logger = structlog.get_logger(__name__)

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str = "1m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate OHLCV data and split into clean and quarantine sets.

    Checks:
    - Required columns present
    - Monotonic event_time per (exchange, symbol)
    - No gaps > 5x bar interval
    - No negative volumes
    - high >= max(open, close)
    - low <= min(open, close)

    Returns:
        (clean_df, quarantine_df) — rows failing any check go to quarantine.
    """
    required_cols = {"event_time", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    if df.empty:
        return df.copy(), pd.DataFrame(columns=df.columns)

    bad_mask = pd.Series(False, index=df.index)
    reasons: list[pd.Series] = []

    # Negative volume
    neg_vol = df["volume"] < 0
    bad_mask |= neg_vol
    if neg_vol.any():
        reasons.append(pd.Series("negative_volume", index=df.index).where(neg_vol, ""))

    # high < max(open, close)
    high_violation = df["high"] < df[["open", "close"]].max(axis=1)
    bad_mask |= high_violation
    if high_violation.any():
        reasons.append(pd.Series("high_lt_max_oc", index=df.index).where(high_violation, ""))

    # low > min(open, close)
    low_violation = df["low"] > df[["open", "close"]].min(axis=1)
    bad_mask |= low_violation
    if low_violation.any():
        reasons.append(pd.Series("low_gt_min_oc", index=df.index).where(low_violation, ""))

    # Monotonicity and gap checks per (exchange, symbol) group
    interval_ms = TIMEFRAME_MS.get(timeframe, 60_000)
    max_gap = pd.Timedelta(milliseconds=interval_ms * 5)

    group_cols = [c for c in ["exchange", "symbol"] if c in df.columns]
    groups = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]

    for _, group in groups:
        sorted_times = group["event_time"].sort_values()
        diffs = sorted_times.diff()

        # Non-monotonic (duplicate or backward timestamps)
        non_mono = diffs < pd.Timedelta(0)
        bad_mask.loc[non_mono[non_mono].index] = True

        # Gaps > 5x interval
        gap_violations = diffs > max_gap
        bad_mask.loc[gap_violations[gap_violations].index] = True

    quarantine_df = df[bad_mask].copy()
    clean_df = df[~bad_mask].copy()

    logger.info(
        "ohlcv_validated",
        total=len(df),
        clean=len(clean_df),
        quarantined=len(quarantine_df),
    )

    return clean_df, quarantine_df


def quarantine_rows(
    df: pd.DataFrame,
    data_root: Path | None = None,
) -> None:
    """Write quarantined rows to the quarantine partition."""
    if df.empty:
        return

    partition_cols = []
    if "exchange" in df.columns:
        partition_cols.append("exchange")
    if "symbol" in df.columns:
        partition_cols.append("symbol")

    write_parquet(
        df,
        "quarantine/ohlcv",
        partition_cols=partition_cols or None,
        data_root=data_root,
    )
    logger.warning("quarantine_written", rows=len(df))

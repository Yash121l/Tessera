"""Partitioned Parquet storage with DuckDB integration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from tessera.config import TesseraSettings

logger = structlog.get_logger(__name__)


def _data_root() -> Path:
    settings = TesseraSettings()
    root = settings.data_root / "raw"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_parquet(
    df: pd.DataFrame,
    table_name: str,
    partition_cols: list[str] | None = None,
    data_root: Path | None = None,
) -> Path:
    """Atomically write a DataFrame to partitioned Parquet.

    Writes to a temp directory, then renames into place.
    """
    root = data_root or _data_root()
    dest = root / table_name
    dest.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    if not partition_cols:
        # Non-partitioned: atomic single-file write (replaces existing data)
        tmp_file = dest / ".tmp_write.parquet"
        final_file = dest / "data.parquet"
        pq.write_table(table, tmp_file)
        os.replace(str(tmp_file), str(final_file))
        # Remove any other parquet files from prior writes
        for f in dest.glob("*.parquet"):
            if f != final_file:
                f.unlink()
    else:
        tmp_dir = tempfile.mkdtemp(dir=dest, prefix=".tmp_")
        try:
            pq.write_to_dataset(
                table,
                root_path=tmp_dir,
                partition_cols=partition_cols,
            )
            for dirpath, _, filenames in os.walk(tmp_dir):
                rel = os.path.relpath(dirpath, tmp_dir)
                target_dir = dest / rel if rel != "." else dest
                target_dir.mkdir(parents=True, exist_ok=True)
                # Remove old parquet files in this partition leaf
                if filenames:
                    for old in target_dir.glob("*.parquet"):
                        old.unlink()
                for fname in filenames:
                    src = Path(dirpath) / fname
                    dst = target_dir / fname
                    os.replace(str(src), str(dst))
        finally:
            _rmtree_empty(Path(tmp_dir))

    logger.info(
        "parquet_written",
        table=table_name,
        rows=len(df),
        partitions=partition_cols,
    )
    return dest


def _rmtree_empty(path: Path) -> None:
    """Remove a directory tree that should be empty after file moves."""
    import contextlib

    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir():
            with contextlib.suppress(OSError):
                child.rmdir()
    with contextlib.suppress(OSError):
        path.rmdir()


def read_parquet(
    table_name: str,
    filters: list[tuple[str, str, str | int | float]] | None = None,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Read a partitioned Parquet dataset with predicate pushdown.

    Args:
        table_name: Name of the table directory under data/raw/.
        filters: PyArrow filter tuples, e.g. [("exchange", "==", "binance")].
        data_root: Override for the data root path.
    """
    root = data_root or _data_root()
    path = root / table_name

    if not path.exists():
        logger.warning("parquet_not_found", table=table_name, path=str(path))
        return pd.DataFrame()

    dataset = pq.ParquetDataset(path, filters=filters)
    table = dataset.read()
    result: pd.DataFrame = table.to_pandas()
    return result


def duckdb_connect(data_root: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with raw Parquet globs registered as views.

    Registers views: ohlcv_1m, funding_rates, universe.
    """
    root = data_root or _data_root()
    conn = duckdb.connect(":memory:")

    # Glob-based views (partitioned datasets)
    glob_views = {
        "ohlcv_1m": root / "ohlcv" / "**" / "*.parquet",
        "funding_rates": root / "funding_rates" / "**" / "*.parquet",
    }

    for view_name, glob_path in glob_views.items():
        glob_str = str(glob_path)
        if _glob_has_files(glob_path):
            sql = (
                f"CREATE VIEW {view_name} AS "
                f"SELECT * FROM parquet_scan('{glob_str}', hive_partitioning=true)"
            )
            conn.execute(sql)
            logger.debug("duckdb_view_registered", view=view_name, glob=glob_str)

    # Single-file views
    universe_path = root / "universe.parquet"
    if universe_path.exists():
        conn.execute(f"CREATE VIEW universe AS SELECT * FROM parquet_scan('{universe_path}')")
        logger.debug("duckdb_view_registered", view="universe", glob=str(universe_path))

    return conn


def _glob_has_files(glob_path: Path) -> bool:
    """Check if a glob pattern matches any files."""
    parent = glob_path.parent
    while "**" in str(parent) or "*" in parent.name:
        parent = parent.parent
    if not parent.exists():
        return False
    return any(parent.rglob("*.parquet"))

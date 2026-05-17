"""Tessera CLI entrypoint.

Every subcommand initializes logging and seeds RNGs on startup.
The paper and backtest subcommands additionally start the Prometheus metrics server.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import structlog
import typer

from tessera.config import Environment, TesseraSettings, generate_run_id, seed_everything
from tessera.log import configure_logging
from tessera.metrics import start_metrics_server

app = typer.Typer(
    name="tessera",
    help="Tessera: Mid-frequency ML trading system for crypto perpetual futures.",
    no_args_is_help=True,
)

ingest_app = typer.Typer(help="Ingest raw market data from exchanges.")
app.add_typer(ingest_app, name="ingest")


def _bootstrap() -> tuple[TesseraSettings, str]:
    """Common startup: load settings, configure logging, seed RNGs."""
    settings = TesseraSettings()
    json_mode = settings.env != Environment.DEV
    configure_logging(level=settings.log_level, json=json_mode)
    seed_everything(settings.random_seed)
    run_id = generate_run_id(settings.random_seed)
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return settings, run_id


@ingest_app.command("universe")
def ingest_universe() -> None:
    """Refresh the tradeable symbol universe from exchanges."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.ingest")
    log.info("universe_refresh_start", run_id=run_id)

    from tessera.data.universe import Universe

    universe = Universe()
    df = universe.refresh()
    typer.echo(f"Universe refreshed: {len(df)} symbols.")
    raise typer.Exit()


@ingest_app.command("ohlcv")
def ingest_ohlcv(
    exchange: str | None = typer.Option(None, help="Exchange ID (e.g. binance)"),
    symbol: str | None = typer.Option(None, help="Symbol (e.g. BTCUSDT)"),
    timeframe: str = typer.Option("1m", help="Bar timeframe"),
    start: str | None = typer.Option(None, help="Start date (YYYY-MM-DD)"),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD)"),
    incremental: bool = typer.Option(False, "--incremental", help="Incremental mode"),
) -> None:
    """Backfill or incrementally ingest OHLCV bars."""
    _bootstrap()

    from tessera.data.ingest_ohlcv import backfill_ohlcv, incremental_ohlcv
    from tessera.data.universe import Universe

    if incremental:
        universe = Universe()
        active_symbols = universe.active_at(datetime.now(UTC))
        if not active_symbols:
            typer.echo("No active symbols in universe. Run 'tessera ingest universe' first.")
            raise typer.Exit(1)

        total = 0
        for sym in active_symbols:
            exc = exchange or "binance"
            rows = incremental_ohlcv(exc, sym, timeframe)
            total += rows
            typer.echo(f"  {exc}/{sym}: {rows} rows")

        typer.echo(f"Incremental ingest complete: {total} total rows.")
    else:
        if not exchange or not symbol:
            typer.echo("--exchange and --symbol required for backfill mode.")
            raise typer.Exit(1)

        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC) if start else None
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC) if end else None

        rows = backfill_ohlcv(exchange, symbol, timeframe, start_dt, end_dt)
        typer.echo(f"Backfill complete: {rows} rows for {exchange}/{symbol}.")

    raise typer.Exit()


features_app = typer.Typer(help="Feature engineering pipeline.")
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    start: str = typer.Option("2021-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)"),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols (default: all active)"),
) -> None:
    """Run the full feature pipeline for all symbols over a date range."""
    import time

    from tessera.features import (
        FeaturePipeline,
        FundingRate,
        FundingZScore,
        GarmanKlass,
        LogReturn,
        MicroPrice,
        OrderFlowImbalance,
        Parkinson,
        RealizedVol,
        SpotPerpBasis,
        SpreadBps,
        VolOfVol,
    )

    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.features")
    log.info("feature_build_start", run_id=run_id, start=start, end=end)

    feature_list = [
        LogReturn(horizon=1),
        LogReturn(horizon=5),
        LogReturn(horizon=15),
        LogReturn(horizon=60),
        LogReturn(horizon=240),
        LogReturn(horizon=1440),
        RealizedVol(window=300),
        RealizedVol(window=60),
        RealizedVol(window=1440),
        Parkinson(window=60),
        GarmanKlass(window=60),
        VolOfVol(window=60),
        OrderFlowImbalance(depth=1),
        MicroPrice(),
        SpreadBps(),
        FundingRate(),
        FundingZScore(window=720),
        SpotPerpBasis(),
    ]

    pipeline = FeaturePipeline(feature_list)

    from tessera.data.store import read_parquet

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        from tessera.data.universe import Universe

        universe = Universe()
        symbol_list = universe.active_at(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC))
        if not symbol_list:
            typer.echo("No active symbols. Run 'tessera ingest universe' first.")
            raise typer.Exit(1)

    t0 = time.time()
    total_rows = 0

    for sym in symbol_list:
        log.info("feature_build_symbol", symbol=sym)
        df = read_parquet(
            "ohlcv",
            filters=[
                ("symbol", "==", sym),
            ],
        )
        if df.empty:
            log.warning("no_data", symbol=sym)
            continue

        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"])
            mask = (df["event_time"] >= start) & (df["event_time"] <= end)
            df = df[mask]

        if df.empty:
            continue

        result = pipeline.compute_multi_day(df, symbol=sym)
        total_rows += len(result)
        typer.echo(f"  {sym}: {len(result)} rows, {len(feature_list)} features")

    elapsed = time.time() - t0
    log.info("feature_build_complete", total_rows=total_rows, elapsed_s=round(elapsed, 1))
    typer.echo(f"Feature build complete: {total_rows} rows in {elapsed:.1f}s")
    raise typer.Exit()


@app.command()
def train() -> None:
    """Train ML models."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.train")
    log.info("subcommand started", subcommand="train", run_id=run_id)
    typer.echo("TODO: train")
    raise typer.Exit()


@app.command()
def backtest() -> None:
    """Run backtest simulation."""
    settings, run_id = _bootstrap()
    start_metrics_server(settings.prometheus_port)
    log = structlog.get_logger("tessera.cli.backtest")
    log.info("subcommand started", subcommand="backtest", run_id=run_id)
    typer.echo("TODO: backtest")
    raise typer.Exit()


@app.command()
def paper() -> None:
    """Run paper trading (live data, simulated execution)."""
    settings, run_id = _bootstrap()
    start_metrics_server(settings.prometheus_port)
    log = structlog.get_logger("tessera.cli.paper")
    log.info("subcommand started", subcommand="paper", run_id=run_id)
    typer.echo("TODO: paper")
    raise typer.Exit()


@app.command()
def report() -> None:
    """Generate performance reports."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.report")
    log.info("subcommand started", subcommand="report", run_id=run_id)
    typer.echo("TODO: report")
    raise typer.Exit()

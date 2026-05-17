"""Tessera CLI entrypoint.

Every subcommand initializes logging and seeds RNGs on startup.
The paper and backtest subcommands additionally start the Prometheus metrics server.
"""

from __future__ import annotations

from datetime import UTC, datetime

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


@app.command()
def features() -> None:
    """Compute features from raw data."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.features")
    log.info("subcommand started", subcommand="features", run_id=run_id)
    typer.echo("TODO: features")
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

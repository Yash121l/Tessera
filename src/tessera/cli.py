"""Tessera CLI entrypoint.

Every subcommand initializes logging and seeds RNGs on startup.
The paper and backtest subcommands additionally start the Prometheus metrics server.
"""

from __future__ import annotations

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


def _bootstrap() -> tuple[TesseraSettings, str]:
    """Common startup: load settings, configure logging, seed RNGs."""
    settings = TesseraSettings()
    json_mode = settings.env != Environment.DEV
    configure_logging(level=settings.log_level, json=json_mode)
    seed_everything(settings.random_seed)
    run_id = generate_run_id(settings.random_seed)
    structlog.contextvars.bind_contextvars(run_id=run_id)
    return settings, run_id


@app.command()
def ingest() -> None:
    """Ingest raw market data from exchanges."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.ingest")
    log.info("subcommand started", subcommand="ingest", run_id=run_id)
    typer.echo("TODO: ingest")
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

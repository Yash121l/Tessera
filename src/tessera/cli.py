"""Tessera CLI entrypoint.

Usage:
    tessera ingest --symbol BTC/USDT:USDT
    tessera features --config configs/features.yaml
    tessera train --config configs/model.yaml
    tessera backtest --config configs/backtest.yaml
    tessera paper --config configs/live.yaml
    tessera report --output reports/
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="tessera",
    help="Tessera: Mid-frequency ML trading system for crypto perpetual futures.",
    no_args_is_help=True,
)


@app.command()
def ingest() -> None:
    """Ingest raw market data from exchanges."""
    typer.echo("TODO: ingest")
    raise typer.Exit()


@app.command()
def features() -> None:
    """Compute features from raw data."""
    typer.echo("TODO: features")
    raise typer.Exit()


@app.command()
def train() -> None:
    """Train ML models."""
    typer.echo("TODO: train")
    raise typer.Exit()


@app.command()
def backtest() -> None:
    """Run backtest simulation."""
    typer.echo("TODO: backtest")
    raise typer.Exit()


@app.command()
def paper() -> None:
    """Run paper trading (live data, simulated execution)."""
    typer.echo("TODO: paper")
    raise typer.Exit()


@app.command()
def report() -> None:
    """Generate performance reports."""
    typer.echo("TODO: report")
    raise typer.Exit()

"""Tessera CLI entrypoint.

Every subcommand initializes logging and seeds RNGs on startup.
The paper and backtest subcommands additionally start the Prometheus metrics server.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
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


train_app = typer.Typer(help="Train ML models.")
app.add_typer(train_app, name="train")


def _assemble_training_data(
    symbol_list: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray, pd.Series]:  # type: ignore[return-value]
    """Load OHLCV → features → triple-barrier labels for a list of symbols.

    Returns (features, y, t1, sample_weight, forward_returns).  Each returned
    Series/array is aligned on the same index.  Returns empty containers if no
    symbols have sufficient data.
    """
    from tessera.data.store import read_parquet
    from tessera.features import (
        FeaturePipeline,
        GarmanKlass,
        LogReturn,
        MicroPrice,
        OrderFlowImbalance,
        Parkinson,
        RealizedVol,
        SpreadBps,
        VolOfVol,
    )
    from tessera.labels.sample_weights import get_sample_weights_by_return
    from tessera.labels.triple_barrier import (
        apply_triple_barrier,
        compute_volatility,
        get_bins,
        make_events,
    )

    feature_list = [
        LogReturn(horizon=1),
        LogReturn(horizon=5),
        LogReturn(horizon=60),
        RealizedVol(window=60),
        Parkinson(window=60),
        GarmanKlass(window=60),
        VolOfVol(window=60),
        OrderFlowImbalance(depth=1),
        MicroPrice(),
        SpreadBps(),
    ]
    pipeline = FeaturePipeline(feature_list)
    feature_cols = [f.name for f in feature_list]

    all_features: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    all_t1: list[pd.Series] = []
    all_sw: list[np.ndarray] = []
    all_ret: list[pd.Series] = []

    for sym in symbol_list:
        df = read_parquet("ohlcv", filters=[("symbol", "==", sym)])
        if df.empty:
            continue
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"])
            df = df[(df["event_time"] >= start) & (df["event_time"] <= end)]
        if len(df) < 500:
            continue

        feat_df = pipeline.compute(df, symbol=sym)
        close = feat_df["close"] if "close" in feat_df.columns else feat_df.iloc[:, 3]

        vol = compute_volatility(close, span=1440, min_periods=60)
        timestamps = pd.DatetimeIndex(close.dropna().index)
        valid = timestamps[vol.reindex(timestamps).notna()]
        if len(valid) < 300:
            continue

        events = make_events(close, valid, vol, vertical_bars=240)
        barriers = apply_triple_barrier(events, close)
        labels = get_bins(barriers, close)

        x_sym = feat_df.loc[labels.index, feature_cols].dropna()
        common = x_sym.index.intersection(labels.index)
        x_sym = x_sym.loc[common]
        y_sym = labels.loc[common, "bin"]
        t1_sym = events.loc[common, "t1"]

        sw = get_sample_weights_by_return(t1_sym, close)
        fwd = close.pct_change().shift(-1).reindex(common).fillna(0)

        all_features.append(x_sym)
        all_y.append(y_sym)
        all_t1.append(t1_sym)
        all_sw.append(np.asarray(sw.values))
        all_ret.append(fwd)

    if not all_features:
        return (
            pd.DataFrame(),
            pd.Series(dtype="int64"),
            pd.Series(dtype="object"),
            np.array([]),
            pd.Series(dtype="float64"),
        )

    return (
        pd.concat(all_features).sort_index(),
        pd.concat(all_y).sort_index(),
        pd.concat(all_t1).sort_index(),
        np.concatenate(all_sw),
        pd.concat(all_ret).sort_index(),
    )


@train_app.command("primary")
def train_primary(
    start: str = typer.Option("2021-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2024-06-30", help="End date (YYYY-MM-DD)"),
    n_trials: int = typer.Option(100, help="Optuna trial budget"),
    n_splits: int = typer.Option(5, help="PurgedKFold splits"),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols (default: all active)"),
    min_sharpe: float = typer.Option(0.0, help="Minimum CV Sharpe for auto-promotion"),
) -> None:
    """Train the primary {-1, 0, +1} LightGBM model."""
    import time

    import numpy as np

    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.train")
    log.info("train_primary_start", run_id=run_id, start=start, end=end, n_trials=n_trials)

    from tessera.data.store import read_parquet
    from tessera.data.universe import Universe
    from tessera.features import (
        FeaturePipeline,
        GarmanKlass,
        LogReturn,
        MicroPrice,
        OrderFlowImbalance,
        Parkinson,
        RealizedVol,
        SpreadBps,
        VolOfVol,
    )
    from tessera.labels.sample_weights import get_sample_weights_by_return
    from tessera.labels.triple_barrier import (
        apply_triple_barrier,
        compute_volatility,
        get_bins,
        make_events,
    )
    from tessera.models.lightgbm_model import PrimaryLightGBMModel
    from tessera.models.registry import ModelRegistry

    feature_list = [
        LogReturn(horizon=1),
        LogReturn(horizon=5),
        LogReturn(horizon=60),
        RealizedVol(window=60),
        Parkinson(window=60),
        GarmanKlass(window=60),
        VolOfVol(window=60),
        OrderFlowImbalance(depth=1),
        MicroPrice(),
        SpreadBps(),
    ]
    pipeline = FeaturePipeline(feature_list)

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        universe = Universe()
        symbol_list = universe.active_at(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC))
        if not symbol_list:
            typer.echo("No active symbols. Run 'tessera ingest universe' first.")
            raise typer.Exit(1)

    all_features: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    all_t1: list[pd.Series] = []
    all_sw: list[np.ndarray] = []
    all_ret: list[pd.Series] = []

    for sym in symbol_list:
        df = read_parquet("ohlcv", filters=[("symbol", "==", sym)])
        if df.empty:
            continue
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"])
            df = df[(df["event_time"] >= start) & (df["event_time"] <= end)]
        if len(df) < 500:
            continue

        feat_df = pipeline.compute(df, symbol=sym)
        close = feat_df["close"] if "close" in feat_df.columns else feat_df.iloc[:, 3]

        vol = compute_volatility(close, span=1440, min_periods=60)
        timestamps = pd.DatetimeIndex(close.dropna().index)
        valid = timestamps[vol.reindex(timestamps).notna()]
        if len(valid) < 300:
            continue

        events = make_events(close, valid, vol, vertical_bars=240)
        barriers = apply_triple_barrier(events, close)
        labels = get_bins(barriers, close)

        feature_cols = [f.name for f in feature_list]
        x_sym = feat_df.loc[labels.index, feature_cols].dropna()
        common = x_sym.index.intersection(labels.index)
        x_sym = x_sym.loc[common]
        y_sym = labels.loc[common, "bin"]
        t1_sym = events.loc[common, "t1"]

        sw = get_sample_weights_by_return(t1_sym, close)
        fwd = close.pct_change().shift(-1).reindex(common).fillna(0)

        all_features.append(x_sym)
        all_y.append(y_sym)
        all_t1.append(t1_sym)
        all_sw.append(np.asarray(sw.values))
        all_ret.append(fwd)

    if not all_features:
        typer.echo("No training data assembled. Check data ingestion.")
        raise typer.Exit(1)

    features = pd.concat(all_features).sort_index()
    y = pd.concat(all_y).sort_index()
    t1 = pd.concat(all_t1).sort_index()
    sample_weight = np.concatenate(all_sw)
    forward_returns = pd.concat(all_ret).sort_index()

    typer.echo(f"Training data: {len(features)} samples, {features.shape[1]} features")
    t0 = time.time()

    model = PrimaryLightGBMModel(seed=settings.random_seed)
    study = model.tune(
        features,
        y,
        t1,
        sample_weight=sample_weight,
        forward_returns=forward_returns,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    elapsed = time.time() - t0
    card = model.get_model_card()
    typer.echo(f"Best Optuna accuracy: {study.best_value:.4f}")
    if card.cv_scores:
        typer.echo(
            f"CV Sharpe: {card.cv_scores.mean_sharpe:.4f} "
            f"± {card.cv_scores.std_sharpe:.4f} "
            f"(deflated: {card.cv_scores.deflated_sharpe:.4f}, "
            f"trials={card.cv_scores.n_trials})"
        )

    registry = ModelRegistry()
    path = registry.save_model(model, "primary")
    registry.promote(path, min_sharpe=min_sharpe)
    typer.echo(f"Model saved and promoted: {path}  ({elapsed:.1f}s)")
    raise typer.Exit()


@train_app.command("meta")
def train_meta(
    primary_model: str = typer.Option(
        "models/primary/current", help="Path to promoted primary model"
    ),
    start: str = typer.Option("2021-01-01", help="Start date"),
    end: str = typer.Option("2024-06-30", help="End date"),
    n_trials: int = typer.Option(100, help="Optuna trial budget"),
    n_splits: int = typer.Option(5, help="PurgedKFold splits"),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols"),
    min_sharpe: float = typer.Option(0.0, help="Minimum CV Sharpe for auto-promotion"),
) -> None:
    """Train the meta-labeling model on top of the primary."""
    import time
    from pathlib import Path

    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.train")
    log.info("train_meta_start", run_id=run_id, primary=primary_model)

    from tessera.models.lightgbm_model import PrimaryLightGBMModel
    from tessera.models.meta_model import MetaModel
    from tessera.models.registry import ModelRegistry

    primary = PrimaryLightGBMModel.load(Path(primary_model))
    typer.echo(f"Primary model loaded from {primary_model}")

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        from tessera.data.universe import Universe

        universe = Universe()
        symbol_list = universe.active_at(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC))
        if not symbol_list:
            typer.echo("No active symbols. Run 'tessera ingest universe' first.")
            raise typer.Exit(1)

    features, y, t1, sample_weight, forward_returns = _assemble_training_data(
        symbol_list, start, end
    )
    if features.empty:
        typer.echo("No training data assembled. Check data ingestion.")
        raise typer.Exit(1)

    # Restrict to the feature columns the primary was trained on
    primary_cols = [c for c in primary.get_model_card().features if c in features.columns]
    X = features[primary_cols]  # noqa: N806

    typer.echo(f"Training data: {len(X)} samples, {X.shape[1]} features")
    t0 = time.time()

    meta = MetaModel(primary=primary, seed=settings.random_seed)
    meta.fit(
        X,
        y,
        sample_weight=sample_weight,
        t1=t1,
        forward_returns=forward_returns,
        n_trials=n_trials,
    )

    elapsed = time.time() - t0
    meta_card = meta.meta.get_model_card()
    if meta_card.cv_scores:
        typer.echo(
            f"Meta CV Sharpe: {meta_card.cv_scores.mean_sharpe:.4f} "
            f"± {meta_card.cv_scores.std_sharpe:.4f} "
            f"(deflated: {meta_card.cv_scores.deflated_sharpe:.4f}, "
            f"trials={meta_card.cv_scores.n_trials})"
        )

    registry = ModelRegistry()
    meta_path = registry.save_model(meta.meta, "meta")
    registry.promote(meta_path, min_sharpe=min_sharpe)

    # Save the full MetaModel bundle (primary + meta) for inference
    full_path = settings.models_root / "meta_model"
    meta.save(full_path)

    log.info("train_meta_complete", path=str(meta_path), elapsed_s=round(elapsed, 1))
    typer.echo(f"Meta-model saved and promoted: {meta_path}  ({elapsed:.1f}s)")
    raise typer.Exit()


@train_app.command("ensemble")
def train_ensemble(
    start: str = typer.Option("2021-01-01", help="Start date"),
    end: str = typer.Option("2024-06-30", help="End date"),
    val_fraction: float = typer.Option(0.2, help="Held-out fraction for weight optimisation"),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols"),
) -> None:
    """Train a Sharpe-optimal ensemble of the promoted primary and a diverse variant."""
    import time

    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.train")
    log.info("train_ensemble_start", run_id=run_id)

    from tessera.models.ensemble import EnsembleModel
    from tessera.models.lightgbm_model import PrimaryLightGBMModel
    from tessera.models.registry import ModelRegistry

    registry = ModelRegistry()
    primary = registry.load_current("primary", PrimaryLightGBMModel)
    typer.echo("Primary model loaded from registry.")

    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]
    else:
        from tessera.data.universe import Universe

        universe = Universe()
        symbol_list = universe.active_at(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC))
        if not symbol_list:
            typer.echo("No active symbols. Run 'tessera ingest universe' first.")
            raise typer.Exit(1)

    features, y, t1, sample_weight, forward_returns = _assemble_training_data(
        symbol_list, start, end
    )
    if features.empty:
        typer.echo("No training data assembled. Check data ingestion.")
        raise typer.Exit(1)

    primary_cols = [c for c in primary.get_model_card().features if c in features.columns]
    X = features[primary_cols]  # noqa: N806

    # Chronological train/val split — never shuffle time-series data
    n_val = max(10, int(len(X) * val_fraction))
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]  # noqa: N806
    y_train = y.iloc[:-n_val]
    sw_train = sample_weight[:-n_val]
    fwd_ret_val = forward_returns.iloc[-n_val:]

    typer.echo(f"Train: {len(X_train)} samples  |  Val: {len(X_val)} samples")

    # Train a smaller, diverse variant with a different seed and shallower trees
    typer.echo("Training diversity variant (num_leaves=31, depth=6, n_estimators=200)…")
    t0 = time.time()
    variant = PrimaryLightGBMModel(
        seed=settings.random_seed + 1,
        n_estimators=200,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
    )
    variant.fit(X_train, y_train, sample_weight=sw_train)

    ensemble = EnsembleModel([primary, variant])
    weights = ensemble.fit_weights(X_val, fwd_ret_val)
    elapsed = time.time() - t0

    typer.echo(f"Ensemble weights: primary={weights[0]:.3f}  variant={weights[1]:.3f}")

    path = registry.save_model(ensemble, "ensemble")
    registry.promote(path)

    log.info("train_ensemble_complete", path=str(path), elapsed_s=round(elapsed, 1))
    typer.echo(f"Ensemble saved and promoted: {path}  ({elapsed:.1f}s)")
    raise typer.Exit()


backtest_app = typer.Typer(help="Event-driven Nautilus Trader backtest.")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    config_path: str = typer.Option(
        "configs/backtest.yaml", "--config", help="Backtest YAML config"
    ),
    primary_model: str = typer.Option("", "--primary-model", help="Path to primary model dir"),
    meta_model: str = typer.Option("", "--meta-model", help="Path to meta model dir"),
    latency_ms: int = typer.Option(
        -1, "--latency-ms", help="Override fixed latency in ms (-1 = use config range)"
    ),
    output_tearsheet: bool = typer.Option(
        True, "--tearsheet/--no-tearsheet", help="Generate QuantStats HTML tearsheet"
    ),
) -> None:
    """Run a full event-driven backtest using Nautilus Trader.

    Loads OHLCV bars from the Parquet store for the configured universe and date
    range, runs the ML Directional strategy, and writes results to
    data/backtest_runs/<run_id>/.
    """
    import time
    from pathlib import Path

    settings, run_id = _bootstrap()
    start_metrics_server(settings.prometheus_port)
    log = structlog.get_logger("tessera.cli.backtest")
    log.info("backtest_run_start", run_id=run_id, config=config_path)

    from tessera.backtest.engine import TesseraBacktestEngine
    from tessera.config import BacktestConfig, load_yaml
    from tessera.strategies.ml_directional import MLDirectionalConfig, MLDirectionalStrategy

    cfg: BacktestConfig = load_yaml(config_path)  # type: ignore[assignment]

    # Collect instrument_ids from all configured venues
    instrument_ids: list[str] = []
    for venue_cfg in cfg.backtest.venues:
        for ccxt_sym in venue_cfg.symbols:
            nautilus_id = f"{ccxt_sym.split('/')[0]}-USDT-PERP.{venue_cfg.exchange.upper()}"
            instrument_ids.append(nautilus_id)

    if not instrument_ids:
        typer.echo("No instruments configured in backtest.yaml. Add venues.symbols.")
        raise typer.Exit(1)

    log_dir = str(settings.data_root / "backtest_runs" / run_id)

    strategy_config = MLDirectionalConfig(
        instrument_ids=tuple(instrument_ids),
        bar_type_suffix=cfg.backtest.bar_type,
        primary_model_path=primary_model,
        meta_model_path=meta_model,
        max_drawdown_pct=cfg.costs.maker_fee_bps * 2.5,  # rough kill-switch threshold
        signal_delay_bars=cfg.backtest.signal_delay_bars,
        log_dir=log_dir,
    )
    strategy = MLDirectionalStrategy(config=strategy_config)

    if latency_ms > 0:
        # Override: fix both min and max to get a deterministic latency
        from tessera.config import BacktestEngineConfig

        cfg.backtest = BacktestEngineConfig(
            **{
                **cfg.backtest.model_dump(),
                "latency_min_ms": latency_ms,
                "latency_max_ms": latency_ms,
            }
        )

    engine = TesseraBacktestEngine.from_config(
        cfg, settings, strategy, run_id, seed=settings.random_seed
    )

    t0 = time.time()
    result = engine.run()
    elapsed = time.time() - t0

    # Summary output
    typer.echo("\n" + "=" * 60)
    typer.echo(f"Run ID:          {result.run_id}")
    typer.echo(f"Date range:      {result.start_date} → {result.end_date}")
    typer.echo(f"Bars processed:  {result.n_bars:,}")
    typer.echo(f"Trades:          {result.n_trades:,}")
    typer.echo(f"Total PnL:       ${result.total_pnl:,.2f}")
    typer.echo(f"  Trading:       ${result.trading_pnl:,.2f}")
    typer.echo(f"  Funding:       ${result.funding_pnl:,.2f}")
    typer.echo(f"  Fees:          ${result.fee_pnl:,.2f}")
    typer.echo(f"Total return:    {result.total_return:.2%}")
    typer.echo(f"Sharpe ratio:    {result.sharpe_ratio:.3f}")
    typer.echo(f"Sortino ratio:   {result.sortino_ratio:.3f}")
    typer.echo(f"Max drawdown:    {result.max_drawdown:.2%}")
    typer.echo(f"Wall-clock time: {elapsed:.1f}s")
    typer.echo(f"Log dir:         {result.log_dir}")
    typer.echo("=" * 60)

    if output_tearsheet and not result.equity_curve.empty:
        try:
            import quantstats as qs

            daily_returns = result.equity_curve.pct_change().dropna()
            if not daily_returns.empty:
                tearsheet_path = Path("docs/figures") / f"tearsheet_{run_id}.html"
                tearsheet_path.parent.mkdir(parents=True, exist_ok=True)
                qs.reports.html(  # type: ignore[no-untyped-call]
                    daily_returns, output=str(tearsheet_path), title=f"Tessera Backtest {run_id}"
                )
                typer.echo(f"Tearsheet:       {tearsheet_path}")
        except ImportError:
            typer.echo("(QuantStats not installed — skipping tearsheet)")
        except Exception as e:
            typer.echo(f"(Tearsheet generation failed: {e})")

    raise typer.Exit()


report_app = typer.Typer(help="Generate performance evaluation reports.")
app.add_typer(report_app, name="report")


@report_app.command("backtest")
def report_backtest(
    run_id: str = typer.Option(
        ..., "--run-id", help="Backtest run ID (reads from data/backtest_runs/<run_id>/)"
    ),
    output: str = typer.Option("docs/figures/", "--output", help="Output directory"),
    n_trials: int = typer.Option(100, "--n-trials", help="Total trial count (Optuna + manual)"),
    benchmark_path: str | None = typer.Option(
        None, "--benchmark", help="Parquet path for benchmark daily returns"
    ),
    test_start: str | None = typer.Option(
        None, "--test-start", help="First OOS date (YYYY-MM-DD) for IS/OOS labelling"
    ),
    n_bootstrap: int = typer.Option(10_000, "--n-bootstrap", help="Bootstrap resamples for SR CI"),
    sr_std: float | None = typer.Option(
        None, "--sr-std", help="Cross-trial SR std (default: 1/√T)"
    ),
) -> None:
    """Generate a full evaluation report for a completed backtest run.

    Reads equity_curve.parquet and summary.json from the run directory and
    produces an enhanced QuantStats HTML tearsheet with DSR, PSR, bootstrap CI,
    and stress-window PnLs written to <output>/<run_id>_report.html.
    """
    import json
    from pathlib import Path

    settings, _ = _bootstrap()
    log = structlog.get_logger("tessera.cli.report")

    run_dir = settings.data_root / "backtest_runs" / run_id
    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}", err=True)
        raise typer.Exit(1)

    equity_path = run_dir / "equity_curve.parquet"
    if not equity_path.exists():
        typer.echo(f"equity_curve.parquet not found in {run_dir}", err=True)
        raise typer.Exit(1)

    equity = pd.read_parquet(equity_path)["equity"]
    returns = equity.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)

    summary_path = run_dir / "summary.json"
    title = f"Tessera Backtest — {run_id}"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        sr_point = summary.get("sharpe_ratio", None)
        title = (
            f"Tessera Backtest — {run_id} "
            f"[{summary.get('start_date', '')} → {summary.get('end_date', '')}]"
        )
    else:
        sr_point = None

    benchmark_returns: pd.Series | None = None
    if benchmark_path:
        try:
            bm_df = pd.read_parquet(benchmark_path)
            benchmark_returns = bm_df.iloc[:, 0] if isinstance(bm_df, pd.DataFrame) else bm_df
        except Exception as exc:
            typer.echo(f"Warning: could not load benchmark: {exc}", err=True)

    from tessera.backtest.reports.tearsheet import generate_tearsheet

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_id}_report.html"

    manual_metrics: dict[str, object] | None = None
    if sr_point is not None:
        manual_metrics = {"point_sr": sr_point}

    log.info("report_start", run_id=run_id, n_trials=n_trials, output=str(output_file))
    typer.echo(f"Generating report for run {run_id!r}…")

    out = generate_tearsheet(
        returns=returns,
        benchmark_returns=benchmark_returns,
        output_path=output_file,
        trial_count=n_trials,
        manual_metrics_to_overlay=manual_metrics,
        sr_std=sr_std,
        n_bootstrap=n_bootstrap,
        test_start_date=test_start,
        title=title,
    )
    typer.echo(f"Report written: {out}")
    raise typer.Exit()


paper_app = typer.Typer(help="Paper trading against Binance Testnet and Bybit Demo.")
app.add_typer(paper_app, name="paper")


@paper_app.command("start")
def paper_start(
    config_path: str = typer.Option("configs/live.yaml", "--config", help="Live YAML config path"),
    healthcheck_port: int = typer.Option(8080, "--healthcheck-port", help="Port for /healthz"),
) -> None:
    """Start the paper trading runner (blocks until stopped or crash limit hit)."""
    from tessera.config import LiveConfig, load_yaml
    from tessera.live.paper import PaperRunner, read_pid

    settings, run_id = _bootstrap()
    start_metrics_server(settings.prometheus_port)
    log = structlog.get_logger("tessera.cli.paper")

    # Warn if already running
    existing_pid = read_pid()
    if existing_pid is not None:
        typer.echo(f"Warning: PID file exists (pid={existing_pid}). Already running?")

    cfg: LiveConfig = load_yaml(config_path)  # type: ignore[assignment]
    log.info("paper_start", run_id=run_id, config=config_path)

    runner = PaperRunner(cfg, settings, run_id, healthcheck_port=healthcheck_port)
    typer.echo(f"Paper runner starting (run_id={run_id}, healthcheck=:{healthcheck_port}/healthz)")
    runner.run()
    raise typer.Exit()


@paper_app.command("status")
def paper_status() -> None:
    """Print current paper runner state from Postgres."""
    import psycopg2

    settings, _ = _bootstrap()

    try:
        conn = psycopg2.connect(settings.postgres_dsn, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, status, pid, crash_reason, started_at, updated_at
                FROM paper_runner_state
                ORDER BY updated_at DESC
                LIMIT 5
                """
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        typer.echo(f"Failed to query Postgres: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not rows:
        typer.echo("No paper runner state found in Postgres.")
        raise typer.Exit()

    from tessera.live.paper import read_pid

    live_pid = read_pid()

    typer.echo("\nPaper runner state (most recent first):")
    typer.echo("=" * 70)
    for run_id, status, pid, crash_reason, started_at, updated_at in rows:
        is_current = live_pid is not None and pid == live_pid
        marker = " ← CURRENT" if is_current else ""
        typer.echo(f"  run_id:    {run_id}{marker}")
        typer.echo(f"  status:    {status}")
        typer.echo(f"  pid:       {pid}")
        if crash_reason:
            typer.echo(f"  reason:    {crash_reason}")
        typer.echo(f"  started:   {started_at}")
        typer.echo(f"  updated:   {updated_at}")
        typer.echo("-" * 70)

    raise typer.Exit()


@paper_app.command("stop")
def paper_stop() -> None:
    """Send SIGTERM to the running paper trader via PID file."""
    from tessera.live.paper import send_stop_signal

    _bootstrap()
    sent = send_stop_signal()
    if sent:
        typer.echo("SIGTERM sent. Runner will flatten positions and stop.")
    else:
        typer.echo("No running paper trader found (PID file missing or stale).")
        raise typer.Exit(1)
    raise typer.Exit()


@app.command()
def report() -> None:
    """Generate performance reports."""
    settings, run_id = _bootstrap()
    log = structlog.get_logger("tessera.cli.report")
    log.info("subcommand started", subcommand="report", run_id=run_id)
    typer.echo("TODO: report")
    raise typer.Exit()

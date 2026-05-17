"""Application configuration and YAML config loading.

Centralizes all settings: environment variables via Pydantic BaseSettings,
YAML config files via typed Pydantic models, and reproducibility via seed_everything.
"""

from __future__ import annotations

import hashlib
import os
import random
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Environment settings (loaded from .env + ENV vars)
# ---------------------------------------------------------------------------


class Environment(StrEnum):
    """Deployment environment."""

    DEV = "dev"
    PAPER = "paper"
    LIVE = "live"


class TesseraSettings(BaseSettings):
    """Global application settings loaded from environment.

    All fields can be set via TESSERA_<FIELD_NAME> environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="TESSERA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Environment = Field(default=Environment.DEV, description="Deployment environment")
    exchanges: list[str] = Field(default=["binance"], description="Active exchanges")
    symbols: list[str] = Field(
        default=["BTC/USDT:USDT", "ETH/USDT:USDT"], description="Trading symbols"
    )

    binance_api_key: SecretStr | None = Field(default=None)
    binance_api_secret: SecretStr | None = Field(default=None)
    bybit_api_key: SecretStr | None = Field(default=None)
    bybit_api_secret: SecretStr | None = Field(default=None)
    tardis_api_key: SecretStr | None = Field(default=None)

    postgres_dsn: str = Field(default="postgresql://tessera:tessera@localhost:5432/tessera")
    redis_url: str = Field(default="redis://localhost:6379/0")
    prometheus_port: int = Field(default=9090)
    log_level: str = Field(default="INFO")
    sentry_dsn: str | None = Field(default=None)

    data_root: Path = Field(default=Path("./data"))
    models_root: Path = Field(default=Path("./models"))
    random_seed: int = Field(default=42)


# ---------------------------------------------------------------------------
# YAML config models (for configs/*.yaml files)
# ---------------------------------------------------------------------------


class ExchangeConfig(BaseModel):
    """Configuration for a single exchange data source."""

    name: str
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default=["1m", "5m", "1h"])


class StorageConfig(BaseModel):
    """Data storage configuration."""

    format: Literal["parquet", "csv"] = "parquet"
    path: str = "./data/raw"
    partition_by: list[str] = Field(default=["exchange", "symbol", "date"])


class DataConfig(BaseModel):
    """Data ingestion configuration (configs/data.yaml)."""

    exchanges: list[ExchangeConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class FeatureSpec(BaseModel):
    """Single feature specification."""

    name: str
    lookback: int = 100
    params: dict[str, Any] = Field(default_factory=dict)


class FeatureConfig(BaseModel):
    """Feature engineering configuration (configs/features.yaml)."""

    features: list[FeatureSpec] = Field(default_factory=list)
    bar_type: str = "volume"


class CVConfig(BaseModel):
    """Cross-validation configuration."""

    method: str = "purged_kfold"
    n_splits: int = 5
    embargo_pct: float = 0.01


class OptimizationConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    engine: str = "optuna"
    n_trials: int = 100
    metric: str = "sharpe_ratio"
    direction: str = "maximize"


class ModelParams(BaseModel):
    """Model hyperparameters."""

    type: str = "lightgbm"
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model training configuration (configs/model.yaml)."""

    model: ModelParams = Field(default_factory=ModelParams)
    cv: CVConfig = Field(default_factory=CVConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


class CostConfig(BaseModel):
    """Transaction cost assumptions."""

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 1.0
    funding_rate: bool = True


class ReportingConfig(BaseModel):
    """Backtest reporting configuration."""

    output_dir: str = "./reports"
    generate_tearsheet: bool = True
    deflated_sharpe: bool = True
    trial_count: int = 10


class VenueConfig(BaseModel):
    """Per-venue configuration for the backtest engine."""

    exchange: str = "binance"
    symbols: list[str] = Field(default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    vip_tier: int = 0
    initial_balance_usdt: float = 100000.0
    default_leverage: float = 3.0
    # Funding cadence: 8h for most perps on Binance/Bybit; 4h for some Bybit pairs.
    funding_period_hours: int = 8


class BacktestEngineConfig(BaseModel):
    """Backtest engine settings."""

    engine: str = "nautilus"
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000.0
    currency: str = "USDT"
    # Uniform-random latency range for order submission [min, max] ms.
    # A single value is drawn from U[latency_min_ms, latency_max_ms] using
    # the backtest seed and applied via Nautilus LatencyModel.
    latency_min_ms: int = 100
    latency_max_ms: int = 500
    # Signal delay expressed in bar units. 0 = act on current bar's signal.
    # Derived automatically from latency / bar_period; overridable for ablation.
    signal_delay_bars: int = 0
    bar_type: str = "1-MINUTE-LAST-EXTERNAL"
    venues: list[VenueConfig] = Field(default_factory=lambda: [VenueConfig()])


class BacktestConfig(BaseModel):
    """Backtest configuration (configs/backtest.yaml)."""

    backtest: BacktestEngineConfig = Field(default_factory=BacktestEngineConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


class PositionConfig(BaseModel):
    """Live position limits."""

    max_notional_usd: float = 10000.0
    max_leverage: float = 3.0
    funding_carry: bool = True


class KillSwitchConfig(BaseModel):
    """Emergency stop configuration."""

    max_drawdown_pct: float = 5.0
    max_daily_loss_usd: float = 500.0
    enabled: bool = True


class LiveEngineConfig(BaseModel):
    """Live trading engine settings."""

    mode: Literal["paper", "live"] = "paper"
    exchange: str = "binance"
    symbols: list[str] = Field(default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    poll_interval_ms: int = 1000


class LiveConfig(BaseModel):
    """Live/paper trading configuration (configs/live.yaml)."""

    live: LiveEngineConfig = Field(default_factory=LiveEngineConfig)
    position: PositionConfig = Field(default_factory=PositionConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)


class SizingConfig(BaseModel):
    """Position sizing parameters."""

    method: str = "kelly_fractional"
    kelly_fraction: float = 0.25
    max_size_usd: float = 5000.0


class RiskLimits(BaseModel):
    """Risk limit thresholds."""

    max_open_orders: int = 10
    max_daily_trades: int = 50
    max_drawdown_pct: float = 5.0
    cooldown_after_loss_minutes: int = 15


class RiskParams(BaseModel):
    """Core risk parameters."""

    max_position_pct: float = 0.1
    max_portfolio_heat: float = 0.2
    max_correlated_positions: int = 3
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0


class RiskConfig(BaseModel):
    """Risk management configuration (configs/risk.yaml)."""

    risk: RiskParams = Field(default_factory=RiskParams)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    limits: RiskLimits = Field(default_factory=RiskLimits)


# ---------------------------------------------------------------------------
# Config model registry (maps YAML filename stems to model classes)
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: dict[str, type[BaseModel]] = {
    "data": DataConfig,
    "features": FeatureConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
    "live": LiveConfig,
    "risk": RiskConfig,
}


def load_yaml(path: str | Path) -> BaseModel:
    """Load and validate a YAML config file against its Pydantic model.

    The model class is selected based on the file stem (e.g., 'data.yaml' → DataConfig).

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ValueError: If the file stem doesn't match a known config type.
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    stem = path.stem
    model_class = _CONFIG_REGISTRY.get(stem)
    if model_class is None:
        msg = f"Unknown config type '{stem}'. Known types: {list(_CONFIG_REGISTRY.keys())}"
        raise ValueError(msg)

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    return model_class.model_validate(raw)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> int:
    """Seed all random number generators for reproducibility.

    Seeds: Python random, NumPy, PYTHONHASHSEED, and PyTorch (if available).
    Reference: AFML Chapter 8 — ensuring deterministic experiment results.

    Args:
        seed: Integer seed value.

    Returns:
        The seed that was set (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def generate_run_id(seed: int) -> str:
    """Generate a short deterministic run ID for tracing.

    Args:
        seed: Seed to incorporate into the hash.

    Returns:
        8-character hex string.
    """
    raw = f"{seed}-{os.getpid()}-{id(object())}".encode()
    return hashlib.sha256(raw).hexdigest()[:8]

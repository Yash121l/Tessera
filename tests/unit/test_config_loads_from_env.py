"""Test that TesseraSettings loads from environment variables."""

from __future__ import annotations

import os
from unittest.mock import patch

from tessera.config import Environment, TesseraSettings


def test_settings_defaults() -> None:
    """Settings should load with all defaults when no env vars set."""
    with patch.dict(os.environ, {}, clear=False):
        settings = TesseraSettings()
    assert settings.env == Environment.DEV
    assert settings.random_seed == 42
    assert settings.prometheus_port == 9090
    assert settings.log_level == "INFO"


def test_settings_from_env() -> None:
    """Settings should load values from TESSERA_-prefixed env vars."""
    env = {
        "TESSERA_ENV": "live",
        "TESSERA_LOG_LEVEL": "DEBUG",
        "TESSERA_RANDOM_SEED": "123",
        "TESSERA_PROMETHEUS_PORT": "8080",
        "TESSERA_POSTGRES_DSN": "postgresql://user:pass@db:5432/prod",
        "TESSERA_REDIS_URL": "redis://redis:6379/1",
        "TESSERA_BINANCE_API_KEY": "key123",
        "TESSERA_BINANCE_API_SECRET": "secret456",
    }
    with patch.dict(os.environ, env, clear=False):
        settings = TesseraSettings()

    assert settings.env == Environment.LIVE
    assert settings.log_level == "DEBUG"
    assert settings.random_seed == 123
    assert settings.prometheus_port == 8080
    assert settings.postgres_dsn == "postgresql://user:pass@db:5432/prod"
    assert settings.redis_url == "redis://redis:6379/1"
    assert settings.binance_api_key is not None
    assert settings.binance_api_key.get_secret_value() == "key123"


def test_settings_list_fields() -> None:
    """List fields (exchanges, symbols) should parse from JSON-style env vars."""
    env = {
        "TESSERA_EXCHANGES": '["binance","bybit"]',
        "TESSERA_SYMBOLS": '["BTC/USDT:USDT"]',
    }
    with patch.dict(os.environ, env, clear=False):
        settings = TesseraSettings()

    assert settings.exchanges == ["binance", "bybit"]
    assert settings.symbols == ["BTC/USDT:USDT"]


def test_settings_optional_fields_none() -> None:
    """Optional fields should be None when not set."""
    settings = TesseraSettings()
    assert settings.binance_api_key is None
    assert settings.bybit_api_key is None
    assert settings.tardis_api_key is None
    assert settings.sentry_dsn is None

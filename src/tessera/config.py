"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Global application configuration.

    All settings are prefixed with TESSERA_ in environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="TESSERA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    redis_url: str = Field(default="redis://localhost:6379/0")
    postgres_dsn: str = Field(default="postgresql://tessera:tessera@localhost:5432/tessera")
    data_dir: str = Field(default="./data")
    kill_switch: bool = Field(default=False, description="Emergency stop all trading")

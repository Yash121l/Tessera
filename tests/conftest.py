"""Shared test fixtures."""

from __future__ import annotations

import pytest

from tessera.config import TesseraSettings


@pytest.fixture
def settings() -> TesseraSettings:
    """Create test settings instance with defaults."""
    return TesseraSettings(log_level="DEBUG")

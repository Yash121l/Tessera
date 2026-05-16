"""Shared test fixtures."""

from __future__ import annotations

import pytest

from tessera.config import AppSettings


@pytest.fixture
def settings() -> AppSettings:
    """Create test settings instance with defaults."""
    return AppSettings(debug=True, log_level="DEBUG")

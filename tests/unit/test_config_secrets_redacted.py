"""Test that SecretStr fields don't leak in repr or str."""

from __future__ import annotations

import os
from unittest.mock import patch

from tessera.config import TesseraSettings


def test_secrets_hidden_in_repr() -> None:
    """SecretStr values must not appear in repr output."""
    env = {
        "TESSERA_BINANCE_API_KEY": "super_secret_key_123",
        "TESSERA_BINANCE_API_SECRET": "ultra_secret_789",
        "TESSERA_BYBIT_API_KEY": "bybit_key_abc",
        "TESSERA_TARDIS_API_KEY": "tardis_xyz",
    }
    with patch.dict(os.environ, env, clear=False):
        settings = TesseraSettings()

    text = repr(settings)
    assert "super_secret_key_123" not in text
    assert "ultra_secret_789" not in text
    assert "bybit_key_abc" not in text
    assert "tardis_xyz" not in text


def test_secrets_hidden_in_str() -> None:
    """SecretStr values must not appear in str output."""
    env = {"TESSERA_BINANCE_API_KEY": "do_not_leak_this"}
    with patch.dict(os.environ, env, clear=False):
        settings = TesseraSettings()

    text = str(settings)
    assert "do_not_leak_this" not in text


def test_secrets_hidden_in_model_dump_json() -> None:
    """SecretStr values must not appear in JSON serialization."""
    env = {"TESSERA_BINANCE_API_KEY": "json_leak_test"}
    with patch.dict(os.environ, env, clear=False):
        settings = TesseraSettings()

    json_str = settings.model_dump_json()
    assert "json_leak_test" not in json_str

"""Smoke tests to verify the package is importable and correctly configured."""

from __future__ import annotations


def test_import_tessera() -> None:
    """Verify the tessera package can be imported."""
    import tessera

    assert tessera.__version__ == "0.1.0"


def test_version_is_semver() -> None:
    """Verify version string follows semver format."""
    import tessera

    parts = tessera.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_config_defaults() -> None:
    """Verify TesseraSettings can be instantiated with environment defaults."""
    from tessera.config import TesseraSettings

    settings = TesseraSettings()
    assert settings.log_level == "INFO"
    assert settings.random_seed == 42

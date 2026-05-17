"""Test that structured logging emits parseable JSON."""

from __future__ import annotations

import json
import logging
from io import StringIO

from tessera.log import configure_logging, get_logger


def test_json_log_parseable() -> None:
    """A log line in JSON mode must be valid JSON with expected fields."""
    configure_logging(level="DEBUG", json=True)

    capture = StringIO()
    handler = logging.StreamHandler(capture)
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().addHandler(handler)

    try:
        log = get_logger("test.json")
        log.info("test event", key="value")

        output = capture.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["event"] == "test event"
        assert parsed["level"] == "info"
        assert "timestamp" in parsed
    finally:
        logging.getLogger().removeHandler(handler)


def test_get_logger_returns_logger() -> None:
    """get_logger should return a usable structlog logger."""
    configure_logging(level="INFO", json=False)
    log = get_logger("test.module")
    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    assert hasattr(log, "error")


def test_log_level_filtering() -> None:
    """Messages below configured level should not appear."""
    configure_logging(level="WARNING", json=True)
    root = logging.getLogger()
    assert root.level == logging.WARNING

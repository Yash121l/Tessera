"""Structured logging configuration using structlog."""

from __future__ import annotations

import structlog


def setup_logging(*, debug: bool = False) -> None:
    """Configure structlog processors for the application.

    Args:
        debug: If True, use colored console output. Otherwise, JSON.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

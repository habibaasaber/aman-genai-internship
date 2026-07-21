"""
utils/logger.py
===============
Structured logging configuration for the Week 2 RAG project.

Sets up ``structlog`` with:
- ISO-8601 timestamps.
- Log level filtering controlled by the ``LOG_LEVEL`` env var (default INFO).
- Human-readable coloured output in development (TTY detected automatically).
- JSON output in production / CI (no TTY).

Usage
-----
    from utils import get_logger
    log = get_logger(__name__)
    log.info("chunk_created", chunk_id=42, size=512)
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def _configure_structlog() -> None:
    """
    Configure structlog processors and stdlib integration.

    Called once at module import time. Subsequent calls are safe (idempotent
    because structlog checks whether it has already been configured).
    """
    log_level_name: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level: int = getattr(logging, log_level_name, logging.INFO)

    # Wire stdlib logging through structlog so third-party libraries
    # (e.g. sentence-transformers, qdrant-client) also use structured output.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Detect whether we are writing to a terminal to pick the renderer.
    is_tty: bool = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    shared_processors: list[structlog.types.Processor] = [
        # Add log level name to every event dict.
        structlog.stdlib.add_log_level,
        # Add ISO-8601 timestamp.
        structlog.processors.TimeStamper(fmt="iso"),
        # Render exception info as a string if present.
        structlog.processors.format_exc_info,
        # Add the logger name (module) to the event dict.
        structlog.stdlib.add_logger_name,
    ]

    if is_tty:
        # Coloured, human-readable output for local development.
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for production / CI environments.
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare the event dict for the final renderer.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib formatter to use structlog's processors.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Replace any existing handlers to avoid duplicate output.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


# Configure on import — safe to call multiple times.
_configure_structlog()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to the given module name.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A structured logger with ``info``, ``debug``, ``warning``, ``error``
        and ``exception`` methods.

    Example
    -------
        log = get_logger(__name__)
        log.info("ingestion_started", pdf_path="data/guide.pdf")
    """
    return structlog.get_logger(name)

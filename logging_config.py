"""Structlog configuration: environment-aware rendering and log-level filtering.

Call configure_logging() once at startup — in the API lifespan or a script's
__main__ block. structlog's make_filtering_bound_logger skips disabled levels
before the event dict is even built, so debug calls in production cost nothing.
"""
import logging

import structlog


def configure_logging(environment: str, log_level: str) -> None:
    """Configure structlog and the stdlib root logger.

    Args:
        environment: "development" → coloured ConsoleRenderer.
                     Anything else  → newline-delimited JSONRenderer.
        log_level:   stdlib level name (DEBUG | INFO | WARNING | ERROR | CRITICAL).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    is_dev = environment == "development"

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.dev.ConsoleRenderer()
        if is_dev
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.ExceptionRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Stdlib logging stays active for third-party libraries (transformers, datasets, torch).
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

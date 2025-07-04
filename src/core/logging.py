"""Structured logging configuration for the multi-agent system."""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory


def configure_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None, json_logs: bool = False
) -> None:
    """Configure structured logging for the application."""

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for the given name."""
    return structlog.get_logger(name)


class AgentLogger:
    """Logger with agent-specific context."""

    def __init__(self, agent_name: str):
        self.logger = get_logger("agent").bind(agent_name=agent_name)

    def info(self, event: str, **kwargs):
        self.logger.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        self.logger.error(event, **kwargs)

    def warning(self, event: str, **kwargs):
        self.logger.warning(event, **kwargs)

    def debug(self, event: str, **kwargs):
        self.logger.debug(event, **kwargs)

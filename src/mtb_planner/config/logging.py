"""Logging setup.

`print()` is banned project-wide (ruff T20). Long-running CLI progress goes
through `rich`; everything else goes through the standard `logging` module
configured here.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, TextIO

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_TIME_FORMAT = "%H:%M:%S"


class StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """A stream handler that resolves `sys.stderr` at emit time.

    `logging.StreamHandler` captures the stream object when it is built. Any
    caller that later replaces `sys.stderr` — a test runner capturing output,
    or an embedding host — leaves the handler holding a stream that eventually
    closes, and every subsequent log record raises instead of being written.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self) -> TextIO:
        """The current standard error stream."""
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        """Ignored: the stream is always looked up fresh."""


def build_config(level: str = "INFO", log_file: Path | None = None) -> dict[str, Any]:
    """Return a `logging.config.dictConfig` mapping.

    Kept as a pure function so tests can assert on the configuration without
    mutating global logging state.
    """
    handlers: dict[str, Any] = {
        "console": {
            "()": "mtb_planner.config.logging.StderrHandler",
            "formatter": "console",
        }
    }
    if log_file is not None:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "console",
            "filename": str(log_file),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "encoding": "utf-8",
        }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {"format": _CONSOLE_FORMAT, "datefmt": _TIME_FORMAT},
        },
        "handlers": handlers,
        "loggers": {
            "mtb_planner": {
                "level": level,
                "handlers": list(handlers),
                "propagate": False,
            },
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    }


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure logging for the process. Safe to call more than once."""
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(build_config(level=level, log_file=log_file))


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `mtb_planner` namespace.

    Pass `__name__`; a bare module name is namespaced automatically so it
    inherits the package handler rather than falling through to root.
    """
    if not name.startswith("mtb_planner"):
        name = f"mtb_planner.{name}"
    return logging.getLogger(name)

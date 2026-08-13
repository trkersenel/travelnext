"""Logging helpers shared by the pipeline, API and UI."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str | None = None) -> None:
    """Configure root logging once, writing to stderr.

    The level can be overridden with the ``TRAVELNEXT_LOG_LEVEL`` environment
    variable (e.g. ``DEBUG``). Repeat calls are no-ops so importing modules in
    any order stays safe.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = level or os.environ.get("TRAVELNEXT_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Third-party libraries are noisy at INFO; keep the pipeline output readable.
    for noisy in ("urllib3", "matplotlib", "PIL", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    setup_logging()
    return logging.getLogger(name)

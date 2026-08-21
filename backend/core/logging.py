"""One logger configuration for the whole system.

Use `get_logger(__name__)` everywhere. Agent steps, data pulls and backtest runs
all share this format so the demo terminal stays readable.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False

_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    from backend.config import settings

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.setLevel(level or settings.log_level)
    root.handlers = [handler]

    # These are noisy and tell us nothing we want during a demo.
    for noisy in ("httpx", "urllib3", "yfinance", "peewee", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)

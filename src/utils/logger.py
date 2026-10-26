"""Logging setup shared by the whole app."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.utils.config import PROJECT_ROOT, get_config

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configures handlers on first use only."""
    global _CONFIGURED
    if not _CONFIGURED:
        level = get_config().get("app", {}).get("log_level", "INFO")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(fmt)

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)

        logging.basicConfig(level=level, handlers=[console, file_handler])
        _CONFIGURED = True
    return logging.getLogger(name)

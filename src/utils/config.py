"""Central configuration loader.

Design pattern: SINGLETON (via functools.lru_cache).
The YAML file is read once; every module that calls get_config() gets
the same dict back instead of re-reading the file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Raised when configuration or a required secret is missing."""


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml (and .env) once and cache the result."""
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_secret(name: str, required: bool = True) -> str | None:
    """Read a secret (API key) from environment variables / .env."""
    get_config()  # makes sure .env has been loaded
    value = os.getenv(name)
    if required and not value:
        raise ConfigError(
            f"Missing environment variable {name}. "
            "Add it to your .env file (see .env.example)."
        )
    return value

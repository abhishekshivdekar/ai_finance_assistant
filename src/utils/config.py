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
    """Raised when configuration or a required secret is missing or invalid."""


def _validate_sma_windows(windows: Any) -> None:
    """market_analysis.sma_windows must be [short, long]: two distinct positive ints, short < long."""
    ok = (
        isinstance(windows, list) and len(windows) == 2
        and all(isinstance(w, int) and not isinstance(w, bool) and w > 0 for w in windows)
        and windows[0] < windows[1]
    )
    if not ok:
        raise ConfigError(
            f"market_analysis.sma_windows must be two positive integers, short first "
            f"and smaller than long (for example [20, 50]); got {windows!r}."
        )


def _validate_goal_planning(goal: dict[str, Any]) -> None:
    """goal_planning: every risk profile needs all scenarios, lowest return first; limits must be consistent."""
    returns = goal["assumptions"]["returns_pct"]
    for profile in goal["risk_profiles"]:
        values = [returns[profile][name] for name in goal["scenarios"]]
        if values != sorted(values):
            raise ConfigError(f"goal_planning returns for {profile!r} must not decrease from conservative to optimistic.")
    if goal["max_months"] < goal["max_horizon_years"] * 12:
        raise ConfigError("goal_planning.max_months must cover max_horizon_years.")
    if goal["min_horizon_years"] < 1 or goal["min_horizon_years"] > goal["max_horizon_years"]:
        raise ConfigError("goal_planning horizon limits must satisfy 1 <= min <= max.")


def _validate(cfg: dict[str, Any]) -> None:
    """Fail clearly on settings that would otherwise break an agent later."""
    market = cfg.get("market_analysis")
    if market is not None:
        _validate_sma_windows(market.get("sma_windows"))
    goal = cfg.get("goal_planning")
    if goal is not None:
        _validate_goal_planning(goal)


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml (and .env) once and cache the result."""
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    _validate(cfg)
    return cfg


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

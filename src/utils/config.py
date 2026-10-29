"""Central configuration loader.

Design pattern: SINGLETON (via functools.lru_cache).
The YAML file is read once; every module that calls get_config() gets
the same dict back instead of re-reading the file.
"""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    """goal_planning: every risk profile needs all scenarios, lowest return first, no negative rates; limits must be consistent."""
    returns = goal["assumptions"]["returns_pct"]
    for profile in goal["risk_profiles"]:
        values = [returns[profile][name] for name in goal["scenarios"]]
        if min(values) < 0:
            raise ConfigError(f"goal_planning returns for {profile!r} must not be negative.")
        if values != sorted(values):
            raise ConfigError(f"goal_planning returns for {profile!r} must not decrease from conservative to optimistic.")
    if goal["assumptions"]["inflation"]["annual_pct"] < 0:
        raise ConfigError("goal_planning inflation must not be negative.")
    if goal["max_months"] < goal["max_horizon_years"] * 12:
        raise ConfigError("goal_planning.max_months must cover max_horizon_years.")
    if goal["min_horizon_years"] < 1 or goal["min_horizon_years"] > goal["max_horizon_years"]:
        raise ConfigError("goal_planning horizon limits must satisfy 1 <= min <= max.")


def _validate_news(news: dict[str, Any]) -> None:
    """news: https base_url on an allowed host, positive limits, a query template for every kind of ticker."""
    parts = urlparse(news["base_url"])
    if parts.scheme != "https" or parts.hostname not in news["allowed_hosts"]:
        raise ConfigError("news.base_url must be an https URL on a host listed in news.allowed_hosts.")
    for key in ("timeout_seconds", "max_response_bytes", "max_items", "max_age_days", "max_query_chars", "title_max_chars"):
        if not news[key] > 0:
            raise ConfigError(f"news.{key} must be greater than zero.")
    if not {"us", "nse", "index"} <= set(news["query_templates"]):
        raise ConfigError("news.query_templates needs us, nse and index.")


TAX_FIGURE_KEYS = (
    "equity_stcg_pct", "equity_ltcg_pct", "equity_ltcg_exemption", "holding_months_equity",
    "holding_months_other", "other_ltcg_pct", "cess_pct", "ppf_pct", "nsc_pct", "ssy_pct",
)


def _validate_tax(tax: dict[str, Any]) -> None:
    """tax: every figure the code reads exists and carries a label, a number, an ISO as_of date, a source and a verify flag."""
    figures = tax["figures"]
    for key in TAX_FIGURE_KEYS:
        if key not in figures:
            raise ConfigError(f"tax.figures.{key} is missing.")
    for key, fig in figures.items():
        value = fig.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"tax.figures.{key}.value must be a number, zero or more.")
        try:
            date.fromisoformat(str(fig.get("as_of")))
        except ValueError as exc:
            raise ConfigError(f"tax.figures.{key}.as_of must be an ISO date (YYYY-MM-DD).") from exc
        if not str(fig.get("source") or "").strip() or not str(fig.get("label") or "").strip():
            raise ConfigError(f"tax.figures.{key} needs a label and a source.")
        if not isinstance(fig.get("verify"), bool):
            raise ConfigError(f"tax.figures.{key}.verify must be true or false.")
        if fig.get("unit") not in tax["units"]:
            raise ConfigError(f"tax.figures.{key}.unit must be one of {sorted(tax['units'])}.")
    for scheme, key in tax["schemes"].items():
        if key not in figures:
            raise ConfigError(f"tax.schemes.{scheme} points at {key!r}, which is not a figure.")


def _validate(cfg: dict[str, Any]) -> None:
    """Fail clearly on settings that would otherwise break an agent later."""
    market = cfg.get("market_analysis")
    if market is not None:
        _validate_sma_windows(market.get("sma_windows"))
    goal = cfg.get("goal_planning")
    if goal is not None:
        _validate_goal_planning(goal)
    news = cfg.get("news")
    if news is not None:
        _validate_news(news)
    tax = cfg.get("tax")
    if tax is not None:
        _validate_tax(tax)


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

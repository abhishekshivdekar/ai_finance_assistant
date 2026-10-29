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


def _validate_rag(rag: dict[str, Any]) -> None:
    """rag.query_expansions (optional): a mapping of non-empty acronym -> non-empty text."""
    expansions = rag.get("query_expansions")
    if expansions is None:
        return
    if not isinstance(expansions, dict):
        raise ConfigError("rag.query_expansions must be a mapping of acronym to words.")
    for acronym, words in expansions.items():
        if not str(acronym).strip() or not isinstance(words, str) or not words.strip():
            raise ConfigError(f"rag.query_expansions.{acronym} needs a non-empty expansion text.")


TAX_TOP_KEYS = (
    "topic", "label", "currency_symbol", "figures", "units", "asset_types", "equity_asset_types", "slab_asset_types", "schemes",
    "input_limits", "stt_assumption", "exclusions", "errors", "needs_input_message",
    "question_needs_input", "no_answer_message", "discarded_message", "no_answer_sentinel", "footer",
    "fallback_commentary", "texts", "system_prompt", "fact_line", "fact_condition", "commentary_prompt",
)
TAX_SUB_KEYS = {
    "input_limits": ("max_amount", "max_months", "max_years", "max_pct"),
    "exclusions": ("capital_gains", "small_savings"),
    "errors": ("not_a_number", "must_be_positive", "must_not_be_negative", "too_large", "whole_number",
               "unknown_choice", "slab_rate_needed", "unknown_calculation"),
    "texts": (
        "commentary_unavailable", "commentary_empty", "commentary_discarded", "loss_note", "exemption_note",
        "approximate_note", "slab_taxed_note", "step_gain", "step_term_long", "step_term_short", "step_term_slab",
        "step_exemption", "step_taxable", "step_tax", "step_cess", "step_total", "step_loss",
        "step_savings_rate", "step_savings_formula", "step_savings_year",
    ),
}


def _validate_tax(tax: dict[str, Any]) -> None:
    """tax: every key the code reads exists, and every figure has a label, a number, a display text, an ISO as_of date, a source and a verify flag."""
    for key in TAX_TOP_KEYS:
        if key not in tax:
            raise ConfigError(f"tax.{key} is missing.")
    for section, keys in TAX_SUB_KEYS.items():
        if not isinstance(tax[section], dict):
            raise ConfigError(f"tax.{section} must be a mapping.")
        for key in keys:
            if key not in tax[section]:
                raise ConfigError(f"tax.{section}.{key} is missing.")
    for section in ("figures", "units", "schemes"):
        if not isinstance(tax[section], dict):
            raise ConfigError(f"tax.{section} must be a mapping.")
    figures = tax["figures"]
    for key in TAX_FIGURE_KEYS:
        if key not in figures:
            raise ConfigError(f"tax.figures.{key} is missing.")
    for key, fig in figures.items():
        if not isinstance(fig, dict):
            raise ConfigError(f"tax.figures.{key} must be a mapping.")
        value = fig.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"tax.figures.{key}.value must be a number, zero or more.")
        try:
            date.fromisoformat(str(fig.get("as_of")))
        except ValueError as exc:
            raise ConfigError(f"tax.figures.{key}.as_of must be an ISO date (YYYY-MM-DD).") from exc
        if not str(fig.get("source") or "").strip() or not str(fig.get("label") or "").strip():
            raise ConfigError(f"tax.figures.{key} needs a label and a source.")
        if not isinstance(fig.get("source_short"), str) or not fig["source_short"].strip():
            raise ConfigError(f"tax.figures.{key}.source_short must be a short non-empty source text.")
        if not str(fig.get("display") or "").strip():
            raise ConfigError(f"tax.figures.{key}.display must be a non-empty text such as '12.5%'.")
        if not isinstance(fig.get("verify"), bool):
            raise ConfigError(f"tax.figures.{key}.verify must be true or false.")
        if fig.get("unit") not in tax["units"]:
            raise ConfigError(f"tax.figures.{key}.unit must be one of {sorted(tax['units'])}.")
    for scheme, key in tax["schemes"].items():
        if key not in figures:
            raise ConfigError(f"tax.schemes.{scheme} points at {key!r}, which is not a figure.")
    for section in ("equity_asset_types", "slab_asset_types"):
        extra = set(tax[section]) - set(tax["asset_types"])
        if extra:
            raise ConfigError(f"tax.{section} has {sorted(extra)}, which are not in tax.asset_types.")


WORKFLOW_MESSAGES = (
    "clarify_low_confidence", "out_of_scope", "advice_static", "fallback", "needs_input_note", "error_note",
    "section_heading", "unsourced_marker",
)
WORKFLOW_REQUIRED_INTENTS = ("clarify", "out_of_scope", "specific_advice")


def _validate_workflow(wf: dict[str, Any]) -> None:
    """workflow: thresholds in range, intent lists consistent, re-route targets real, every message present."""
    confidence = wf["router_min_confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ConfigError("workflow.router_min_confidence must be a number between 0 and 1.")
    for key, minimum in (("max_route_attempts", 0), ("recursion_limit", 1), ("max_agents_per_turn", 1), ("max_history_turns", 1)):
        value = wf[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ConfigError(f"workflow.{key} must be a whole number of at least {minimum}.")
    intents, agents = list(wf["intents"]), list(wf["agent_intents"])
    if not set(WORKFLOW_REQUIRED_INTENTS) <= set(intents):
        raise ConfigError(f"workflow.intents must include {', '.join(WORKFLOW_REQUIRED_INTENTS)}.")
    if not agents or not set(agents) <= set(intents):
        raise ConfigError("workflow.agent_intents must be a non-empty subset of workflow.intents.")
    if not set(wf["hint_nodes"]) <= set(agents):
        raise ConfigError("workflow.hint_nodes must be a subset of workflow.agent_intents.")
    if set(agents) - set(wf["agent_labels"]):
        raise ConfigError("workflow.agent_labels needs a label for every agent in workflow.agent_intents.")
    for source, candidates in wf["reroute"].items():
        if source not in agents:
            raise ConfigError(f"workflow.reroute.{source} is not an agent in workflow.agent_intents.")
        for candidate in candidates:
            if candidate.get("agent") not in agents or not str(candidate.get("needs") or "").strip():
                raise ConfigError(f"workflow.reroute.{source} needs entries with an agent from workflow.agent_intents and a `needs` key.")
    missing = [key for key in WORKFLOW_MESSAGES if not str(wf["messages"].get(key) or "").strip()]
    if missing:
        raise ConfigError(f"workflow.messages is missing {missing}.")


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
    rag = cfg.get("rag")
    if rag is not None:
        _validate_rag(rag)
    tax = cfg.get("tax")
    if tax is not None:
        _validate_tax(tax)
    workflow = cfg.get("workflow")
    if workflow is not None:
        _validate_workflow(workflow)


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

"""Central configuration loader.

Design pattern: SINGLETON (via functools.lru_cache).
The YAML file is read once; every module that calls get_config() gets
the same dict back instead of re-reading the file.
"""
from __future__ import annotations

import os
import re
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


TICKER_LOOKUP_MESSAGES = (
    "ambiguous", "option", "not_found", "unavailable", "invalid", "disclosure_with_others", "disclosure_name_only", "other_listing",
    "not_found_suggestions", "bare_symbol_note",
)


def _validate_ticker_lookup(lookup: dict[str, Any], periods: list[str]) -> None:
    """ticker_lookup: positive limits, a known validation period, string lists, alias notes and every message."""
    for key in ("timeout_seconds", "validation_budget_seconds", "max_query_chars", "min_prefix_chars", "max_results", "max_validated",
                "max_candidates_shown", "max_others_shown", "max_suggestions", "cache_ttl_minutes", "cache_max_entries"):
        value = lookup[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ConfigError(f"ticker_lookup.{key} must be a number greater than zero.")
    if lookup["validation_period"] not in periods:
        raise ConfigError("ticker_lookup.validation_period must be one of market_data.history_periods.")
    for key in ("allowed_quote_types", "etf_quote_types", "etf_words", "exchange_words", "legal_suffixes", "preferred_markets"):
        values = lookup[key]
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v.strip() for v in values):
            raise ConfigError(f"ticker_lookup.{key} must be a non-empty list of non-empty text.")
    suffixes, names = lookup["bare_symbol_suffixes"], lookup["suffix_exchange_names"]
    if not isinstance(suffixes, list) or not all(isinstance(v, str) and v.startswith(".") and names.get(v) for v in suffixes):
        raise ConfigError("ticker_lookup.bare_symbol_suffixes must be a list of suffixes (like .NS), each named in ticker_lookup.suffix_exchange_names.")
    for symbol, note in lookup["alias_notes"].items():
        if not str(symbol).strip() or not isinstance(note, str) or not note.strip():
            raise ConfigError(f"ticker_lookup.alias_notes.{symbol} needs a non-empty note.")
    for key in TICKER_LOOKUP_MESSAGES:
        if not str(lookup["messages"].get(key) or "").strip():
            raise ConfigError(f"ticker_lookup.messages.{key} must not be empty.")


TAX_FIGURE_KEYS = (
    "equity_stcg_pct", "equity_ltcg_pct", "equity_ltcg_exemption", "holding_months_equity",
    "holding_months_other", "other_ltcg_pct", "cess_pct", "ppf_pct", "nsc_pct", "ssy_pct",
)


def _validate_rag(rag: dict[str, Any]) -> None:
    """rag.snippet_max_chars: a positive int; rag.query_expansions (optional): a mapping of non-empty acronym -> non-empty text."""
    limit = rag.get("snippet_max_chars")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ConfigError("rag.snippet_max_chars must be a positive whole number.")
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
    "clarify_low_confidence", "out_of_scope", "advice_static", "fallback", "needs_input_note", "error_note", "no_answer_note",
    "section_heading", "unsourced_marker", "budget_exceeded", "partial_note", "advice_opening",
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
    for key in ("router_history_max_chars", "part_max_chars"):
        chars = wf[key]
        if isinstance(chars, bool) or not isinstance(chars, int) or chars < 1:
            raise ConfigError(f"workflow.{key} must be a positive whole number.")
    for prompt in ("router_prompt", "synthesizer_prompt", "safe_response_prompt"):
        if not str(wf.get(prompt) or "").strip():
            raise ConfigError(f"workflow.{prompt} must not be empty.")
    for pattern in wf["advice_patterns"]:
        try:
            re.compile(str(pattern))
        except re.error as exc:
            raise ConfigError(f"workflow.advice_patterns has an invalid regex {pattern!r}: {exc}") from exc
    banned, opening = wf.get("merge_banned_phrases"), wf.get("advice_opening_pattern")
    if not isinstance(banned, list) or not banned or not all(isinstance(p, str) and p.strip() for p in banned):
        raise ConfigError("workflow.merge_banned_phrases must be a non-empty list of regexes.")
    if not isinstance(opening, str) or not opening.strip():
        raise ConfigError("workflow.advice_opening_pattern must be a non-empty regex.")
    abbreviations = wf.get("sentence_abbreviations")
    if not isinstance(abbreviations, list) or not all(isinstance(a, str) and a.strip() for a in abbreviations):
        raise ConfigError("workflow.sentence_abbreviations must be a list of non-empty words (it may be empty).")
    for pattern in [*banned, opening]:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"workflow has an invalid regex {pattern!r} (merge_banned_phrases or advice_opening_pattern): {exc}") from exc
    aliases = wf["ticker_aliases"]
    if not isinstance(aliases, dict) or any(not str(k).strip() or not isinstance(v, str) or not v.strip() for k, v in aliases.items()):
        raise ConfigError("workflow.ticker_aliases must map a name to a non-empty symbol.")
    missing = [key for key in WORKFLOW_MESSAGES if not str(wf["messages"].get(key) or "").strip()]
    if missing:
        raise ConfigError(f"workflow.messages is missing {missing}.")


UI_TEXT_KEYS = (
    "page_title", "app_title", "app_tagline", "tabs.chat", "tabs.market", "tabs.news",
    "tabs.portfolio", "portfolio.intro", "portfolio.submit", "portfolio.working", "portfolio.rows_required", "portfolio.query_template", "portfolio.allocation_heading", "portfolio.metrics_heading", "portfolio.positions_heading", "portfolio.currency_heading", "portfolio.not_available", "portfolio.columns.ticker", "portfolio.columns.quantity", "portfolio.columns.avg_cost", "portfolio.columns.instrument_type", "portfolio.columns.asset_class", "portfolio.columns.expense_ratio_pct", "portfolio.column_help.ticker", "portfolio.column_help.quantity", "portfolio.column_help.avg_cost", "portfolio.column_help.instrument_type", "portfolio.column_help.asset_class", "portfolio.column_help.expense_ratio_pct", "portfolio.metrics.total_value", "portfolio.metrics.position_count", "portfolio.metrics.largest", "portfolio.metrics.top3", "portfolio.metrics.diversification", "portfolio.metrics.risk", "portfolio.metrics.equity_share", "portfolio.metrics.expense_ratio", "portfolio.positions.ticker", "portfolio.positions.instrument_type", "portfolio.positions.asset_class", "portfolio.positions.quantity", "portfolio.positions.avg_cost", "portfolio.positions.price", "portfolio.positions.value", "portfolio.positions.weight", "portfolio.positions.pnl", "portfolio.positions.pnl_pct",
    "freshness.with_time", "freshness.no_time",
    "market.intro", "market.tickers_label", "market.tickers_placeholder", "market.tickers_help", "market.period_label",
    "market.submit", "market.working", "market.tickers_required", "market.query_template", "market.chart_heading",
    "market.chart_title", "market.chart_y", "market.chart_y_plain", "market.table_heading", "market.not_available",
    "market.columns.ticker", "market.columns.currency", "market.columns.period_return_pct", "market.columns.volatility_pct",
    "market.columns.max_drawdown_pct", "market.columns.trend",
    "news.intro", "news.mode_label", "news.mode_query", "news.mode_ticker", "news.text_label", "news.text_placeholder",
    "news.text_help", "news.submit", "news.working", "news.text_required", "news.query_template_query",
    "news.query_template_ticker", "news.headlines_heading", "news.published", "news.open_article",
    "sidebar.usage_title", "sidebar.usage_tokens", "sidebar.usage_tokens_help", "sidebar.usage_limit", "sidebar.usage_no_limit", "sidebar.usage_cost",
    "sidebar.usage_cost_unknown", "sidebar.new_conversation", "sidebar.new_conversation_help", "sidebar.status_title",
    "sidebar.warmup_running", "sidebar.warmup_ok", "sidebar.warmup_failed",
    "chat.placeholder", "chat.placeholder_busy", "chat.placeholder_stopped", "chat.empty_state", "chat.thinking",
    "chat.sources_title", "chat.source_link",
    "messages.budget_stopped", "messages.empty_answer", "messages.needs_input_hint", "messages.no_answer_hint",
)


def _validate_portfolio(portfolio: dict[str, Any], ui: Any) -> None:
    """portfolio: the table choices equal the Literals of portfolio_models.py, defaults are choices, every choice has a UI label."""
    from typing import get_args

    from src.core.portfolio_types import AssetClass, InstrumentType   # imported here, not at the top: keeps config free of app imports

    for key, literal in (("instrument_types", InstrumentType), ("asset_classes", AssetClass)):
        values = portfolio[key]
        if not isinstance(values, list) or sorted(values) != sorted(get_args(literal)):
            raise ConfigError(f"portfolio.{key} must list exactly {', '.join(get_args(literal))} (the Literal in portfolio_models.py).")
    defaults = portfolio["defaults"]
    if defaults.get("instrument_type") not in portfolio["instrument_types"] or defaults.get("asset_class") not in portfolio["asset_classes"]:
        raise ConfigError("portfolio.defaults must name an instrument type and an asset class from the lists.")
    for key in ("row_problem_message", "resolved_note", "lookup_limit_message"):
        if not str(portfolio.get(key) or "").strip():
            raise ConfigError(f"portfolio.{key} must not be empty.")
    if isinstance(ui, dict) and "portfolio" in ui:
        for labels, choices in (("instrument_labels", portfolio["instrument_types"]), ("asset_class_labels", portfolio["asset_classes"])):
            shown = ui["portfolio"].get(labels) or {}
            if sorted(shown) != sorted(choices) or not all(isinstance(v, str) and v.strip() for v in shown.values()):
                raise ConfigError(f"ui.portfolio.{labels} needs a non-empty label for every choice in portfolio.")


def _validate_ui(ui: Any) -> None:
    """Every key in UI_TEXT_KEYS must be a non-empty string, so a typo fails at load, not in the browser."""
    if not isinstance(ui, dict):
        raise ConfigError("ui must be a mapping of UI texts.")
    for dotted in UI_TEXT_KEYS:
        node: Any = ui
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, str) or not node.strip():
            raise ConfigError(f"ui.{dotted} must be a non-empty text.")


def _validate_llm(llm: Any) -> None:
    """llm.max_session_tokens (optional): null for no limit, else a positive whole number."""
    limit = llm.get("max_session_tokens") if isinstance(llm, dict) else None
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ConfigError("llm.max_session_tokens must be a positive whole number or null.")


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
    portfolio = cfg.get("portfolio")
    if portfolio is not None and "instrument_types" in portfolio:
        _validate_portfolio(portfolio, cfg.get("ui"))
    lookup = cfg.get("ticker_lookup")
    if lookup is not None:
        _validate_ticker_lookup(lookup, cfg["market_data"]["history_periods"])
    rag = cfg.get("rag")
    if rag is not None:
        _validate_rag(rag)
    tax = cfg.get("tax")
    if tax is not None:
        _validate_tax(tax)
    workflow = cfg.get("workflow")
    if workflow is not None:
        _validate_workflow(workflow)
    _validate_llm(cfg.get("llm"))
    if "ui" in cfg:
        _validate_ui(cfg["ui"])


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

"""Turn state["news_params"] into one search phrase: pure, no I/O.

Design pattern: PURE FUNCTIONS. Example: {"ticker": "RELIANCE.NS"} becomes "Reliance
Industries" through config `news.query_overrides`; {"ticker": "INFY.NS"} without an
override becomes "INFY NSE stock"; {"query": "Nifty 50"} is used as typed.

Input: {"ticker": "RELIANCE.NS"} or {"query": "Nifty 50"}, optional "max_items".
If both are given the query wins (it is the more specific, user-typed text).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.market.provider import InvalidSymbolError, validate_news_query, validate_symbol


@dataclass(frozen=True)
class NewsRequest:
    """A resolved request: a search phrase and a headline limit, or a message asking for better input."""

    query: str | None
    max_items: int
    message: str | None = None


def ticker_query(symbol: str, cfg: dict[str, Any]) -> str:
    """Search phrase for a validated ticker: config override, else a template ('<SYMBOL> stock')."""
    override = cfg["query_overrides"].get(symbol)
    if override:
        return str(override)
    kind = "index" if symbol.startswith("^") else "nse" if symbol.endswith(".NS") else "us"
    return cfg["query_templates"][kind].format(symbol=symbol.removeprefix("^").removesuffix(".NS"))


def _limit(raw: Any, cfg: dict[str, Any]) -> int:
    """The requested headline count clamped to 1..news.max_items; anything else means the maximum."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return max(1, min(raw, cfg["max_items"]))
    return int(cfg["max_items"])


def resolve_query(params: Any, cfg: dict[str, Any]) -> NewsRequest:
    """Validate news_params and build the search phrase (never raises)."""
    params = params if isinstance(params, dict) else {}
    limit = _limit(params.get("max_items"), cfg)
    messages = cfg["messages"]
    try:
        if str(params.get("query") or "").strip():
            return NewsRequest(validate_news_query(params["query"]), limit)
        if str(params.get("ticker") or "").strip():
            return NewsRequest(ticker_query(validate_symbol(params["ticker"]), cfg), limit)
    except InvalidSymbolError as exc:
        return NewsRequest(None, limit, messages["invalid_input"].format(detail=exc))
    return NewsRequest(None, limit, messages["needs_input"])

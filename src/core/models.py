"""Shared data models: the AgentResult envelope, payloads and MarketResult.

Payloads are minimal on purpose; later build steps extend them. They are
TypedDicts (plain dicts at runtime), so they pass through LangGraph state and
JSON without conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict


class Source(TypedDict):
    """A cited RAG chunk, shown in the UI "Sources" expander."""

    title: str
    url: str | None
    chunk_id: str
    score: float


class MarketPayload(TypedDict, total=False):
    """Market agent data: quotes (with currency), history and freshness."""

    quotes: list[dict[str, Any]]
    history: list[dict[str, Any]]
    as_of: str
    source: str
    is_stale: bool


class PortfolioPayload(TypedDict, total=False):
    """Portfolio agent data. Values are kept per currency, never summed."""

    weights: dict[str, float]
    total_value_by_currency: dict[str, float]
    top_holding_share: float
    sector_split: dict[str, float]


class GoalPayload(TypedDict, total=False):
    """Goal planning data: scenario projections and required contribution."""

    projections: list[dict[str, Any]]
    required_monthly_contribution: float


class TaxPayload(TypedDict, total=False):
    """Tax education data: inputs, config values used and calculation steps."""

    inputs: dict[str, Any]
    config_used: dict[str, Any]
    steps: list[str]


class QAPayload(TypedDict, total=False):
    """Finance Q&A data: the search query actually used and how many chunks were retrieved."""

    rewritten_query: str
    retrieved: int


class AgentResult(TypedDict):
    """Common envelope every agent returns.

    `answer` is plain text with NO disclaimer; final-answer nodes add it once.
    """

    agent: str
    turn_id: str
    status: Literal["ok", "no_answer", "wrong_agent", "error", "needs_input"]
    answer: str
    sources: list[Source]
    error: str | None
    data: MarketPayload | PortfolioPayload | GoalPayload | TaxPayload | QAPayload | None


@dataclass
class MarketResult:
    """Result of a market-data call, always labelled with freshness.

    A dataclass (not a TypedDict) because the market layer returns it as an
    object and the defaults (`is_stale=False`, `error=None`) are useful.
    """

    data: Any
    currency: str | None
    as_of: datetime | None
    source: str  # live | cache | stale cache | sample data
    is_stale: bool = False
    error: str | None = None

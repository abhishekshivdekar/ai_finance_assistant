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
    """A cited RAG chunk, shown in the UI "Sources" expander.

    `snippet` is clean display text cut from the chunk in code (src/rag/snippet.py), never written by a model.
    """

    title: str
    url: str | None
    chunk_id: str
    score: float
    snippet: str


class MarketPayload(TypedDict, total=False):
    """Market agent data: trend metrics, price history and freshness.

    `metrics` is MarketMetrics as a dict (one entry per ticker, each in its own
    currency), `comparison` holds the table rows, `source` is the least live
    source among the analysed tickers, `freshness_texts` the price sentence part of every ticker whose price is not
    live (a merge must keep them) and `warnings` lists everything left out.
    """

    history: list[dict[str, Any]]
    as_of: str
    source: str
    is_stale: bool
    freshness_texts: list[str]
    metrics: dict[str, Any]
    comparison: list[dict[str, Any]]
    warnings: list[str]


class PortfolioPayload(TypedDict, total=False):
    """Portfolio agent data: PortfolioMetrics as a plain dict, plus warnings.

    Values are kept per currency and never summed across currencies.
    """

    metrics: dict[str, Any]
    warnings: list[str]


class GoalPayload(TypedDict, total=False):
    """Goal planning data: scenario results, chart rows and the assumptions behind them.

    `projections` are flat rows {scenario, year, value, value_todays_money} for a line
    chart. `assumptions` (illustrative label, as_of, source, rates, timing notes) is
    always filled from config, never by the LLM. Per-scenario dicts are keyed by name.
    """

    inputs: dict[str, Any]
    target_amount: float
    scenarios: list[dict[str, Any]]
    projections: list[dict[str, Any]]
    required_monthly_contribution: dict[str, float]
    required_labels: dict[str, str]
    months_to_goal: dict[str, int | None]
    shortfall: dict[str, float]
    inflation: dict[str, Any]
    assumptions: dict[str, Any]
    warnings: list[str]


class NewsPayload(TypedDict, total=False):
    """News agent data: the headlines actually fetched, their freshness and the summary's citations.

    `items` are {number, title, publisher, published (ISO, UTC), link}; they are always the
    fetched data, never LLM output. `summary_citations` lists the headline numbers the summary
    cites (empty when the fixed fallback line is shown).
    """

    query: str
    items: list[dict[str, Any]]
    as_of: str
    source: str
    is_stale: bool
    warnings: list[str]
    summary_citations: list[int]


class TaxPayload(TypedDict, total=False):
    """Tax education data for both paths (architecture 2.6).

    Calculator path: `calculation`, `inputs`, `results`, `steps`, `notes`, `assumptions`.
    Question path: `facts` (every numbered config fact shown to the LLM) and `fact_citations`.
    Both: `config_used` (label, value, unit, as_of, source, verify per figure), `exclusions`, `label`
    ("illustrative, verify before relying") and `warnings`, always filled from config in code, never by the LLM.
    """

    path: str
    calculation: str
    inputs: dict[str, Any]
    results: dict[str, Any]
    steps: list[str]
    notes: list[str]
    assumptions: list[str]
    config_used: dict[str, Any]
    exclusions: list[str]
    label: str
    facts: list[dict[str, Any]]
    fact_citations: list[int]
    warnings: list[str]


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
    data: MarketPayload | PortfolioPayload | GoalPayload | NewsPayload | TaxPayload | QAPayload | None


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

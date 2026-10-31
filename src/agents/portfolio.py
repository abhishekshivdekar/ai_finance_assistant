"""Portfolio agent: deterministic metrics, plus one LLM call for a plain-language summary.

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the llm and the market provider are constructor arguments
  with defaults from the existing factories. Example: a test passes
  PortfolioAgent(llm=FakeLLM(), provider=FakeProvider()) and needs no network.

Principle: ALL numbers come from pure Python (src/core/portfolio_metrics.py). The
LLM sees only the already-computed metrics as JSON and writes words; a number
guard rejects any summary that contains a number that was not in that JSON, and
the agent then uses a fixed fallback text. The summary can never fail the request.

A row whose ticker is a company name (not a symbol) goes to the injected TickerResolver first; if any such row needs the user
(ambiguous, not found, lookup down) the answer is needs_input listing ALL those rows and nothing is analysed, because a
portfolio missing a holding would show wrong weights.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.market_metrics import worst_source
from src.core.models import AgentResult, PortfolioPayload
from src.core.number_guard import collect_numbers, numbers_in
from src.core.portfolio_metrics import build_metrics, summary_input
from src.core.portfolio_models import PortfolioMetrics
from src.core.portfolio_valuation import parse_holdings, value_holdings
from src.data.market import get_market_provider
from src.data.market.provider import InvalidSymbolError, MarketDataProvider, validate_symbol
from src.data.tickers.resolver import TickerResolver, build_resolver
from src.utils.config import get_config
from src.workflow.state import GraphState


class PortfolioAgent(BaseAgent):
    """Explains a portfolio: allocation, concentration, diversification and risk (educational)."""

    name = "portfolio"
    llm_role = "portfolio_summary"

    def __init__(self, llm: Any = None, provider: MarketDataProvider | None = None, resolver: TickerResolver | None = None) -> None:
        """Defaults: llm from role portfolio_summary, provider from get_market_provider(), resolver built on that provider."""
        super().__init__(llm=llm)
        self.provider = provider if provider is not None else get_market_provider()
        self.resolver = resolver if resolver is not None else build_resolver(self.provider)

    def _result(self, status: str, answer: str, payload: PortfolioPayload) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=[], error=None, data=payload,
        )

    def _summarise(self, metrics: PortfolioMetrics) -> tuple[str, str | None]:
        """(summary, warning). Falls back to the fixed text on any failure or a bad number."""
        cfg = get_config()["portfolio"]
        fallback = cfg["summary_fallback"].strip()
        payload = summary_input(metrics)
        try:
            reply = self.llm.invoke(cfg["summary_prompt"].format(metrics_json=json.dumps(payload, indent=2)))
            text = str(getattr(reply, "content", reply)).strip()
        except Exception:
            self.log.warning("Portfolio summary failed; using the fallback text", exc_info=True)
            return fallback, "The plain-language summary was unavailable, so a standard description is shown."
        if not text:
            return fallback, "The plain-language summary was empty, so a standard description is shown."
        invented = numbers_in(text) - collect_numbers(payload)
        if invented:
            self.log.warning("Summary contains numbers not in the metrics: %s", sorted(invented))
            return fallback, "The plain-language summary was discarded because it contained numbers that were not calculated."
        return text, None

    @staticmethod
    def _typed_name(row: Any) -> str:
        """The ticker text of a row when it is a company name (not a symbol), else "" (symbols and odd rows are parse_holdings' job)."""
        ticker = row.get("ticker") if isinstance(row, dict) else None
        text = ticker.strip() if isinstance(ticker, str) else ""
        try:
            validate_symbol(text)
        except InvalidSymbolError:
            return text
        return ""

    def _resolve_rows(self, rows: list[Any], cfg: dict[str, Any]) -> tuple[list[Any], list[str], list[str], list[str]]:
        """(rows with names replaced by symbols, notes, problems, warnings).

        At most `max_holdings` names are looked up (more are left out with a warning), concurrently with at most
        `max_workers` threads, each lookup under the resolver's own time limits. A row that needs the user is a
        problem line naming the row by the text typed; every problem row is reported, none hides another.
        """
        named = [(index, self._typed_name(row)) for index, row in enumerate(rows)]
        named = [(index, typed) for index, typed in named if typed]
        looked, extra = named[: cfg["max_holdings"]], named[cfg["max_holdings"]:]
        results: list[Any] = []
        if looked:
            with ThreadPoolExecutor(max_workers=max(1, min(int(cfg["max_workers"]), len(looked)))) as pool:
                results = list(pool.map(lambda item: self.resolver.resolve(item[1]), looked))
        out, notes, problems = list(rows), [], []
        warnings = [cfg["lookup_limit_message"].format(limit=cfg["max_holdings"], typed=typed) for _, typed in extra]
        dropped = {index for index, _ in extra}
        for (index, typed), found in zip(looked, results):
            if found.status == "resolved":
                out[index] = {**rows[index], "ticker": found.symbols[0]}
                notes.append(found.note or cfg["resolved_note"].format(typed=typed, symbol=found.symbols[0]))
            else:
                problems.append(cfg["row_problem_message"].format(typed=typed, message=found.message))
        return [row for index, row in enumerate(out) if index not in dropped], list(dict.fromkeys(notes)), problems, warnings

    def _execute(self, state: GraphState) -> AgentResult:
        """Resolve names, validate, value, compute metrics, then summarise."""
        cfg = get_config()["portfolio"]
        needs_input = cfg["needs_input_message"].strip()
        rows, notes, problems, warnings = self._resolve_rows(state.get("portfolio") or [], cfg)
        if problems:
            return self._result("needs_input", "\n\n".join(problems), {"warnings": warnings})
        holdings, parse_warnings = parse_holdings(rows, cfg["max_holdings"])
        warnings += parse_warnings
        if not holdings:
            return self._result("needs_input", needs_input, {"warnings": warnings})

        positions, price_warnings = value_holdings(holdings, self.provider)
        warnings += price_warnings
        if not positions:
            return self._result("no_answer", cfg["unpriced_message"].strip(), {"warnings": warnings})

        metrics = build_metrics(positions, warnings, cfg)
        summary, note = self._summarise(metrics)
        all_warnings = metrics.warnings + ([note] if note else [])
        payload: PortfolioPayload = {
            "metrics": metrics.model_dump(mode="json"), "warnings": all_warnings,
            "as_of": metrics.as_of.isoformat() if metrics.as_of else "",
            "source": worst_source([p.price_source for p in metrics.positions]),
            "is_stale": metrics.uses_stale_or_sample,
        }
        if notes:
            payload["lookup_notes"] = notes
            summary = f"{summary}\n\n{' '.join(notes)}"
        return self._result("ok", summary, payload)

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
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.models import AgentResult, PortfolioPayload
from src.core.number_guard import collect_numbers, numbers_in
from src.core.portfolio_metrics import build_metrics, summary_input
from src.core.portfolio_models import PortfolioMetrics
from src.core.portfolio_valuation import parse_holdings, value_holdings
from src.data.market import get_market_provider
from src.data.market.provider import MarketDataProvider
from src.utils.config import get_config
from src.workflow.state import GraphState


class PortfolioAgent(BaseAgent):
    """Explains a portfolio: allocation, concentration, diversification and risk (educational)."""

    name = "portfolio"
    llm_role = "portfolio_summary"

    def __init__(self, llm: Any = None, provider: MarketDataProvider | None = None) -> None:
        """Defaults: llm from role portfolio_summary, provider from get_market_provider()."""
        super().__init__(llm=llm)
        self.provider = provider if provider is not None else get_market_provider()

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

    def _execute(self, state: GraphState) -> AgentResult:
        """Validate, value, compute metrics, then summarise."""
        cfg = get_config()["portfolio"]
        needs_input = cfg["needs_input_message"].strip()
        rows = state.get("portfolio") or []
        holdings, warnings = parse_holdings(rows, cfg["max_holdings"])
        if not holdings:
            return self._result("needs_input", needs_input, {"warnings": warnings})

        positions, price_warnings = value_holdings(holdings, self.provider)
        warnings += price_warnings
        if not positions:
            return self._result("no_answer", cfg["unpriced_message"].strip(), {"warnings": warnings})

        metrics = build_metrics(positions, warnings, cfg)
        summary, note = self._summarise(metrics)
        all_warnings = metrics.warnings + ([note] if note else [])
        payload: PortfolioPayload = {"metrics": metrics.model_dump(mode="json"), "warnings": all_warnings}
        return self._result("ok", summary, payload)

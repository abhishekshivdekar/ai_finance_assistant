"""News agent: recent headlines for a ticker or search phrase, plus one guarded LLM summary (architecture 2.5).

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the llm and the market provider are constructor arguments
  with defaults from the existing factories. Example: a test passes
  NewsAgent(llm=FakeLLM(), provider=FakeProvider()) and needs no network.

Principle: the headline list in the payload is ALWAYS the fetched data, whatever the
LLM does. The LLM sees only the numbered headlines as JSON (untrusted DATA, never
instructions) and writes a short themed summary. The summary is accepted only if
(1) it cites at least one existing headline number (invalid citations are dropped,
not renumbered, because the reader sees the full numbered list) and (2) every
number in it appears in a headline (the number guard shared with the other agents).
Otherwise a fixed fallback line from config is shown, with a warning, status ok.

Input: state["news_params"] = {"ticker": "RELIANCE.NS"} or {"query": "Nifty 50"}, optional "max_items".
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.citations import drop_invalid_citations, strip_citations
from src.core.models import AgentResult, NewsPayload
from src.core.news_query import resolve_query
from src.core.portfolio_metrics import collect_numbers, numbers_in
from src.data.market import get_market_provider
from src.data.market.provider import MarketDataProvider
from src.utils.config import get_config
from src.workflow.state import GraphState


class NewsAgent(BaseAgent):
    """Summarises recent headlines (educational, no predictions or advice)."""

    name = "news"
    llm_role = "news_summary"

    def __init__(self, llm: Any = None, provider: MarketDataProvider | None = None) -> None:
        """Defaults: llm from role news_summary, provider from get_market_provider()."""
        super().__init__(llm=llm)
        self.provider = provider if provider is not None else get_market_provider()

    def _result(self, status: str, answer: str, payload: NewsPayload) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=[], error=None, data=payload,
        )

    def _summarise(self, items: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, list[int], str | None]:
        """(summary, cited headline numbers, warning). Falls back to the fixed line on any failure."""
        fallback, messages = cfg["fallback_summary"].strip(), cfg["messages"]
        headlines = [{"n": i["number"], "title": i["title"], "publisher": i["publisher"]} for i in items]
        prompt = cfg["prompt"].format(headlines_json=json.dumps(headlines, indent=2, ensure_ascii=False))
        try:
            reply = self.llm.invoke(prompt)
            text = str(getattr(reply, "content", reply)).strip()
        except Exception:
            self.log.warning("News summary failed; using the fallback line", exc_info=True)
            return fallback, [], messages["summary_unavailable"]
        if not text:
            return fallback, [], messages["summary_empty"]
        text, cited = drop_invalid_citations(text, len(items))
        if not cited:
            return fallback, [], messages["summary_no_citation"]
        allowed = collect_numbers([[h["title"], h["publisher"]] for h in headlines])
        invented = numbers_in(strip_citations(text)) - allowed
        if invented:
            self.log.warning("Summary contains numbers not in the headlines: %s", sorted(invented))
            return fallback, [], messages["summary_discarded"]
        return text, cited, None

    def _execute(self, state: GraphState) -> AgentResult:
        """Resolve the query, fetch headlines through the provider chain, then summarise."""
        cfg = get_config()["news"]
        request = resolve_query(state.get("news_params"), cfg)
        if request.query is None:
            return self._result("needs_input", str(request.message), {"query": "", "items": [], "warnings": []})

        found = self.provider.get_news(request.query)
        if found.error:
            self.log.warning("News unavailable for %r: %s", request.query, found.error)
            return self._result("no_answer", cfg["messages"]["unavailable"], {"query": request.query, "items": [], "warnings": []})
        items = [{"number": n, **item} for n, item in enumerate((found.data or [])[: request.max_items], start=1)]
        if not items:
            self.log.info("No headlines for %r; skipping the summary call", request.query)
            return self._result("no_answer", cfg["messages"]["nothing_found"], {"query": request.query, "items": [], "warnings": []})

        summary, cited, note = self._summarise(items, cfg)
        stale = bool(found.is_stale)
        warnings = ([cfg["messages"]["stale"].format(source=found.source)] if stale else []) + ([note] if note else [])
        payload: NewsPayload = {
            "query": request.query, "items": items,
            "as_of": found.as_of.isoformat() if found.as_of else "",
            "source": found.source, "is_stale": stale,
            "warnings": warnings, "summary_citations": cited,
        }
        return self._result("ok", summary, payload)

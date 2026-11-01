"""Market agent: deterministic trend metrics, plus one LLM call for a plain-language summary.

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the llm and the market provider are constructor arguments
  with defaults from the existing factories. Example: a test passes
  MarketAgent(llm=FakeLLM(), provider=FakeProvider()) and needs no network.

Principle: ALL numbers come from pure Python (src/core/market_metrics.py). The
LLM sees only the already-computed metrics as JSON and writes words; the number
guard shared with the Portfolio agent rejects any summary that contains a number
that was not in that JSON. Only the summary step falls back (to a fixed text);
any other failure is turned into an error result by BaseAgent.run().

Input: state["market_params"] = {"tickers": ["AAPL", "TCS.NS", "Suzlon"], "period": "6mo"}. An entry that is not a symbol is
a company name: the injected TickerResolver turns it into a price-checked symbol, or the agent asks the user (needs_input)
for the symbol. A symbol that has no prices is looked up once as a name ("apple" reads as the symbol APPLE).
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.market_history import fetch_histories, parse_request
from src.core.market_metrics import (
    LIVE_SOURCES, build_metrics, comparison_rows, history_points, price_series, price_text, summary_input, ticker_metrics, worst_source,
)
from src.core.market_models import MarketMetrics, TickerMetrics
from src.core.format_guard import format_warning, mentions_data_format
from src.core.models import AgentResult, MarketPayload
from src.core.number_guard import collect_numbers, numbers_in
from src.core.portfolio_valuation import NOT_LIVE_SOURCES
from src.data.market import get_market_provider
from src.data.market.provider import MarketDataProvider
from src.data.tickers.entries import resolve_entries
from src.data.tickers.resolver import TickerResolver, build_resolver
from src.utils.config import get_config
from src.workflow.state import GraphState


class MarketAgent(BaseAgent):
    """Explains recent price behaviour of one or more tickers (educational, no predictions)."""

    name = "market"
    llm_role = "market_summary"

    def __init__(self, llm: Any = None, provider: MarketDataProvider | None = None, resolver: TickerResolver | None = None) -> None:
        """Defaults: llm from role market_summary, provider from get_market_provider(), resolver built on that provider."""
        super().__init__(llm=llm)
        self.provider = provider if provider is not None else get_market_provider()
        self.resolver = resolver if resolver is not None else build_resolver(self.provider)

    def _result(self, status: str, answer: str, payload: MarketPayload) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=[], error=None, data=payload,
        )

    def _summarise(self, metrics: MarketMetrics, cfg: dict[str, Any]) -> tuple[str, str | None]:
        """(summary, warning). Falls back to the fixed text on any failure or a bad number."""
        fallback = cfg["summary_fallback"].strip()
        payload = summary_input(metrics, cfg)
        try:
            reply = self.llm.invoke(cfg["summary_prompt"].format(metrics_json=json.dumps(payload, indent=2)))
            text = str(getattr(reply, "content", reply)).strip()
        except Exception:
            self.log.warning("Market summary failed; using the fallback text", exc_info=True)
            return fallback, "The plain-language summary was unavailable, so a standard description is shown."
        if not text:
            return fallback, "The plain-language summary was empty, so a standard description is shown."
        if mentions_data_format(text):
            return fallback, format_warning()
        invented = numbers_in(text) - collect_numbers(payload)
        if invented:
            self.log.warning("Summary contains numbers not in the metrics: %s", sorted(invented))
            return fallback, "The plain-language summary was discarded because it contained numbers that were not calculated."
        return text, None

    def _execute(self, state: GraphState) -> AgentResult:
        """Validate the request, fetch histories, compute metrics, then summarise."""
        cfg = get_config()["market_analysis"]
        params = state.get("market_params")
        params = params if isinstance(params, dict) else {}
        entries = resolve_entries(params.get("tickers"), self.resolver, cfg["max_tickers"])
        tickers, period, warnings = parse_request({**params, "tickers": entries.tickers}, cfg)
        warnings = entries.warnings + warnings
        if not tickers:
            asked = "\n\n".join(entries.asks) or cfg["needs_input_message"].strip()
            return self._result("needs_input", asked, {"warnings": warnings})

        results = fetch_histories(tickers, self.provider, cfg)
        lookup_notes, asks = list(entries.notes), list(entries.asks)
        for index, ticker in enumerate(tickers):                 # a symbol-looking word with no prices may be a name
            if (results[index].error or not results[index].currency) and ticker in entries.names:
                found = self.resolver.resolve(entries.names[ticker])
                if found.status == "resolved" and found.symbols[0] not in tickers:
                    tickers[index] = found.symbols[0]
                    results[index] = fetch_histories([found.symbols[0]], self.provider, cfg)[0]
                    lookup_notes += [found.note] if found.note else []
                elif found.status in ("ambiguous", "unavailable"):
                    asks.append(found.message)
        warnings += asks

        analysed: list[TickerMetrics] = []
        history: list[dict[str, Any]] = []
        for ticker, result in zip(tickers, results):
            series = [] if result.error or not result.currency else price_series(result.data)
            stale = bool(result.is_stale or result.source in NOT_LIVE_SOURCES)
            found, notes = ticker_metrics(
                ticker, series, result.currency or "", period, cfg,
                price_source=result.source, price_stale=stale, as_of=result.as_of,
            )
            if found is None:
                self.log.warning("No usable history for %s: %s", ticker, result.error)
                warnings.append(f"{ticker} was left out: no price history is available right now.")
                continue
            warnings += notes
            analysed.append(found)
            history += history_points(ticker, series, cfg["period_days"][period])

        if not analysed:
            return self._result("no_answer", cfg["unavailable_message"].strip(), {"warnings": warnings})

        metrics = build_metrics(analysed, warnings, period)
        summary, note = self._summarise(metrics, cfg)
        all_warnings = metrics.warnings + ([note] if note else [])
        payload: MarketPayload = {
            "metrics": metrics.model_dump(mode="json"),
            "comparison": comparison_rows(metrics) if len(analysed) > 1 else [],
            "history": history,
            "as_of": metrics.as_of.isoformat() if metrics.as_of else "",
            "source": worst_source([t.price_source for t in metrics.tickers]),
            "is_stale": metrics.uses_stale_or_sample,
            "freshness_texts": [price_text(t, cfg) for t in metrics.tickers if t.price_source not in LIVE_SOURCES],
            "warnings": all_warnings,
        }
        if lookup_notes:
            payload["lookup_notes"] = lookup_notes
            summary = f"{summary}\n\n{' '.join(lookup_notes)}"
        return self._result("ok", summary, payload)

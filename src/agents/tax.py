"""Tax Education agent: grounded tax Q&A and deterministic illustrative calculators (architecture 2.6).

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the llm and the retrieve function are constructor arguments with
  defaults from the existing factories. Example: a test passes
  TaxAgent(llm=FakeLLM(), retrieve_fn=fake_retrieve) and needs no network or index.

Two paths, chosen by the state:
(a) QUESTION: no state["tax_params"] "calculation" -> retrieval locked to the config topic
    (never the Chat dropdown), numbered chunks [1].. plus numbered config facts [F1]..; the
    answer is kept only if it cites at least one real chunk or fact and every number in it
    appears in the retrieved chunks, the question or the display text of a figure (fact ids,
    as_of dates and source strings are NOT a whitelist). A figure counts as present when the
    answer cites it or states its number; the footer, exclusions and config_used use that union
    (a coincidental number match can add an extra footer; accepted). Otherwise: no_answer.
(b) CALCULATOR: state["tax_params"] = {"calculation": "capital_gains" | "small_savings_growth", ...}
    -> pure functions in src/core/tax_metrics.py; one optional LLM call writes plain-language
    commentary from the computed JSON only, with a fixed fallback line.

Principle: the payload (inputs, config values used with as_of and source, steps, exclusions and the
"illustrative, verify before relying" label) is built from config and code, so it is complete
whatever the LLM does. Rates and limits are never in code or prompts: they come from config facts.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.core.citations import apply_citations, drop_invalid_fact_citations, strip_citations, strip_fact_citations
from src.core.models import AgentResult, TaxPayload
from src.core.portfolio_metrics import collect_numbers, numbers_in
from src.core.tax_inputs import CALCULATIONS, parse_capital_gains, parse_small_savings
from src.core.tax_metrics import (
    exclusions_for,
    build_facts,
    capital_gains,
    fact_lines,
    figure_entry,
    footer,
    quoted_fact_keys,
    small_savings_growth,
    summary_input,
)
from src.core.tax_models import TaxCalculation
from src.rag.retriever import RetrievedChunk, format_context, merge_retrieval, retrieve, to_sources
from src.utils.config import get_config
from src.workflow.state import GraphState


class TaxAgent(BaseAgent):
    """Explains Indian tax concepts and runs illustrative calculators (educational, no personal advice)."""

    name = "tax_education"
    llm_role = "tax_summary"

    def __init__(self, llm: Any = None, retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve) -> None:
        """Defaults: llm from role tax_summary, retrieve_fn from the shared retriever."""
        super().__init__(llm=llm)
        self.retrieve_fn = retrieve_fn

    def _result(self, status: str, answer: str, payload: TaxPayload, sources: list[Any] | None = None) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=sources or [], error=None, data=payload,
        )

    def _base_payload(self, path: str, cfg: dict[str, Any], exclusions: list[str]) -> TaxPayload:
        """The parts of the payload that always come from config."""
        return {"path": path, "label": cfg["label"], "exclusions": exclusions, "warnings": []}

    # ---- question path ----

    def _answer_question(self, question: str, cfg: dict[str, Any]) -> AgentResult:
        """Retrieve from the locked topic, answer from chunks and facts, then check citations and numbers."""
        payload = self._base_payload("question", cfg, [])
        if not question:
            return self._result("needs_input", cfg["question_needs_input"], payload)
        chunks = merge_retrieval(self.retrieve_fn, [question], cfg["topic"])
        facts = build_facts(cfg)
        payload["facts"] = facts
        no_answer = self._result("no_answer", " ".join(cfg["no_answer_message"].split()), payload)
        if not chunks:
            self.log.info("No chunks retrieved; skipping the answer call")
            return no_answer

        lines = fact_lines(facts, cfg)
        reply = self.llm.invoke([
            SystemMessage(content=cfg["system_prompt"].format(stt_assumption=cfg["stt_assumption"])),
            HumanMessage(content=f"Context:\n{format_context(chunks)}\n\nFacts:\n{lines}\n\nQuestion: {question}"),
        ])
        text = str(getattr(reply, "content", reply)).strip()
        if text.strip("`\"'* .").upper() == cfg["no_answer_sentinel"]:
            self.log.info("Model replied %s", cfg["no_answer_sentinel"])
            return no_answer

        text, fact_cited = drop_invalid_fact_citations(text, len(facts))
        answer, cited_chunks = apply_citations(text, chunks)
        discarded = self._result("no_answer", " ".join(cfg["discarded_message"].split()), payload)
        if not fact_cited and not cited_chunks:
            self.log.warning("Answer cites no chunk or fact; discarding it")
            return discarded
        allowed = collect_numbers([[c.text, c.title] for c in chunks] + [[f["display"] for f in facts], question])
        bare = strip_fact_citations(strip_citations(answer))
        invented = numbers_in(bare) - allowed
        if invented:
            self.log.warning("Answer contains numbers not in the chunks, facts or question: %s", sorted(invented))
            return discarded

        # A figure is present when the answer cites it ([Fn]) OR states its number. The footer, the
        # exclusions (with stt_condition) and config_used all use that union; a coincidental number
        # match can add an extra footer, which is accepted.
        keys = list(dict.fromkeys([facts[n - 1]["key"] for n in fact_cited] + quoted_fact_keys(bare, facts)))
        used = {key: figure_entry(cfg, key) for key in keys}
        if used:
            answer = f"{answer}\n\n{footer(used, cfg)}"
            payload["exclusions"] = exclusions_for("capital_gains", cfg, equity=any(e["stt_condition"] for e in used.values()))
        payload["fact_citations"] = fact_cited
        payload["config_used"] = used
        return self._result("ok", answer, payload, to_sources(cited_chunks))

    # ---- calculator path ----

    def _commentary(self, calc: TaxCalculation, cfg: dict[str, Any]) -> tuple[str, str | None]:
        """(commentary, warning). Falls back to the fixed text on any failure or an invented number.

        The number whitelist is the inputs, the computed results and the rates used. Dates and
        sources are shown to nobody in the prompt, so an invented year is rejected too.
        """
        fallback, warn = " ".join(cfg["fallback_commentary"].split()), cfg["texts"]
        payload = summary_input(calc, cfg)
        try:
            reply = self.llm.invoke(cfg["commentary_prompt"].format(results_json=json.dumps(payload, indent=2, ensure_ascii=False)))
            text = str(getattr(reply, "content", reply)).strip()
        except Exception:
            self.log.warning("Tax commentary failed; using the fallback text", exc_info=True)
            return fallback, warn["commentary_unavailable"]
        if not text:
            return fallback, warn["commentary_empty"]
        allowed = collect_numbers({k: payload[k] for k in ("inputs", "results", "rates")})
        invented = numbers_in(text) - allowed
        if invented:
            self.log.warning("Commentary contains numbers not in the results: %s", sorted(invented))
            return fallback, warn["commentary_discarded"]
        return text, None

    def _calculate(self, params: dict[str, Any], cfg: dict[str, Any]) -> AgentResult:
        """Validate the request, run the calculator, then write the commentary."""
        name = params.get("calculation")
        payload = self._base_payload("calculation", cfg, [])
        if name == "capital_gains":
            payload["exclusions"] = exclusions_for("capital_gains", cfg, equity=params.get("asset_type") in cfg["equity_asset_types"])
        elif name == "small_savings_growth":
            payload["exclusions"] = exclusions_for("small_savings", cfg)
        if name not in CALCULATIONS:
            errors = [cfg["errors"]["unknown_calculation"].format(choices=", ".join(CALCULATIONS))]
            inputs = None
        elif name == "capital_gains":
            inputs, errors = parse_capital_gains(params, cfg)
        else:
            inputs, errors = parse_small_savings(params, cfg)
        if inputs is None:
            payload["warnings"] = errors
            message = "\n".join([" ".join(cfg["needs_input_message"].split()), *[f"- {e}" for e in errors]])
            return self._result("needs_input", message, payload)

        calc = capital_gains(inputs, cfg) if name == "capital_gains" else small_savings_growth(inputs, cfg)  # type: ignore[arg-type]
        commentary, note = self._commentary(calc, cfg)
        payload.update({
            "calculation": calc.calculation, "inputs": calc.inputs, "results": calc.results, "steps": calc.steps,
            "notes": calc.notes, "assumptions": calc.assumptions, "exclusions": calc.exclusions,
            "config_used": calc.config_used,
            "warnings": [note] if note else [],
        })
        return self._result("ok", f"{commentary}\n\n{footer(calc.config_used, cfg)}", payload)

    def _execute(self, state: GraphState) -> AgentResult:
        """A tax_params request with a "calculation" runs a calculator; anything else is a question."""
        cfg = get_config()["tax"]
        params = state.get("tax_params")
        if isinstance(params, dict) and params.get("calculation") is not None:
            return self._calculate(params, cfg)
        return self._answer_question((state.get("query") or "").strip(), cfg)

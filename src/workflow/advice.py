"""Educational framing for an advice request (architecture 4.3, `safe_response`).

Design pattern: GUARDED GENERATION. Retrieval supplies the only facts, the prompt forbids any recommendation, and code
checks the reply before it is shown. Example: "Should I buy Tesla?" gets a short explanation of how people weigh risk,
time horizon and diversification, with [1] citations from the knowledge base; if the model recommends a purchase,
the output guard replaces the text, and with no chunks or any failed check the static advice text is used instead.

The user's message reaches the prompt as a JSON data block, the retrieved chunks as a delimited context block.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.citations import apply_citations, strip_citations
from src.core.models import Source
from src.core.number_guard import collect_numbers, numbers_in
from src.core.output_guard import guard_answer
from src.rag.retriever import RetrievedChunk, format_context, merge_retrieval, to_sources
from src.utils.logger import get_logger
from src.workflow.routing import workflow_cfg
from src.workflow.state import GraphState

log = get_logger(__name__)


def educational_answer(
    state: GraphState, llm: Any, retrieve_fn: Callable[..., list[RetrievedChunk]],
) -> tuple[str, list[Source]] | None:
    """(answer, sources) for an advice request, or None when the static text should be used.

    None when: no chunks, the model call fails, the reply cites no retrieved chunk, or it contains a number that is
    not in the chunks or the question. The output guard (reword once, then a safe message) always runs; a reworded
    text must pass the number check again, and a replaced text carries no sources.
    """
    cfg = workflow_cfg()
    query = state.get("query") or ""
    chunks = merge_retrieval(retrieve_fn, [query], state.get("topic_filter"))
    if not chunks:
        log.info("No chunks for the advice request; using the static text")
        return None
    data = json.dumps({"query": query}, ensure_ascii=False).replace("<", chr(92) + "u003c")
    messages = [
        SystemMessage(content=cfg["safe_response_prompt"]),
        HumanMessage(content=f"<context>\n{format_context(chunks)}\n</context>\n<user_input>\n{data}\n</user_input>"),
    ]
    try:
        reply = llm.invoke(messages)
        text = str(getattr(reply, "content", reply)).strip()
    except Exception:
        log.warning("Educational answer failed; using the static text", exc_info=True)
        return None
    answer, cited = apply_citations(text, chunks)
    allowed = collect_numbers([[c.text, c.title] for c in chunks] + [query])
    if not cited or numbers_in(strip_citations(answer)) - allowed:
        log.warning("Educational answer cites nothing or has an unsupported number; using the static text")
        return None
    guarded = guard_answer(answer, llm)
    if guarded == answer:
        return answer, to_sources(cited)
    if numbers_in(guarded) - allowed:
        return None
    return guarded, []

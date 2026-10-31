"""Educational framing for an advice request (architecture 4.3, `safe_response`).

Design pattern: GUARDED GENERATION. Retrieval supplies the only facts, the prompt forbids any recommendation, and code
checks the reply before it is shown. Example: "Should I buy Tesla?" gets a short explanation of how people weigh risk,
time horizon and diversification, with [1] citations from the knowledge base; if the model recommends a purchase,
the output guard replaces the text, and with no chunks or any failed check the static advice text is used instead.

The user's message reaches the prompt as a JSON data block, the retrieved chunks as a delimited context block.
"""
from __future__ import annotations

import json
import re
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

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str, abbreviations: list[str]) -> str:
    """The first sentence of the text; a full stop after an abbreviation or an initialism does not end it.

    Example: "Rs. 5 lakh is common. Next." -> "Rs. 5 lakh is common."; "Funds in the U.S. matter, e.g. index funds. More."
    -> "Funds in the U.S. matter, e.g. index funds."; "It rose 3.5 percent. More." splits after "percent." only (a
    decimal point has no space after it). `abbreviations` (`workflow.sentence_abbreviations`) are compared in
    lower case without the final dot; a word with a dot inside ("U.S.", "e.g.", "i.e.") is always an abbreviation.
    """
    text = text.strip()
    known = {a.casefold() for a in abbreviations}
    for boundary in _SENTENCE_END.finditer(text):
        word = text[: boundary.start()].rsplit(None, 1)[-1]
        if word.endswith(".") and (word[:-1].casefold() in known or "." in word[:-1]):
            continue
        return text[: boundary.start()]
    return text


def with_advice_opening(text: str, cfg: dict[str, Any]) -> str:
    """The text, starting with `messages.advice_opening` unless its first sentence already refuses the request.

    Design pattern: GUARD. The prompt only asks the model for the opening, so code makes sure of it. Example: "People weigh risk [1]." becomes "I can't recommend a
    specific investment or predict prices. People weigh risk [1]."; "I can't predict prices. People weigh risk [1]."
    is left alone (no duplicate); "Investors can't ignore risk [1]." is not a refusal, so it gets the opening. The test
    is `workflow.advice_opening_pattern` (first person: I can't / cannot / am unable ... recommend / predict / advise /
    tell) on the first sentence only.
    """
    first = first_sentence(text, cfg["sentence_abbreviations"])
    if re.search(cfg["advice_opening_pattern"], first, re.IGNORECASE):
        return text
    return f"{' '.join(cfg['messages']['advice_opening'].split())} {text}"


def educational_answer(
    state: GraphState, llm: Any, retrieve_fn: Callable[..., list[RetrievedChunk]],
) -> tuple[str, list[Source]] | None:
    """(answer, sources) for an advice request, or None when the static text should be used.

    The reply starts with the "cannot recommend" opening (added when the model left it out). None when: no chunks, the
    model call fails, the reply cites no retrieved chunk, or it contains a number that is not in the chunks or the
    question. The output guard (reword once, then a safe message) always runs; a reworded text must pass the number
    check again, and a reworded or replaced text carries no sources (and no added opening).
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
    answer, cited = apply_citations(with_advice_opening(text, cfg), chunks)       # the opening is part of every check below
    allowed = collect_numbers([[c.text, c.title] for c in chunks] + [query])
    if not cited or numbers_in(strip_citations(answer)) - allowed:
        log.warning("Educational answer cites nothing or has an unsupported number; using the static text")
        return None
    guarded = guard_answer(answer, llm, check_numbers=False)      # the number check below uses the chunks, a wider set
    if guarded == answer:
        return answer, to_sources(cited)
    if numbers_in(guarded) - allowed:
        return None
    return guarded, []

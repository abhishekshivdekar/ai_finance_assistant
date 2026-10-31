"""The LLM merge of two or more agent answers, with checks and a verbatim fallback (architecture 3.2).

Design pattern: GUARD around an untrusted step. The model only rewrites; code decides whether the rewrite may be
used. Example: if the merged text contains a number that no agent answer contained, or drops the Tax footer, the
merge is thrown away and the answers are joined verbatim under headings (always safe).

The agent answers go to the model as a JSON data block; the prompt says they are data and never instructions.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.citations import CITATION, FACT_CITATION, drop_invalid_citations
from src.core.models import AgentResult, Source
from src.core.number_guard import collect_numbers, numbers_in
from src.core.output_guard import find_violation
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.workflow.answers import join_sections, renumber_results
from src.workflow.routing import workflow_cfg

log = get_logger(__name__)

# "[News 1]" and "[News 1, News 2]": the markers merge_verbatim gives to an answer that has no source list.
LABELLED = re.compile(r"\[[A-Za-z][A-Za-z ]*? \d+(?:\s*,\s*[A-Za-z][A-Za-z ]*? \d+)*\]")
_LABELLED_TOKEN = re.compile(r"([A-Za-z][A-Za-z ]*?) (\d+)")
_FACT_TOKEN = re.compile(r"F\d+")
_BRACKET = re.compile(r"\[[^\[\]\n]{0,60}\]")                   # any short bracket token; malformed_markers() filters it
_VALID_MARKERS = (CITATION, FACT_CITATION, LABELLED)


def _norm(text: str) -> str:
    """Whitespace-collapsed, case-folded text for 'does it still appear' checks."""
    return " ".join(text.split()).casefold()


def strip_markers(text: str) -> str:
    """The text without [n], [Fn] and [Label n] markers, so their digits are not mistaken for figures."""
    return LABELLED.sub("", FACT_CITATION.sub("", CITATION.sub("", text)))


def marker_tokens(text: str) -> set[str]:
    """The labelled and fact markers in the text as comparable tokens ("News 1", "F3")."""
    tokens: set[str] = set()
    for match in LABELLED.finditer(text):
        tokens |= {f"{label} {n}" for label, n in _LABELLED_TOKEN.findall(match.group(0))}
    for match in FACT_CITATION.finditer(text):
        tokens |= set(_FACT_TOKEN.findall(match.group(0)))
    return tokens


def malformed_markers(text: str) -> set[str]:
    """Bracket tokens with a digit that look like a marker but are not a valid one.

    Valid: "[1]", "[1, 2]", "[F1]", "[News 1]". Not valid: "[News4]", "[Market1]", "[ News 4 ]", "[ 1 ]". Such a token
    would pass unseen (the checks only know valid markers), so validate_merge rejects it.
    """
    return {t for t in _BRACKET.findall(text) if re.search(r"\d", t) and not any(p.fullmatch(t) for p in _VALID_MARKERS)}


def must_keep(result: AgentResult) -> list[str]:
    """Texts the agent put in its answer from its payload that a merge must not drop.

    The payload `label` (Tax: "illustrative, verify before relying"), the `source` of stale or sample prices
    every whole price sentence of a price that is not live (Market `freshness_texts`), the company-lookup notes (Market `lookup_notes`), every `exclusions` line
    and the `as_of` dates of `config_used` figures, each only when the agent's own answer contains it (an agent
    that left it out leaves nothing to keep).
    Example: a Tax answer ending "Figures quoted are illustrative, verify before relying (as of 2026-10-29)..."
    keeps the label, the date and, for equity figures, the securities-transaction-tax line; a Market answer with
    "AAPL last price USD 270.12 (cache, as of 2026-10-30 09:15 UTC)" keeps that whole sentence part.
    """
    data = result.get("data") or {}
    candidates: list[Any] = [data.get("label"), data.get("source") if data.get("is_stale") else None]
    candidates += list(data.get("freshness_texts") or [])
    candidates += list(data.get("lookup_notes") or [])
    candidates += list(data.get("exclusions") or [])
    candidates += [entry.get("as_of") for entry in (data.get("config_used") or {}).values() if isinstance(entry, dict)]
    answer = _norm(result["answer"])
    return [str(c) for c in dict.fromkeys(candidates) if isinstance(c, str) and c.strip() and _norm(c) in answer]


def validate_merge(
    text: str, sections: list[tuple[str, str]], source_count: int, keep: list[str], banned: list[str] | None = None,
) -> str | None:
    """The merged text with invalid [n] dropped, or None when any check fails.

    Checks: not empty; the output guard patterns do not match; the disclaimer is not in it (it is added once,
    later); no `banned` phrase (default `workflow.merge_banned_phrases`, regexes) appears unless an input answer has
    it too (the model must not talk about its input); no malformed marker such as "[News4]" or "[ News 4 ]" (unless an
    input answer has it); [Fn] and [Label n] markers all appear in the input answers;
    every number is one of the numbers in the input answers (markers removed first); every text in `keep` appears.
    """
    text = text.strip()
    if not text or find_violation(text) or _norm(get_config()["app"]["disclaimer"]) in _norm(text):
        return None
    inputs = [body for _, body in sections]
    for pattern in workflow_cfg()["merge_banned_phrases"] if banned is None else banned:
        phrase = re.compile(pattern, re.IGNORECASE)
        if phrase.search(text) and not any(phrase.search(body) for body in inputs):
            return None
    if malformed_markers(text) - set().union(*(malformed_markers(body) for body in inputs)):
        return None
    if not marker_tokens(text) <= set().union(*(marker_tokens(body) for body in inputs)):
        return None
    cleaned, _ = drop_invalid_citations(text, source_count)
    if numbers_in(strip_markers(cleaned)) - collect_numbers([strip_markers(body) for body in inputs]):
        return None
    normalised = _norm(cleaned)
    if any(_norm(item) not in normalised for item in keep):
        return None
    return cleaned


MERGE_LLM, MERGE_VERBATIM = "llm", "verbatim"


def merge_with_llm(
    results: list[AgentResult], llm: Any, cfg: dict[str, Any] | None = None,
) -> tuple[str, list[Source], str]:
    """One LLM call joins the ok answers; any failed check or error gives the verbatim join instead.

    Returns (text, sources, mode) with mode MERGE_LLM or MERGE_VERBATIM, so the caller can tell which text it got.
    """
    cfg = cfg or workflow_cfg()
    sections, sources = renumber_results(results, cfg)
    verbatim = (join_sections(sections, cfg), sources, MERGE_VERBATIM)
    data = json.dumps({"answers": [{"label": label, "text": body} for label, body in sections]}, ensure_ascii=False)
    messages = [
        SystemMessage(content=cfg["synthesizer_prompt"]),
        HumanMessage(content=f"<agent_answers>\n{data.replace('<', chr(92) + 'u003c')}\n</agent_answers>"),
    ]
    try:
        reply = llm.invoke(messages)
        text = str(getattr(reply, "content", reply))
    except Exception:
        log.warning("Merge call failed; joining the answers verbatim", exc_info=True)
        return verbatim
    keep = [item for result in results for item in must_keep(result)]
    merged = validate_merge(text, sections, len(sources), keep, cfg["merge_banned_phrases"])
    if merged is None:
        log.warning("Merged text failed a check; joining the answers verbatim")
        return verbatim
    return merged, sources, MERGE_LLM

"""Building the final answer text: disclaimer, notes and the verbatim merge of several agent answers.

Design pattern: PURE FUNCTIONS. Example: merge_verbatim joins a Finance Q&A answer citing [1] and a Tax answer
citing [1] into one text where the second one reads [2], and the Sources list is the two source lists in that order.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from src.core.citations import CITATION
from src.core.models import AgentResult, Source
from src.utils.config import get_config
from src.workflow.routing import workflow_cfg


def add_disclaimer(answer: str) -> str:
    """The answer followed by the disclaimer from config. Called once per final node, through nodes.finish()."""
    return f"{answer.strip()}\n\n{' '.join(get_config()['app']['disclaimer'].split())}"


def _map_citations(text: str, rename: Callable[[int], str]) -> str:
    """Rewrite every number inside [n] / [n, m] markers with rename(n); the brackets stay."""

    def swap(match: re.Match[str]) -> str:
        return "[" + ", ".join(rename(int(n)) for n in re.findall(r"\d+", match.group(1))) + "]"

    return CITATION.sub(swap, text)


def renumber_results(results: list[AgentResult], cfg: dict[str, Any]) -> tuple[list[tuple[str, str]], list[Source]]:
    """[(label, text)] per result with unique [n] markers, plus the concatenated sources.

    An answer with sources has its markers shifted by the number of sources before it (agent A cites [1][2],
    agent B's [1] becomes [3]) and the source lists are concatenated in the same order. An answer without sources
    (News cites its numbered headline list) gets its markers labelled, "[1]" -> "[News 1]", so they cannot be
    mistaken for a source number. Used before the LLM merge and for the verbatim join.
    """
    messages = cfg["messages"]
    sections: list[tuple[str, str]] = []
    sources: list[Source] = []
    for result in results:
        label = cfg["agent_labels"][result["agent"]]
        if result["sources"]:
            offset = len(sources)
            text = _map_citations(result["answer"], lambda n, offset=offset: str(n + offset))
            sources = [*sources, *result["sources"]]
        else:
            text = _map_citations(result["answer"], lambda n, label=label: messages["unsourced_marker"].format(label=label, n=n)[1:-1])
        sections.append((label, text))
    return sections, sources


def join_sections(sections: list[tuple[str, str]], cfg: dict[str, Any]) -> str:
    """The sections under their headings, each answer in its own words."""
    return "\n\n".join(f"{cfg['messages']['section_heading'].format(label=label)}\n{text}" for label, text in sections)


def merge_verbatim(results: list[AgentResult], cfg: dict[str, Any] | None = None) -> tuple[str, list[Source]]:
    """Join the ok answers under headings with unique [n] markers: the safe path whenever an LLM merge is rejected."""
    cfg = cfg or workflow_cfg()
    sections, sources = renumber_results(results, cfg)
    return join_sections(sections, cfg), sources


def notes_for(results: list[AgentResult], cfg: dict[str, Any] | None = None) -> list[str]:
    """One code-written note per needs_input / error / no_answer result (used when other agents did answer)."""
    cfg = cfg or workflow_cfg()
    messages = cfg["messages"]
    notes: list[str] = []
    for result in results:
        label = cfg["agent_labels"].get(result["agent"], result["agent"])
        if result["status"] == "needs_input":
            notes.append(" ".join(messages["needs_input_note"].format(label=label, message=result["answer"]).split()))
        elif result["status"] == "error":
            notes.append(" ".join(messages["error_note"].format(label=label).split()))
        elif result["status"] == "no_answer":
            notes.append(" ".join(messages["no_answer_note"].format(label=label, message=result["answer"]).split()))
    return notes

"""Finance Q&A agent: grounded, cited answers from the knowledge base (architecture 2.1).

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the answer llm, the rewrite llm and the retrieve function
  are constructor arguments with defaults from the existing factories. Example:
  a test passes FinanceQAAgent(llm=FakeLLM(), rewrite_llm=FakeLLM(),
  retrieve_fn=fake_retrieve) and needs no network or index, while the app just
  calls FinanceQAAgent().

Pipeline: rewrite the question into a plain finance-term query -> retrieve with
the original AND the rewritten query -> merge -> if nothing, no_answer without
an answer call -> one grounded LLM call -> parse NO_ANSWER and [n] citations.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base_agent import BaseAgent
from src.core.llm_factory import get_llm
from src.core.models import AgentResult, QAPayload
from src.rag.retriever import (
    RetrievedChunk,
    diversity_caps,
    format_context,
    retrieve,
    select_chunks,
    to_sources,
)
from src.utils.config import get_config
from src.workflow.state import GraphState

# "[1]", "[1, 2]" and "[1,2]" are one group each; "[1][2]" is two groups.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def clean_rewrite(text: str, max_chars: int) -> str:
    """Return a usable search query from the rewrite reply, or "" if it is malformed.

    Takes the first non-empty line and strips quotes. Over max_chars it is cut at
    a word boundary; a reply more than twice max_chars, without letters or
    digits, or with no word boundary to cut at counts as malformed.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    query = lines[0].strip(" \t\"'`")
    if not re.search(r"[A-Za-z0-9]", query) or len(query) > 2 * max_chars:
        return ""
    if len(query) > max_chars:
        head = query[: max_chars + 1]
        if " " not in head:
            return ""
        query = head.rsplit(" ", 1)[0].strip(" ,;:-")
    return query


def _numbers(group: str) -> list[int]:
    """'1, 2' -> [1, 2]."""
    return [int(n) for n in re.findall(r"\d+", group)]


def apply_citations(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    """Keep only citations that exist, renumber them 1..m, return the cited chunks.

    Example: with 4 context chunks the reply "A [3]. B [1][9]. C [3, 1]." becomes
    "A [1]. B [2]. C [1, 2]." and the cited chunks are [chunk 3, chunk 1], so the
    Sources list lines up with the numbers the reader sees. [9] does not exist
    and is removed.
    """
    order: list[int] = []
    for match in _CITATION.finditer(answer):
        for n in _numbers(match.group(1)):
            if 1 <= n <= len(chunks) and n not in order:
                order.append(n)
    new_number = {n: i for i, n in enumerate(order, start=1)}

    def renumber(match: re.Match[str]) -> str:
        nums = list(dict.fromkeys(new_number[n] for n in _numbers(match.group(1)) if n in new_number))
        return "[" + ", ".join(str(n) for n in nums) + "]" if nums else ""

    text = _CITATION.sub(renumber, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # space left before punctuation
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)  # double space left in the middle
    return text.strip(), [chunks[n - 1] for n in order]


class FinanceQAAgent(BaseAgent):
    """Answers general finance-education questions with cited knowledge-base sources."""

    name = "finance_qa"
    llm_role = "finance_qa"

    def __init__(
        self,
        llm: Any = None,
        rewrite_llm: Any = None,
        retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve,
    ) -> None:
        """Defaults: answer llm from role finance_qa, rewrite llm from finance_qa_rewrite."""
        super().__init__(llm=llm)
        self.rewrite_llm = (
            rewrite_llm if rewrite_llm is not None else get_llm(role="finance_qa_rewrite")
        )
        self.retrieve_fn = retrieve_fn

    def _rewrite(self, question: str) -> str:
        """One cheap LLM call; any failure or empty/malformed reply returns the question."""
        cfg = get_config()["finance_qa"]
        try:
            reply = self.rewrite_llm.invoke(cfg["rewrite_prompt"].format(question=question))
            rewritten = clean_rewrite(
                str(getattr(reply, "content", reply)), cfg["rewrite_max_chars"]
            )
        except Exception:
            self.log.warning("Query rewrite failed; using the original question", exc_info=True)
            return question
        return rewritten or question

    def _retrieve(self, queries: list[str], topic: str | None) -> list[RetrievedChunk]:
        """Retrieve for each query, merge by chunk (best score), re-apply the shared rules.

        retrieve() already applied the threshold and caps per query; select_chunks
        runs again on the merged list so the per-article and glossary caps and top_k
        still hold after merging (no second implementation of those rules).
        """
        rag = get_config()["rag"]
        best: dict[str, RetrievedChunk] = {}
        for query in queries:
            for chunk in self.retrieve_fn(query, topic=topic):
                known = best.get(chunk.chunk_id)
                if known is None or chunk.score > known.score:
                    best[chunk.chunk_id] = chunk
        hits = [({"chunk_id": cid}, chunk.score) for cid, chunk in best.items()]
        max_per_article, max_per_topic = diversity_caps(rag)
        kept = select_chunks(hits, rag["top_k"], rag["score_threshold"], max_per_article, max_per_topic)
        return [best[rec["chunk_id"]] for rec, _ in kept]

    def _result(
        self, status: str, answer: str, sources: list[Any], payload: QAPayload
    ) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=sources, error=None, data=payload,
        )

    def _execute(self, state: GraphState) -> AgentResult:
        """Rewrite, retrieve, answer from the context, then parse sentinel and citations."""
        cfg = get_config()["finance_qa"]
        question = (state.get("query") or "").strip()
        if not question:
            return self._result("needs_input", "Please type a finance question.", [], {})

        rewritten = self._rewrite(question)
        queries = [question] if rewritten == question else [question, rewritten]
        chunks = self._retrieve(queries, state.get("topic_filter"))  # type: ignore[misc]
        payload: QAPayload = {"rewritten_query": rewritten, "retrieved": len(chunks)}
        no_answer = self._result("no_answer", cfg["no_answer_message"].strip(), [], payload)
        if not chunks:
            self.log.info("No chunks retrieved; skipping the answer call")
            return no_answer

        reply = self.llm.invoke([
            SystemMessage(content=cfg["system_prompt"]),
            HumanMessage(content=f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"),
        ])
        text = str(getattr(reply, "content", reply)).strip()
        if text.strip("`\"'* .").upper() == cfg["no_answer_sentinel"]:
            self.log.info("Model replied %s", cfg["no_answer_sentinel"])
            return no_answer

        answer, cited = apply_citations(text, chunks)
        if not cited:
            self.log.warning("Answer has no valid citation; returning it with no sources")
        return self._result("ok", answer, to_sources(cited), payload)

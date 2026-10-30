"""The single entry point the UI calls: run_query() (architecture sections 3.2 and 4).

Design pattern: FACADE. The UI knows one function and one result object; the graph, router, agents and
checkpointer stay behind it. Example: the Chat tab calls run_query("What is a P/E ratio?", "session-1") and
shows result.answer and result.sources; the Market tab calls run_query("", "session-1", intent_hint="market",
market_params={"tickers": ["AAPL"]}) and draws result.results[0]["data"]. A second pattern, SINGLETON
(lru_cache), makes get_graph() build the compiled graph once per process.

Whatever goes wrong inside the graph (recursion backstop, an unexpected exception, an empty answer, the session token
budget) the caller gets a QueryResult with a friendly message, never an exception or a traceback. Turns on the SAME
thread_id run one after the other (a lock per thread), so a double click cannot make two turns read each other's state.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langgraph.errors import GraphRecursionError

from src.core.models import AgentResult, Source
from src.core.usage_tracker import is_budget_exceeded
from src.utils.logger import get_logger
from src.workflow.answers import add_disclaimer
from src.workflow.graph import build_graph
from src.workflow.router import Router
from src.workflow.routing import workflow_cfg
from src.workflow.state import current_results
from src.workflow.turn import new_turn_input

log = get_logger(__name__)

STOPPED_BUDGET, STOPPED_RECURSION, STOPPED_ERROR = "budget", "recursion", "error"


@dataclass(frozen=True)
class QueryResult:
    """What one turn gives back to the UI.

    `results` are the agent results of THIS turn (their `data` payloads feed charts, tables and warnings: the UI
    shows Market freshness and payload warnings from them, never from `answer`). `merge_mode` is "llm" or
    "verbatim" when two or more answers were merged, else None. `stopped` is None for a normal turn, else
    "budget", "recursion" or "error" (then `answer` is a friendly message).
    """

    answer: str
    sources: list[Source] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    confidence: float = 0.0
    nodes: list[str] = field(default_factory=list)
    merge_mode: str | None = None
    results: list[AgentResult] = field(default_factory=list)
    stopped: str | None = None


_thread_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _thread_lock(thread_id: str) -> threading.Lock:
    """The lock of one conversation thread, created on first use under a guard lock (one lock per thread_id)."""
    with _locks_guard:
        return _thread_locks.setdefault(thread_id, threading.Lock())


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """The compiled graph with the real router and agents and an in-memory checkpointer (Singleton)."""
    return build_graph(Router())


def _stopped(answer: str, why: str, nodes: list[str]) -> QueryResult:
    """A QueryResult for a turn that did not finish normally."""
    return QueryResult(answer=answer, nodes=nodes, stopped=why)


def _fallback(why: str, nodes: list[str]) -> QueryResult:
    """The friendly fallback text (with the disclaimer) for a turn that failed inside the graph."""
    return _stopped(add_disclaimer(" ".join(workflow_cfg()["messages"]["fallback"].split())), why, nodes)


def run_query(query: str, thread_id: str, intent_hint: str | None = None, **params: Any) -> QueryResult:
    """Run one user turn on a conversation thread and return the answer plus the data behind it.

    `params` are the per-turn inputs of new_turn_input (topic_filter, portfolio, goal_params, market_params,
    news_params, tax_params); an unknown name raises TypeError at once, because that is a bug in the caller, not
    a runtime failure. The input is complete and fresh on every call (new turn_id, per-turn keys reset, earlier agent
    results filtered out by turn_id). The conversation history of the thread is kept on purpose: the router reads it, so
    "And what about Apple?" works after a Microsoft question. Calls on one thread_id wait for each other.
    """
    turn = new_turn_input(query, intent_hint=intent_hint, **params)
    messages = workflow_cfg()["messages"]
    if is_budget_exceeded():
        log.warning("Session token budget reached; turn not started")
        return _stopped(" ".join(messages["budget_exceeded"].split()), STOPPED_BUDGET, [])
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": workflow_cfg()["recursion_limit"]}
    nodes: list[str] = []
    try:
        graph = get_graph()
        with _thread_lock(thread_id):
            for update in graph.stream(turn, config, stream_mode="updates"):
                nodes += list(update)
            state = graph.get_state(config).values
    except GraphRecursionError:
        log.warning("Graph hit the recursion limit after nodes %s", nodes)
        return _fallback(STOPPED_RECURSION, nodes)
    except Exception:
        log.exception("Workflow failed after nodes %s", nodes)
        return _fallback(STOPPED_ERROR, nodes)
    if not (state.get("final_answer") or "").strip():
        log.warning("Graph finished with an empty answer after nodes %s", nodes)
        return _fallback(STOPPED_ERROR, nodes)
    return QueryResult(
        answer=state["final_answer"],
        sources=list(state.get("sources") or []),
        intents=list(state.get("intents") or []),
        confidence=float(state.get("router_confidence") or 0.0),
        nodes=nodes,
        merge_mode=state.get("merge_mode"),
        results=current_results(state),
    )

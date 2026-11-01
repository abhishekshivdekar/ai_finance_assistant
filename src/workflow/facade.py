"""The single entry point the UI calls: run_query() (architecture sections 3.2 and 4).

Design pattern: FACADE. The UI knows one function and one result object; the graph, router, agents and
checkpointer stay behind it. Example: the Chat tab calls run_query("What is a P/E ratio?", "session-1") and
shows result.answer and result.sources; the Market tab calls run_query("Market analysis for AAPL.", "session-1",
intent_hint="market", market_params={"tickers": ["AAPL"]}) and draws result.results[0]["data"]. (Give a hinted turn a
short real sentence, not "": the query is kept in the thread history that the router reads on later chat turns.) A second pattern, SINGLETON
(lru_cache), makes get_graph() build the compiled graph once per process.

Whatever goes wrong inside the graph (recursion backstop, an unexpected exception, an empty answer, the session token
budget) the caller gets a QueryResult with a friendly message, never an exception or a traceback. Turns on the SAME
thread_id run one after the other (a lock per thread), so a double click cannot make two turns read each other's state.
"""
from __future__ import annotations

import atexit
import threading
import time
import weakref
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langgraph.errors import GraphRecursionError

from src.core.models import AgentResult, Source
from src.core.usage_tracker import get_usage_registry, is_budget_exceeded
from src.rag.embeddings import get_embedder
from src.rag.vector_store import get_vector_store
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.workflow.answers import add_disclaimer
from src.workflow.graph import build_graph
from src.workflow.persistence import ConversationStore, PersistenceStatus, open_store
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


@dataclass(frozen=True)
class WarmUp:
    """How the start-up warm-up went. `message` is "" on success, else the error type only (details are in the log)."""

    ok: bool
    seconds: float
    message: str = ""


@dataclass(frozen=True)
class UsageSummary:
    """LLM usage since the PROCESS started (all conversations and visitors share it; a new thread does not reset it).

    `cost` is USD, or None when any call used a model with no price in config.yaml. `limit` is llm.max_session_tokens
    (None = no limit); `exceeded` is True when run_query will refuse new turns.
    """

    tokens: int
    cost: float | None
    limit: int | None
    exceeded: bool


@dataclass(frozen=True)
class HistoryItem:
    """One message of a stored conversation, for redrawing the Chat tab after a reload.

    `role` is "user" or "assistant". `sources` are the citations of an assistant message. `tab` is the tab that made the
    turn ("market", "portfolio", "goal_planning", ...) or None for a Chat turn; an assistant message carries the tab of the
    user message it answers.
    """

    role: str
    text: str
    sources: list[Source] = field(default_factory=list)
    tab: str | None = None


_WARM_UP_TEXT = "warm up"


class _ThreadLock:
    """A lock that can be weakly referenced (a plain threading.Lock cannot), so unused locks disappear."""

    __slots__ = ("_lock", "__weakref__")

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


_thread_locks: "weakref.WeakValueDictionary[str, _ThreadLock]" = weakref.WeakValueDictionary()
_locks_guard = threading.Lock()


def _thread_lock(thread_id: str) -> _ThreadLock:
    """The lock of one conversation thread (one object per thread_id while any turn holds or waits for it).

    The dictionary holds the locks weakly: a `with _thread_lock(id):` block keeps its lock alive, and once no turn uses it
    the entry vanishes, so the dictionary no longer grows with every conversation the process has ever seen.
    """
    with _locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = _thread_locks[thread_id] = _ThreadLock()
        return lock


@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    """The conversation store (Singleton): the SQLite file when possible, else memory. Old conversations are cleaned up here."""
    store = open_store()
    settings = get_config()["persistence"]
    try:
        store.cleanup(settings["max_age_days"], settings["max_threads"])
    except Exception:
        log.exception("Retention cleanup failed")
    atexit.register(store.close)
    return store


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """The compiled graph with the real router and agents on the conversation store's checkpointer (Singleton)."""
    return build_graph(Router(), checkpointer=get_store().saver)


def persistence_status() -> PersistenceStatus:
    """Whether conversations are saved in this run and the sentence to show (never raises)."""
    try:
        return get_store().status
    except Exception:
        log.exception("Persistence status unavailable")
        return PersistenceStatus(False, " ".join(get_config()["persistence"]["messages"]["unavailable"].split()))


def history(thread_id: str) -> list[HistoryItem]:
    """The stored messages of a conversation, oldest first. Unknown thread or any failure -> []. Never raises, never logs text."""
    try:
        values = get_graph().get_state({"configurable": {"thread_id": thread_id}}).values
    except Exception:
        log.exception("Reading a conversation failed")
        return []
    items: list[HistoryItem] = []
    tab: str | None = None
    for message in values.get("messages") or []:
        extra = getattr(message, "additional_kwargs", None) or {}
        if message.type == "human":
            tab = extra.get("tab")
            items.append(HistoryItem("user", str(message.content), [], tab))
        elif message.type == "ai":
            items.append(HistoryItem("assistant", str(message.content), list(extra.get("sources") or []), tab))
    return items


def delete_thread(thread_id: str) -> bool:
    """Delete a conversation from the store (waits for a running turn of that thread). True when it succeeded."""
    try:
        with _thread_lock(thread_id):
            get_store().delete(thread_id)
    except Exception:
        log.exception("Deleting a conversation failed")
        return False
    return True


def warm_up() -> WarmUp:
    """Load the embedding model, the vector store and the graph now, so the first question does not wait for them.

    Never raises: a failure is logged and returned as WarmUp(ok=False), and the app still works (the first question
    then pays the loading time). Makes no LLM call (the embedding model may be downloaded the very first time it is used).
    """
    started = time.perf_counter()
    try:
        get_embedder().embed_query(_WARM_UP_TEXT)
        get_vector_store()
        get_graph()
    except Exception as exc:
        log.exception("Warm-up failed")
        return WarmUp(ok=False, seconds=time.perf_counter() - started, message=type(exc).__name__)
    return WarmUp(ok=True, seconds=time.perf_counter() - started)


def usage_summary() -> UsageSummary:
    """Tokens, cost and budget state of this process, read from the usage registry and llm.max_session_tokens."""
    registry = get_usage_registry()
    costs = [entry["cost"] for entry in registry.totals().values()]
    return UsageSummary(
        tokens=registry.session_tokens(),
        cost=None if any(c is None for c in costs) else float(sum(costs)),
        limit=get_config().get("llm", {}).get("max_session_tokens"),
        exceeded=is_budget_exceeded(),
    )


def _remember(thread_id: str) -> None:
    """Note that the thread was used (for retention). A failure is logged and never breaks the turn."""
    try:
        get_store().touch(thread_id)
    except Exception:
        log.exception("Recording conversation activity failed")


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
            for update in graph.stream(turn, config, stream_mode="updates", durability="exit"):     # one checkpoint per turn, not per node
                nodes += list(update)
            state = graph.get_state(config).values
            _remember(thread_id)
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

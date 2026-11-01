"""The only module of the UI that imports the workflow facade (architecture section 13).

Design pattern: ADAPTER / SEAM. The UI calls these three functions and nothing else of the backend, so the tests replace
them with fakes (`monkeypatch.setattr(backend, "ask", fake)`) and no test needs a graph, an LLM or the network. Example:
tab_chat calls `backend.ask(question, thread_id)`; in a test `ask` returns a ready-made QueryResult.
"""
from __future__ import annotations

from typing import Any

from src.workflow.facade import (
    HistoryItem, PersistenceStatus, QueryResult, UsageSummary, WarmUp, delete_thread, history, persistence_status, run_query, usage_summary, warm_up,
)

__all__ = [
    "HistoryItem", "PersistenceStatus", "QueryResult", "UsageSummary", "WarmUp", "ask", "forget", "persistence", "read_history", "start_up", "usage",
]


def ask(query: str, thread_id: str, intent_hint: str | None = None, **params: Any) -> QueryResult:
    """Run one turn on a conversation thread (the facade's run_query; it never raises)."""
    return run_query(query, thread_id, intent_hint, **params)


def start_up() -> WarmUp:
    """Load the embedder, the index and the graph now (the facade's warm_up; it never raises)."""
    return warm_up()


def usage() -> UsageSummary:
    """Token usage and budget state of this process (the facade's usage_summary)."""
    return usage_summary()


def read_history(thread_id: str) -> list[HistoryItem]:
    """The stored messages of a conversation (the facade's history; [] when there is none)."""
    return history(thread_id)


def forget(thread_id: str) -> bool:
    """Delete a conversation from the store (the facade's delete_thread)."""
    return delete_thread(thread_id)


def persistence() -> PersistenceStatus:
    """Whether conversations are saved in this run, with the sentence for the sidebar."""
    return persistence_status()

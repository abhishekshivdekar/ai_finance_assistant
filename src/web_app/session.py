"""Per-browser-session state: the conversation thread, the chat on screen, the waiting request and each tab's last result.

Design pattern: a thin STATE HOLDER over st.session_state (one entry per browser session). Example: "New conversation"
gives a new thread_id, so the backend's memory of the old thread is no longer used, and clears the chat; it does not
touch the usage counters (those belong to the running app, not to the conversation) and keeps the Market and News
results (they are not part of a conversation).

At most ONE request waits at a time, for any tab. That is what disables every submit button while a call runs.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from src.web_app.backend import QueryResult

THREAD_ID, CHAT, PENDING, RESULTS, ACTIVE_TAB = "thread_id", "chat", "pending_request", "tab_results", "active_tab"


@dataclass(frozen=True)
class Request:
    """One facade call waiting to be made: which tab asked, the query text, the tab's intent hint and its params."""

    tab: str
    query: str
    intent_hint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _new_thread_id() -> str:
    """A random id, one per conversation."""
    return f"web-{uuid.uuid4().hex}"


def init() -> None:
    """Create the session entries on the first run of a browser session."""
    st.session_state.setdefault(THREAD_ID, _new_thread_id())
    st.session_state.setdefault(CHAT, [])
    st.session_state.setdefault(PENDING, None)
    st.session_state.setdefault(RESULTS, {})
    st.session_state.setdefault(ACTIVE_TAB, None)


def thread_id() -> str:
    """The id of the current conversation thread."""
    return str(st.session_state[THREAD_ID])


def chat() -> list[dict[str, Any]]:
    """The messages on screen: {"role": "user", "text": ...} or {"role": "assistant", "result": QueryResult}."""
    return st.session_state[CHAT]  # type: ignore[no-any-return]


def is_busy() -> bool:
    """True while a submitted request has not been answered yet (for any tab)."""
    return st.session_state[PENDING] is not None


def submit(tab: str, query: str, intent_hint: str | None = None, **params: Any) -> None:
    """Mark a request as waiting for its answer and remember the tab, so the page reopens on it."""
    st.session_state[PENDING] = Request(tab, query, intent_hint, params)
    st.session_state[ACTIVE_TAB] = tab


def take_pending(tab: str) -> Request | None:
    """The waiting request of this tab, cleared so that a rerun cannot make the call a second time (None if not its)."""
    request = st.session_state[PENDING]
    if request is None or request.tab != tab:
        return None
    st.session_state[PENDING] = None
    return request  # type: ignore[no-any-return]


def has_pending(tab: str) -> bool:
    """True when the waiting request belongs to this tab (it is not taken)."""
    request = st.session_state[PENDING]
    return request is not None and request.tab == tab


def active_tab() -> str | None:
    """The tab that made the last request (None before any)."""
    return st.session_state[ACTIVE_TAB]  # type: ignore[no-any-return]


def set_result(tab: str, result: QueryResult) -> None:
    """Keep the last result of a tab (Market, News) so later reruns redraw it without calling the backend."""
    st.session_state[RESULTS][tab] = result


def result_for(tab: str) -> QueryResult | None:
    """The last result of a tab, or None."""
    return st.session_state[RESULTS].get(tab)


def add_answer(result: QueryResult) -> None:
    """Add the assistant's answer to the chat."""
    chat().append({"role": "assistant", "result": result})


def new_conversation() -> None:
    """A fresh thread and an empty chat. Does not reset token usage; tab results stay."""
    st.session_state[THREAD_ID] = _new_thread_id()
    st.session_state[CHAT] = []
    st.session_state[PENDING] = None

"""Per-browser-session state: the conversation thread, the chat on screen, the waiting request and each tab's last result.

Design pattern: a thin STATE HOLDER over st.session_state (one entry per browser session). Example: "New conversation"
gives a new thread_id, so the backend's memory of the old thread is no longer used, and clears the chat; it does not
touch the usage counters (those belong to the running app, not to the conversation) and keeps the Market and News
results (they are not part of a conversation).

The thread id also lives in the page URL (`?t=web-<32 hex>`), because a reload gives a NEW browser session and discards
st.session_state. Example: after the first question the URL is http://localhost:8501/?t=web-3f2a...; pressing F5 starts a
fresh session that reads `t`, finds it valid, and redraws the Chat from the stored conversation. A missing or malformed
value (anything not matching THREAD_PATTERN) is ignored and a new id is made, so the URL cannot be used to pick another
kind of key. Anyone with the full URL can open that conversation (see the README); two tabs on the same URL share one thread.

At most ONE request waits at a time, for any tab. That is what disables every submit button while a call runs.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from src.web_app import backend
from src.web_app.backend import QueryResult

THREAD_ID, CHAT, PENDING, RESULTS, ACTIVE_TAB = "thread_id", "chat", "pending_request", "tab_results", "active_tab"
NOTICE, CONFIRM_DELETE, URL_KEY = "notice", "confirm_delete", "t"
THREAD_PATTERN = re.compile(r"web-[0-9a-f]{32}")


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


def _restored_chat(thread: str) -> list[dict[str, Any]]:
    """The Chat messages of a stored conversation: only untagged (Chat) turns; Market/Portfolio/Goals/News turns stay out."""
    chat: list[dict[str, Any]] = []
    for item in backend.read_history(thread):
        if item.tab is not None:
            continue
        if item.role == "user":
            chat.append({"role": "user", "text": item.text})
        else:
            chat.append({"role": "assistant", "result": QueryResult(answer=item.text, sources=list(item.sources))})
    return chat


def init() -> None:
    """Create the session entries on the first run of a browser session; take the thread from the URL when it is valid."""
    if THREAD_ID not in st.session_state:
        wanted = st.query_params.get(URL_KEY)
        valid = isinstance(wanted, str) and THREAD_PATTERN.fullmatch(wanted) is not None
        thread = str(wanted) if valid else _new_thread_id()
        st.session_state[THREAD_ID] = thread
        st.session_state[CHAT] = _restored_chat(thread) if valid else []
        st.query_params[URL_KEY] = thread
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
    """The tab that made the last request, until the render that shows its answer has been drawn (else None)."""
    return st.session_state[ACTIVE_TAB]  # type: ignore[no-any-return]


def clear_active_tab() -> None:
    """Forget the tab of the last request, so a later rerun does not pass `default=` to st.tabs again."""
    st.session_state[ACTIVE_TAB] = None


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
    """A fresh thread (also written to the URL) and an empty chat. Does not reset token usage; tab results stay."""
    thread = _new_thread_id()
    st.session_state[THREAD_ID] = thread
    st.session_state[CHAT] = []
    st.session_state[PENDING] = None
    st.query_params[URL_KEY] = thread


def delete_conversation() -> None:
    """Delete the stored conversation, then start a new one. The outcome is kept as a notice for the next draw.

    The confirmation box is cleared either way. When the delete fails nothing else changes: the chat and the thread stay.
    """
    st.session_state[CONFIRM_DELETE] = False
    if backend.forget(thread_id()):
        new_conversation()
        st.session_state[NOTICE] = "delete_done"
    else:
        st.session_state[NOTICE] = "delete_failed"


def take_notice() -> str | None:
    """The key of the one-time notice (a ui.sidebar text), cleared when read."""
    return st.session_state.pop(NOTICE, None)  # type: ignore[no-any-return]

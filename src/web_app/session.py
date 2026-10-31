"""Per-browser-session state: the conversation thread, the chat shown on screen and the question being answered.

Design pattern: a thin STATE HOLDER over st.session_state (one entry per browser session). Example: "New conversation"
gives a new thread_id, so the backend's memory of the old thread is no longer used, and clears the chat; it does not
touch the usage counters (those belong to the running app, not to the conversation).
"""
from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from src.web_app.backend import QueryResult

THREAD_ID, CHAT, PENDING = "thread_id", "chat", "pending_question"


def _new_thread_id() -> str:
    """A random id, one per conversation."""
    return f"web-{uuid.uuid4().hex}"


def init() -> None:
    """Create the session entries on the first run of a browser session."""
    st.session_state.setdefault(THREAD_ID, _new_thread_id())
    st.session_state.setdefault(CHAT, [])
    st.session_state.setdefault(PENDING, None)


def thread_id() -> str:
    """The id of the current conversation thread."""
    return str(st.session_state[THREAD_ID])


def chat() -> list[dict[str, Any]]:
    """The messages on screen: {"role": "user", "text": ...} or {"role": "assistant", "result": QueryResult}."""
    return st.session_state[CHAT]  # type: ignore[no-any-return]


def is_busy() -> bool:
    """True while a submitted question has not been answered yet."""
    return st.session_state[PENDING] is not None


def submit(question: str) -> None:
    """Show the user's message and mark the question as waiting for its answer."""
    chat().append({"role": "user", "text": question})
    st.session_state[PENDING] = question


def take_pending() -> str | None:
    """The waiting question, cleared so that a rerun cannot ask it a second time."""
    question = st.session_state[PENDING]
    st.session_state[PENDING] = None
    return question  # type: ignore[no-any-return]


def add_answer(result: QueryResult) -> None:
    """Add the assistant's answer to the chat."""
    chat().append({"role": "assistant", "result": result})


def new_conversation() -> None:
    """A fresh thread and an empty chat. Does not reset token usage."""
    st.session_state[THREAD_ID] = _new_thread_id()
    st.session_state[CHAT] = []
    st.session_state[PENDING] = None

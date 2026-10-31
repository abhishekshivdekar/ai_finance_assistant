"""The Chat tab: the router decides which agent answers (no intent hint).

Flow (Streamlit reruns the whole script on every click, so the backend call is guarded):
1. Submit run: the user's message is stored, a request is marked as waiting, `st.rerun()` redraws the page with the
   input disabled.
2. Answer run: the page is drawn with every input disabled, THEN the returned action takes the request (exactly once),
   calls `backend.ask`, stores the answer and reruns with the input enabled again. Any other rerun has no request and
   calls nothing. The call is deferred until every tab is drawn, so no tab shows a stale enabled button during it.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

from src.web_app import backend, session
from src.web_app.render_common import plain_text, render_result
from src.web_app.text import ui_text

TAB = "chat"


def _render_history() -> None:
    """Draw every stored message."""
    for message in session.chat():
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(plain_text(message["text"]))
            else:
                render_result(message["result"])


def _answer(slot: st.delta_generator.DeltaGenerator) -> None:
    """Make the waiting call (once), keep the answer and redraw."""
    request = session.take_pending(TAB)
    if request is not None:
        with slot, st.chat_message("assistant"), st.spinner(ui_text("chat.thinking")):
            result = backend.ask(request.query, session.thread_id())
        session.add_answer(result)
        st.rerun()


def render(budget_exceeded: bool) -> Callable[[], None] | None:
    """Draw the chat and handle one submitted question; returns the deferred call when a request is waiting."""
    busy = session.is_busy()
    if not session.chat() and not busy:
        st.info(ui_text("chat.empty_state"))
    _render_history()
    slot = st.container()                                  # the waiting answer is drawn here, above the input
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    placeholder = "chat.placeholder_stopped" if budget_exceeded else "chat.placeholder_busy" if busy else "chat.placeholder"
    question = st.chat_input(ui_text(placeholder), disabled=busy or budget_exceeded)
    if question and question.strip():
        session.chat().append({"role": "user", "text": question.strip()})
        session.submit(TAB, question.strip())
        st.rerun()
    return (lambda: _answer(slot)) if session.has_pending(TAB) else None

"""The Chat tab: the router decides which agent answers (no intent hint).

Flow (Streamlit reruns the whole script on every click, so the backend call is guarded):
1. Submit run: the user's message is stored, `pending` is set, `st.rerun()` redraws the page with the input disabled.
2. Answer run: `take_pending()` hands out the question exactly once, `backend.ask` runs, the answer is stored and
   `st.rerun()` redraws the page with the input enabled again. Any other rerun finds no pending question and calls nothing.
"""
from __future__ import annotations

import streamlit as st

from src.web_app import backend, session
from src.web_app.render_common import plain_text, render_result
from src.web_app.text import ui_text


def _render_history() -> None:
    """Draw every stored message."""
    for message in session.chat():
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(plain_text(message["text"]))
            else:
                render_result(message["result"])


def render(budget_exceeded: bool) -> None:
    """Draw the chat and handle one submitted question."""
    busy = session.is_busy()
    if not session.chat() and not busy:
        st.info(ui_text("chat.empty_state"))
    _render_history()
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    placeholder = "chat.placeholder_stopped" if budget_exceeded else "chat.placeholder_busy" if busy else "chat.placeholder"
    question = st.chat_input(ui_text(placeholder), disabled=busy or budget_exceeded)
    if question and question.strip():
        session.submit(question.strip())
        st.rerun()
    pending = session.take_pending()
    if pending is not None:
        with st.chat_message("assistant"), st.spinner(ui_text("chat.thinking")):
            result = backend.ask(pending, session.thread_id())
        session.add_answer(result)
        st.rerun()

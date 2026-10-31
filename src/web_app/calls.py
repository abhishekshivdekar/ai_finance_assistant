"""The one place where a tab's waiting request becomes a backend call (Market and News share it).

Design pattern: TEMPLATE FUNCTION. Example: tab_market returns `lambda: run_waiting("market", slot, "market.working")`;
main.render() calls it after all tabs are drawn, so the facade call happens at most once per submit and never on a plain rerun.
"""
from __future__ import annotations

import streamlit as st

from src.web_app import backend, session
from src.web_app.text import ui_text


def run_waiting(tab: str, slot: st.delta_generator.DeltaGenerator, spinner_key: str) -> None:
    """Take the tab's waiting request (exactly once), call the backend inside `slot`, keep the result and redraw."""
    request = session.take_pending(tab)
    if request is not None:
        with slot, st.spinner(ui_text(spinner_key)):
            result = backend.ask(request.query, session.thread_id(), request.intent_hint, **request.params)
        session.set_result(tab, result)
        st.rerun()

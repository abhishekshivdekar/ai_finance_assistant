"""Reload and delete (UI step 5b): the thread id lives in ?t=, the Chat is redrawn from the stored conversation, and the
sidebar can delete it. The backend is faked (no graph, no file): the fake returns scripted history items."""
from __future__ import annotations

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.utils.config import get_config
from src.web_app import backend, session
from web_support import APP, HOSTILE, SOURCE, answer_result, ask, fake, shown, ui  # noqa: F401

THREAD = "web-" + "ab" * 16


def item(role, text, sources=None, tab=None):
    return backend.HistoryItem(role, text, sources or [], tab)


def conversation():
    return [
        item("user", "What is a bond?"),
        item("assistant", "A bond is a loan [1].", [SOURCE]),
        item("user", "Plan my savings goal from the Goals form.", tab="goal_planning"),
        item("assistant", "A goal plan the Goals tab already showed.", tab="goal_planning"),
        item("user", "And a stock?"),
        item("assistant", "A stock is a share of ownership."),
    ]


def open_page(thread=None):
    at = AppTest.from_file(APP, default_timeout=10)
    if thread is not None:
        at.query_params["t"] = thread
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def messages(at):
    return [m.markdown[0].value for m in at.chat_message]


def sidebar_buttons(at):
    return {b.label: b for b in at.sidebar.button}


# ============================== the thread id in the URL ==============================

def test_a_first_visit_makes_a_valid_id_and_writes_it_to_the_url_without_reading_history(fake):
    at = open_page()
    thread = at.session_state["thread_id"]
    assert session.THREAD_PATTERN.fullmatch(thread) and at.query_params["t"] == thread
    assert fake.history_reads == [] and ui("chat.empty_state") in shown(at)


def test_a_valid_id_in_the_url_is_kept_and_its_chat_turns_are_redrawn(fake):
    fake.histories[THREAD] = conversation()
    at = open_page(THREAD)
    assert at.session_state["thread_id"] == THREAD and at.query_params["t"] == THREAD
    texts = " ".join(messages(at))
    assert "What is a bond?" in texts and "A bond is a loan" in texts and "And a stock?" in texts and "A stock is a share" in texts
    assert "Plan my savings goal" not in texts and "Goals tab already showed" not in texts               # tab turns are not Chat turns
    assert [e.label for e in at.expander] == [ui("chat.sources_title", count=1)] and ui("chat.empty_state") not in shown(at)
    assert [m.avatar for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert fake.history_reads == [THREAD]


def test_reruns_do_not_read_the_history_again_and_a_new_question_uses_the_restored_thread(fake):
    fake.histories[THREAD] = conversation()
    at = open_page(THREAD)
    at.run()
    at.run()
    assert fake.history_reads == [THREAD]
    ask(at, "Another question?")
    assert fake.asks[0][0][1] == THREAD and len(fake.asks) == 1
    assert "Another question?" in " ".join(messages(at)) and at.query_params["t"] == THREAD


def test_a_valid_id_with_no_stored_conversation_gives_an_empty_chat_on_that_thread(fake):
    at = open_page(THREAD)
    assert at.session_state["thread_id"] == THREAD and ui("chat.empty_state") in shown(at) and fake.history_reads == [THREAD]


@pytest.mark.parametrize("bad", [
    "", "abc", "web-" + "g" * 32, "web-" + "AB" * 16, "web-" + "a" * 31, "web-" + "a" * 33, "../../etc/passwd", "web-" + "ab" * 16 + "x",
    " web-" + "ab" * 16, "WEB-" + "ab" * 16, "<script>alert(1)</script>",
])
def test_an_invalid_id_is_ignored_a_new_one_is_made_and_the_url_is_corrected(fake, bad):
    fake.histories[bad] = [item("user", "should never be shown")]
    at = open_page(bad)
    thread = at.session_state["thread_id"]
    assert thread != bad and session.THREAD_PATTERN.fullmatch(thread) and at.query_params["t"] == thread
    assert fake.history_reads == [] and "should never be shown" not in " ".join(shown(at))


def test_a_repeated_t_parameter_with_a_bad_value_is_ignored(fake):
    at = open_page(["junk", "web-" + "x" * 32])
    assert session.THREAD_PATTERN.fullmatch(at.session_state["thread_id"]) and fake.history_reads == []


def test_two_browser_tabs_on_the_same_url_share_one_thread(fake):
    fake.histories[THREAD] = conversation()
    first, second = open_page(THREAD), open_page(THREAD)
    assert first.session_state["thread_id"] == second.session_state["thread_id"] == THREAD
    assert messages(first) == messages(second) and fake.history_reads == [THREAD, THREAD]
    ask(first, "From the first tab")
    ask(second, "From the second tab")
    assert [call[0][1] for call in fake.asks] == [THREAD, THREAD]          # both turns go to the one stored thread
    assert "From the second tab" not in " ".join(messages(first))           # each page only redraws on its own rerun / reload


def test_a_tab_without_the_parameter_gets_its_own_new_thread(fake):
    fake.histories[THREAD] = conversation()
    shared, fresh = open_page(THREAD), open_page()
    assert fresh.session_state["thread_id"] != THREAD and shared.session_state["thread_id"] == THREAD


# ============================== hostile stored text ==============================

def test_hostile_text_in_a_stored_conversation_is_shown_as_plain_text(fake):
    fake.histories[THREAD] = [
        item("user", HOSTILE), item("assistant", f"Answer: {HOSTILE}", [{**SOURCE, "title": HOSTILE, "snippet": HOSTILE, "url": "javascript:alert(1)"}]),
    ]
    at = open_page(THREAD)
    page = shown(at) + [m for m in messages(at)]
    assert not [t for t in page if "<script" in t or "](javascript:" in t]
    assert any("&lt;script" in t for t in messages(at))
    assert not [link for link in at.get("markdown") if "(javascript:" in str(link.value)]


# ============================== New conversation ==============================

def test_new_conversation_writes_the_new_id_to_the_url_and_empties_the_chat(fake):
    fake.histories[THREAD] = conversation()
    at = open_page(THREAD)
    sidebar_buttons(at)[ui("sidebar.new_conversation")].click().run()
    thread = at.session_state["thread_id"]
    assert thread != THREAD and session.THREAD_PATTERN.fullmatch(thread) and at.query_params["t"] == thread
    assert messages(at) == [] and fake.forgotten == []                       # the old conversation stays stored
    at.run()
    assert at.query_params["t"] == thread and fake.history_reads == [THREAD]


# ============================== Delete my conversation ==============================

def test_the_delete_button_needs_the_confirmation_box_first(fake):
    at = open_page(THREAD)
    assert sidebar_buttons(at)[ui("sidebar.delete_button")].disabled is True
    at.sidebar.checkbox[0].check().run()
    assert sidebar_buttons(at)[ui("sidebar.delete_button")].disabled is False and fake.forgotten == []


def test_confirmed_delete_removes_the_stored_thread_clears_the_chat_and_starts_a_new_thread(fake):
    fake.histories[THREAD] = conversation()
    at = open_page(THREAD)
    at.sidebar.checkbox[0].check().run()
    sidebar_buttons(at)[ui("sidebar.delete_button")].click().run()
    assert fake.forgotten == [THREAD]
    thread = at.session_state["thread_id"]
    assert thread != THREAD and at.query_params["t"] == thread and messages(at) == [] and ui("chat.empty_state") in shown(at)
    assert [s.value for s in at.sidebar.success if s.value == ui("sidebar.delete_done")] == [ui("sidebar.delete_done")]
    assert at.sidebar.checkbox[0].value is False and sidebar_buttons(at)[ui("sidebar.delete_button")].disabled is True
    at.run()
    assert ui("sidebar.delete_done") not in [s.value for s in at.sidebar.success]          # the notice is shown once


def test_a_failed_delete_keeps_the_conversation_the_thread_and_the_url_and_says_so(fake):
    fake.histories[THREAD] = conversation()
    fake.forget_ok = False
    at = open_page(THREAD)
    at.sidebar.checkbox[0].check().run()
    sidebar_buttons(at)[ui("sidebar.delete_button")].click().run()
    assert fake.forgotten == [THREAD] and at.session_state["thread_id"] == THREAD and at.query_params["t"] == THREAD
    assert "A bond is a loan" in " ".join(messages(at)) and ui("sidebar.delete_failed") in [w.value for w in at.sidebar.warning]
    assert at.sidebar.checkbox[0].value is False


def test_the_delete_controls_are_drawn_disabled_while_a_call_runs(fake, monkeypatch):
    seen, real_button = [], st.button
    monkeypatch.setattr(st, "button", lambda label, *a, **kw: seen.append((label, kw.get("disabled"))) or real_button(label, *a, **kw))
    fake.reply = answer_result("A bond is a loan.", sources=[])
    at = open_page(THREAD)
    at.sidebar.checkbox[0].check().run()
    seen.clear()
    ask(at, "What is a bond?")
    delete = ui("sidebar.delete_button")
    assert [s for s in seen if s[0] == delete] == [(delete, False), (delete, True), (delete, False)]


# ============================== unchanged behaviour ==============================

def test_a_reached_budget_still_shows_the_restored_chat_with_the_input_disabled(fake):
    fake.usage_value = backend.UsageSummary(tokens=2_000_001, cost=1.0, limit=2_000_000, exceeded=True)
    fake.histories[THREAD] = conversation()
    at = open_page(THREAD)
    assert at.chat_input[0].disabled is True and "A stock is a share" in " ".join(messages(at)) and fake.asks == []


def test_the_persistence_status_is_still_shown_with_a_restored_chat(fake):
    fake.histories[THREAD] = conversation()
    saved = " ".join(get_config()["persistence"]["messages"]["saving"].split())
    assert saved in [c.value for c in open_page(THREAD).sidebar.caption]

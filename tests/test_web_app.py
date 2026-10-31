"""The Streamlit app, driven with streamlit.testing.v1.AppTest and a fake backend (no graph, no LLM, no network)."""
from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.core.usage_tracker import UsageRegistry
from src.utils.config import get_config
from src.web_app import backend

APP = str(Path(__file__).resolve().parent.parent / "app.py")
HOSTILE = "<script>alert(1)</script> [x](javascript:alert(1))"
SOURCE = {"title": "Capital Gains Basics", "url": "https://www.incometax.gov.in", "chunk_id": "a/b#0", "score": 0.7,
          "snippet": "A capital gain is the profit from selling an asset"}


def disclaimer() -> str:
    return " ".join(get_config()["app"]["disclaimer"].split())


def ui(path: str, **values) -> str:
    node = get_config()["ui"]
    for part in path.split("."):
        node = node[part]
    text = " ".join(node.split())
    return text.format(**values) if values else text


def answer_result(text="A bond is a loan [1].", sources=None, **kw) -> backend.QueryResult:
    """What the facade returns for a normal turn: the backend has already added the disclaimer."""
    return backend.QueryResult(answer=f"{text}\n\n{disclaimer()}", sources=[SOURCE] if sources is None else sources,
                               intents=["finance_qa"], **kw)


class FakeBackend:
    """Stands in for src.web_app.backend: records every call, returns scripted results."""

    def __init__(self):
        self.asks, self.start_ups, self.usage_calls = [], 0, 0
        self.reply = answer_result()
        self.usage_value = backend.UsageSummary(tokens=1234, cost=0.0123, limit=2_000_000, exceeded=False)
        self.warm = backend.WarmUp(ok=True, seconds=1.5)
        self.start_up_error = None

    def ask(self, *args, **kwargs):
        self.asks.append((args, kwargs))
        return self.reply

    def start_up(self):
        self.start_ups += 1
        if self.start_up_error:
            raise self.start_up_error
        return self.warm

    def usage(self):
        self.usage_calls += 1
        return self.usage_value


@pytest.fixture
def fake(monkeypatch):
    st.cache_resource.clear()
    fb = FakeBackend()
    monkeypatch.setattr(backend, "ask", fb.ask)
    monkeypatch.setattr(backend, "start_up", fb.start_up)
    monkeypatch.setattr(backend, "usage", fb.usage)
    yield fb
    st.cache_resource.clear()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=10)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def shown(at: AppTest) -> list[str]:
    """Every text the page shows, as passed to Streamlit (markdown, captions, boxes, metrics, expander labels)."""
    out: list[str] = []
    for block in (at.main, at.sidebar):
        for kind in ("markdown", "caption", "warning", "info", "success", "error", "title", "subheader"):
            out += [e.value for e in getattr(block, kind)]
        out += [m.label + " " + m.value for m in block.metric]
    out += [x.label for x in at.expander]
    return out


# ---- first load ----

def test_first_load_shows_the_shell_the_usage_and_an_enabled_input_and_asks_nothing(fake):
    at = start()
    texts = shown(at)
    assert ui("app_title") in texts and ui("app_tagline") in texts and ui("chat.empty_state") in texts
    assert [t.label for t in at.tabs] == [ui("tabs.chat")]
    metric = at.sidebar.metric[0]
    assert metric.label == ui("sidebar.usage_tokens") == "Tokens used since the app started (all visitors)" and metric.value == "1,234"
    assert ui("sidebar.usage_limit", limit="2,000,000") in texts and ui("sidebar.usage_cost", cost="0.0123") in texts
    assert ui("sidebar.warmup_ok") in texts
    assert at.chat_input[0].disabled is False and at.chat_input[0].placeholder == ui("chat.placeholder")
    assert fake.asks == [] and fake.start_ups == 1


def test_no_limit_and_an_unknown_cost_are_said_so(fake):
    fake.usage_value = backend.UsageSummary(tokens=5, cost=None, limit=None, exceeded=False)
    texts = shown(start())
    assert ui("sidebar.usage_no_limit") in texts and ui("sidebar.usage_cost_unknown") in texts


def test_the_real_config_sets_the_runaway_guard_limit():
    assert get_config()["llm"]["max_session_tokens"] == 2_000_000


# ---- one question ----

def test_a_question_goes_to_the_facade_once_without_a_hint_and_the_answer_and_sources_are_shown(fake):
    at = start()
    thread = at.session_state["thread_id"]
    ask(at, "  What is a bond?  ")
    assert len(fake.asks) == 1
    args, kwargs = fake.asks[0]
    assert args == ("What is a bond?", thread) and kwargs == {}                      # no intent_hint, no params
    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant"]
    assert "What is a bond?" in [e.value for e in at.chat_message[0].markdown]
    assert any("A bond is a loan \\[1]." in t or "A bond is a loan [1]." in t for t in shown(at))
    assert at.expander[0].label == ui("chat.sources_title", count=1)
    inside = [e.value for e in at.expander[0].markdown] + [e.value for e in at.expander[0].caption]
    assert any("Capital Gains Basics" in t for t in inside) and any("profit from selling an asset…" in t for t in inside)
    assert f"[{ui('chat.source_link')}](https://www.incometax.gov.in)" in inside
    assert at.chat_input[0].disabled is False and at.chat_input[0].placeholder == ui("chat.placeholder")


def test_the_disclaimer_comes_from_the_backend_and_is_shown_exactly_once(fake):
    at = start()
    ask(at, "What is a bond?")
    assert sum(t.count(disclaimer()) for t in shown(at)) == 1
    ask(at, "And an ETF?")
    assert sum(t.count(disclaimer()) for t in shown(at)) == 2                         # once per answer, two answers


def test_the_facade_is_called_only_by_a_submit_and_never_by_a_rerun(fake):
    at = start()
    ask(at, "What is a bond?")
    assert len(fake.asks) == 1
    for _ in range(3):
        at.run()
    at.sidebar.button[0].click().run()                                               # a click on another widget reruns the script
    assert len(fake.asks) == 1
    ask(at, "again")
    assert len(fake.asks) == 2 and fake.start_ups == 1                               # warm-up is cached across runs


def test_the_input_is_disabled_while_a_call_runs_and_enabled_again_afterwards(fake, monkeypatch):
    seen = []
    real = st.chat_input
    monkeypatch.setattr(st, "chat_input", lambda *a, **kw: seen.append((kw.get("disabled"), a[0])) or real(*a, **kw))
    at = start()
    seen.clear()
    ask(at, "What is a bond?")
    assert seen == [(False, ui("chat.placeholder")), (True, ui("chat.placeholder_busy")), (False, ui("chat.placeholder"))]


def test_a_blank_question_is_ignored(fake):
    at = start()
    ask(at, "   ")
    assert fake.asks == [] and at.chat_message == []


# ---- stopped, needs_input, no answer ----

@pytest.mark.parametrize("stopped", ["error", "recursion"])
def test_an_error_or_recursion_stop_shows_the_backends_friendly_text_in_a_warning(fake, stopped):
    fake.reply = backend.QueryResult(answer=f"Sorry, I couldn't prepare an answer this time.\n\n{disclaimer()}", stopped=stopped)
    at = start()
    ask(at, "q")
    warnings = [w.value for w in at.main.warning]
    assert len(warnings) == 1 and "couldn't prepare an answer" in warnings[0] and "Traceback" not in warnings[0]
    assert sum(t.count(disclaimer()) for t in shown(at)) == 1


def test_a_budget_stop_shows_the_ui_message_not_the_backend_text(fake):
    fake.reply = backend.QueryResult(answer="backend budget text", stopped="budget")
    at = start()
    ask(at, "q")
    assert [w.value for w in at.main.warning] == [ui("messages.budget_stopped")]
    assert "backend budget text" not in " ".join(shown(at))


def test_a_blank_answer_gets_a_friendly_message(fake):
    fake.reply = backend.QueryResult(answer="  ")
    at = start()
    ask(at, "q")
    assert [w.value for w in at.main.warning] == [ui("messages.empty_answer")]


def _result(status):
    return {"agent": "market", "turn_id": "t", "status": status, "answer": "", "sources": [], "error": None, "data": None}


@pytest.mark.parametrize("statuses, hint", [
    (["needs_input"], "messages.needs_input_hint"),
    (["no_answer", "needs_input"], "messages.needs_input_hint"),
    (["no_answer"], "messages.no_answer_hint"),
    (["no_answer", "no_answer"], "messages.no_answer_hint"),
    (["no_answer", "error"], None),
    (["ok", "no_answer"], None),
    ([], None),
])
def test_a_hint_is_added_when_no_agent_answered(fake, statuses, hint):
    fake.reply = answer_result("Some text.", sources=[], results=[_result(s) for s in statuses])
    at = start()
    ask(at, "q")
    captions = [c.value for c in at.main.caption]
    hints = {ui("messages.needs_input_hint"), ui("messages.no_answer_hint")}
    assert [c for c in captions if c in hints] == ([ui(hint)] if hint else [])


def test_no_sources_means_no_expander(fake):
    fake.reply = answer_result(sources=[])
    at = start()
    ask(at, "q")
    assert list(at.expander) == []


# ---- budget ----

def test_a_reached_budget_shows_the_message_and_disables_the_input(fake):
    fake.usage_value = backend.UsageSummary(tokens=2_000_001, cost=1.0, limit=2_000_000, exceeded=True)
    at = start()
    assert [w.value for w in at.main.warning] == [ui("messages.budget_stopped")]
    assert at.chat_input[0].disabled is True and at.chat_input[0].placeholder == ui("chat.placeholder_stopped")
    assert fake.asks == []


# ---- new conversation ----

def test_new_conversation_gives_a_new_thread_and_an_empty_chat_and_keeps_the_usage(fake, monkeypatch):
    monkeypatch.setattr(UsageRegistry, "reset", lambda self: pytest.fail("the usage registry must not be reset"))
    at = start()
    first_thread = at.session_state["thread_id"]
    ask(at, "What is a bond?")
    assert len(at.chat_message) == 2
    at.sidebar.button[0].click().run()
    assert at.sidebar.button[0].label == ui("sidebar.new_conversation")
    assert at.session_state["thread_id"] != first_thread and at.chat_message == []
    assert ui("chat.empty_state") in shown(at) and at.sidebar.metric[0].value == "1,234"
    ask(at, "next")
    assert fake.asks[-1][0][1] == at.session_state["thread_id"] != first_thread


# ---- start-up (warm-up) ----

def test_a_failed_warm_up_is_a_warning_with_the_error_type_and_the_app_still_works(fake):
    fake.warm = backend.WarmUp(ok=False, seconds=0.1, message="RuntimeError")
    at = start()
    assert ui("sidebar.warmup_failed", error="RuntimeError") in [w.value for w in at.sidebar.warning]
    ask(at, "What is a bond?")
    assert len(fake.asks) == 1


def test_a_warm_up_that_raises_is_caught_and_never_shows_the_exception_text(fake):
    fake.start_up_error = RuntimeError("secret detail sk-abc123")
    at = start()
    assert ui("sidebar.warmup_failed", error="RuntimeError") in [w.value for w in at.sidebar.warning]
    assert "secret detail" not in " ".join(shown(at)) and "sk-abc123" not in " ".join(shown(at))
    at.run()
    assert fake.start_ups == 1                                                         # the failure is cached too


# ---- untrusted text ----

def test_hostile_text_in_an_answer_and_a_source_is_not_rendered_as_html_or_as_a_javascript_link(fake):
    source = {"title": HOSTILE, "url": "javascript:alert(1)", "chunk_id": "n/1", "score": 0.5, "snippet": HOSTILE}
    good = {"title": "Real", "url": "https://www.sebi.gov.in", "chunk_id": "n/2", "score": 0.5, "snippet": "Plain."}
    fake.reply = answer_result(f"News: {HOSTILE} [1]", sources=[source, good])
    at = start()
    ask(at, HOSTILE)
    text = "\n".join(shown(at) + [e.value for e in at.expander[0].markdown] + [e.value for e in at.expander[0].caption])
    assert "<script" not in text and "javascript:" not in text and "](javascript" not in text
    assert "&lt;script>alert\\(1)&lt;/script>" in text or "&lt;script&gt;alert" in text    # shown as literal text
    links = [e.value for e in at.expander[0].markdown if e.value.startswith(f"[{ui('chat.source_link')}]")]
    assert links == [f"[{ui('chat.source_link')}](https://www.sebi.gov.in)"]            # only the real http(s) source has a link

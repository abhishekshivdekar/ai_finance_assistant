"""Shared helpers for the web app tests: a fake backend, the AppTest driver and readers of what the page shows."""
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
    kw.setdefault("intents", ["finance_qa"])
    return backend.QueryResult(answer=f"{text}\n\n{disclaimer()}", sources=[SOURCE] if sources is None else sources, **kw)


class FakeBackend:
    """Stands in for src.web_app.backend: records every call, returns scripted results."""

    def __init__(self):
        self.asks, self.start_ups, self.usage_calls = [], 0, 0
        self.reply = answer_result()
        self.usage_value = backend.UsageSummary(tokens=1234, cost=0.0123, limit=2_000_000, exceeded=False)
        self.warm = backend.WarmUp(ok=True, seconds=1.5)
        self.start_up_error = None
        self.histories, self.history_reads, self.forgotten, self.forget_ok = {}, [], [], True
        self.persistence_status = backend.PersistenceStatus(True, " ".join(get_config()["persistence"]["messages"]["saving"].split()))

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

    def persistence(self):
        return self.persistence_status

    def read_history(self, thread_id):
        self.history_reads.append(thread_id)
        return list(self.histories.get(thread_id, []))

    def forget(self, thread_id):
        self.forgotten.append(thread_id)
        return self.forget_ok


@pytest.fixture
def fake(monkeypatch):
    st.cache_resource.clear()
    fb = FakeBackend()
    monkeypatch.setattr(backend, "ask", fb.ask)
    monkeypatch.setattr(backend, "start_up", fb.start_up)
    monkeypatch.setattr(backend, "usage", fb.usage)
    monkeypatch.setattr(backend, "persistence", fb.persistence)
    monkeypatch.setattr(backend, "read_history", fb.read_history)
    monkeypatch.setattr(backend, "forget", fb.forget)
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
    for table in list(at.table) + dataframes(at):
        frame = table.value
        out += [str(c) for c in frame.columns] + [str(v) for v in frame.to_numpy().ravel()]
    return out


def labelled(elements, label: str):
    """The one widget with this label (tests must not depend on the position of a tab's widgets)."""
    found = [e for e in elements if e.label == label]
    assert len(found) == 1, [e.label for e in elements]
    return found[0]


def dataframes(at: AppTest) -> list:
    """The tables on the page without the Portfolio tab's editor (it is on every page, empty until the user types)."""
    return [frame for frame in at.dataframe if getattr(frame, "key", None) != "portfolio_table"]


def charts(at: AppTest) -> list[dict]:
    """The Plotly figures on the page as dicts (data and layout)."""
    import json

    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]

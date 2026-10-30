"""run_query: a fresh input every call, friendly failures, the result object and the real graph behind it (no LLM, no network)."""
import copy
import re
from typing import get_type_hints

import pytest
from langgraph.errors import GraphRecursionError

from src.utils.config import get_config
from src.workflow import facade
from src.workflow.facade import QueryResult, get_graph, run_query
from src.workflow.graph import build_graph
from src.workflow.routing import Decision
from src.workflow.state import GraphState
from src.workflow.turn import APPEND_ONLY_KEYS, new_turn_input

CFG = get_config()
WF = CFG["workflow"]
MSG = WF["messages"]
DISCLAIMER = " ".join(CFG["app"]["disclaimer"].split())
SRC = {"title": "T", "url": None, "chunk_id": "a/b#0", "score": 0.5}


class FakeAgent:
    """Answers ok with a small payload; `answer` can be scripted."""

    def __init__(self, name, answer=None, sources=None):
        self.name, self.answer, self.sources = name, answer, sources or []

    def run(self, state):
        return {"agent": self.name, "turn_id": state["turn_id"], "status": "ok", "answer": self.answer or f"{self.name} answer",
                "sources": self.sources, "error": None, "data": {"source": "live", "warnings": ["w"]}}


class Classify:
    def __init__(self, *intents, confidence=0.9):
        self.decision = Decision(list(intents), confidence)

    def __call__(self, state):
        return self.decision


class FakeLLM:
    def __init__(self, text):
        self.text = text

    def invoke(self, messages):
        return type("Msg", (), {"content": self.text})()


def real_graph(*intents, **kwargs):
    return build_graph(Classify(*intents), {name: FakeAgent(name) for name in WF["agent_intents"]}, **kwargs)


class SpyGraph:
    """Stands in for a compiled graph: records stream() calls, yields scripted updates, then optionally raises."""

    def __init__(self, updates=(), error=None):
        self.updates, self.error, self.calls = list(updates), error, []

    def stream(self, turn, config, stream_mode):
        self.calls.append((turn, config, stream_mode))
        yield from self.updates
        if self.error:
            raise self.error

    def get_state(self, config):
        return type("Snap", (), {"values": {"final_answer": "spy", "turn_id": self.calls[-1][0]["turn_id"]}})()


@pytest.fixture
def use(monkeypatch):
    """use(graph) makes the facade run that graph."""
    def install(graph):
        monkeypatch.setattr(facade, "get_graph", lambda: graph)
        return graph
    return install


# ---- the input of every call ----

def test_every_call_sends_a_complete_fresh_input(use):
    spy = use(SpyGraph())
    run_query("hello", "t1")
    turn = spy.calls[0][0]
    assert set(turn) == set(get_type_hints(GraphState)) - set(APPEND_ONLY_KEYS)   # the two append-reducers cannot be cleared
    assert turn["query"] == "hello" and turn["intent_hint"] is None and turn["intents"] == [] and turn["agents_to_run"] == []
    assert turn["route_attempts"] == 0 and turn["final_answer"] == "" and turn["merge_mode"] is None and turn["sources"] == []
    assert all(turn[key] is None for key in ("portfolio", "goal_params", "market_params", "news_params", "tax_params", "topic_filter"))


def test_each_call_has_its_own_turn_id_and_a_hint_and_params_never_leak_into_the_next_call(use):
    spy = use(SpyGraph())
    run_query("a", "t1", intent_hint="market", market_params={"tickers": ["AAPL"]}, topic_filter="bonds")
    run_query("b", "t1")
    first, second = spy.calls[0][0], spy.calls[1][0]
    assert re.fullmatch(r"[0-9a-f]{32}", first["turn_id"]) and first["turn_id"] != second["turn_id"]
    assert (first["intent_hint"], first["market_params"], first["topic_filter"]) == ("market", {"tickers": ["AAPL"]}, "bonds")
    assert (second["intent_hint"], second["market_params"], second["topic_filter"]) == (None, None, None)


def test_the_thread_the_recursion_limit_and_the_updates_mode_come_from_the_call_and_config(use):
    spy = use(SpyGraph())
    run_query("hello", "thread-7")
    _, config, mode = spy.calls[0]
    assert config == {"configurable": {"thread_id": "thread-7"}, "recursion_limit": WF["recursion_limit"]} and mode == "updates"


def test_an_unknown_param_is_a_bug_in_the_caller_and_raises(use):
    spy = use(SpyGraph())
    with pytest.raises(TypeError):
        run_query("hello", "t1", colour="red")
    assert spy.calls == []


# ---- the result object on the real graph ----

def test_one_agent_gives_the_answer_the_nodes_the_intents_and_the_payloads():
    graph = real_graph("market")
    result = run_query_on(graph, "How is Apple doing?")
    assert isinstance(result, QueryResult) and result.stopped is None
    assert result.nodes == ["router", "market", "synthesizer"] and result.intents == ["market"] and result.confidence == 0.9
    assert result.answer == f"market answer\n\n{DISCLAIMER}" and result.merge_mode is None
    assert [r["agent"] for r in result.results] == ["market"] and result.results[0]["data"] == {"source": "live", "warnings": ["w"]}


def test_two_agents_merged_by_the_llm_report_llm_mode():
    merged = "Apple is up. A bond is a loan."
    graph = build_graph(Classify("market", "finance_qa"), {name: FakeAgent(name, "Apple is up." if name == "market" else "A bond is a loan.")
                                                           for name in WF["agent_intents"]}, synthesizer_llm=FakeLLM(merged))
    result = run_query_on(graph, "q")
    assert result.merge_mode == "llm" and merged in result.answer and result.nodes == ["router", "market", "finance_qa", "synthesizer"]


def test_two_agents_whose_merge_fails_a_check_report_verbatim_mode():
    graph = build_graph(Classify("market", "finance_qa"), {name: FakeAgent(name) for name in WF["agent_intents"]},
                        synthesizer_llm=FakeLLM("An invented 77 percent figure."))
    result = run_query_on(graph, "q")
    assert result.merge_mode == "verbatim" and "market answer" in result.answer and "finance_qa answer" in result.answer


def test_sources_come_from_the_final_state():
    graph = build_graph(Classify("finance_qa"), {name: FakeAgent(name, sources=[SRC]) for name in WF["agent_intents"]})
    assert run_query_on(graph, "q").sources == [SRC]


def test_a_follow_up_on_the_same_thread_sees_only_its_own_turn(use):
    graph = use(real_graph("market"))
    first = run_query("one", "same")
    second = run_query("two", "same")
    assert len(first.results) == 1 and len(second.results) == 1
    assert first.results[0]["turn_id"] != second.results[0]["turn_id"]
    assert len(graph.get_state({"configurable": {"thread_id": "same"}}).values["messages"]) == 4


def test_a_clarify_turn_has_no_results_and_no_merge_mode():
    result = run_query_on(real_graph("clarify"), "hm")
    assert result.nodes == ["router", "clarify"] and result.results == [] and result.merge_mode is None and result.stopped is None


def run_query_on(graph, query, thread="t1"):
    """run_query with the facade pointed at this graph."""
    original = facade.get_graph
    facade.get_graph = lambda: graph
    try:
        return run_query(query, thread)
    finally:
        facade.get_graph = original


# ---- failures become friendly messages ----

def test_the_real_recursion_limit_gives_the_friendly_message_with_the_disclaimer(monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["workflow"]["recursion_limit"] = 1
    monkeypatch.setattr(facade, "workflow_cfg", lambda: cfg["workflow"])
    with pytest.raises(GraphRecursionError):          # the limit really is hit by this graph
        real_graph("market").invoke(new_turn_input("q"), {"configurable": {"thread_id": "probe"}, "recursion_limit": 1})
    result = run_query_on(real_graph("market"), "q")
    assert result.stopped == "recursion" and " ".join(MSG["fallback"].split()) in result.answer and DISCLAIMER in result.answer
    assert result.nodes and result.results == []


def test_an_unexpected_error_never_shows_a_traceback_or_its_message(use):
    use(SpyGraph(updates=[{"router": {}}], error=RuntimeError("secret internals")))
    result = run_query("q", "t1")
    assert result.stopped == "error" and result.nodes == ["router"]
    assert "secret internals" not in result.answer and "Traceback" not in result.answer and DISCLAIMER in result.answer


def test_a_graph_that_cannot_be_built_is_also_a_friendly_error(monkeypatch):
    def boom():
        raise RuntimeError("no key")

    monkeypatch.setattr(facade, "get_graph", boom)
    result = run_query("q", "t1")
    assert result.stopped == "error" and result.nodes == [] and " ".join(MSG["fallback"].split()) in result.answer


def test_a_reached_budget_stops_before_the_graph_with_the_config_message(monkeypatch):
    monkeypatch.setattr(facade, "is_budget_exceeded", lambda: True)
    monkeypatch.setattr(facade, "get_graph", lambda: pytest.fail("the graph must not start"))
    result = run_query("q", "t1")
    assert result.stopped == "budget" and result.answer == " ".join(MSG["budget_exceeded"].split()) and result.nodes == []


def test_the_budget_check_uses_the_real_session_totals(monkeypatch):
    from src.core.usage_tracker import get_usage_registry

    cfg = copy.deepcopy(CFG)
    cfg["llm"]["max_session_tokens"] = 10
    monkeypatch.setattr("src.core.usage_tracker.get_config", lambda: cfg)
    registry = get_usage_registry()
    registry.reset()
    try:
        registry.record("router", "m", 6, 5)
        monkeypatch.setattr(facade, "get_graph", lambda: pytest.fail("the graph must not start"))
        assert run_query("q", "t1").stopped == "budget"
    finally:
        registry.reset()


# ---- the shared graph ----

def test_get_graph_builds_the_real_router_graph_once(monkeypatch):
    built = []
    monkeypatch.setattr(facade, "Router", lambda: "router")
    monkeypatch.setattr(facade, "build_graph", lambda classify: built.append(classify) or object())
    get_graph.cache_clear()
    try:
        assert get_graph() is get_graph() and built == ["router"]
    finally:
        get_graph.cache_clear()

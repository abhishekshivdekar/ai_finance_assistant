"""run_query: a fresh input every call, friendly failures, the result object and the real graph behind it (no LLM, no network)."""
import copy
import re
import threading
from typing import get_type_hints

import pytest
from langgraph.errors import GraphRecursionError

from src.utils.config import get_config
from test_workflow_router import FakeLLM as RouterLLM, reply
from src.workflow import facade
from src.workflow.facade import QueryResult, get_graph, run_query
from src.workflow.graph import build_graph
from src.workflow.router import Router
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

    def __init__(self, name, answer=None, sources=None, status="ok"):
        self.name, self.answer, self.sources, self.status = name, answer, sources or [], status

    def run(self, state):
        return {"agent": self.name, "turn_id": state["turn_id"], "status": self.status, "answer": self.answer or f"{self.name} answer",
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

    def __init__(self, updates=(), error=None, answer="spy"):
        self.updates, self.error, self.answer, self.calls = list(updates), error, answer, []

    def stream(self, turn, config, stream_mode):
        self.calls.append((turn, config, stream_mode))
        yield from self.updates
        if self.error:
            raise self.error

    def get_state(self, config):
        return type("Snap", (), {"values": {"final_answer": self.answer, "turn_id": self.calls[-1][0]["turn_id"]}})()


@pytest.fixture
def use(monkeypatch):
    """use(graph) makes the facade run that graph."""
    def install(graph):
        monkeypatch.setattr(facade, "get_graph", lambda: graph)
        return graph
    return install


@pytest.fixture
def run_on(use):
    """run_on(graph, query, thread="t1") runs run_query with the facade pointed at this graph."""
    def go(graph, query, thread="t1"):
        use(graph)
        return run_query(query, thread)
    return go


# ---- the input of every call ----

def test_every_call_sends_a_complete_fresh_input(use):
    spy = use(SpyGraph())
    run_query("hello", "t1")
    turn = spy.calls[0][0]
    assert set(turn) == set(get_type_hints(GraphState)) - set(APPEND_ONLY_KEYS)   # the two append-reducers cannot be cleared
    assert turn["query"] == "hello" and turn["intent_hint"] is None and turn["intents"] == [] and turn["agents_to_run"] == []
    assert turn["route_attempts"] == 0 and turn["final_answer"] == "" and turn["merge_mode"] is None and turn["sources"] == []
    assert turn["agent_parts"] is None and turn["partial_handled"] is False
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

def test_one_agent_gives_the_answer_the_nodes_the_intents_and_the_payloads(run_on):
    graph = real_graph("market")
    result = run_on(graph, "How is Apple doing?")
    assert isinstance(result, QueryResult) and result.stopped is None
    assert result.nodes == ["router", "market", "synthesizer"] and result.intents == ["market"] and result.confidence == 0.9
    assert result.answer == f"market answer\n\n{DISCLAIMER}" and result.merge_mode is None
    assert [r["agent"] for r in result.results] == ["market"] and result.results[0]["data"] == {"source": "live", "warnings": ["w"]}


def test_two_agents_merged_by_the_llm_report_llm_mode(run_on):
    merged = "Apple is up. A bond is a loan."
    graph = build_graph(Classify("market", "finance_qa"), {name: FakeAgent(name, "Apple is up." if name == "market" else "A bond is a loan.")
                                                           for name in WF["agent_intents"]}, synthesizer_llm=FakeLLM(merged))
    result = run_on(graph, "q")
    assert result.merge_mode == "llm" and merged in result.answer and result.nodes == ["router", "market", "finance_qa", "synthesizer"]


def test_two_agents_whose_merge_fails_a_check_report_verbatim_mode(run_on):
    graph = build_graph(Classify("market", "finance_qa"), {name: FakeAgent(name) for name in WF["agent_intents"]},
                        synthesizer_llm=FakeLLM("An invented 77 percent figure."))
    result = run_on(graph, "q")
    assert result.merge_mode == "verbatim" and "market answer" in result.answer and "finance_qa answer" in result.answer


def test_sources_come_from_the_final_state(run_on):
    graph = build_graph(Classify("finance_qa"), {name: FakeAgent(name, sources=[SRC]) for name in WF["agent_intents"]})
    assert run_on(graph, "q").sources == [SRC]


def test_a_follow_up_on_the_same_thread_sees_only_its_own_turn(use):
    graph = use(real_graph("market"))
    first = run_query("one", "same")
    second = run_query("two", "same")
    assert len(first.results) == 1 and len(second.results) == 1
    assert first.results[0]["turn_id"] != second.results[0]["turn_id"]
    assert len(graph.get_state({"configurable": {"thread_id": "same"}}).values["messages"]) == 4


def test_a_clarify_turn_has_no_results_and_no_merge_mode(run_on):
    result = run_on(real_graph("clarify"), "hm")
    assert result.nodes == ["router", "clarify"] and result.results == [] and result.merge_mode is None and result.stopped is None


@pytest.mark.parametrize("intent", WF["agent_intents"])
def test_every_agent_intent_runs_through_run_query_with_the_disclaimer_once(run_on, intent):
    result = run_on(real_graph(intent), "q")
    assert result.nodes == ["router", intent, "synthesizer"] and result.intents == [intent] and result.stopped is None
    assert result.answer == f"{intent} answer\n\n{DISCLAIMER}" and result.answer.count(DISCLAIMER) == 1
    assert [r["agent"] for r in result.results] == [intent]


@pytest.mark.parametrize("intent", ["out_of_scope", "specific_advice"])
def test_the_safe_response_intents_answer_without_agents(run_on, intent):
    result = run_on(real_graph(intent), "q")
    assert result.nodes == ["router", "safe_response"] and result.results == [] and result.stopped is None
    assert result.answer.strip() and result.answer.count(DISCLAIMER) <= 1


def test_a_hint_turn_skips_the_router_and_has_no_confidence(use):
    def no_router(state):
        pytest.fail("the router must not run for a hint")

    use(build_graph(no_router, {name: FakeAgent(name) for name in WF["agent_intents"]}))
    result = run_query("", "t1", intent_hint="market", market_params={"tickers": ["AAPL"]})
    assert result.nodes == ["market", "synthesizer"] and result.confidence == 0.0 and result.stopped is None
    assert [r["agent"] for r in result.results] == ["market"]


def test_every_agent_failing_gives_the_graph_fallback_text(run_on):
    agents = {name: FakeAgent(name, "boom", status="error") for name in WF["agent_intents"]}
    result = run_on(build_graph(Classify("market"), agents), "q")
    assert result.nodes[-1] == "fallback" and " ".join(MSG["fallback"].split()) in result.answer and result.stopped is None


def test_an_agent_needing_input_ends_in_clarify_with_its_message(run_on):
    agents = {name: FakeAgent(name, "Please tell me which tickers.", status="needs_input") for name in WF["agent_intents"]}
    result = run_on(build_graph(Classify("market"), agents), "q")
    assert result.nodes[-1] == "clarify" and "Please tell me which tickers." in result.answer and result.stopped is None
    assert [r["status"] for r in result.results] == ["needs_input"]


def test_the_router_sees_the_history_of_the_same_thread_but_not_of_another(use):
    llm = RouterLLM(reply(("finance_qa",)))
    use(build_graph(Router(llm), {name: FakeAgent(name, "Microsoft answer") for name in WF["agent_intents"]}))
    run_query("How is Microsoft doing?", "thread-a")
    run_query("And what about Apple?", "thread-a")
    run_query("And what about Tesla?", "thread-b")
    first, second, other = (" ".join(m.content for m in call) for call in llm.runner.calls)
    assert "Microsoft" not in first.replace("How is Microsoft doing?", "")
    assert "How is Microsoft doing?" in second and "Microsoft answer" in second
    assert "Microsoft" not in other


# ---- failures become friendly messages ----

def test_the_real_recursion_limit_gives_the_friendly_message_with_the_disclaimer(monkeypatch, run_on):
    cfg = copy.deepcopy(CFG)
    cfg["workflow"]["recursion_limit"] = 1
    monkeypatch.setattr(facade, "workflow_cfg", lambda: cfg["workflow"])
    with pytest.raises(GraphRecursionError):          # the limit really is hit by this graph
        real_graph("market").invoke(new_turn_input("q"), {"configurable": {"thread_id": "probe"}, "recursion_limit": 1})
    result = run_on(real_graph("market"), "q")
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


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_a_finished_graph_with_a_blank_answer_gives_the_fallback_with_stopped_error(use, caplog, answer):
    use(SpyGraph(updates=[{"router": {}}], answer=answer))
    with caplog.at_level("WARNING"):
        result = run_query("q", "t1")
    assert result.stopped == "error" and result.nodes == ["router"] and " ".join(MSG["fallback"].split()) in result.answer
    assert DISCLAIMER in result.answer and "empty answer" in caplog.text


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


# ---- one turn at a time per thread ----

class BlockingGraph(SpyGraph):
    """Every stream() waits for `release`; records how many streams run at once."""

    def __init__(self):
        super().__init__()
        self.entered, self.release, self.active, self.max_active = threading.Semaphore(0), threading.Event(), 0, 0
        self.count_lock = threading.Lock()

    def stream(self, turn, config, stream_mode):
        with self.count_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.release()
        self.release.wait(timeout=5)
        try:
            yield from super().stream(turn, config, stream_mode)
        finally:
            with self.count_lock:
                self.active -= 1


def _start_two(use, ids):
    graph = use(BlockingGraph())
    threads = [threading.Thread(target=run_query, args=(f"q{i}", thread_id)) for i, thread_id in enumerate(ids)]
    for t in threads:
        t.start()
    return graph, threads


def test_two_calls_on_one_thread_run_one_after_the_other(use):
    graph, threads = _start_two(use, ["same", "same"])
    assert graph.entered.acquire(timeout=5)
    assert not graph.entered.acquire(timeout=0.3)      # the second call is waiting for the lock
    graph.release.set()
    for t in threads:
        t.join(timeout=5)
    assert graph.max_active == 1 and len(graph.calls) == 2


def test_calls_on_different_threads_do_not_wait_for_each_other(use):
    graph, threads = _start_two(use, ["one", "two"])
    assert graph.entered.acquire(timeout=5) and graph.entered.acquire(timeout=5)
    graph.release.set()
    for t in threads:
        t.join(timeout=5)
    assert graph.max_active == 2


def test_one_lock_per_thread_id():
    assert facade._thread_lock("x") is facade._thread_lock("x") and facade._thread_lock("x") is not facade._thread_lock("y")


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


# ---- warm_up and usage_summary ----

def test_warm_up_loads_the_embedder_the_store_and_the_graph_and_reports_success(monkeypatch):
    calls = []

    class Embedder:
        def embed_query(self, text):
            calls.append(("embed", text))

    monkeypatch.setattr(facade, "get_embedder", lambda: Embedder())
    monkeypatch.setattr(facade, "get_vector_store", lambda: calls.append("store"))
    monkeypatch.setattr(facade, "get_graph", lambda: calls.append("graph"))
    result = facade.warm_up()
    assert result.ok and result.message == "" and result.seconds >= 0
    assert calls == [("embed", "warm up"), "store", "graph"]


@pytest.mark.parametrize("failing", ["get_embedder", "get_vector_store", "get_graph"])
def test_a_failed_warm_up_never_raises_and_reports_only_the_error_type(monkeypatch, caplog, failing):
    class Embedder:
        def embed_query(self, text):
            return None

    for name, value in {"get_embedder": lambda: Embedder(), "get_vector_store": lambda: None, "get_graph": lambda: None}.items():
        monkeypatch.setattr(facade, name, value)

    def boom():
        raise RuntimeError("secret detail sk-should-not-show")

    monkeypatch.setattr(facade, failing, boom)
    result = facade.warm_up()
    assert not result.ok and result.message == "RuntimeError" and "sk-" not in result.message
    assert "Warm-up failed" in caplog.text


def test_warm_up_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr("src.core.llm_factory.get_llm", lambda *a, **k: pytest.fail("no LLM call in warm_up"))
    monkeypatch.setattr(facade, "get_embedder", lambda: type("E", (), {"embed_query": lambda self, text: None})())
    monkeypatch.setattr(facade, "get_vector_store", lambda: None)
    monkeypatch.setattr(facade, "get_graph", lambda: None)
    assert facade.warm_up().ok


@pytest.fixture
def clean_registry():
    from src.core.usage_tracker import get_usage_registry

    registry = get_usage_registry()
    registry.reset()
    yield registry
    registry.reset()


def test_usage_summary_reads_the_registry_and_the_limit_from_the_config(clean_registry):
    limit = get_config()["llm"]["max_session_tokens"]
    assert limit == 2_000_000
    assert facade.usage_summary() == facade.UsageSummary(tokens=0, cost=0.0, limit=limit, exceeded=False)
    clean_registry.record("router", get_config()["llm"]["model"], 1000, 500)
    clean_registry.record("synthesizer", get_config()["llm"]["model"], 100, 50)
    summary = facade.usage_summary()
    price = get_config()["pricing"][get_config()["llm"]["model"]]
    assert summary.tokens == 1650 and summary.limit == limit and not summary.exceeded
    assert summary.cost == pytest.approx((1100 * price["input"] + 550 * price["output"]) / 1_000_000)


def test_usage_summary_cost_is_unknown_when_a_model_has_no_price(clean_registry):
    clean_registry.record("router", "no-such-model", 10, 10)
    assert facade.usage_summary().cost is None and facade.usage_summary().tokens == 20


def test_usage_summary_shows_the_limit_and_whether_it_is_reached(clean_registry, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["max_session_tokens"] = 100
    monkeypatch.setattr("src.core.usage_tracker.get_config", lambda: cfg)
    monkeypatch.setattr(facade, "get_config", lambda: cfg)
    clean_registry.record("router", "m", 40, 20)
    assert facade.usage_summary() == facade.UsageSummary(tokens=60, cost=None, limit=100, exceeded=False)
    clean_registry.record("router", "m", 30, 10)
    assert facade.usage_summary().exceeded is True


def test_the_budget_message_does_not_promise_that_a_new_session_helps():
    text = " ".join(MSG["budget_exceeded"].split()).lower()
    assert "new session" not in text and "restart" in text and "run of the app" in text

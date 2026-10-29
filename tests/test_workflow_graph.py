"""The LangGraph workflow with fake agents and a fake router: one test per edge of architecture 4.4."""
import copy
from typing import get_type_hints

import pytest
from langgraph.errors import GraphRecursionError

from src.utils.config import get_config
from src.workflow import graph as graph_module
from src.workflow import routing
from src.workflow.graph import build_graph
from src.workflow.routing import Decision
from src.workflow.state import GraphState
from src.workflow.turn import APPEND_ONLY_KEYS, new_turn_input

CFG = get_config()
DISCLAIMER = " ".join(CFG["app"]["disclaimer"].split())
MSG = CFG["workflow"]["messages"]
AGENTS = CFG["workflow"]["agent_intents"]
SRC = {"title": "T", "url": None, "chunk_id": "a/b#0", "score": 0.5}


class FakeAgent:
    """Returns the scripted statuses in order (the last one repeats); records every call."""

    def __init__(self, name, *statuses, answer=None, sources=None, error=None):
        self.name, self.statuses, self.answer = name, list(statuses or ["ok"]), answer
        self.sources, self.error, self.calls = sources or [], error, []

    def run(self, state):
        status = self.statuses[min(len(self.calls), len(self.statuses) - 1)]
        self.calls.append(state["turn_id"])
        text = self.answer if self.answer is not None else f"{self.name} {status}"
        error = self.error if status == "error" else None
        return {"agent": self.name, "turn_id": state["turn_id"], "status": status, "answer": text,
                "sources": self.sources, "error": error, "data": None}


class Classify:
    """A fake router: fixed decision, records the route_attempts it saw."""

    def __init__(self, intents=(), confidence=0.9, market_params=None, news_params=None):
        self.decision = Decision(list(intents), confidence, market_params, news_params)
        self.seen = []

    def __call__(self, state):
        self.seen.append({"route_attempts": state["route_attempts"], "intent_hint": state["intent_hint"]})
        return self.decision


def agents(**scripted):
    """All six fake agents; scripted = name -> FakeAgent replacing the default ok one."""
    return {name: scripted.get(name) or FakeAgent(name) for name in AGENTS}


def run(graph, query="hello", thread="t1", **kwargs):
    """Run one turn; returns (nodes visited in order, final state values)."""
    config = {"configurable": {"thread_id": thread}, "recursion_limit": CFG["workflow"]["recursion_limit"]}
    visited = []
    for update in graph.stream(new_turn_input(query, **kwargs), config, stream_mode="updates"):
        visited += list(update)
    return visited, graph.get_state(config).values


def count(text, part):
    return text.count(part)


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Nothing in the workflow tests may create a real LLM."""
    def fail(**kwargs):
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr("src.agents.base_agent.get_llm", fail)


def patch_workflow(monkeypatch, **changes):
    """Replace keys of the workflow config for the routing helpers (edges and nodes read it through routing)."""
    cfg = copy.deepcopy(get_config())
    cfg["workflow"].update(changes)
    monkeypatch.setattr(routing, "get_config", lambda: cfg)


# ---- START edge ----

@pytest.mark.parametrize("hint", ["portfolio", "market", "goal_planning", "tax_education", "news", "finance_qa"])
def test_every_hint_goes_straight_to_its_agent_without_the_router(hint):
    classify = Classify(["finance_qa"])
    visited, state = run(build_graph(classify, agents()), intent_hint=hint)
    assert visited == [hint, "synthesizer"] and classify.seen == []
    assert state["final_answer"].startswith(f"{hint} ok")


def test_no_hint_goes_to_the_router():
    visited, _ = run(build_graph(Classify(["market"]), agents()))
    assert visited == ["router", "market", "synthesizer"]


@pytest.mark.parametrize("hint", ["clarify", "nonsense", "specific_advice"])
def test_an_unknown_hint_is_ignored_and_the_router_decides(hint):
    classify = Classify(["market"])
    visited, _ = run(build_graph(classify, agents()), intent_hint=hint)
    assert visited[0] == "router" and len(classify.seen) == 1


# ---- router edges ----

@pytest.mark.parametrize("intents", [
    ["specific_advice"], ["specific_advice", "market"], ["market", "specific_advice", "finance_qa"],
])
def test_specific_advice_goes_to_safe_response_and_no_agent_runs(intents):
    fakes = agents()
    visited, state = run(build_graph(Classify(intents), fakes), "Should I buy Tesla?")
    assert visited == ["router", "safe_response"]
    assert all(not fake.calls for fake in fakes.values())
    assert " ".join(MSG["advice_static"].split()) in state["final_answer"]


def test_specific_advice_wins_even_when_confidence_is_low():
    visited, _ = run(build_graph(Classify(["specific_advice"], confidence=0.1), agents()))
    assert visited == ["router", "safe_response"]


def test_out_of_scope_alone_goes_to_safe_response_with_its_own_message():
    fakes = agents()
    visited, state = run(build_graph(Classify(["out_of_scope"]), fakes), "Best pasta recipe?")
    assert visited == ["router", "safe_response"] and not any(f.calls for f in fakes.values())
    assert " ".join(MSG["out_of_scope"].split()) in state["final_answer"]


def test_out_of_scope_next_to_a_real_agent_is_dropped_and_the_agent_runs():
    visited, _ = run(build_graph(Classify(["out_of_scope", "market"]), agents()))
    assert visited == ["router", "market", "synthesizer"]


def test_low_confidence_goes_to_clarify_and_no_agent_runs():
    fakes = agents()
    visited, state = run(build_graph(Classify(["market"], confidence=0.2), fakes))
    assert visited == ["router", "clarify"] and not any(f.calls for f in fakes.values())
    assert state["final_answer"] == " ".join(MSG["clarify_low_confidence"].split())


def test_confidence_exactly_at_the_threshold_is_enough():
    threshold = CFG["workflow"]["router_min_confidence"]
    visited, _ = run(build_graph(Classify(["market"], confidence=threshold), agents()))
    assert visited == ["router", "market", "synthesizer"]


def test_the_clarify_intent_goes_to_clarify_even_with_high_confidence():
    visited, _ = run(build_graph(Classify(["clarify"], confidence=1.0), agents()))
    assert visited == ["router", "clarify"]


@pytest.mark.parametrize("intents", [[], ["not_an_intent"]])
def test_a_router_result_with_no_usable_intent_goes_to_fallback(intents):
    visited, state = run(build_graph(Classify(intents), agents()))
    assert visited == ["router", "fallback"] and " ".join(MSG["fallback"].split()) in state["final_answer"]


def test_two_agents_run_in_the_routers_order_then_the_synthesizer_merges():
    fakes = agents(market=FakeAgent("market", answer="Apple is at 190."),
                   finance_qa=FakeAgent("finance_qa", answer="P/E is price over earnings."))
    visited, state = run(build_graph(Classify(["market", "finance_qa"]), fakes))
    assert visited == ["router", "market", "finance_qa", "synthesizer"]
    assert state["final_answer"].index("Apple is at 190.") < state["final_answer"].index("P/E is price")


def test_more_agents_than_the_cap_are_dropped():
    visited, _ = run(build_graph(Classify(["market", "news", "finance_qa"]), agents()))
    assert visited == ["router", "market", "news", "synthesizer"]


def test_the_router_keeps_a_tab_param_when_it_found_nothing_and_writes_what_it_found():
    _, state = run(build_graph(Classify(["finance_qa"]), agents()), market_params={"tickers": ["TCS.NS"]})
    assert state["market_params"] == {"tickers": ["TCS.NS"]}
    _, state = run(build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}, news_params={"ticker": "AAPL"}), agents()))
    assert state["market_params"] == {"tickers": ["AAPL"]} and state["news_params"] == {"ticker": "AAPL"}


# ---- agent edges ----

def test_ok_goes_to_the_synthesizer_which_adds_the_disclaimer_once():
    _, state = run(build_graph(Classify(["finance_qa"]), agents(finance_qa=FakeAgent("finance_qa", answer="Answer [1].", sources=[SRC]))))
    assert state["final_answer"] == f"Answer [1].\n\n{DISCLAIMER}" and state["sources"] == [SRC]


@pytest.mark.parametrize("status", ["no_answer", "wrong_agent"])
def test_no_answer_and_wrong_agent_reroute_when_the_router_found_a_ticker(status):
    fakes = agents(finance_qa=FakeAgent("finance_qa", status), market=FakeAgent("market", answer="Apple is at 190."))
    graph = build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}), fakes)
    visited, state = run(graph, "what is NVDA doing today")
    assert visited == ["router", "finance_qa", "reroute", "market", "synthesizer"]
    assert state["route_attempts"] == 1 and "Apple is at 190." in state["final_answer"]
    assert "finance_qa" not in state["final_answer"]


def test_no_answer_without_a_ticker_is_presented_by_the_synthesizer_with_the_disclaimer():
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer", answer="The knowledge base has no good match."))
    visited, state = run(build_graph(Classify(["finance_qa"]), fakes))
    assert visited == ["router", "finance_qa", "synthesizer"]
    assert state["final_answer"] == f"The knowledge base has no good match.\n\n{DISCLAIMER}" and state["route_attempts"] == 0


def test_no_answer_goes_on_to_the_next_queued_agent_before_the_end_rules():
    fakes = agents(market=FakeAgent("market", "no_answer"), finance_qa=FakeAgent("finance_qa", answer="Concept."))
    visited, state = run(build_graph(Classify(["market", "finance_qa"]), fakes))
    assert visited == ["router", "market", "finance_qa", "synthesizer"] and "Concept." in state["final_answer"]


def test_the_reroute_cap_stops_a_chain_of_no_answers(monkeypatch):
    patch_workflow(monkeypatch, reroute={
        "finance_qa": [{"agent": "market", "needs": "market_params"}],
        "market": [{"agent": "news", "needs": "market_params"}],
        "news": [{"agent": "tax_education", "needs": "market_params"}],
    })
    fakes = agents(**{n: FakeAgent(n, "no_answer") for n in ("finance_qa", "market", "news", "tax_education")})
    visited, state = run(build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}), fakes))
    assert visited == ["router", "finance_qa", "reroute", "market", "reroute", "news", "synthesizer"]
    assert state["route_attempts"] == CFG["workflow"]["max_route_attempts"] == 2
    assert fakes["tax_education"].calls == []


def test_a_reroute_target_that_already_ran_is_not_tried_again(monkeypatch):
    patch_workflow(monkeypatch, reroute={"finance_qa": [{"agent": "market", "needs": "market_params"}],
                                         "market": [{"agent": "finance_qa", "needs": "market_params"}]})
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer"), market=FakeAgent("market", "no_answer"))
    visited, _ = run(build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}), fakes))
    assert visited == ["router", "finance_qa", "reroute", "market", "synthesizer"]


def test_a_reroute_is_skipped_when_the_candidate_is_already_queued(monkeypatch):
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer"))
    visited, _ = run(build_graph(Classify(["finance_qa", "market"], market_params={"tickers": ["AAPL"]}), fakes))
    assert visited == ["router", "finance_qa", "market", "synthesizer"]       # market runs from the queue, no reroute node


def test_a_zero_reroute_cap_never_reroutes(monkeypatch):
    patch_workflow(monkeypatch, max_route_attempts=0)
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer"))
    visited, _ = run(build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}), fakes))
    assert visited == ["router", "finance_qa", "synthesizer"]


# ---- needs_input (deviation from architecture 4.4: the queue goes on) ----

def test_a_single_agent_needs_input_goes_to_clarify_with_the_agents_message_and_no_disclaimer():
    fakes = agents(market=FakeAgent("market", "needs_input", answer="Which ticker do you mean?"))
    visited, state = run(build_graph(Classify(["market"]), fakes), "How is it doing?")
    assert visited == ["router", "market", "clarify"]
    assert state["final_answer"] == "Which ticker do you mean?" and DISCLAIMER not in state["final_answer"]


def test_a_tab_hint_with_needs_input_goes_to_clarify():
    fakes = agents(goal_planning=FakeAgent("goal_planning", "needs_input", answer="Enter your goal amount."))
    visited, state = run(build_graph(Classify(), fakes), intent_hint="goal_planning")
    assert visited == ["goal_planning", "clarify"] and state["final_answer"] == "Enter your goal amount."


@pytest.mark.parametrize("order", [("market", "finance_qa"), ("finance_qa", "market")])
def test_needs_input_in_a_two_agent_queue_continues_and_the_ok_answer_gets_a_note(order):
    fakes = agents(market=FakeAgent("market", "needs_input", answer="Which ticker?"),
                   finance_qa=FakeAgent("finance_qa", answer="A P/E ratio is price over earnings."))
    visited, state = run(build_graph(Classify(list(order)), fakes))
    assert visited == ["router", *order, "synthesizer"]
    answer = state["final_answer"]
    assert answer.startswith("A P/E ratio is price over earnings.")
    assert MSG["needs_input_note"].format(label="Market", message="Which ticker?") in answer
    assert count(answer, DISCLAIMER) == 1


def test_two_needs_input_results_go_to_clarify_with_both_messages():
    fakes = agents(market=FakeAgent("market", "needs_input", answer="Which ticker?"),
                   news=FakeAgent("news", "needs_input", answer="Which company?"))
    visited, state = run(build_graph(Classify(["market", "news"]), fakes))
    assert visited == ["router", "market", "news", "clarify"]
    assert state["final_answer"] == "Which ticker?\n\nWhich company?"


def test_needs_input_beats_no_answer_when_nothing_is_ok():
    fakes = agents(market=FakeAgent("market", "needs_input", answer="Which ticker?"),
                   finance_qa=FakeAgent("finance_qa", "no_answer"))
    visited, _ = run(build_graph(Classify(["market", "finance_qa"]), fakes))
    assert visited[-1] == "clarify"


# ---- errors ----

def test_one_agent_error_gives_the_other_answer_with_a_note_and_records_the_error():
    fakes = agents(market=FakeAgent("market", "error", error="market failed"),
                   finance_qa=FakeAgent("finance_qa", answer="A P/E ratio is price over earnings."))
    visited, state = run(build_graph(Classify(["market", "finance_qa"]), fakes))
    assert visited == ["router", "market", "finance_qa", "synthesizer"]
    assert " ".join(MSG["error_note"].format(label="Market").split()) in state["final_answer"]
    assert [e["message"] for e in state["errors"]] == ["market failed"] and state["errors"][0]["turn_id"] == state["turn_id"]


@pytest.mark.parametrize("queue", [["market"], ["market", "news"]])
def test_every_result_an_error_goes_to_fallback(queue):
    fakes = agents(market=FakeAgent("market", "error", error="x"), news=FakeAgent("news", "error", error="y"))
    visited, state = run(build_graph(Classify(queue), fakes))
    assert visited[-1] == "fallback" and " ".join(MSG["fallback"].split()) in state["final_answer"]


def test_a_tab_hint_agent_error_goes_to_fallback():
    fakes = agents(market=FakeAgent("market", "error", error="x"))
    visited, _ = run(build_graph(Classify(), fakes), intent_hint="market")
    assert visited == ["market", "fallback"]


def test_error_plus_no_answer_presents_the_no_answer_message():
    fakes = agents(market=FakeAgent("market", "error", error="x"), finance_qa=FakeAgent("finance_qa", "no_answer", answer="No match."))
    visited, state = run(build_graph(Classify(["market", "finance_qa"]), fakes))
    assert visited[-1] == "synthesizer" and state["final_answer"].startswith("No match.")


def test_only_wrong_agent_results_go_to_fallback():
    fakes = agents(market=FakeAgent("market", "wrong_agent"))
    visited, _ = run(build_graph(Classify(["market"]), fakes))
    assert visited[-1] == "fallback"


# ---- the disclaimer: exactly once on final answers, never on clarify ----

SCENARIOS = {
    "synthesizer one ok": (["finance_qa"], {}),
    "synthesizer two ok": (["market", "finance_qa"], {}),
    "synthesizer no_answer": (["finance_qa"], {"finance_qa": FakeAgent("finance_qa", "no_answer", answer="No match.")}),
    "synthesizer partial": (["market", "finance_qa"], {"market": FakeAgent("market", "error", error="x")}),
    "safe_response advice": (["specific_advice"], {}),
    "safe_response out_of_scope": (["out_of_scope"], {}),
    "fallback": ([], {}),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_disclaimer_appears_exactly_once(name):
    intents, scripted = SCENARIOS[name]
    _, state = run(build_graph(Classify(intents), agents(**scripted)))
    assert count(state["final_answer"], DISCLAIMER) == 1 and state["final_answer"].endswith(DISCLAIMER)


def test_clarify_adds_no_disclaimer_and_no_financial_content():
    _, state = run(build_graph(Classify(["market"], confidence=0.1), agents()))
    assert DISCLAIMER not in state["final_answer"]


def test_agent_answers_never_carry_the_disclaimer_so_the_merge_cannot_double_it():
    _, state = run(build_graph(Classify(["market", "news"]), agents()))
    assert count(state["final_answer"], DISCLAIMER) == 1


def test_the_final_answer_is_added_to_the_history_as_an_assistant_message():
    _, state = run(build_graph(Classify(["finance_qa"]), agents()), "What is a bond?")
    assert [m.type for m in state["messages"]] == ["human", "ai"] and state["messages"][1].content == state["final_answer"]


# ---- per-turn hygiene (architecture 3.2) ----

def test_a_second_turn_on_the_same_thread_starts_clean():
    classify = Classify(["finance_qa"])
    fakes = agents(market=FakeAgent("market", answer="Turn one market."), finance_qa=FakeAgent("finance_qa", answer="Turn two concept."))
    graph = build_graph(classify, fakes)
    visited1, state1 = run(graph, "Analyse AAPL", intent_hint="market", market_params={"tickers": ["AAPL"]})
    visited2, state2 = run(graph, "What is a bond?")
    assert visited1 == ["market", "synthesizer"] and visited2 == ["router", "finance_qa", "synthesizer"]
    assert classify.seen == [{"route_attempts": 0, "intent_hint": None}]          # the router ran only in turn 2, from a clean start
    assert "Turn one" not in state2["final_answer"] and "Turn two concept." in state2["final_answer"]
    assert state2["turn_id"] != state1["turn_id"] and state2["intents"] == ["finance_qa"] and state2["route_attempts"] == 0
    assert state2["market_params"] is None                                         # turn 1's params did not carry over
    assert len(state2["agent_outputs"]) == 2                                       # history is kept, but filtered by turn_id
    assert [r["turn_id"] for r in state2["agent_outputs"]] == [state1["turn_id"], state2["turn_id"]]


def test_the_previous_turns_reroute_count_and_queue_do_not_leak():
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer"))
    graph = build_graph(Classify(["finance_qa"], market_params={"tickers": ["AAPL"]}), fakes)
    _, first = run(graph)
    _, second = run(graph)
    assert first["route_attempts"] == 1 and second["route_attempts"] == 1          # each turn counts from 0, not 1 -> 2


def test_new_turn_input_sets_every_key_except_the_append_reducers():
    keys = set(get_type_hints(GraphState))
    first, second = new_turn_input("a"), new_turn_input("b", intent_hint="news", topic_filter="stocks", tax_params={"calculation": "x"})
    assert set(first) == keys - set(APPEND_ONLY_KEYS)
    assert first["turn_id"] != second["turn_id"] and first["intent_hint"] is None
    assert second["intent_hint"] == "news" and second["topic_filter"] == "stocks" and second["tax_params"] == {"calculation": "x"}
    assert first["intents"] == [] and first["agents_to_run"] == [] and first["route_attempts"] == 0 and first["final_answer"] == ""


# ---- graph wiring ----

def test_the_recursion_limit_is_a_backstop():
    graph = build_graph(Classify(["market", "news", "finance_qa"]), agents())
    config = {"configurable": {"thread_id": "r"}, "recursion_limit": 2}
    with pytest.raises(GraphRecursionError):
        graph.invoke(new_turn_input("x"), config)


def test_the_default_agents_are_the_six_real_agents(monkeypatch):
    monkeypatch.setattr("src.agents.base_agent.get_llm", lambda **kw: object())
    built = graph_module.default_agents()
    assert set(built) == set(AGENTS) and built["tax_education"].name == "tax_education"


def test_build_graph_uses_the_default_agents_when_none_are_given(monkeypatch):
    fakes = agents(market=FakeAgent("market", answer="From the default set."))
    monkeypatch.setattr(graph_module, "default_agents", lambda: fakes)
    _, state = run(build_graph(Classify(["market"])))
    assert "From the default set." in state["final_answer"]

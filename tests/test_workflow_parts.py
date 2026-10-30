"""Commit 5 of the workflow: re-route only without an ok result, per-agent parts, more than the maximum agents, safe_response prompt rules."""
import pytest
from langchain_core.messages import HumanMessage

from src.core.models import AgentResult
from src.utils.config import get_config
from src.workflow.graph import build_graph
from src.workflow.router import RouterOutput, RouterPart, TooManyAgentsError, read_output, read_parts
from src.workflow.routing import Decision, reroute_target
from test_workflow_graph import Classify, FakeAgent, agents, run
from test_workflow_router import EMPTY, MIN, classify, reply

CFG = get_config()
WF = CFG["workflow"]
MSG = WF["messages"]
CAP = WF["max_agents_per_turn"]
MARKET_Q = {"agent": "market", "question": "What is Apple trading at?"}
QA_Q = {"agent": "finance_qa", "question": "What is a P/E ratio?"}
PARAMS = {"market_params": {"tickers": ["AAPL"]}, "news_params": {"ticker": "AAPL"}}


def result(agent, status):
    return AgentResult(agent=agent, turn_id="t", status=status, answer="a", sources=[], error=None, data=None)


class QueryRecorder(FakeAgent):
    """A fake agent that remembers the query it was given."""

    def run(self, state):
        self.queries = [*getattr(self, "queries", []), state["query"]]
        return super().run(state)


# ---- 1. re-route only when no result is ok ----

def test_an_ok_result_anywhere_in_the_turn_means_no_reroute_target():
    state = {"turn_id": "t", "agent_outputs": [result("market", "ok"), result("finance_qa", "no_answer")], **PARAMS}
    assert reroute_target(state, WF) is None
    state["agent_outputs"] = [result("finance_qa", "no_answer")]
    assert reroute_target(state, WF) == "market"


def test_market_ok_and_finance_qa_no_answer_does_not_pull_in_an_unasked_news_agent():
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer", answer="No match."))
    graph = build_graph(Classify(["market", "finance_qa"], market_params=PARAMS["market_params"], news_params=PARAMS["news_params"]), fakes)
    visited, state = run(graph, "What is Apple trading at, and what is a P/E ratio?")
    assert visited == ["router", "market", "finance_qa", "synthesizer"] and fakes["news"].calls == []
    assert state["route_attempts"] == 0 and MSG["no_answer_note"].format(label="Concepts", message="No match.") in state["final_answer"]


def test_a_lone_no_answer_with_a_ticker_still_reroutes_once_and_nothing_else_is_pulled_in():
    fakes = agents(finance_qa=FakeAgent("finance_qa", "no_answer"), market=FakeAgent("market", "no_answer"))
    visited, state = run(build_graph(Classify(["finance_qa"], **PARAMS), fakes))
    assert visited == ["router", "finance_qa", "reroute", "market", "synthesizer"] and fakes["news"].calls == []
    assert state["route_attempts"] == 1          # the cap itself is tested in test_workflow_graph.py


# ---- 2. parts ----

def test_read_parts_keeps_valid_parts_for_queued_agents():
    assert read_parts([MARKET_Q, QA_Q], ["market", "finance_qa"], WF) == {"market": "What is Apple trading at?", "finance_qa": "What is a P/E ratio?"}
    assert read_parts({"market": " What  is\nApple? "}, ["market", "news"], WF) == {"market": "What is Apple?"}      # a mapping works too
    assert read_parts([RouterPart(**MARKET_Q)], ["market", "news"], WF) == {"market": "What is Apple trading at?"}


@pytest.mark.parametrize("raw, expected", [
    ([MARKET_Q, {"agent": "news", "question": "Latest news?"}], {"market": "What is Apple trading at?"}),    # news is not queued
    ([MARKET_Q, {"agent": "finance_qa", "question": "   "}], {"market": "What is Apple trading at?"}),      # empty
    ([MARKET_Q, {"agent": "finance_qa", "question": 7}], {"market": "What is Apple trading at?"}),          # not a string
    ([MARKET_Q, {"agent": "finance_qa", "question": "x" * (WF["part_max_chars"] + 1)}], {"market": "What is Apple trading at?"}),
    ([MARKET_Q, {"agent": "market", "question": "A second market question"}], {"market": "What is Apple trading at?"}),   # first wins
    (["junk", None, 3, MARKET_Q], {"market": "What is Apple trading at?"}),
    ([{"agent": "finance_qa", "question": "x" * WF["part_max_chars"]}], {"finance_qa": "x" * WF["part_max_chars"]}),   # exactly the cap is fine
    ([], None), (None, None), ("junk", None), ([{"agent": "other", "question": "q"}], None),
])
def test_read_parts_drops_everything_that_is_not_valid(raw, expected):
    assert read_parts(raw, ["market", "finance_qa"], WF) == expected


def test_parts_need_two_or_more_agents():
    assert read_parts([MARKET_Q], ["market"], WF) is None and read_parts([MARKET_Q], [], WF) is None


def test_the_router_returns_the_parts_of_a_compound_question():
    decision, _ = classify(reply(("market", "finance_qa"), parts=[MARKET_Q, QA_Q], tickers=["Apple"]), "What is Apple trading at, and what is a P/E ratio?")
    assert decision.parts == {"market": "What is Apple trading at?", "finance_qa": "What is a P/E ratio?"} and decision.partial is False


def test_a_single_agent_reply_has_no_parts_even_if_the_model_sent_one():
    decision, _ = classify(reply(("market",), parts=[MARKET_Q]), "Apple?")
    assert decision.parts is None


def test_advice_never_has_parts():
    decision, _ = classify(reply(("market", "finance_qa", "specific_advice"), parts=[MARKET_Q, QA_Q]), "Should I buy Apple?")
    assert decision.parts is None


def test_a_reply_without_a_parts_field_is_still_valid():
    reply_without = {k: v for k, v in reply(("market",)).items() if k != "parts"}
    decision, _ = classify(reply_without, "Apple?")
    assert decision.intents == ["market"] and decision.parts is None
    assert RouterOutput(intents=["market"], confidence=1, entities=EMPTY, parts=[]).parts == []


def test_each_agent_gets_its_own_part_and_the_full_query_stays_in_the_state():
    qa, market = QueryRecorder("finance_qa"), QueryRecorder("market")
    parts = {"market": "What is Apple trading at?", "finance_qa": "What is a P/E ratio?"}
    classifier = Classify(["market", "finance_qa"])
    classifier.decision = Decision(["market", "finance_qa"], 0.9, parts=parts)
    full = "What is Apple trading at, and what is a P/E ratio?"
    _, state = run(build_graph(classifier, agents(finance_qa=qa, market=market)), full)
    assert market.queries == [parts["market"]] and qa.queries == [parts["finance_qa"]]
    assert state["query"] == full and state["messages"][0].content == full and state["agent_parts"] == parts


def test_an_agent_without_a_part_and_a_rerouted_agent_get_the_full_query():
    market, qa = QueryRecorder("market"), QueryRecorder("finance_qa", "no_answer")
    classifier = Classify(["finance_qa"], market_params=PARAMS["market_params"])
    _, state = run(build_graph(classifier, agents(finance_qa=qa, market=market)), "full question")
    assert qa.queries == ["full question"] and market.queries == ["full question"] and state["agent_parts"] is None


def test_parts_do_not_leak_into_the_next_turn_on_the_same_thread():
    market, qa = QueryRecorder("market"), QueryRecorder("finance_qa")
    classifier = Classify(["market", "finance_qa"])
    classifier.decision = Decision(["market", "finance_qa"], 0.9, parts={"market": "part"})
    graph = build_graph(classifier, agents(finance_qa=qa, market=market))
    run(graph, "first")
    classifier.decision = Decision(["market", "finance_qa"], 0.9)
    run(graph, "second")
    assert market.queries == ["part", "second"]


# ---- 3. more agents than the maximum ----

def test_the_maximum_is_three_agents():
    assert CAP == 3
    assert len(read_output(reply(("market", "news", "finance_qa")), WF)[0]) == 3
    with pytest.raises(TooManyAgentsError):
        read_output(reply(("market", "news", "finance_qa", "portfolio")), WF)


def test_too_many_agent_intents_go_to_finance_qa_marked_partial():
    decision, _ = classify(reply(("market", "news", "finance_qa", "portfolio")), "Everything about Apple")
    assert decision == Decision(["finance_qa"], MIN, partial=True)


def test_other_unusable_replies_are_not_marked_partial():
    decision, _ = classify(reply(("not_an_intent",)), "hello")
    assert decision.partial is False


def test_a_partial_turn_adds_the_configured_note_after_the_answer():
    classifier = Classify(["finance_qa"])
    classifier.decision = Decision(["finance_qa"], MIN, partial=True)
    _, state = run(build_graph(classifier, agents()))
    assert state["partial_handled"] is True
    assert " ".join(MSG["partial_note"].split()) in state["final_answer"] and state["final_answer"].index("finance_qa ok") < state["final_answer"].index("Note:")


def test_a_normal_turn_has_no_partial_note():
    _, state = run(build_graph(Classify(["finance_qa"]), agents()))
    assert state["partial_handled"] is False and " ".join(MSG["partial_note"].split()) not in state["final_answer"]


def test_the_partial_flag_is_reset_for_the_next_turn():
    classifier = Classify(["finance_qa"])
    classifier.decision = Decision(["finance_qa"], MIN, partial=True)
    graph = build_graph(classifier, agents())
    run(graph, "one")
    classifier.decision = Decision(["finance_qa"], 0.9)
    _, state = run(graph, "two")
    assert state["partial_handled"] is False and " ".join(MSG["partial_note"].split()) not in state["final_answer"]


# ---- 5. safe_response prompt rules ----

def test_the_safe_response_prompt_carries_the_new_rules():
    prompt = WF["safe_response_prompt"]
    for word in ("context", "material", "provided", "supplied"):
        assert f'"{word}"' in prompt
    assert "Never use the words" in prompt and "Never say what the context or sources do not cover" in prompt
    assert "Discuss only the points the context supports" in prompt and "ascending order" in prompt
    assert "Never recommend, rank or predict" in prompt and "Use only" in prompt


def test_the_router_prompt_allows_three_agents_and_asks_for_parts():
    prompt = WF["router_prompt"]
    assert "at most three agent intents" in prompt and "parts:" in prompt and "What is a P/E ratio?" in prompt


def test_a_new_turn_input_has_empty_parts_and_no_partial_flag():
    from src.workflow.turn import new_turn_input

    turn = new_turn_input("q")
    assert turn["agent_parts"] is None and turn["partial_handled"] is False and isinstance(turn["messages"][0], HumanMessage)

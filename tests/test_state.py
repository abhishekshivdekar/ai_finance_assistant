import operator
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph

from src.workflow.state import GraphState, current_errors, current_results


def _res(agent, turn_id="t1", status="ok"):
    return {
        "agent": agent, "turn_id": turn_id, "status": status, "answer": agent,
        "sources": [], "error": None, "data": None,
    }


def test_reducer_keys_use_operator_add():
    hints = get_type_hints(GraphState, include_extras=True)
    for key in ("agent_outputs", "errors"):
        assert operator.add in hints[key].__metadata__


def _market(state):
    return {
        "agent_outputs": [_res("market")],
        "errors": [{"turn_id": "t1", "message": "m failed"}],
        "final_answer": "first",
        "route_attempts": 1,
    }


def _qa(state):
    return {
        "agent_outputs": [_res("finance_qa")],
        "errors": [{"turn_id": "t1", "message": "qa failed"}],
        "final_answer": "second",
        "route_attempts": 2,
    }


def _run_two_node_graph():
    graph = StateGraph(GraphState)
    graph.add_node("market", _market)
    graph.add_node("finance_qa", _qa)
    graph.add_edge(START, "market")
    graph.add_edge("market", "finance_qa")
    graph.add_edge("finance_qa", END)
    return graph.compile().invoke(
        {"turn_id": "t1", "agent_outputs": [], "errors": [], "route_attempts": 0}
    )


def test_two_agents_results_both_present_in_order():
    out = _run_two_node_graph()
    assert [r["agent"] for r in out["agent_outputs"]] == ["market", "finance_qa"]
    assert [e["message"] for e in out["errors"]] == ["m failed", "qa failed"]


def test_single_writer_fields_are_replaced():
    out = _run_two_node_graph()
    assert out["final_answer"] == "second"
    assert out["route_attempts"] == 2


def test_current_results_ignores_previous_turn():
    state = {
        "turn_id": "t2",
        "agent_outputs": [_res("market", "t1"), _res("finance_qa", "t2")],
    }
    assert [r["agent"] for r in current_results(state)] == ["finance_qa"]


def test_current_errors_ignores_previous_turn():
    state = {
        "turn_id": "t2",
        "errors": [
            {"turn_id": "t1", "message": "old"},
            {"turn_id": "t2", "message": "new"},
        ],
    }
    assert [e["message"] for e in current_errors(state)] == ["new"]


def test_missing_keys_give_empty_lists():
    assert current_results({"turn_id": "t1"}) == []
    assert current_errors({"turn_id": "t1"}) == []


def test_news_params_is_a_typed_state_key():
    assert "news_params" in get_type_hints(GraphState)

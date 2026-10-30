"""Conditional edges of the graph (architecture sections 4.2 to 4.4).

Design pattern: PURE FUNCTIONS as graph edges. An edge only reads the state and returns a node name; the
nodes do the writing. Example: route_after_router({"intents": ["specific_advice", "market"], ...}) is
"safe_response", whatever else the router found, so no other agent can leak advice.
"""
from __future__ import annotations

from src.utils.logger import get_logger
from src.workflow.routing import next_agent, reroute_target, route_at_end, workflow_cfg
from src.workflow.state import GraphState, current_results

log = get_logger(__name__)


def route_from_start(state: GraphState) -> str:
    """A tab hint (any of workflow.hint_nodes) goes straight to that agent; anything else goes to the router."""
    hint = state.get("intent_hint")
    if hint and hint in workflow_cfg()["hint_nodes"]:
        return str(hint)
    if hint:
        log.warning("Unknown intent_hint %r ignored; using the router", hint)
    return "router"


def route_after_router(state: GraphState) -> str:
    """Priority: specific_advice, then out_of_scope with no agent, then clarify, then the first agent, else fallback."""
    cfg = workflow_cfg()
    intents = state.get("intents") or []
    agents = state.get("agents_to_run") or []
    if "specific_advice" in intents:
        return "safe_response"
    if not agents and "out_of_scope" in intents:
        return "safe_response"
    if "clarify" in intents or (agents and state.get("router_confidence", 0.0) < cfg["router_min_confidence"]):
        return "clarify"
    return agents[0] if agents else "fallback"


def route_after_agent(state: GraphState) -> str:
    """After an agent node: re-route on no_answer / wrong_agent while attempts remain and no result is ok, else the next agent, else the end rules.

    needs_input is an ordinary non-ok result: the queue goes on, and route_at_end decides (architecture 4.4
    deviation, see the Decisions list).
    """
    cfg = workflow_cfg()
    results = current_results(state)
    if results and results[-1]["status"] in ("no_answer", "wrong_agent"):
        if state.get("route_attempts", 0) < cfg["max_route_attempts"] and reroute_target(state, cfg):
            return "reroute"
    return next_agent(state) or route_at_end(state)


def route_after_reroute(state: GraphState) -> str:
    """The re-route node queued its target, so this is normally that agent."""
    return next_agent(state) or route_at_end(state)

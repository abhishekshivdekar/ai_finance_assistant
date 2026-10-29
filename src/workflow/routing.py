"""Pure routing helpers shared by the edges and the nodes (architecture section 4.4).

Design pattern: PURE FUNCTIONS. Every decision "which node runs next" is a function of the state and
the workflow config, so each edge can be tested with a plain dict. Example: after `finance_qa` answered
no_answer and the router found a ticker, reroute_target(state) is "market"; without a ticker it is None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.config import get_config
from src.workflow.state import GraphState, current_results


def workflow_cfg() -> dict[str, Any]:
    """The `workflow:` block of config.yaml (one place, so tests patch one function)."""
    return get_config()["workflow"]


@dataclass(frozen=True)
class Decision:
    """What the router decided: intents, confidence and the params it found in the query.

    `market_params` / `news_params` are None when the router found nothing; the router node then leaves
    the state's own value (for example one sent by a tab) untouched.
    """

    intents: list[str]
    confidence: float
    market_params: dict[str, Any] | None = None
    news_params: dict[str, Any] | None = None


def plan_agents(intents: list[str], cfg: dict[str, Any]) -> list[str]:
    """The ordered agent queue for the intents.

    `specific_advice` anywhere gives an empty queue, so no agent can run (architecture 4.3). Other
    non-agent intents (clarify, out_of_scope) are dropped from the queue; at most
    workflow.max_agents_per_turn agents are kept. Example: ["market", "out_of_scope"] -> ["market"].
    """
    if "specific_advice" in intents:
        return []
    agents = [intent for intent in dict.fromkeys(intents) if intent in cfg["agent_intents"]]
    return agents[: cfg["max_agents_per_turn"]]


def ran_agents(state: GraphState) -> list[str]:
    """Names of the agents that produced a result in the current turn, in order."""
    return [result["agent"] for result in current_results(state)]


def next_agent(state: GraphState) -> str | None:
    """The first queued agent that has not produced a result this turn, or None at the end of the queue."""
    ran = set(ran_agents(state))
    return next((agent for agent in state.get("agents_to_run") or [] if agent not in ran), None)


def reroute_target(state: GraphState, cfg: dict[str, Any]) -> str | None:
    """The agent to try after the latest result, or None.

    Candidates come from workflow.reroute[latest agent]; one qualifies when it has not run or been queued
    and the state holds the params it needs. Example: finance_qa -> market only if state["market_params"]
    was filled by the router (the query contained a ticker).
    """
    results = current_results(state)
    if not results:
        return None
    ran = set(ran_agents(state)) | set(state.get("agents_to_run") or [])
    for candidate in cfg["reroute"].get(results[-1]["agent"], []):
        if candidate["agent"] not in ran and state.get(candidate["needs"]):
            return str(candidate["agent"])
    return None


def route_at_end(state: GraphState) -> str:
    """Where the graph goes when no agent is left to run.

    ok -> synthesizer (it adds a note for every needs_input / error result); otherwise needs_input -> clarify;
    otherwise no_answer -> synthesizer (it presents the no_answer message); otherwise fallback.
    """
    statuses = {result["status"] for result in current_results(state)}
    if "ok" in statuses:
        return "synthesizer"
    if "needs_input" in statuses:
        return "clarify"
    if "no_answer" in statuses:
        return "synthesizer"
    return "fallback"

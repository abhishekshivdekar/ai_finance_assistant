"""Builds the LangGraph (architecture section 4).

Design pattern: BUILDER with DEPENDENCY INJECTION. build_graph() takes the router's classify function and the
agent objects, so a test passes fakes and the app passes default_agents(). Example: a test builds the graph with
fake agents and a classify that always answers "market", and checks which nodes ran.
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.agents.tax import TaxAgent
from src.workflow.edges import route_after_agent, route_after_reroute, route_after_router, route_from_start
from src.rag.retriever import retrieve
from src.workflow.nodes import (
    LazyLLM, clarify_node, fallback_node, make_agent_node, make_router_node, make_safe_response_node, make_synthesizer_node,
    reroute_node,
)
from src.workflow.routing import Decision, workflow_cfg
from src.workflow.state import GraphState

FINAL_NODES = ("clarify", "safe_response", "synthesizer", "fallback")


def default_agents() -> dict[str, Any]:
    """The six real agents, keyed by intent name (each asks get_llm for its own role)."""
    return {
        "finance_qa": FinanceQAAgent(), "portfolio": PortfolioAgent(), "market": MarketAgent(),
        "goal_planning": GoalPlanningAgent(), "news": NewsAgent(), "tax_education": TaxAgent(),
    }


def build_graph(
    classify: Callable[[GraphState], Decision],
    agents: dict[str, Any] | None = None,
    checkpointer: Any = None,
    synthesizer_llm: Any = None,
    safe_response_llm: Any = None,
    retrieve_fn: Any = None,
) -> Any:
    """Compile the graph. The checkpointer defaults to the in-memory MemorySaver (architecture 12.1).

    The two LLMs (roles synthesizer and safe_response) are created on first use unless a test injects them;
    retrieve_fn (advice context) defaults to the shared retriever.
    """
    agents = default_agents() if agents is None else agents
    names = list(workflow_cfg()["agent_intents"])
    builder = StateGraph(GraphState)
    builder.add_node("router", make_router_node(classify))
    for name in names:
        builder.add_node(name, make_agent_node(agents[name], name))
    builder.add_node("reroute", reroute_node)
    for name, node in (("clarify", clarify_node), ("fallback", fallback_node),
                       ("safe_response", make_safe_response_node(LazyLLM("safe_response", safe_response_llm), retrieve_fn or retrieve)),
                       ("synthesizer", make_synthesizer_node(LazyLLM("synthesizer", synthesizer_llm)))):
        builder.add_node(name, node)

    after_agent = {*names, "reroute", *FINAL_NODES}
    builder.add_conditional_edges(START, route_from_start, {"router", *names})
    builder.add_conditional_edges("router", route_after_router, {*names, *FINAL_NODES})
    for name in names:
        builder.add_conditional_edges(name, route_after_agent, after_agent)
    builder.add_conditional_edges("reroute", route_after_reroute, after_agent)
    for name in FINAL_NODES:
        builder.add_edge(name, END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())

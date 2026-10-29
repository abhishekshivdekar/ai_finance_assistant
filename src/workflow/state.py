"""Graph state contract.

Design pattern: STATE. One GraphState dict carries all data between LangGraph
nodes. Example: turn 1 is a Market-tab submit, turn 2 is a Chat question on the
same thread. Turn 1's market result is still in the checkpoint, but
current_results() ignores it because its turn_id differs from turn 2's.

Reducers: a node's return value is merged key by key, and without a reducer a
node REPLACES the key. agent_outputs and errors use operator.add so each node
appends its own entry instead of overwriting the others.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.core.models import AgentResult, Source


class ErrorEntry(TypedDict):
    """An error message tagged with the turn that produced it."""

    turn_id: str
    message: str


class GraphState(TypedDict):
    """Everything the graph nodes read and write (architecture section 3.2)."""

    messages: Annotated[list[BaseMessage], add_messages]  # conversation memory
    query: str
    turn_id: str  # new for every user turn (set by the facade)
    intent_hint: str | None  # set by Portfolio/Market/Goals tabs
    intents: list[str]  # from router, e.g. ["market", "finance_qa"]
    router_confidence: float
    agents_to_run: list[str]  # ordered queue (single writer: router)
    route_attempts: int  # re-route counter (section 4.4)
    agent_outputs: Annotated[list[AgentResult], operator.add]
    errors: Annotated[list[ErrorEntry], operator.add]
    sources: list[Source]  # single writer: synthesizer
    market_freshness: dict[str, Any] | None
    portfolio: list[dict[str, Any]] | None  # from Portfolio tab / profile store
    goal_params: dict[str, Any] | None  # from Goals tab
    market_params: dict[str, Any] | None  # from Market tab: {"tickers": [...], "period": "6mo"}
    topic_filter: str | None  # set by the Chat tab dropdown; the facade passes it to the retriever
    final_answer: str


def current_results(state: GraphState) -> list[AgentResult]:
    """Return only the results produced in the current turn."""
    turn_id = state.get("turn_id")
    return [r for r in state.get("agent_outputs", []) if r["turn_id"] == turn_id]


def current_errors(state: GraphState) -> list[ErrorEntry]:
    """Return only the errors recorded in the current turn."""
    turn_id = state.get("turn_id")
    return [e for e in state.get("errors", []) if e["turn_id"] == turn_id]

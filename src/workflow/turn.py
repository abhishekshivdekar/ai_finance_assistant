"""The fresh input of one user turn (architecture section 3.2, "Per-turn state hygiene").

Design pattern: FACTORY FUNCTION. Example: turn 1 is a Market-tab submit and turn 2 a Chat question on the same
thread; new_turn_input("...") for turn 2 has intent_hint None, empty intents and queue, route_attempts 0 and a
new turn_id, so nothing of turn 1 steers turn 2. It resets every single-writer key; agent_outputs and errors are
append-reducers that cannot be cleared, so current_results() / current_errors() filter them by turn_id.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from src.workflow.state import GraphState

# Keys that new_turn_input does not send: the append-reducers keep the thread's history on purpose.
APPEND_ONLY_KEYS = ("agent_outputs", "errors")


def new_turn_input(
    query: str,
    *,
    intent_hint: str | None = None,
    topic_filter: str | None = None,
    portfolio: list[dict[str, Any]] | None = None,
    goal_params: dict[str, Any] | None = None,
    market_params: dict[str, Any] | None = None,
    news_params: dict[str, Any] | None = None,
    tax_params: dict[str, Any] | None = None,
) -> GraphState:
    """A complete GraphState input for one turn: every key except the append-reducers is set."""
    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "turn_id": uuid.uuid4().hex,
        "intent_hint": intent_hint,
        "intents": [],
        "router_confidence": 0.0,
        "agents_to_run": [],
        "route_attempts": 0,
        "sources": [],
        "market_freshness": None,
        "portfolio": portfolio,
        "goal_params": goal_params,
        "market_params": market_params,
        "news_params": news_params,
        "tax_params": tax_params,
        "topic_filter": topic_filter,
        "final_answer": "",
        "merge_mode": None,
    }

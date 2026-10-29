"""Graph nodes (architecture section 4.1).

Design pattern: FACTORY FUNCTIONS for nodes that need a dependency. make_agent_node(agent) returns the
node function for one agent, so the graph builder (and the tests) decide which agent object sits behind
each name. Example: make_agent_node(FakeAgent()) is a node a test can run with no LLM.

A node returns only the keys it writes. agent_outputs and errors are append-reducers, so a node returns
its own entries in a list. Only the final-answer nodes (synthesizer, safe_response, fallback) add the
disclaimer, and each does so exactly once through finish().
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage

from src.core.llm_factory import get_llm
from src.core.models import AgentResult, Source
from src.rag.retriever import RetrievedChunk, retrieve
from src.utils.logger import get_logger
from src.workflow.advice import educational_answer
from src.workflow.answers import add_disclaimer, notes_for
from src.workflow.merge import merge_with_llm
from src.workflow.routing import Decision, plan_agents, reroute_target, workflow_cfg
from src.workflow.state import GraphState, current_results

log = get_logger(__name__)

Node = Callable[[GraphState], dict[str, Any]]


class LazyLLM:
    """An LLM for one role that is created on first use (graphs that never merge or give advice never create it).

    Design pattern: LAZY INITIALISATION behind the Factory. Example: LazyLLM("synthesizer").get() calls
    get_llm(role="synthesizer") once; LazyLLM("synthesizer", fake).get() returns the fake without a factory call.
    """

    def __init__(self, role: str, llm: Any = None) -> None:
        self.role, self._llm = role, llm

    def get(self) -> Any:
        """The LLM, created through get_llm on the first call."""
        if self._llm is None:
            self._llm = get_llm(role=self.role)
        return self._llm


def reply(answer: str, sources: list[Source] | None = None) -> dict[str, Any]:
    """The state update of a final node: the answer text, its sources and the assistant message for the history."""
    return {"final_answer": answer, "sources": sources or [], "messages": [AIMessage(content=answer)]}


def finish(answer: str, sources: list[Source] | None = None) -> dict[str, Any]:
    """A final answer WITH the disclaimer (synthesizer, safe_response, fallback)."""
    return reply(add_disclaimer(answer), sources)


def make_agent_node(agent: Any) -> Node:
    """Wrap one agent: run it and return its result (and its error, tagged with the turn) for the reducers."""

    def node(state: GraphState) -> dict[str, Any]:
        result: AgentResult = agent.run(state)
        update: dict[str, Any] = {"agent_outputs": [result]}
        if result["error"]:
            update["errors"] = [{"turn_id": result["turn_id"], "message": result["error"]}]
        return update

    return node


def make_router_node(classify: Callable[[GraphState], Decision]) -> Node:
    """The router node: `classify` decides, the node writes intents, confidence, the agent queue and found params."""

    def node(state: GraphState) -> dict[str, Any]:
        decision = classify(state)
        update: dict[str, Any] = {
            "intents": list(decision.intents),
            "router_confidence": decision.confidence,
            "agents_to_run": plan_agents(list(decision.intents), workflow_cfg()),
        }
        for key in ("market_params", "news_params"):
            if getattr(decision, key):
                update[key] = getattr(decision, key)
        log.info("Router intents=%s confidence=%.2f queue=%s", update["intents"], decision.confidence, update["agents_to_run"])
        return update

    return node


def reroute_node(state: GraphState) -> dict[str, Any]:
    """Count one re-route hop and queue the target chosen by reroute_target (architecture 4.4)."""
    target = reroute_target(state, workflow_cfg())
    attempts = state.get("route_attempts", 0) + 1
    log.info("Re-route hop %d to %s", attempts, target)
    return {"route_attempts": attempts, "agents_to_run": [*(state.get("agents_to_run") or []), target]}


def clarify_node(state: GraphState) -> dict[str, Any]:
    """Ask for the missing information as an ordinary reply (no LLM call, no disclaimer: no financial content).

    The text is the needs_input message(s) of the agents, or the low-confidence message from config.
    """
    asked = [r["answer"] for r in current_results(state) if r["status"] == "needs_input"]
    return reply("\n\n".join(asked) if asked else " ".join(workflow_cfg()["messages"]["clarify_low_confidence"].split()))


def make_safe_response_node(llm: LazyLLM, retrieve_fn: Callable[..., list[RetrievedChunk]] = retrieve) -> Node:
    """safe_response: no agent runs here. out_of_scope is a static text; specific_advice is an educational answer.

    The educational answer (retrieved context, no recommendation, checked) falls back to the static advice text.
    """

    def node(state: GraphState) -> dict[str, Any]:
        messages = workflow_cfg()["messages"]
        if "specific_advice" not in (state.get("intents") or []):
            return finish(" ".join(messages["out_of_scope"].split()))
        found = educational_answer(state, llm.get(), retrieve_fn)
        if found is None:
            return finish(" ".join(messages["advice_static"].split()))
        return finish(found[0], found[1])

    return node


def fallback_node(state: GraphState) -> dict[str, Any]:
    """A friendly message when routing failed or every agent failed."""
    return finish(" ".join(workflow_cfg()["messages"]["fallback"].split()))


def make_synthesizer_node(llm: LazyLLM) -> Node:
    """synthesizer: one ok result is its answer (no LLM call); two or more are merged by ONE LLM call.

    The merge is checked in code and falls back to a verbatim join (merge_with_llm). Notes for needs_input /
    error / no_answer results come from code. Only no_answer results: that message. Nothing usable: the fallback.
    """

    def node(state: GraphState) -> dict[str, Any]:
        results = current_results(state)
        ok = [r for r in results if r["status"] == "ok"]
        if ok:
            answer, sources = (ok[0]["answer"], ok[0]["sources"]) if len(ok) == 1 else merge_with_llm(ok, llm.get())
            return finish("\n\n".join([answer, *notes_for(results)]), sources)
        no_answer = [r for r in results if r["status"] == "no_answer"]
        if no_answer:
            return finish(no_answer[-1]["answer"])
        return fallback_node(state)

    return node

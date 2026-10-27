"""Abstract parent of all agents. It is never a graph node itself.

Design pattern: TEMPLATE METHOD. run() fixes the shared steps (logging,
turn_id, output guard, error handling) and subclasses fill in _execute().
Example: MarketAgent._execute() only fetches quotes and builds the prompt;
logging, the guard and error handling come free from BaseAgent.run().
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.llm_factory import get_llm
from src.core.models import AgentResult
from src.core.output_guard import guard_answer
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.workflow.state import GraphState


class BaseAgent(ABC):
    """Template Method base class; subclasses implement _execute()."""

    name: str
    llm_role: str = "default"

    def __init__(self, llm: Any = None) -> None:
        """Use the injected llm (tests pass a fake), else ask the Factory."""
        self.llm = llm if llm is not None else get_llm(role=self.llm_role)
        self.log = get_logger(self.name)

    def run(self, state: GraphState) -> AgentResult:
        """Run the agent. Returns a plain answer (no disclaimer); never raises."""
        turn_id = state.get("turn_id", "")
        start = time.perf_counter()
        self.log.info("Start agent=%s turn_id=%s", self.name, turn_id)
        try:
            result = self._execute(state)
            if not result.get("turn_id"):
                result["turn_id"] = turn_id
            if not result.get("agent"):
                result["agent"] = self.name
            if result["status"] == "ok":
                result["answer"] = guard_answer(result["answer"], self.llm)
        except Exception:
            self.log.exception("Agent %s failed", self.name)  # technical detail: log only
            result = AgentResult(
                agent=self.name,
                turn_id=turn_id,
                status="error",
                answer=get_config()["agents"]["error_message"],
                sources=[],
                error=f"{self.name} failed",
                data=None,
            )
        self.log.info(
            "End agent=%s status=%s duration=%.3fs",
            self.name, result["status"], time.perf_counter() - start,
        )
        return result

    @abstractmethod
    def _execute(self, state: GraphState) -> AgentResult:
        """Agent-specific work: build and return an AgentResult."""

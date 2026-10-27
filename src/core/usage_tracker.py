"""Token and cost tracking.

Design patterns:
- OBSERVER: UsageTracker is a LangChain callback. When an LLM call finishes,
  LangChain notifies it; the agent never records usage itself.
- SINGLETON: get_usage_registry() (lru_cache) returns one shared registry, so
  the "router" tracker and the "default" tracker add into the same totals,
  like get_config() returns one shared dict.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def _price_for(model: str) -> dict[str, float] | None:
    """Return {"input", "output"} USD per 1M tokens, or None if unpriced."""
    return (get_config().get("pricing") or {}).get(model)


class UsageRegistry:
    """Per-role token and cost totals. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, dict[str, Any]] = {}

    def record(self, role: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add one call's tokens to a role. Cost becomes None if model is unpriced."""
        price = _price_for(model)
        cost = (
            None
            if price is None
            else (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
        )
        with self._lock:
            entry = self._totals.setdefault(
                role, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            )
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            # Once any call is unpriced the role's cost stays unknown (never understated).
            if cost is None or entry["cost"] is None:
                entry["cost"] = None
            else:
                entry["cost"] += cost
        log.info("Usage role=%s model=%s in=%d out=%d", role, model, input_tokens, output_tokens)

    def totals(self, role: str | None = None) -> dict[str, Any]:
        """Totals for one role, or for all roles keyed by role name."""
        with self._lock:
            if role is not None:
                return dict(self._totals.get(role, {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}))
            return {r: dict(t) for r, t in self._totals.items()}

    def session_tokens(self) -> int:
        """Total input + output tokens across all roles."""
        with self._lock:
            return sum(t["input_tokens"] + t["output_tokens"] for t in self._totals.values())

    def reset(self) -> None:
        """Clear all totals."""
        with self._lock:
            self._totals.clear()


@lru_cache(maxsize=1)
def get_usage_registry() -> UsageRegistry:
    """Return the shared registry (Singleton)."""
    return UsageRegistry()


class UsageTracker(BaseCallbackHandler):
    """Records token usage for one role after every LLM call (Observer)."""

    def __init__(self, role: str, model: str) -> None:
        self.role = role
        self.model = model

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Read usage_metadata from the generations and add it to the registry."""
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    get_usage_registry().record(
                        self.role,
                        self.model,
                        int(usage.get("input_tokens", 0)),
                        int(usage.get("output_tokens", 0)),
                    )


@lru_cache(maxsize=None)
def get_usage_tracker(role: str, model: str) -> UsageTracker:
    """One tracker per (role, model), so repeated get_llm() calls don't stack handlers."""
    return UsageTracker(role, model)


def is_budget_exceeded() -> bool:
    """True when session tokens reach llm.max_session_tokens (missing/null = no limit)."""
    limit = get_config().get("llm", {}).get("max_session_tokens")
    if limit is None:
        return False
    return get_usage_registry().session_tokens() >= limit

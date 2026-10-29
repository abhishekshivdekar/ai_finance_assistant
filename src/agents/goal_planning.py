"""Goal Planning agent: deterministic savings projections, plus one LLM call for short commentary.

Design patterns:
- TEMPLATE METHOD: BaseAgent.run() supplies logging, the output guard and error
  handling; this class only implements _execute().
- DEPENDENCY INJECTION: the llm is a constructor argument with a default from the
  factory. Example: a test passes GoalPlanningAgent(llm=FakeLLM()) and needs no network.

Principle: ALL numbers come from pure Python (src/core/goal_metrics.py). The LLM
sees only the computed results as JSON and writes words; the number guard shared
with the Portfolio and Market agents rejects any commentary containing a number
that was not in that JSON. Only the commentary step falls back (to a fixed text
without numbers); any other failure becomes an error result in BaseAgent.run().
The assumptions block (illustrative label, as_of, source, timing notes) is built
from config in code, so it is in the payload whatever the LLM does.

Input: state["goal_params"] = {"goal_amount": 1_000_000, "horizon_years": 15,
"current_savings": 50_000, "monthly_contribution": 5_000, "risk_profile": "medium",
"annual_step_up_pct": 5, "inflation_adjust": True}.
"""
from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.goal_inputs import parse_goal_params
from src.core.goal_metrics import below_inflation_text, build_plan, summary_input
from src.core.goal_models import GoalPlan
from src.core.models import AgentResult, GoalPayload
from src.core.number_guard import collect_numbers, numbers_in
from src.utils.config import get_config
from src.workflow.state import GraphState


def assumptions_block(plan: GoalPlan, cfg: dict[str, Any]) -> dict[str, Any]:
    """The assumptions behind the numbers, straight from config (always shown by the UI)."""
    a = cfg["assumptions"]
    return {
        "illustrative": True, "label": a["label"], "as_of": a["as_of"], "source": a["source"],
        "currency": cfg["currency"], "risk_profile": plan.inputs.risk_profile,
        "annual_returns_pct": dict(a["returns_pct"][plan.inputs.risk_profile]),
        "inflation": {**a["inflation"], "applied": plan.inputs.inflation_adjust},
        "deposit_timing": a["deposit_timing_note"], "step_up_timing": a["step_up_note"], "return_convention": a["rate_note"],
    }


def inflation_block(plan: GoalPlan, cfg: dict[str, Any]) -> dict[str, Any]:
    """Both goal figures with their labels when inflation is applied; {} otherwise."""
    if not plan.inputs.inflation_adjust:
        return {}
    return {
        "annual_pct": cfg["assumptions"]["inflation"]["annual_pct"],
        "goal_in_todays_money": plan.inputs.goal_amount,
        "goal_in_future_money": plan.target_amount,
        "labels": dict(cfg["labels"]),
    }


class GoalPlanningAgent(BaseAgent):
    """Projects savings toward a goal under illustrative scenarios (educational, no predictions)."""

    name = "goal_planning"
    llm_role = "goal_summary"

    def _result(self, status: str, answer: str, payload: GoalPayload) -> AgentResult:
        """Build the envelope; turn_id is filled in by BaseAgent.run()."""
        return AgentResult(
            agent=self.name, turn_id="", status=status, answer=answer,  # type: ignore[typeddict-item]
            sources=[], error=None, data=payload,
        )

    def _comment(self, plan: GoalPlan, cfg: dict[str, Any]) -> tuple[str, str | None]:
        """(commentary, warning). Falls back to the fixed text on any failure or a bad number.

        The number whitelist is the computed values and the user's own inputs only: the
        `assumptions` entry (as_of date, timing notes, source) is shown to the LLM but
        not whitelisted, so "10 years" or "13 months" taken from it is rejected.
        """
        fallback, warn = cfg["summary_fallback"].strip(), cfg["texts"]
        payload = summary_input(plan, cfg)
        try:
            reply = self.llm.invoke(cfg["summary_prompt"].format(results_json=json.dumps(payload, indent=2)))
            text = str(getattr(reply, "content", reply)).strip()
        except Exception:
            self.log.warning("Goal commentary failed; using the fallback text", exc_info=True)
            return fallback, warn["commentary_unavailable"]
        if not text:
            return fallback, warn["commentary_empty"]
        allowed = collect_numbers({k: v for k, v in payload.items() if k != "assumptions"})
        invented = numbers_in(text) - allowed
        if invented:
            self.log.warning("Commentary contains numbers not in the results: %s", sorted(invented))
            return fallback, warn["commentary_discarded"]
        return text, None

    def _execute(self, state: GraphState) -> AgentResult:
        """Validate the request, compute the plan, then write the commentary."""
        cfg = get_config()["goal_planning"]
        inputs, errors = parse_goal_params(state.get("goal_params"), cfg)
        if inputs is None:
            message = "\n".join([cfg["needs_input_message"].strip(), *[f"- {e}" for e in errors]])
            return self._result("needs_input", message, {"warnings": errors})

        plan = build_plan(inputs, cfg)
        commentary, note = self._comment(plan, cfg)
        scenarios = plan.scenarios
        flags = [cfg["texts"]["below_inflation_warning"].format(scenario=s.name, sentence=below_inflation_text(s, cfg))
                 for s in scenarios if s.below_inflation]
        payload: GoalPayload = {
            "inputs": inputs.model_dump(mode="json"),
            "target_amount": plan.target_amount,
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "projections": plan.projection_rows,
            "required_monthly_contribution": {s.name: s.required_monthly_contribution for s in scenarios},
            "required_labels": {s.name: s.required_label for s in scenarios},
            "months_to_goal": {s.name: s.months_to_goal for s in scenarios},
            "shortfall": {s.name: s.shortfall for s in scenarios},
            "inflation": inflation_block(plan, cfg),
            "assumptions": assumptions_block(plan, cfg),
            "warnings": [*flags, *([note] if note else [])],
        }
        return self._result("ok", commentary, payload)

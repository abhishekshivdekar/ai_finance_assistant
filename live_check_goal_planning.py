"""Live check for the Goal Planning agent (REAL LLM, so it uses network and tokens).

Run from the project root:  python live_check_goal_planning.py
Prints, for two sample goals: the scenario table, required contributions, the assumptions block,
the commentary and the token cost. The numbers come from code; only the commentary uses the LLM.
"""
from src.agents.goal_planning import GoalPlanningAgent
from src.core.usage_tracker import get_usage_registry
from live_checks.transcript import run_with_transcript

GOALS = {
    "15-year retirement-style goal with a yearly step-up": {
        "goal_amount": 10_000_000, "horizon_years": 15, "current_savings": 500_000,
        "monthly_contribution": 20_000, "risk_profile": "medium", "annual_step_up_pct": 5, "inflation_adjust": True,
    },
    "3-year target that is easily met": {
        "goal_amount": 300_000, "horizon_years": 3, "current_savings": 250_000,
        "monthly_contribution": 10_000, "risk_profile": "low", "annual_step_up_pct": 0, "inflation_adjust": False,
    },
}
ROW = "{:<13} {:>8} {:>16} {:>16} {:>16} {:>14} {}"


def months_text(months: int | None) -> str:
    """Months to goal as text ('-' when not reached)."""
    return "-" if months is None else str(months)


def show(title: str, params: dict) -> None:
    """Run the agent on one goal and print everything the UI would show."""
    agent = GoalPlanningAgent()
    result = agent.run({"turn_id": "live", "goal_params": params})
    data = result["data"] or {}
    print(f"\n{'=' * 8} {title} {'=' * 8}\nstatus: {result['status']}")
    if not data.get("scenarios"):
        print(result["answer"])
        return
    print(f"target used for comparison: {data['target_amount']:,.2f} {data['assumptions']['currency']}")
    if data["inflation"]:
        labels = data["inflation"]["labels"]
        print(f"  {labels['goal_future_money']}: {data['inflation']['goal_in_future_money']:,.2f}"
              f" (entered, today's money: {data['inflation']['goal_in_todays_money']:,.2f})")
    print("\n=== Scenarios ===")
    print(ROW.format("scenario", "return %", "value at end", "shortfall", "needed / month", "months", "note"))
    for s in data["scenarios"]:
        print(ROW.format(s["name"], f"{s['annual_return_pct']:.1f}", f"{s['final_value']:,.2f}", f"{s['shortfall']:,.2f}",
                         f"{s['required_monthly_contribution']:,.2f}", months_text(s["months_to_goal"]), s["months_note"] or ""))
    print("\n=== Monthly amount meaning ===")
    for s in data["scenarios"]:
        print(f"  {s['name']:13s}{s['required_monthly_contribution']:,.2f}  = {s['required_label']}")
    if data["inflation"]:
        for s in data["scenarios"]:
            print(f"  {s['name']:13s} value in today's money: {s['final_value_todays_money']:,.2f}")
    print("\n=== Assumptions ===")
    for key, value in data["assumptions"].items():
        print(f"  {key}: {value}")
    print("\n=== Warnings ===")
    for warning in data["warnings"] or ["none"]:
        print(f"  - {warning}")
    print("\n=== Commentary ===")
    print(result["answer"])


def main() -> None:
    for title, params in GOALS.items():
        show(title, params)
    print("\n=== Token cost ===")
    for role, t in sorted(get_usage_registry().totals().items()):
        cost = "unpriced model" if t["cost"] is None else f"${t['cost']:.6f}"
        print(f"  {role:20s} in={t['input_tokens']:>6} out={t['output_tokens']:>6} cost={cost}")


if __name__ == "__main__":
    run_with_transcript(main, __file__)

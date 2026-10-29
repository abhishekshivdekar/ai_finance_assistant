"""Live check for the Tax Education agent (REAL LLM and REAL index, so it uses tokens and local embeddings).

Run from the project root:  python live_check_tax.py
Runs three questions, two capital gains calculations and one small-savings projection, and prints
everything the UI would show: the answer, the facts and config values used with their as_of date and
source, the steps, the exclusions and the token cost. Every figure is illustrative and must be
verified on incometax.gov.in before it is relied on. Do not run this without checking the budget.
"""
from src.agents.tax import TaxAgent
from src.core.usage_tracker import get_usage_registry

QUESTIONS = [
    "What is LTCG?",
    "Which rate applies to equity fund gains held for 2 years?",
    "How does PPF work?",
    "What is STT?",
    "What is the difference between PPF and NPS?",
]
CALCULATIONS = {
    "Equity fund sold after 18 months": {
        "calculation": "capital_gains", "asset_type": "equity_fund", "buy_value": 400_000, "sell_value": 650_000,
        "expenses": 1_000, "holding_months": 18, "ltcg_exemption_already_used": 0,
    },
    "Debt fund bought after the rule change, user supplies the slab rate": {
        "calculation": "capital_gains", "asset_type": "debt_fund_post_2023", "buy_value": 300_000,
        "sell_value": 350_000, "expenses": 0, "holding_months": 30, "slab_rate_pct": 30,
    },
    "PPF with a yearly deposit": {
        "calculation": "small_savings_growth", "scheme": "ppf", "mode": "annual", "amount": 100_000, "years": 15,
    },
}


def show_config_used(config_used: dict) -> None:
    """Print each figure used with its display text, as_of date, short source and verify flag (source_short from config)."""
    print("\n--- Config values used ---")
    for key, entry in config_used.items():
        print(f"  {key}: {entry['display']} ({entry['label']}); as of {entry['as_of']}; source: {entry['source_short']}; verify: {entry['verify']}")


def show(title: str, state: dict) -> None:
    """Run the agent on one request and print the status, answer and payload."""
    result = TaxAgent().run({"turn_id": "live", **state})
    data = result["data"] or {}
    print(f"\n{'=' * 8} {title} {'=' * 8}\nstatus: {result['status']}\n\n{result['answer']}")
    if data.get("steps"):
        print("\n--- Steps ---")
        for step in data["steps"]:
            print(f"  {step}")
    if data.get("results"):
        print("\n--- Results ---")
        for key, value in data["results"].items():
            if key != "yearly":
                print(f"  {key}: {value}")
    if data.get("config_used"):
        show_config_used(data["config_used"])
    if data.get("facts") and result["status"] == "ok":
        cited = data.get("fact_citations") or []
        print(f"\n--- Facts cited by the model ({len(cited)} of {len(data['facts'])} offered) ---")
        for fact in data["facts"]:
            if fact["number"] in cited:
                print(f"  {fact['id']}: {fact['label']} = {fact['display']}; as of {fact['as_of']}; source: {fact['source_short']}")
    if result["sources"]:
        print("\n--- Sources ---")
        for source in result["sources"]:
            print(f"  {source['title']} ({source['chunk_id']}, score {source['score']:.2f})")
    for label in ("notes", "assumptions", "exclusions", "warnings"):
        if data.get(label):
            print(f"\n--- {label.capitalize()} ---")
            for line in data[label]:
                print(f"  - {line}")
    print(f"\nLabel: {data.get('label')}")


def main() -> None:
    for question in QUESTIONS:
        show(f"Question: {question}", {"query": question})
    for title, params in CALCULATIONS.items():
        show(title, {"tax_params": params})
    print("\n=== Token cost ===")
    for role, t in sorted(get_usage_registry().totals().items()):
        cost = "unpriced model" if t["cost"] is None else f"${t['cost']:.6f}"
        print(f"  {role:20s} in={t['input_tokens']:>6} out={t['output_tokens']:>6} cost={cost}")


if __name__ == "__main__":
    main()

"""Live check for the Finance Q&A agent (uses the REAL index and the REAL LLM, so it costs tokens).

Run from the project root after build_index.py:  python live_check_qa.py
Prints, per question: status, rewritten query, answer, cited article slugs and tokens,
then a total token and cost summary by role.
Sources print as numbered lines: [n] is the same n as in the answer text.
"""
from pathlib import Path

import yaml

from src.agents.finance_qa import FinanceQAAgent
from src.core.usage_tracker import get_usage_registry
from src.utils.config import PROJECT_ROOT
from live_checks.transcript import run_with_transcript

ANSWERABLE = [
    "What is the difference between an ETF and a mutual fund?",
    "How does compound interest work?",
    "What is a stop-loss order?",
    "What is the difference between short-term and long-term capital gains in India?",
    "will my pile last after my salary stops",                       # colloquial
    "why did my fund lose money when the RBI raised rates",          # colloquial
]
OFF_TOPIC = ["What is the best recipe for a chocolate cake?"]


def near_miss_questions() -> list[str]:
    """The finance-adjacent questions the knowledge base does not cover."""
    path = Path(PROJECT_ROOT) / "tests" / "rag_eval" / "questions.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["near_miss"])


def tokens_now() -> tuple[int, int]:
    """(input, output) tokens recorded so far across all roles."""
    totals = get_usage_registry().totals()
    return (
        sum(t["input_tokens"] for t in totals.values()),
        sum(t["output_tokens"] for t in totals.values()),
    )


def main() -> None:
    agent = FinanceQAAgent()
    groups = [("answerable", ANSWERABLE), ("near-miss", near_miss_questions()), ("off-topic", OFF_TOPIC)]
    for label, questions in groups:
        for question in questions:
            before = tokens_now()
            result = agent.run({"turn_id": "live", "query": question})
            after = tokens_now()
            data = result["data"] or {}
            print(f"\n[{label}] {question}")
            print(f"  status:    {result['status']}")
            print(f"  rewritten: {data.get('rewritten_query', '-')}")
            print(f"  answer:    {result['answer']}")
            print("  sources:   " + ("-" if not result["sources"] else ""))
            for n, source in enumerate(result["sources"], start=1):
                print(f"    [{n}] {source['chunk_id'].split('#')[0]}")
            print(f"  tokens:    in={after[0] - before[0]} out={after[1] - before[1]}")

    print("\n=== Usage summary ===")
    registry = get_usage_registry()
    total_cost = 0.0
    unpriced = False
    for role, t in sorted(registry.totals().items()):
        cost = t["cost"]
        unpriced = unpriced or cost is None
        total_cost += cost or 0.0
        shown = "unpriced model" if cost is None else f"${cost:.6f}"
        print(f"  {role:20s} in={t['input_tokens']:>6} out={t['output_tokens']:>6} cost={shown}")
    print(f"  total tokens: {registry.session_tokens()}")
    print(f"  total cost:   ${total_cost:.6f}" + (" (some calls unpriced)" if unpriced else ""))


if __name__ == "__main__":
    run_with_transcript(main, __file__)

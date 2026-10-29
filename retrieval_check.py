"""Retrieval-only check for query expansion (LOCAL embeddings and the local index; no LLM, no API money).

Run from the project root:  python retrieval_check.py  ["question" ...]   (no arguments: the default questions)
For each question it prints the retrieval scores before (the question as written, `retrieve`) and
after (`merge_retrieval`, which also searches the acronym-expanded query from rag.query_expansions),
once for the tax topic and once for all topics.
"""
import sys

from src.rag.retriever import RetrievedChunk, expand_query, merge_retrieval, retrieve
from src.utils.config import get_config

QUESTIONS = [
    "What is LTCG?", "What is STT?", "How does PPF work?", "What is the difference between PPF and NPS?",
]
TOPICS = ["accounts_and_taxes_india", None]
SNIPPET = 120


def show(label: str, chunks: list[RetrievedChunk]) -> None:
    """Print one result list: score, chunk id and the first SNIPPET characters of the text."""
    print(f"  {label}: {len(chunks)} chunk(s)")
    for chunk in chunks:
        text = " ".join(chunk.text.split())[:SNIPPET]
        print(f"    {chunk.score:.2f}  {chunk.chunk_id}  {text}")


def main() -> None:
    expansions = get_config()["rag"].get("query_expansions") or {}
    for question in sys.argv[1:] or QUESTIONS:
        print(f"\n=== {question!r} -> expanded: {expand_query(question, expansions)!r}")
        for topic in TOPICS:
            print(f" topic={topic}")
            show("before", retrieve(question, topic=topic))
            show("after ", merge_retrieval(retrieve, [question], topic))


if __name__ == "__main__":
    main()

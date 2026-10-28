"""Run the RAG evaluation sweep (real embedder, no API tokens).
In VS Code: open this file and click the Run (play) button at the top right.

Set EMBEDDING_MODEL to a sentence-transformers model name (for example another
free local model) to evaluate it. The eval builds its own in-memory stores with
that model; the saved app index and config.yaml are not touched.
"""
import runpy
from pathlib import Path

EMBEDDING_MODEL: str | None = None   # None = use embeddings.model from config.yaml

# The sweep runs ONLY these (chunk_size, chunk_overlap, top_k, threshold) configs,
# each with the diversity caps on and off. Held-out hit rate / MRR are the first columns.
SWEEP_CONFIGS: list[tuple[int, int, int, float]] = [
    (300, 120, 6, 0.30),
    (400, 120, 6, 0.30),
    (400, 60, 6, 0.35),
]

HARNESS = Path(__file__).parent / "tests" / "rag_eval" / "rag_eval.py"


def run() -> None:
    """Load the harness by path (tests/ is not a package) and call its main()."""
    runpy.run_path(str(HARNESS), run_name="rag_eval")["main"](EMBEDDING_MODEL, SWEEP_CONFIGS)


if __name__ == "__main__":
    run()

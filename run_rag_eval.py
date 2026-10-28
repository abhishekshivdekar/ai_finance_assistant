"""Run the RAG evaluation sweep (real embedder, no API tokens).
In VS Code: open this file and click the Run (play) button at the top right.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parent / "tests" / "rag_eval" / "rag_eval.py"),
        run_name="__main__",
    )

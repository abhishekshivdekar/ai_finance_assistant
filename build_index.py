"""Build the RAG index from src/data/knowledge_base/.
In VS Code: open this file and click the Run (play) button at the top right.
The first run downloads the embedding model (needs internet once).
"""
from src.rag.ingest import build_index
from src.utils.config import get_config


def main() -> None:
    print("Embedding model:", get_config()["embeddings"]["model"])
    summary = build_index()
    print(f"Indexed {summary['articles']} articles into {summary['chunks']} chunks.")
    print("Index saved to", get_config()["rag"]["vector_store_dir"])


if __name__ == "__main__":
    main()

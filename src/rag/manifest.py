"""Index manifest: records how the index was built so staleness can be detected.

Ingest writes manifest.json next to the index; the vector store checks it when
the index loads. Example: you change rag.chunk_size from 800 to 600 but forget
to rebuild; on the next load a warning names chunk_size as the mismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


def resolve_dir(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def compute_article_hash(kb_dir: Path) -> str:
    """sha256 over the sorted relative paths and bytes of every article."""
    digest = hashlib.sha256()
    for file in sorted(kb_dir.rglob("*.md")):
        digest.update(file.relative_to(kb_dir).as_posix().encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def build_manifest(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    article_hash: str,
    topic_chunking: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict; topic_chunking records the per-topic overrides."""
    return {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "article_hash": article_hash,
        "topic_chunking": topic_chunking or {},
    }


def write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json into the index directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(directory: Path) -> dict[str, Any] | None:
    """Read manifest.json, or None if missing or unreadable."""
    try:
        return json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def manifest_mismatches(
    manifest: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Names of the manifest fields that differ from the current settings."""
    return [key for key, value in current.items() if manifest.get(key) != value]


def warn_if_stale(directory: Path) -> list[str]:
    """Log a warning if the index may be stale; returns the differing fields. Never raises."""
    manifest = read_manifest(directory)
    if manifest is None:
        log.warning("Index manifest missing or unreadable in %s; rebuild the index", directory)
        return ["manifest"]
    cfg = get_config()
    try:
        current = build_manifest(
            cfg["embeddings"]["model"],
            cfg["rag"]["chunk_size"],
            cfg["rag"]["chunk_overlap"],
            compute_article_hash(resolve_dir(cfg["rag"]["knowledge_base_dir"])),
            cfg["rag"].get("topic_chunking") or {},
        )
    except OSError:
        log.warning("Could not read the knowledge base to verify the index manifest")
        return ["article_hash"]
    mismatches = manifest_mismatches(manifest, current)
    if mismatches:
        log.warning(
            "Index may be stale (differs from current config/articles: %s); rebuild with build_index.py",
            ", ".join(mismatches),
        )
    return mismatches

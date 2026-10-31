"""UI texts from the `ui:` block of config.yaml; no label or message is written in code."""
from __future__ import annotations

from typing import Any

from src.utils.config import get_config


def ui_text(path: str, **values: Any) -> str:
    """The text at a dotted path of the `ui:` block, whitespace tidied; `{name}` fields are filled from `values`.

    Example: ui_text("chat.sources_title", count=3) -> "Sources (3)". A path missing from the config raises KeyError
    (the config loader already requires every key the UI uses, so this is a bug in the caller).
    """
    node: Any = get_config()["ui"]
    for part in path.split("."):
        node = node[part]
    text = " ".join(str(node).split())
    return text.format(**values) if values else text

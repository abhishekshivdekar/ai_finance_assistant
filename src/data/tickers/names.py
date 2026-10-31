"""Cleaning and comparing company names (pure functions, no network).

`clean_query` is the only door to a lookup: text that reaches Yahoo has had control characters and unusual symbols
removed and has been length-checked. Example: "Taiwan Semiconductor Manufacturing Co Ltd TPE: 2330" becomes
"Taiwan Semiconductor Manufacturing Co Ltd TPE 2330"; "<script>x</script>" becomes "script x script".
"""
from __future__ import annotations

import re
from typing import Any

_DISALLOWED = re.compile(r"[^\w .&'\-]")          # keeps letters, digits, space and . & ' -
_SEPARATORS = re.compile(r"[^\w&]+")


class InvalidQueryError(ValueError):
    """The name is empty, not text, or longer than ticker_lookup.max_query_chars."""


def clean_query(raw: Any, max_chars: int) -> str:
    """The name with unusual characters replaced by spaces and spaces collapsed; raises InvalidQueryError if unusable."""
    if not isinstance(raw, str):
        raise InvalidQueryError("The name is not text.")
    query = " ".join(_DISALLOWED.sub(" ", raw).split())
    if not query:
        raise InvalidQueryError("The name is empty.")
    if len(query) > max_chars:
        raise InvalidQueryError(f"The name is longer than {max_chars} characters.")
    return query


def tokens(text: str, legal_suffixes: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Lower-case words of a name without legal suffixes. Example: "Infosys Ltd." -> ("infosys",)."""
    suffixes = {suffix.casefold() for suffix in legal_suffixes}
    words = _SEPARATORS.sub(" ", text.casefold()).split()
    return tuple(word for word in words if word not in suffixes)


def is_name_start(query: tuple[str, ...], name: tuple[str, ...]) -> bool:
    """True when the query's words are the first whole words of the name.

    Example: ("suzlon",) starts ("suzlon", "energy"); ("apple",) does not start ("pineapple", "co"); ("taiwan",
    "semiconductors") does not start ("taiwan", "semiconductor", "manufacturing") (a spelling difference means ask).
    """
    return bool(query) and name[: len(query)] == query

"""Display snippets for the Sources list: clean text cut from a retrieved chunk, in code only (never by the LLM).

Design pattern: PURE FUNCTIONS. Example: a chunk "## Gain and loss\n\n**Capital gain** is the profit...<!-- VERIFY -->"
becomes the snippet "Gain and loss Capital gain is the profit..." (no heading marks, bold marks or HTML comment),
cut to at most `rag.snippet_max_chars` characters at a word boundary.

The snippet is always a prefix of `clean_text(chunk text)`, so it is a substring of the cleaned chunk. It is display
text for the UI: it is never sent to a model and never passes through a number guard.
"""
from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)          # also an unterminated "<!--" up to the end
_STRAY_CLOSE = re.compile(r"-->")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"\*\*|__|`")


def _table_row(line: str) -> str:
    """"| a | b |" -> "a · b"; any other line is unchanged."""
    if not line.lstrip().startswith("|"):
        return line
    return " · ".join(cell.strip() for cell in line.strip().strip("|").split("|") if cell.strip())


def clean_text(text: str) -> str:
    """Chunk text as plain display text: no HTML comments, heading marks, bold or code marks, link targets or table pipes.

    Bullet marks become "•"; all whitespace (including line breaks) becomes single spaces.
    """
    text = _STRAY_CLOSE.sub("", _COMMENT.sub("", text))
    text = _TABLE_RULE.sub("", text)
    text = "\n".join(_table_row(line) for line in text.splitlines())
    text = _BULLET.sub("• ", _HEADING.sub("", text))
    text = _EMPHASIS.sub("", _LINK.sub(r"\1", text))
    return " ".join(text.split())


def make_snippet(text: str, max_chars: int) -> str:
    """The first `max_chars` characters of the cleaned text, backed up to a word boundary when the cut falls mid-word.

    Example: with max_chars 12, "Capital gain is the profit" -> "Capital gain" (not "Capital gain "); a text that
    already fits is returned whole. The result is always a substring of clean_text(text).
    """
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    if not cleaned[max_chars].isspace():
        head, space, _ = cut.rpartition(" ")
        if space:
            cut = head
    return cut.rstrip()

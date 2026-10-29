"""Citation helpers shared by the agents that cite numbered items ("[1]", "[1, 2]").

Design pattern: PURE FUNCTIONS. Example: with 3 items, drop_invalid_citations("A [2]. B [7].", 3)
returns ("A [2]. B.", [2]); apply_citations renumbers instead (used by Finance Q&A, whose
Sources list contains only the cited chunks).
"""
from __future__ import annotations

import re
from typing import TypeVar

T = TypeVar("T")

# "[1]", "[1, 2]" and "[1,2]" are one group each; "[1][2]" is two groups.
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers(group: str) -> list[int]:
    """'1, 2' -> [1, 2]."""
    return [int(n) for n in re.findall(r"\d+", group)]


def _tidy(text: str) -> str:
    """Remove the spacing a deleted citation leaves behind."""
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)      # space left before punctuation
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)  # double space left in the middle
    return text.strip()


def strip_citations(text: str) -> str:
    """The text without any [n] marker (so the digits are not mistaken for figures)."""
    return CITATION.sub("", text)


def apply_citations(answer: str, items: list[T]) -> tuple[str, list[T]]:
    """Keep only citations that exist, renumber them 1..m, return the cited items.

    Example: with 4 items the reply "A [3]. B [1][9]. C [3, 1]." becomes
    "A [1]. B [2]. C [1, 2]." and the cited items are [item 3, item 1], so the
    Sources list lines up with the numbers the reader sees. [9] does not exist
    and is removed.
    """
    order: list[int] = []
    for match in CITATION.finditer(answer):
        for n in _numbers(match.group(1)):
            if 1 <= n <= len(items) and n not in order:
                order.append(n)
    new_number = {n: i for i, n in enumerate(order, start=1)}

    def renumber(match: re.Match[str]) -> str:
        nums = list(dict.fromkeys(new_number[n] for n in _numbers(match.group(1)) if n in new_number))
        return "[" + ", ".join(str(n) for n in nums) + "]" if nums else ""

    return _tidy(CITATION.sub(renumber, answer)), [items[n - 1] for n in order]


# Fact markers: "[F1]" and "[F1, F2]" (facts are numbered separately from chunks, which use plain numbers).
FACT_CITATION = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")


def strip_fact_citations(text: str) -> str:
    """The text without any [Fn] marker (so the digits are not mistaken for figures)."""
    return FACT_CITATION.sub("", text)


def drop_invalid_fact_citations(text: str, count: int) -> tuple[str, list[int]]:
    """Remove [Fn] markers outside F1..F<count> WITHOUT renumbering; also return the fact numbers still cited.

    Example: with 3 facts, "Rate [F2]. Other [F9]." gives ("Rate [F2]. Other.", [2]). Facts are not
    renumbered because the reader sees the whole numbered fact list.
    """
    cited: set[int] = set()

    def keep(match: re.Match[str]) -> str:
        nums = list(dict.fromkeys(n for n in _numbers(match.group(1)) if 1 <= n <= count))
        cited.update(nums)
        return "[" + ", ".join(f"F{n}" for n in nums) + "]" if nums else ""

    return _tidy(FACT_CITATION.sub(keep, text)), sorted(cited)


def drop_invalid_citations(text: str, count: int) -> tuple[str, list[int]]:
    """Remove citations of numbers outside 1..count WITHOUT renumbering; also return the numbers still cited.

    No renumbering because the reader sees the full numbered list: "[2]" must keep meaning item 2.
    """
    cited: set[int] = set()

    def keep(match: re.Match[str]) -> str:
        nums = list(dict.fromkeys(n for n in _numbers(match.group(1)) if 1 <= n <= count))
        cited.update(nums)
        return "[" + ", ".join(str(n) for n in nums) + "]" if nums else ""

    return _tidy(CITATION.sub(keep, text)), sorted(cited)

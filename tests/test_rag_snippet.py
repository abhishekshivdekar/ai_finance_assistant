"""Snippets for the Sources list: clean display text cut from a chunk in code (src/rag/snippet.py)."""
import json
from pathlib import Path

import pytest

from src.rag.retriever import RetrievedChunk, format_context, to_sources
from src.rag.snippet import clean_text, make_snippet
from src.utils.config import get_config

REFS = [{"name": "SEBI", "url": "https://www.sebi.gov.in"}]


def chunk(text, chunk_id="t/a#0"):
    return RetrievedChunk(text, "Title", "t", REFS, chunk_id, 0.5)


@pytest.mark.parametrize("raw, expected", [
    ("A gain.<!-- VERIFY: rate --> Taxed.", "A gain. Taxed."),
    ("A gain. <!-- VERIFY --", "A gain."),                                          # unterminated comment: dropped to the end
    ("Tail of a comment --> text", "Tail of a comment text"),                       # stray closing mark
    ("<!--\nmulti\nline\n--> Body", "Body"),
    ("# Capital Gains Basics", "Capital Gains Basics"),
    ("## What counts\n\nA capital asset is owned.", "What counts A capital asset is owned."),
    ("Not a heading: C# is a language", "Not a heading: C# is a language"),
    ("A **bold** and __strong__ and `code` word", "A bold and strong and code word"),
    ("- first\n- second\n* third", "• first • second • third"),
    ("1. one\n2. two", "1. one 2. two"),
    ("See [SEBI](https://www.sebi.gov.in) now", "See SEBI now"),
    ("| Year | Rate |\n|---|---|\n| 2024 | 5 |", "Year · Rate 2024 · 5"),
    ("line one\n\n\n  line   two\t!", "line one line two !"),
    ("", ""),
])
def test_clean_text_gives_plain_display_text(raw, expected):
    assert clean_text(raw) == expected


def test_a_chunk_with_a_verify_comment_gives_a_snippet_without_html_comment_marks():
    text = "## Rates\n\nThe rate is set by law.<!-- VERIFY: check the current rate --> It can change."
    snippet = to_sources([chunk(text)])[0]["snippet"]
    assert "<!--" not in snippet and "-->" not in snippet and "VERIFY" not in snippet and "#" not in snippet
    assert snippet == "Rates The rate is set by law. It can change." and snippet in clean_text(text)


def test_a_snippet_is_a_substring_of_the_cleaned_chunk_and_of_a_plain_chunk_itself():
    plain = "A bond is a loan to a government or company. It pays interest and returns the money later."
    snippet = make_snippet(plain, 60)
    assert snippet in plain and snippet in clean_text(plain) and 0 < len(snippet) <= 60
    marked = "## Bonds\n\n**A bond** is a loan to a government or company. It pays interest. " * 5
    assert make_snippet(marked, 80) in clean_text(marked)


@pytest.mark.parametrize("limit", [1, 5, 12, 13, 40, 80])
def test_a_long_text_is_cut_to_the_limit_at_a_word_boundary(limit):
    text = "Capital gain is the profit from selling an asset for more than it cost"
    snippet = make_snippet(text, limit)
    assert len(snippet) <= limit and text.startswith(snippet) and snippet == snippet.rstrip()
    if limit < len(text) and not text[limit].isspace() and " " in text[:limit]:
        assert text[len(snippet)] == " "                                          # no word is cut in half


def test_a_single_long_word_is_cut_hard_and_a_short_text_is_kept_whole():
    assert make_snippet("Supercalifragilistic", 5) == "Super"
    assert make_snippet("Short text.", 240) == "Short text."
    assert make_snippet("Exactly ten", 11) == "Exactly ten"
    assert make_snippet("Capital gain", 7) == "Capital"                           # the cut falls on the space after a word


def test_to_sources_uses_the_configured_length_unless_one_is_given():
    text = "word " * 200
    limit = get_config()["rag"]["snippet_max_chars"]
    assert 0 < len(to_sources([chunk(text)])[0]["snippet"]) <= limit
    assert len(to_sources([chunk(text)], snippet_chars=20)[0]["snippet"]) <= 20


def test_the_prompt_context_still_uses_the_raw_chunk_text_not_the_snippet():
    c = chunk("## Heading\n\n**Bold** text<!-- VERIFY -->")
    to_sources([c])
    assert format_context([c]) == "[1] Title\n## Heading\n\n**Bold** text<!-- VERIFY -->"


INDEX = Path(__file__).resolve().parent.parent / "src" / "rag" / "vector_store" / "chunks.json"


@pytest.mark.skipif(not INDEX.exists(), reason="no saved index")
def test_no_snippet_from_the_saved_index_has_html_comment_marks_or_heading_marks():
    records = json.loads(INDEX.read_text(encoding="utf-8"))
    limit = get_config()["rag"]["snippet_max_chars"]
    for rec in records:
        snippet = make_snippet(rec["text"], limit)
        assert "<!--" not in snippet and "-->" not in snippet and "**" not in snippet and "\n" not in snippet
        assert not snippet.startswith("#") and len(snippet) <= limit and snippet in clean_text(rec["text"])

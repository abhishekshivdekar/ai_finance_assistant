"""Rendering helpers shared by every tab: untrusted text is made safe here, once.

Design pattern: ADAPTER around Streamlit's Markdown. Text from a model, a news feed or the knowledge base is untrusted.
Rules: `unsafe_allow_html` is never used anywhere in the UI; only http(s) links are rendered; `<` is written as an HTML
entity and `$` is escaped (so a price is not read as a formula). Example: the title
"<script>alert(1)</script> [x](javascript:alert(1))" is shown as the literal text "<script>alert(1)</script> x": no tag,
no link.
"""
from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import streamlit as st

from src.core.models import Source
from src.web_app.backend import QueryResult
from src.web_app.format import display_snippet, format_timestamp
from src.web_app.text import ui_text

STOPPED_BUDGET = "budget"
LIVE_SOURCE = "live"          # the payload `source` of data fetched just now (see LIVE_SOURCES in src/core/market_metrics.py)
_LINK = re.compile(r"(!?)\[([^\]\n]*)\]\(((?:[^()\s]|\([^()]*\))*)(?:\s+\"[^\"]*\")?\)")    # [label](target) and ![alt](target)
_REFERENCE_DEFINITION = re.compile(r"^[ ]{0,3}\[[^\]\n]+\]:[ \t]*\S.*$", re.MULTILINE)     # [ref]: target
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-!|~$])")


def safe_http_url(url: str | None) -> str | None:
    """The URL percent-encoded for use inside a Markdown link, or None unless it is a real http(s) URL with a host.

    Example: "https://www.sebi.gov.in" is kept; "javascript:alert(1)", "data:text/html,x", "//host/x" and "" give None.
    Spaces and parentheses are encoded so the URL cannot end the link early.
    """
    text = (url or "").strip()
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return quote(text, safe=":/?#[]@!$&'*+,;=%~-._")


def _link(match: re.Match[str]) -> str:
    """Keep a link whose target is http(s); otherwise (other schemes, images) keep only the label."""
    bang, label, target = match.groups()
    safe = None if bang else safe_http_url(target)
    return f"[{label}]({safe})" if safe else label


def safe_markdown(text: str) -> str:
    """Model or feed text for st.markdown: formatting stays, but only http(s) links survive and nothing is raw HTML."""
    text = _LINK.sub(_link, _REFERENCE_DEFINITION.sub("", text))
    return text.replace("<", "&lt;").replace("$", "\\$")


def plain_text(text: str) -> str:
    """Untrusted text shown literally: links reduced to their label, HTML and Markdown characters escaped."""
    text = _LINK.sub(lambda m: m.group(2), _REFERENCE_DEFINITION.sub("", text))
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _MARKDOWN_SPECIALS.sub(r"\\\1", text).strip()


def render_sources(sources: list[Source]) -> None:
    """An expander listing the cited sources in citation order: [n] title, snippet, and a link only for a real http(s) URL.

    Every entry is kept (no merging of the same article), because the number in the answer is the position here.
    """
    if not sources:
        return
    with st.expander(ui_text("chat.sources_title", count=len(sources))):
        for number, source in enumerate(sources, start=1):
            st.markdown(f"**\\[{number}\\]** {plain_text(source['title'])}")
            snippet = source.get("snippet") or ""
            if snippet.strip():
                st.caption(plain_text(display_snippet(snippet)))
            link = safe_http_url(source.get("url"))
            if link:
                st.markdown(f"[{ui_text('chat.source_link')}]({link})")


def _status_hint(result: QueryResult) -> str | None:
    """A short hint when no agent produced an answer: needs_input wins over no_answer."""
    statuses = [r["status"] for r in result.results]
    if not statuses or "ok" in statuses:
        return None
    if "needs_input" in statuses:
        return ui_text("messages.needs_input_hint")
    if all(status == "no_answer" for status in statuses):
        return ui_text("messages.no_answer_hint")
    return None


def render_result(result: QueryResult, resolve: Callable[[str], str] | None = None) -> None:
    """One answer: the text, a hint when nothing was found or input is missing, then the sources.

    The backend already put the disclaimer into `result.answer`; this function never adds one. A "budget" stop shows the
    UI's own message (the budget is per running app, see ui.messages.budget_stopped); any other stop shows the
    backend's friendly text in a warning box. `resolve` post-processes the safe answer text (the News tab turns its
    [n] markers into links with it).
    """
    if result.stopped == STOPPED_BUDGET:
        st.warning(ui_text("messages.budget_stopped"))
        return
    if not result.answer.strip():
        st.warning(ui_text("messages.empty_answer"))
        return
    text = safe_markdown(result.answer)
    if result.stopped:
        st.warning(text)
    else:
        st.markdown(resolve(text) if resolve else text)
    hint = None if result.stopped else _status_hint(result)
    if hint:
        st.caption(hint)
    render_sources(result.sources)


def payload_of(result: QueryResult, agent: str) -> dict[str, Any] | None:
    """The data payload of this turn's result for one agent (any status), or None. Payload text is untrusted too."""
    for item in result.results:
        data = item.get("data")
        if item["agent"] == agent and isinstance(data, dict):
            return data
    return None


def render_warnings(data: dict[str, Any]) -> None:
    """Every warning of a payload as a warning box, in plain text (the payload is the source of truth, not the answer).

    The first letter is shown in capitals (display only: the payload keeps its text). Example: "low scenario: ..." -> "Low scenario: ...".
    """
    for warning in data.get("warnings") or []:
        text = str(warning).strip()
        st.warning(plain_text(text[:1].upper() + text[1:]))


def render_freshness(source: str, as_of: str, stale: bool, details: list[str]) -> None:
    """Where the data came from and when, from the payload: green when live, blue for a fresh cache, orange when stale or sample.

    The colour follows `source` and `stale` only (stale/sample = orange; "live" = green; anything else = blue), the same for
    every tab. `details` are the payload's own sentences for prices that are not live (Market `freshness_texts`).
    """
    if not source:
        return
    when = format_timestamp(as_of) if as_of else ""
    line = ui_text("freshness.with_time", source=plain_text(source), as_of=when) if when else ui_text("freshness.no_time", source=plain_text(source))
    show = st.warning if stale else st.success if source == LIVE_SOURCE else st.info
    show(line)
    for detail in details:
        st.caption(plain_text(detail))

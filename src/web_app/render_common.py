"""Rendering helpers shared by every tab: untrusted text is made safe here, once.

Design pattern: ADAPTER around Streamlit's Markdown. Text from a model, a news feed or the knowledge base is untrusted.
Rules: `unsafe_allow_html` is never used anywhere in the UI; only http(s) links are rendered; `<` is written as an HTML
entity and `$` is escaped (so a price is not read as a formula). Example: the title
"<script>alert(1)</script> [x](javascript:alert(1))" is shown as the literal text "<script>alert(1)</script> x": no tag,
no link.
"""
from __future__ import annotations

import re
from urllib.parse import quote, urlsplit

import streamlit as st

from src.core.models import Source
from src.web_app.backend import QueryResult
from src.web_app.format import display_snippet
from src.web_app.text import ui_text

STOPPED_BUDGET = "budget"
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


def render_result(result: QueryResult) -> None:
    """One answer: the text, a hint when nothing was found or input is missing, then the sources.

    The backend already put the disclaimer into `result.answer`; this function never adds one. A "budget" stop shows the
    UI's own message (the budget is per running app, see ui.messages.budget_stopped); any other stop shows the
    backend's friendly text in a warning box.
    """
    if result.stopped == STOPPED_BUDGET:
        st.warning(ui_text("messages.budget_stopped"))
        return
    if not result.answer.strip():
        st.warning(ui_text("messages.empty_answer"))
        return
    if result.stopped:
        st.warning(safe_markdown(result.answer))
    else:
        st.markdown(safe_markdown(result.answer))
    hint = None if result.stopped else _status_hint(result)
    if hint:
        st.caption(hint)
    render_sources(result.sources)

"""The News tab: a topic or a ticker goes to the News agent with intent_hint="news" (the router is skipped).

The answer cites headlines as [n] (or [News n]) and the result has no sources: the numbers refer to the `items` of the
NewsPayload. Each marker becomes a link to its article when the item exists and has a real http(s) link, and stays plain
text otherwise. The headline list shows every article once. Titles and publishers are untrusted feed text: plain text only.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import streamlit as st

from src.web_app import backend, session
from src.web_app.calls import run_waiting
from src.web_app.format import format_timestamp
from src.web_app.render_common import payload_of, plain_text, render_freshness, render_result, render_warnings, safe_http_url
from src.web_app.text import ui_text

TAB, HINT = "news", "news"
MODES = ("query", "ticker")                     # the keys of news_params the agent understands
_CITE = r"(?:News )?\d+"                      # "3" or "News 3": the two forms the backend writes (src/workflow/answers.py)
_MARKER = re.compile(rf"\[{_CITE}(?:\s*,\s*{_CITE})*\](?!\()")      # a "(" right after is a Markdown link, not a citation


def resolve_markers(text: str, items: list[dict[str, Any]]) -> str:
    """Turn every [n] / [News n] / [1, 2] marker into per-number markers, linked to the article when it has a real link.

    Only those forms are citations. Other brackets ("[FY 2026]", "[Table 3]", "[Q3 2026]") and a marker already followed
    by "(" (a Markdown link such as "[1](https://...)") are left as written.

    Example: with item 1 linked to https://a.example, "see [1][3]" becomes "see [[1]](https://a.example)\\[3\\]": the
    first is a link, the missing item 3 is plain text (never an error). Nothing is renumbered or merged.
    """
    links = {item.get("number"): safe_http_url(item.get("link")) for item in items}

    def link(match: re.Match[str]) -> str:
        parts = []
        for number in (int(n) for n in re.findall(r"\d+", match.group(0))):
            url = links.get(number)
            parts.append(f"[\\[{number}\\]]({url})" if url else f"\\[{number}\\]")
        return "".join(parts)

    return _MARKER.sub(link, text)


def _headlines(items: list[dict[str, Any]]) -> None:
    """Every fetched headline once: [n] title, publisher and time, and a link only when it is a real http(s) URL."""
    unique: list[dict[str, Any]] = []
    for item in items:
        if all(item.get("number") != kept.get("number") for kept in unique):
            unique.append(item)
    if not unique:
        return
    st.subheader(ui_text("news.headlines_heading", count=len(unique)))
    for item in unique:
        st.markdown(f"**\\[{item.get('number')}\\]** {plain_text(str(item.get('title') or ''))}")
        details = [plain_text(str(item[key])) for key in ("publisher",) if item.get(key)]
        if item.get("published"):
            details.append(ui_text("news.published", when=format_timestamp(str(item["published"]))))
        if details:
            st.caption(" · ".join(details))
        url = safe_http_url(item.get("link"))
        if url:
            st.markdown(f"[{ui_text('news.open_article')}]({url})")


def _render_result(result: backend.QueryResult) -> None:
    """The answer with linked markers, then freshness and warnings of the payload, then the headline list."""
    data = payload_of(result, "news")
    items = list((data or {}).get("items") or [])
    render_result(result, resolve=lambda text: resolve_markers(text, items))
    if data is None:
        return
    render_freshness(str(data.get("source") or ""), str(data.get("as_of") or ""), bool(data.get("is_stale")), [])
    render_warnings(data)
    _headlines(items)


def render(budget_exceeded: bool) -> Callable[[], None] | None:
    """Draw the form and the last result; returns the deferred call when a request is waiting (made after every tab is drawn)."""
    busy = session.is_busy()
    st.markdown(ui_text("news.intro"))
    if budget_exceeded:
        st.warning(ui_text("messages.budget_stopped"))
    with st.form("news_form"):
        mode = st.radio(
            ui_text("news.mode_label"), MODES, format_func=lambda m: ui_text(f"news.mode_{m}"), horizontal=True, disabled=busy,
        )
        text = st.text_input(ui_text("news.text_label"), placeholder=ui_text("news.text_placeholder"), help=ui_text("news.text_help"), disabled=busy)
        submitted = st.form_submit_button(ui_text("news.submit"), disabled=busy or budget_exceeded)
    if submitted:
        text = text.strip()
        if not text:
            st.warning(ui_text("news.text_required"))
        else:
            query = ui_text(f"news.query_template_{mode}", text=text)
            session.submit(TAB, query, HINT, news_params={mode: text})
            st.rerun()
    slot = st.container()                                  # the spinner of the waiting call is drawn here
    last = session.result_for(TAB)
    if last is not None:
        _render_result(last)
    return (lambda: run_waiting(TAB, slot, "news.working")) if session.has_pending(TAB) else None

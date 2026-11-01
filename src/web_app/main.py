"""The page: header, sidebar (usage, new conversation, start-up status) and the tabs.

Design pattern: COMPOSITION ROOT for the UI. `render()` is the only function app.py calls; each tab is one `render`
function, so a new tab is one new module and one line here. A tab's `render` draws the tab and may return an action
(the backend call for a waiting request); `render()` runs it only after ALL tabs are drawn, so while the call runs
every submit button on the page is already disabled.
"""
from __future__ import annotations

import streamlit as st

from src.utils.logger import get_logger
from src.web_app import backend, session, tab_chat, tab_goals, tab_market, tab_news, tab_portfolio
from src.web_app.format import format_count, format_usd
from src.web_app.text import ui_text

log = get_logger(__name__)
TABS = (tab_chat.TAB, tab_portfolio.TAB, tab_goals.TAB, tab_market.TAB, tab_news.TAB)


@st.cache_resource(show_spinner=False)
def _warm_up() -> backend.WarmUp:
    """Warm the backend once per server process (Singleton via cache_resource); a failure is a result, not a crash.

    The result is cached either way, so a failed start-up is not retried on every click; restart the app to retry.
    """
    try:
        return backend.start_up()
    except Exception as exc:                                  # the facade never raises; this protects the page anyway
        log.exception("Start-up warm-up raised")
        return backend.WarmUp(ok=False, seconds=0.0, message=type(exc).__name__)


def _render_sidebar(usage: backend.UsageSummary, warm: backend.WarmUp) -> None:
    """Usage since the app started, the New conversation button and the start-up status. Never shows exception text."""
    with st.sidebar:
        st.subheader(ui_text("sidebar.usage_title"))
        st.metric(ui_text("sidebar.usage_tokens"), format_count(usage.tokens), help=ui_text("sidebar.usage_tokens_help"))
        st.caption(ui_text("sidebar.usage_limit", limit=format_count(usage.limit)) if usage.limit else ui_text("sidebar.usage_no_limit"))
        st.caption(ui_text("sidebar.usage_cost", cost=format_usd(usage.cost)) if usage.cost is not None else ui_text("sidebar.usage_cost_unknown"))
        st.button(
            ui_text("sidebar.new_conversation"), help=ui_text("sidebar.new_conversation_help"),
            on_click=session.new_conversation, disabled=session.is_busy(),
        )
        st.subheader(ui_text("sidebar.status_title"))
        if warm.ok:
            st.success(ui_text("sidebar.warmup_ok"))
        else:
            st.warning(ui_text("sidebar.warmup_failed", error=warm.message))


def render() -> None:
    """Draw the whole page for one script run."""
    st.set_page_config(page_title=ui_text("page_title"), layout="wide")
    session.init()
    with st.spinner(ui_text("sidebar.warmup_running")):
        warm = _warm_up()
    usage = backend.usage()
    _render_sidebar(usage, warm)
    st.title(ui_text("app_title"))
    st.caption(ui_text("app_tagline"))
    active = session.active_tab()
    chat_tab, portfolio_tab, goals_tab, market_tab, news_tab = st.tabs(
        [ui_text(f"tabs.{key}") for key in TABS], default=ui_text(f"tabs.{active}") if active else None,
    )
    deferred = []
    for tab, render_tab in (
        (chat_tab, tab_chat.render), (portfolio_tab, tab_portfolio.render), (goals_tab, tab_goals.render),
        (market_tab, tab_market.render), (news_tab, tab_news.render),
    ):
        with tab:
            deferred.append(render_tab(usage.exceeded))
    for action in deferred:                                   # the one waiting call (if any), after every tab is drawn disabled
        if action is not None:
            action()
    if not any(action is not None for action in deferred):    # the answer is on screen: later reruns must not pull the user back to this tab
        session.clear_active_tab()

"""Pure helpers of the web app, the `ui:` config rules and the architectural rules of src/web_app."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import yaml

from src.utils.config import UI_TEXT_KEYS, ConfigError, get_config
from src.web_app import session
from src.web_app.format import display_snippet, format_count, format_usd
from src.web_app.render_common import plain_text, safe_http_url, safe_markdown
from src.web_app.text import ui_text

ROOT = Path(__file__).resolve().parent.parent
WEB_APP = ROOT / "src" / "web_app"
HOSTILE = "<script>alert(1)</script> [x](javascript:alert(1))"


# ---- formatting ----

def test_formatting_helpers():
    assert format_count(0) == "0" and format_count(1234567) == "1,234,567"
    assert format_usd(0.00597) == "0.0060" and format_usd(1234.5) == "1,234.5000"
    assert display_snippet("Ends here.") == "Ends here." and display_snippet("Cut mid sentence") == "Cut mid sentence…"
    assert display_snippet("  Really?  ") == "Really?" and display_snippet("Wow!") == "Wow!" and display_snippet("   ") == ""


# ---- untrusted text ----

@pytest.mark.parametrize("url, expected", [
    ("https://www.sebi.gov.in", "https://www.sebi.gov.in"),
    ("http://example.com/a?b=c#d", "http://example.com/a?b=c#d"),
    ("  HTTPS://Example.com/x  ", "HTTPS://Example.com/x"),
    ("https://example.com/a b(c)", "https://example.com/a%20b%28c%29"),
    ("javascript:alert(1)", None), ("data:text/html,<b>x</b>", None), ("//example.com/x", None), ("ftp://example.com", None),
    ("https://", None), ("http://[::1", None), ("", None), (None, None), ("   ", None), ("example.com", None),
])
def test_only_real_http_urls_become_links(url, expected):
    assert safe_http_url(url) == expected


def test_safe_markdown_keeps_formatting_and_http_links_but_drops_everything_else():
    text = "**Bold** [ok](https://www.sebi.gov.in/x) [bad](javascript:alert(1)) ![img](https://evil.example/p.png) <b>hi</b> $5 and $6"
    out = safe_markdown(text)
    assert out == "**Bold** [ok](https://www.sebi.gov.in/x) bad img &lt;b>hi&lt;/b> \\$5 and \\$6"
    assert "javascript" not in out and "<b>" not in out and "![" not in out


def test_safe_markdown_drops_reference_style_link_definitions_and_titled_links():
    out = safe_markdown('Read [this][1] now.\n[1]: javascript:alert(1)\n[t](https://example.com "title") [u](javascript:x "t")')
    assert "javascript" not in out and "[1]:" not in out and "[t](https://example.com)" in out and out.endswith("[t](https://example.com) u")


def test_plain_text_shows_markup_literally():
    out = plain_text(HOSTILE)
    assert out == "&lt;script&gt;alert\\(1\\)&lt;/script&gt; x"
    assert "<" not in out and "javascript" not in out
    assert plain_text("A & B *bold* `code` # not a heading $5 | pipe_") == "A &amp; B \\*bold\\* \\`code\\` \\# not a heading \\$5 \\| pipe\\_"
    assert plain_text("[a](https://x.example) [b](javascript:1)\n[r]: javascript:2") == "a b"


# ---- the ui: config block ----

def test_ui_text_reads_the_config_and_fills_fields():
    assert ui_text("tabs.chat") == get_config()["ui"]["tabs"]["chat"]
    assert ui_text("chat.sources_title", count=3) == "Sources (3)"
    with pytest.raises(KeyError):
        ui_text("chat.no_such_key")


def test_every_ui_key_the_code_asks_for_is_in_the_config():
    ui = get_config()["ui"]
    for dotted in UI_TEXT_KEYS:
        node = ui
        for part in dotted.split("."):
            node = node[part]
        assert isinstance(node, str) and node.strip()


def _write_ui(tmp_path, change):
    cfg = {"ui": copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["ui"])}
    change(cfg["ui"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("change", [
    lambda ui: ui.pop("page_title"), lambda ui: ui["tabs"].pop("chat"), lambda ui: ui["chat"].update(placeholder="  "),
    lambda ui: ui["messages"].update(budget_stopped=5), lambda ui: ui["sidebar"].update(warmup_failed=None),
    lambda ui: ui.pop("sidebar"), lambda ui: ui.update(tabs="Chat"),
])
def test_a_broken_ui_block_is_rejected_at_load(tmp_path, change):
    with pytest.raises(ConfigError, match="ui."):
        get_config(_write_ui(tmp_path, change))


def test_a_ui_block_that_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ui: text\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ui must be a mapping"):
        get_config(str(path))


def test_the_complete_ui_block_loads(tmp_path):
    assert get_config(_write_ui(tmp_path, lambda ui: None))["ui"]["chat"]["thinking"]


@pytest.mark.parametrize("value, ok", [("2000000", True), ("null", True), ("1", True), ("0", False), ("-5", False), ("'10'", False), ("1.5", False), ("true", False)])
def test_max_session_tokens_is_null_or_a_positive_whole_number(tmp_path, value, ok):
    path = tmp_path / "config.yaml"
    path.write_text(f"llm:\n  max_session_tokens: {value}\n", encoding="utf-8")
    if ok:
        assert "llm" in get_config(str(path))
    else:
        with pytest.raises(ConfigError, match="max_session_tokens"):
            get_config(str(path))


def test_an_llm_block_without_a_limit_is_fine(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  model: m\n", encoding="utf-8")
    assert get_config(str(path))["llm"] == {"model": "m"}


# ---- rules for the code in src/web_app and app.py ----

def _sources() -> list[Path]:
    return sorted(WEB_APP.glob("*.py")) + [ROOT / "app.py"]


def _code_strings(path: Path) -> list[str]:
    """String literals of a module that are not docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def test_unsafe_allow_html_is_never_used():
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        keywords = [n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword)]
        names = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)] + [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
        assert "unsafe_allow_html" not in keywords + names + _code_strings(path), path.name


ALLOWED_SRC_IMPORTS = ("src.web_app", "src.utils", "src.core.models")


def test_only_backend_py_imports_the_workflow_and_no_ui_module_imports_an_agent_retriever_provider_or_llm():
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                if not name.startswith("src"):
                    continue
                if path.name == "backend.py":
                    assert name == "src.workflow.facade", (path.name, name)
                else:
                    assert name.startswith(ALLOWED_SRC_IMPORTS), (path.name, name)


def test_no_known_label_is_written_in_the_code():
    """Spot-check: labels and messages come from ui_text(), so no string literal in the UI code is a known UI text."""
    texts = [t for t in _flatten(get_config()["ui"]) if len(t) > 8]
    for path in _sources():
        literals = [lit for lit in _code_strings(path) if " " in lit]                 # identifiers and keys have no spaces
        for text in texts:
            assert not any(text in literal or literal in text and len(literal) > 8 for literal in literals), (path.name, text)


def _flatten(node) -> list[str]:
    if isinstance(node, dict):
        return [t for value in node.values() for t in _flatten(value)]
    return [" ".join(node.split())]


# ---- backend.py is a pure pass-through to the facade ----

def test_backend_passes_calls_through_to_the_facade(monkeypatch):
    from src.web_app import backend

    calls = []
    monkeypatch.setattr(backend, "run_query", lambda *args, **kwargs: calls.append((args, kwargs)) or "result")
    monkeypatch.setattr(backend, "warm_up", lambda: "warm")
    monkeypatch.setattr(backend, "usage_summary", lambda: "usage")
    assert backend.ask("q", "t1") == "result" and backend.ask("q", "t1", "market", market_params={"tickers": ["AAPL"]}) == "result"
    assert calls == [(("q", "t1", None), {}), (("q", "t1", "market"), {"market_params": {"tickers": ["AAPL"]}})]
    assert backend.start_up() == "warm" and backend.usage() == "usage"


# ---- times, and the request hand-over ----

@pytest.mark.parametrize("iso, expected", [
    ("2026-10-30T08:32:00+00:00", "2026-10-30 08:32 UTC"),
    ("2026-10-30T14:02:00+05:30", "2026-10-30 08:32 UTC"),           # an aware time is shown in UTC
    ("2026-10-30T08:32:00", "2026-10-30 08:32 UTC"),                  # a naive time is taken as UTC (as the backend does)
    ("last Tuesday", "last Tuesday"),                                 # not a time: shown as it is
    ("", ""),
])
def test_format_timestamp(iso, expected):
    from src.web_app.format import format_timestamp

    assert format_timestamp(iso) == expected


def test_format_percent():
    from src.web_app.format import format_percent

    assert format_percent(12.34, "n/a") == "+12.3%" and format_percent(-4.56, "n/a") == "-4.6%" and format_percent(0.0, "n/a") == "+0.0%"
    assert format_percent(None, "n/a") == "n/a"


def test_a_waiting_request_is_handed_out_once_and_only_to_its_own_tab():
    from streamlit.testing.v1 import AppTest

    def script():
        import streamlit as st

        from src.web_app import session

        session.init()
        session.submit("market", "q", "market", market_params={"tickers": ["A"]})
        st.session_state["seen"] = [
            session.has_pending("news"), session.take_pending("news"), session.is_busy(), session.has_pending("market"),
            session.take_pending("market"), session.take_pending("market"), session.is_busy(), session.active_tab(),
        ]

    at = AppTest.from_function(script).run()
    assert not at.exception
    request = session.Request("market", "q", "market", {"market_params": {"tickers": ["A"]}})
    assert at.session_state["seen"] == [False, None, True, True, request, None, False, "market"]


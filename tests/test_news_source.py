import copy
import urllib.error
import urllib.parse
import urllib.request

import pytest

from src.data.errors import PermanentProviderError, TransientProviderError
from src.data.news import google_rss as g
from src.utils.config import get_config

CFG = get_config()["news"]
RSS = (b"<rss><channel><item><title>Stocks rise - Mint</title><link>https://news.test/a</link>"
       b"<pubDate>Thu, 29 Oct 2026 10:00:00 GMT</pubDate><source>Mint</source></item></channel></rss>")
from datetime import datetime, timezone

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


class Recorder:
    """A fetcher that records its arguments and returns fixed bytes."""

    def __init__(self, body=RSS):
        self.body, self.calls = body, []

    def __call__(self, url, headers, timeout, max_bytes):
        self.calls.append((url, headers, timeout, max_bytes))
        return self.body


def source(fetcher):
    return g.GoogleNewsRssSource(fetcher=fetcher, clock=lambda: NOW)


def patch_cfg(monkeypatch, **changes):
    full = copy.deepcopy(get_config())
    full["news"].update(changes)
    monkeypatch.setattr(g, "get_config", lambda: full)


# ---- the source with an injected fetcher ----

def test_fetch_builds_the_url_from_config_and_parses_the_feed():
    rec = Recorder()
    items = source(rec).fetch("Nifty 50")
    url, headers, timeout, max_bytes = rec.calls[0]
    parts = urllib.parse.urlparse(url)
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "news.google.com", "/rss/search")
    assert urllib.parse.parse_qs(parts.query) == {"q": ["Nifty 50"], "hl": ["en-IN"], "gl": ["IN"], "ceid": ["IN:en"]}
    assert headers == {"User-Agent": CFG["user_agent"]}
    assert timeout == CFG["timeout_seconds"] and max_bytes == CFG["max_response_bytes"]
    assert items == [{"title": "Stocks rise", "publisher": "Mint", "link": "https://news.test/a",
                      "published": "2026-10-29T10:00:00+00:00"}]


def test_a_host_that_is_not_allowed_is_rejected_before_any_request(monkeypatch):
    patch_cfg(monkeypatch, base_url="https://evil.test/rss/search")
    rec = Recorder()
    with pytest.raises(PermanentProviderError, match="not allowed"):
        source(rec).fetch("x")
    assert rec.calls == []


def test_plain_http_is_rejected(monkeypatch):
    patch_cfg(monkeypatch, base_url="http://news.google.com/rss/search")
    with pytest.raises(PermanentProviderError):
        source(Recorder()).fetch("x")


def test_oversize_body_from_any_fetcher_is_rejected_before_parsing(monkeypatch):
    patch_cfg(monkeypatch, max_response_bytes=10)
    with pytest.raises(PermanentProviderError, match="larger"):
        source(Recorder(RSS)).fetch("x")


@pytest.mark.parametrize("body", [b"<rss><channel><item>", b"<rss/>", b"<!DOCTYPE rss><rss><channel/></rss>"])
def test_malformed_xml_becomes_a_permanent_error(body):
    with pytest.raises(PermanentProviderError):
        source(Recorder(body)).fetch("x")


def test_an_empty_feed_is_not_an_error():
    assert source(Recorder(b"<rss><channel></channel></rss>")).fetch("zzz") == []


def test_fetcher_errors_propagate_unchanged():
    def down(*args):
        raise TransientProviderError("down")

    with pytest.raises(TransientProviderError):
        source(down).fetch("x")


def test_default_clock_and_fetcher_are_wired():
    s = g.GoogleNewsRssSource()
    assert s._fetcher is g.http_fetch and s._clock().tzinfo is not None


# ---- http_fetch with a fake opener (no network) ----

class FakeResponse:
    def __init__(self, body=b"ok", length=None):
        self._body = body
        self.headers = {} if length is None else {"Content-Length": length}

    def read(self, n):
        return self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, outcome):
        self.outcome, self.requests = outcome, []

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fetch_with(monkeypatch, outcome, max_bytes=100):
    opener = FakeOpener(outcome)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: opener)
    body = g.http_fetch("https://news.google.com/rss/search?q=x", {"User-Agent": "UA"}, 7, max_bytes)
    return body, opener


def test_http_fetch_sends_the_user_agent_and_timeout(monkeypatch):
    body, opener = fetch_with(monkeypatch, FakeResponse(b"hello"))
    request, timeout = opener.requests[0]
    assert body == b"hello" and timeout == 7
    assert request.get_header("User-agent") == "UA" and request.full_url.startswith("https://news.google.com/")


def test_http_fetch_rejects_a_declared_size_over_the_limit(monkeypatch):
    with pytest.raises(PermanentProviderError, match="larger"):
        fetch_with(monkeypatch, FakeResponse(b"x", length="101"))


def test_http_fetch_ignores_a_non_numeric_content_length(monkeypatch):
    assert fetch_with(monkeypatch, FakeResponse(b"abc", length="n/a"))[0] == b"abc"


def test_http_fetch_rejects_a_body_over_the_limit_without_a_header(monkeypatch):
    with pytest.raises(PermanentProviderError, match="larger"):
        fetch_with(monkeypatch, FakeResponse(b"x" * 500))


def test_http_fetch_accepts_a_body_exactly_at_the_limit(monkeypatch):
    assert len(fetch_with(monkeypatch, FakeResponse(b"x" * 100), max_bytes=100)[0]) == 100


@pytest.mark.parametrize("code, expected", [
    (500, TransientProviderError), (503, TransientProviderError), (429, TransientProviderError),
    (404, PermanentProviderError), (403, PermanentProviderError),
])
def test_http_errors_are_classified(monkeypatch, code, expected):
    error = urllib.error.HTTPError("https://news.google.com/x", code, "msg", {}, None)
    with pytest.raises(expected):
        fetch_with(monkeypatch, error)


@pytest.mark.parametrize("error", [urllib.error.URLError("no route"), TimeoutError("timed out"), ConnectionResetError("reset")])
def test_network_problems_are_transient(monkeypatch, error):
    with pytest.raises(TransientProviderError):
        fetch_with(monkeypatch, error)


def test_http_fetch_builds_its_opener_with_the_redirect_guard(monkeypatch):
    seen = {}

    def build(*handlers):
        seen["handlers"] = handlers
        return FakeOpener(FakeResponse(b"ok"))

    monkeypatch.setattr(urllib.request, "build_opener", build)
    g.http_fetch("https://news.google.com/x", {}, 1, 10)
    assert isinstance(seen["handlers"][0], g._AllowedRedirects)


# ---- redirects ----

def redirect(handler, url):
    request = urllib.request.Request("https://news.google.com/rss/search?q=x")
    return handler.redirect_request(request, None, 302, "Found", {}, url)


def test_redirect_to_another_host_or_to_http_is_refused():
    handler = g._AllowedRedirects(["news.google.com"])
    for target in ("https://evil.test/x", "http://news.google.com/x", "file:///etc/passwd"):
        with pytest.raises(PermanentProviderError, match="not allowed"):
            redirect(handler, target)


def test_redirect_within_the_allowed_host_is_followed():
    new = redirect(g._AllowedRedirects(["news.google.com"]), "https://news.google.com/rss/other")
    assert new.full_url == "https://news.google.com/rss/other"


def test_host_allowed():
    assert g.host_allowed("https://news.google.com/x", ["news.google.com"])
    assert not g.host_allowed("https://news.google.com.evil.test/x", ["news.google.com"])
    assert not g.host_allowed("not a url", ["news.google.com"])

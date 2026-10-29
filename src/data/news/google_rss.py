"""Google News RSS implementation of NewsSource (standard library only).

Design patterns:
- STRATEGY: one NewsSource implementation (see source.py).
- DEPENDENCY INJECTION: the network call is a constructor argument (`fetcher`).
  Example: a test passes GoogleNewsRssSource(fetcher=lambda *a: RSS_BYTES) and
  needs no network, while the app uses the default http_fetch.

Safety: only hosts listed in `news.allowed_hosts` over https are contacted
(redirects included), the response size is capped before parsing, and the
timeout and User-Agent come from config.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

from src.core.news_parse import FeedError, parse_feed
from src.data.errors import PermanentProviderError, TransientProviderError
from src.data.news.source import NewsSource
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

Fetcher = Callable[[str, dict[str, str], float, int], bytes]   # (url, headers, timeout, max_bytes) -> body


def host_allowed(url: str, allowed_hosts: list[str]) -> bool:
    """True for an https URL whose host is in the allow-list."""
    parts = urllib.parse.urlparse(url)
    return parts.scheme == "https" and (parts.hostname or "") in allowed_hosts


class _AllowedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only when its target host is allowed."""

    def __init__(self, allowed_hosts: list[str]) -> None:
        self._allowed = allowed_hosts

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        """Refuse redirects to other hosts (or to http)."""
        if not host_allowed(newurl, self._allowed):
            raise PermanentProviderError("The news feed redirected to a host that is not allowed.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_fetch(url: str, headers: dict[str, str], timeout: float, max_bytes: int) -> bytes:
    """GET `url`; the ONLY place in the news layer that touches the network.

    Reads at most max_bytes + 1 and rejects anything larger, so a huge response is never held or parsed.
    """
    opener = urllib.request.build_opener(_AllowedRedirects(get_config()["news"]["allowed_hosts"]))
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise PermanentProviderError("The news response is larger than the allowed size.")
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:       # before URLError: HTTPError is a subclass
        if exc.code >= 500 or exc.code == 429:
            raise TransientProviderError(f"News feed returned HTTP {exc.code}.") from exc
        raise PermanentProviderError(f"News feed returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, OSError) as exc:     # timeouts and connection problems
        raise TransientProviderError(f"News feed unreachable: {exc}") from exc
    if len(body) > max_bytes:
        raise PermanentProviderError("The news response is larger than the allowed size.")
    return body


class GoogleNewsRssSource(NewsSource):
    """Headlines from the Google News RSS search feed."""

    def __init__(self, fetcher: Fetcher = http_fetch, clock: Callable[[], datetime] | None = None) -> None:
        self._fetcher = fetcher
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def fetch(self, query: str) -> list[dict[str, Any]]:
        """Fetch and parse the feed for `query` (raises Transient/PermanentProviderError)."""
        cfg = get_config()["news"]
        url = cfg["base_url"] + "?" + urllib.parse.urlencode({"q": query, **cfg["params"]})
        if not host_allowed(url, cfg["allowed_hosts"]):
            raise PermanentProviderError("The news feed host is not allowed.")
        body = self._fetcher(url, {"User-Agent": cfg["user_agent"]}, cfg["timeout_seconds"], cfg["max_response_bytes"])
        if len(body) > cfg["max_response_bytes"]:
            raise PermanentProviderError("The news response is larger than the allowed size.")
        try:
            return parse_feed(body, self._clock(), cfg)
        except FeedError as exc:
            log.warning("Unusable news feed for %r: %s", query, exc)
            raise PermanentProviderError(str(exc)) from exc

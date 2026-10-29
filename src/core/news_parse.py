"""Pure parsing and cleaning of a Google News RSS feed: no I/O, no hidden state.

Design pattern: PURE FUNCTIONS. Example: parse_feed(xml, now, cfg) on a feed with
the same story twice returns it once, newest first, so a test needs only an XML
string and a fixed `now`.

Everything in a feed is untrusted text: titles are unescaped, stripped of tags and
control characters and length-capped; links must be http(s); anything unusable is
dropped (never guessed).
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

_TAG = re.compile(r"<[^>]*>")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_NON_ALNUM = re.compile(r"[\W_]+")
_DTD = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_DECLARED_ENCODING = re.compile(r"^<\?xml[^>]*\bencoding\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)


class FeedError(ValueError):
    """The response is not a usable RSS feed (not UTF-8, malformed XML, DTD/entities, no channel)."""


def decode_utf8(xml_bytes: bytes) -> str:
    """The body as text, only when it is plain UTF-8 (no BOM, no NUL bytes, no other declared encoding).

    Checks run on decoded text, so a UTF-16 payload cannot hide a DOCTYPE from the later DTD check.
    """
    try:
        text = xml_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FeedError("The news feed is not valid UTF-8.") from exc
    if text.startswith("\ufeff") or "\x00" in text:
        raise FeedError("The news feed must be plain UTF-8 (no byte-order mark, no NUL characters).")
    declared = _DECLARED_ENCODING.match(text)
    if declared and declared.group(1).lower().replace("_", "-") not in ("utf-8", "utf8"):
        raise FeedError("The news feed declares an encoding other than UTF-8.")
    return text


def clean_text(text: str | None) -> str:
    """Unescape entities, drop tags and control characters, collapse whitespace."""
    plain = _TAG.sub(" ", html.unescape(text or ""))
    return " ".join(_CONTROL.sub(" ", plain).split())


def cap(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, ending with an ellipsis when something was removed."""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def strip_publisher(title: str, publisher: str) -> str:
    """Remove a trailing ' - Publisher' only when it matches the <source> element."""
    suffix = f" - {publisher}"
    return title[: -len(suffix)].rstrip() if publisher and title.endswith(suffix) else title


def parse_date(text: str | None) -> datetime | None:
    """An RFC 822 date as an aware UTC datetime (a date without zone counts as UTC); None if unreadable."""
    try:
        parsed = parsedate_to_datetime((text or "").strip())
    except (TypeError, ValueError, IndexError):
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def valid_link(link: str) -> bool:
    """Only absolute http(s) links are kept (the UI will render them)."""
    parts = urlparse(link)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def parse_item(item: ET.Element, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """One <item> as {title, publisher, link, published}, or None when it is unusable."""
    publisher = clean_text(item.findtext("source"))
    title = cap(strip_publisher(clean_text(item.findtext("title")), publisher), cfg["title_max_chars"])
    link = clean_text(item.findtext("link"))
    published = parse_date(item.findtext("pubDate"))
    if not title or published is None or not valid_link(link):
        return None
    return {"title": title, "publisher": publisher or None, "link": link, "published": published.isoformat()}


def parse_feed(xml_bytes: bytes, now: datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Clean, de-duplicate, age-filter and sort the items of a feed; at most cfg["max_items"], newest first.

    An empty feed returns []. Raises FeedError for a response that is not a usable feed.
    """
    text = decode_utf8(xml_bytes)
    if _DTD.search(text):
        raise FeedError("The news feed contains a DTD or entity declaration.")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FeedError(f"The news feed is not valid XML: {exc}") from exc
    channel = root.find("channel")
    if channel is None:
        raise FeedError("The news feed has no channel.")
    oldest = now - timedelta(days=cfg["max_age_days"])
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for element in channel.findall("item"):
        item = parse_item(element, cfg)
        if item is None or datetime.fromisoformat(item["published"]) < oldest:
            continue
        keys = {_NON_ALNUM.sub("", item["title"].casefold()), item["link"]}
        if keys & seen:
            continue
        seen |= keys
        items.append(item)
    items.sort(key=lambda i: i["published"], reverse=True)
    return items[: cfg["max_items"]]

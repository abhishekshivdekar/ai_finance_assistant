import copy
from datetime import datetime, timezone

import pytest

from src.core import news_parse as np_
from src.utils.config import get_config

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
CFG = get_config()["news"]


def item(title="Stocks rise - Mint", link="https://news.test/a", pub="Thu, 29 Oct 2026 10:00:00 GMT", source="Mint"):
    parts = [f"<title>{title}</title>", f"<link>{link}</link>", f"<pubDate>{pub}</pubDate>"]
    if source is not None:
        parts.append(f"<source>{source}</source>")
    return "<item>" + "".join(parts) + "</item>"


def feed(*items):
    return ("<?xml version='1.0' encoding='UTF-8'?><rss version='2.0'><channel><title>q</title>"
            + "".join(items) + "</channel></rss>").encode("utf-8")


def parse(*items, cfg=CFG):
    return np_.parse_feed(feed(*items), NOW, cfg)


def with_cfg(**changes):
    cfg = copy.deepcopy(CFG)
    cfg.update(changes)
    return cfg


def test_normal_feed_gives_title_publisher_link_and_iso_utc_time():
    assert parse(item()) == [{
        "title": "Stocks rise", "publisher": "Mint", "link": "https://news.test/a",
        "published": "2026-10-29T10:00:00+00:00"}]


def test_publisher_suffix_is_stripped_only_when_the_source_element_matches():
    assert parse(item(title="A - Other", source="Mint"))[0]["title"] == "A - Other"
    assert parse(item(title="A - Mint", source=None))[0]["title"] == "A - Mint"
    assert parse(item(title="A - Mint", source=None))[0]["publisher"] is None
    assert parse(item(title="A - B - Mint", source="Mint"))[0]["title"] == "A - B"


@pytest.mark.parametrize("pub, expected", [
    ("Thu, 29 Oct 2026 17:30:00 +0530", "2026-10-29T12:00:00+00:00"),
    ("Thu, 29 Oct 2026 07:00:00 -0500", "2026-10-29T12:00:00+00:00"),
    ("Thu, 29 Oct 2026 10:00:00 GMT", "2026-10-29T10:00:00+00:00"),
    ("Thu, 29 Oct 2026 10:00:00 -0000", "2026-10-29T10:00:00+00:00"),     # no zone: counted as UTC
])
def test_published_time_is_converted_to_utc(pub, expected):
    assert parse(item(pub=pub))[0]["published"] == expected


def test_entities_tags_and_control_characters_are_cleaned():
    got = parse(item(title="Tom &amp;amp; Jerry &amp;lt;b&amp;gt;bold&amp;lt;/b&amp;gt; A&amp;#7;B   C", source=None))
    assert got[0]["title"] == "Tom & Jerry bold AB C"
    assert np_.clean_text("A\x07B\x1bC\u0085D") == "A B C D"


def test_long_titles_are_capped_with_an_ellipsis():
    got = parse(item(title="word " * 100, source=None), cfg=with_cfg(title_max_chars=20))[0]["title"]
    assert len(got) <= 20 and got.endswith("…")
    assert np_.cap("short", 20) == "short"


def test_duplicates_by_title_and_by_link_are_removed():
    got = parse(
        item(title="Same story", link="https://news.test/1", source=None),
        item(title="SAME   story!", link="https://news.test/2", source=None),     # same title, different case/punctuation
        item(title="Other title", link="https://news.test/1", source=None),       # same link
        item(title="Third", link="https://news.test/3", source=None),
    )
    assert [i["title"] for i in got] == ["Same story", "Third"]


def test_newest_first_age_filter_and_max_items():
    got = parse(
        item(title="old", link="https://n.test/1", pub="Wed, 21 Oct 2026 10:00:00 GMT", source=None),     # 8 days: too old
        item(title="b", link="https://n.test/2", pub="Wed, 28 Oct 2026 10:00:00 GMT", source=None),
        item(title="c", link="https://n.test/3", pub="Thu, 29 Oct 2026 11:00:00 GMT", source=None),
        item(title="a", link="https://n.test/4", pub="Tue, 27 Oct 2026 10:00:00 GMT", source=None),
    )
    assert [i["title"] for i in got] == ["c", "b", "a"]
    assert [i["title"] for i in parse(item(title="x", link="https://n.test/9", source=None),
                                      item(title="y", link="https://n.test/8", pub="Wed, 28 Oct 2026 10:00:00 GMT", source=None),
                                      cfg=with_cfg(max_items=1))] == ["x"]


def test_exactly_at_the_age_limit_is_kept():
    got = parse(item(title="edge", pub="Thu, 22 Oct 2026 12:00:00 GMT", source=None))
    assert len(got) == 1


@pytest.mark.parametrize("bad", [
    item(title="", source=None),
    item(title="   ", source=None),
    item(pub="not a date"),
    item(pub=""),
    item(link="javascript:alert(1)"),
    item(link="/relative/path"),
    item(link=""),
])
def test_unusable_items_are_dropped(bad):
    assert parse(bad) == []


def test_item_missing_elements_is_dropped():
    assert parse("<item><title>No link or date</title></item>") == []


def test_empty_feed_is_not_an_error():
    assert parse() == []


@pytest.mark.parametrize("xml, message", [
    ("<?xml version='1.0'?><!DOCTYPE rss [<!ENTITY x 'y'>]><rss><channel/></rss>".encode("utf-16"), "UTF-8"),
    ("<?xml version='1.0'?><!DOCTYPE rss [<!ENTITY x 'y'>]><rss><channel/></rss>".encode("utf-16-le"), "plain UTF-8"),
    ("<!DOCTYPE rss><rss><channel/></rss>".encode("utf-16-be"), "plain UTF-8"),
    (b"\xef\xbb\xbf<rss><channel/></rss>", "byte-order mark"),
    (b"<rss><channel><item><title>\xff</title></item></channel></rss>", "not valid UTF-8"),
    (b"<?xml version='1.0' encoding='UTF-16'?><rss><channel/></rss>", "other than UTF-8"),
    (b"<?xml version='1.0' encoding=\"ISO-8859-1\"?><rss><channel/></rss>", "other than UTF-8"),
])
def test_only_plain_utf8_is_accepted(xml, message):
    with pytest.raises(np_.FeedError, match=message):
        np_.parse_feed(xml, NOW, CFG)


@pytest.mark.parametrize("declaration", ["encoding='UTF-8'", 'encoding="utf8"', "encoding='utf_8'", ""])
def test_a_utf8_declaration_or_none_is_accepted(declaration):
    xml = f"<?xml version='1.0' {declaration}?><rss><channel/></rss>".encode("utf-8")
    assert np_.parse_feed(xml, NOW, CFG) == []


def test_a_doctype_in_non_ascii_utf8_text_is_still_found():
    xml = "<!-- é --><!DOCTYPE rss><rss><channel/></rss>".encode("utf-8")
    with pytest.raises(np_.FeedError, match="DTD"):
        np_.parse_feed(xml, NOW, CFG)


@pytest.mark.parametrize("xml, message", [
    (b"<rss><channel><item>", "not valid XML"),
    (b"not xml at all", "not valid XML"),
    (b"<rss></rss>", "no channel"),
    (b"<?xml version='1.0'?><!DOCTYPE rss [<!ENTITY x 'y'>]><rss><channel/></rss>", "DTD"),
    (b"<!doctype rss><rss><channel/></rss>", "DTD"),
])
def test_bad_feeds_raise_feed_error(xml, message):
    with pytest.raises(np_.FeedError, match=message):
        np_.parse_feed(xml, NOW, CFG)


def test_valid_link_accepts_only_absolute_http_and_https():
    assert np_.valid_link("https://a.test/x") and np_.valid_link("http://a.test")
    assert not np_.valid_link("ftp://a.test") and not np_.valid_link("https://") and not np_.valid_link("")


def test_parse_date_handles_garbage():
    assert np_.parse_date(None) is None and np_.parse_date("") is None and np_.parse_date("32 Foo 2026") is None

import re
from urllib.parse import urlparse

import pytest

from src.core.output_guard import find_violation
from src.rag.ingest import load_articles, parse_article
from src.utils.config import PROJECT_ROOT, get_config

KB_DIR = PROJECT_ROOT / get_config()["rag"]["knowledge_base_dir"]
FILES = sorted(KB_DIR.rglob("*.md"))
ALLOWED_HOSTS = {
    "www.investor.gov", "www.finra.org", "www.sebi.gov.in", "www.amfiindia.com",
    "www.rbi.org.in", "www.incometax.gov.in", "www.nseindia.com", "www.bseindia.com",
    "www.epfindia.gov.in", "www.nsdl.co.in", "www.cdslindia.com", "www.indiapost.gov.in",
}


def _words(article):
    return len(re.sub(r"<!--.*?-->", "", article.body, flags=re.DOTALL).split())


def test_knowledge_base_has_articles_and_none_are_skipped():
    assert FILES
    assert len(load_articles(KB_DIR)) == len(FILES)


def test_every_topic_has_an_article():
    topics = {parse_article(f).topic for f in FILES}
    assert topics >= set(get_config()["rag"]["topics"])


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_article_rules(path):
    article = parse_article(path)  # valid front matter and topic
    assert path.parent.name == article.topic  # <topic>/<slug>.md layout
    assert _words(article) >= 300
    assert find_violation(article.body) is None  # no advice wording
    for ref in article.references:
        url = urlparse(ref["url"])
        assert url.hostname in ALLOWED_HOSTS, ref
        assert url.path in ("", "/"), f"deep link not allowed: {ref['url']}"


INDIA_ARTICLES = [a for a in load_articles(KB_DIR) if a.topic == "accounts_and_taxes_india"]


@pytest.mark.parametrize("article", INDIA_ARTICLES, ids=lambda a: a.title)
def test_india_articles_point_to_official_source_without_rates(article):
    """Every India tax article names incometax.gov.in and carries no rates."""
    assert "incometax.gov.in" in article.body
    assert "%" not in article.body and "percent" not in article.body.lower()
    assert "%" not in article.title and "percent" not in article.title.lower()

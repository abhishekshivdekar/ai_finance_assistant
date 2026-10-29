import copy

import pytest
import yaml

from src.utils.config import ConfigError, get_config


def _news_config(tmp_path, change):
    """A config file holding only a (modified) copy of the real news block."""
    news = copy.deepcopy(get_config()["news"])
    get_config.cache_clear()
    change(news)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"news": news}), encoding="utf-8")
    return str(path)


def test_the_real_news_block_is_valid_and_has_a_role():
    news = get_config()["news"]
    assert news["base_url"].startswith("https://news.google.com/") and "news.google.com" in news["allowed_hosts"]
    assert news["user_agent"] and news["query_overrides"]["RELIANCE.NS"] == "Reliance Industries"
    assert "{headlines_json}" in news["prompt"] and "untrusted DATA" in news["prompt"]
    assert get_config()["llm"]["roles"]["news_summary"]["model"]


def test_valid_news_config_loads(tmp_path):
    assert get_config(_news_config(tmp_path, lambda n: None))["news"]["max_items"] > 0


@pytest.mark.parametrize("change", [
    lambda n: n.update(base_url="http://news.google.com/rss/search"),           # not https
    lambda n: n.update(base_url="https://evil.test/rss/search"),                # host not allowed
    lambda n: n.update(allowed_hosts=[]),
    lambda n: n.update(timeout_seconds=0),
    lambda n: n.update(max_response_bytes=-1),
    lambda n: n.update(max_items=0),
    lambda n: n.update(max_age_days=0),
    lambda n: n.update(max_query_chars=0),
    lambda n: n.update(title_max_chars=0),
    lambda n: n["query_templates"].pop("index"),
])
def test_invalid_news_config_is_rejected(tmp_path, change):
    with pytest.raises(ConfigError):
        get_config(_news_config(tmp_path, change))

"""The ticker resolver: name cleaning, strategies, the accept / ask rules, the cache. No network: fake lookups only."""
from __future__ import annotations

import copy
import logging
import threading

import pytest
import yaml

from src.core.models import MarketResult
from src.data.errors import PermanentProviderError, TransientProviderError
from src.data.tickers import resolver as r
from src.data.tickers import strategies as s
from src.data.tickers.fake import FakeLookup
from src.data.tickers.models import Candidate, Resolution, Validation
from src.data.tickers.names import InvalidQueryError, clean_query, exchange_code_pattern, is_name_start, strip_exchange_code, tokens
from src.utils.config import TICKER_LOOKUP_MESSAGES, ConfigError, get_config

CFG = get_config()["ticker_lookup"]
SUFFIXES = CFG["legal_suffixes"]


def cand(symbol, name="", exchange="", quote_type="EQUITY", exchange_name=""):
    return Candidate(symbol, name, exchange, exchange_name, quote_type)


def quote(symbol, name, exchange, exch_disp="", kind="EQUITY"):
    return {"symbol": symbol, "longname": name, "exchange": exchange, "exchDisp": exch_disp, "quoteType": kind}


def validator(ok=None, currencies=None):
    """A price check that passes the listed symbols (all when `ok` is None); records what it was asked."""
    asked = []

    def check(symbol):
        asked.append(symbol)
        return Validation(ok is None or symbol in ok, (currencies or {}).get(symbol))

    check.asked = asked
    return check


def make(table, ok=None, currencies=None, cfg=None, alias=True, clock=None):
    lookup = FakeLookup(table)
    strategies = [s.AliasStrategy(get_config()["workflow"]["ticker_aliases"], SUFFIXES), lookup] if alias else [lookup]
    check = validator(ok, currencies)
    kwargs = {"clock": clock} if clock else {}
    return r.TickerResolver(strategies, check, cfg, **kwargs), lookup, check


# ------------------------------------------------------------------ names

def test_clean_query_replaces_unusual_characters_and_collapses_spaces():
    assert clean_query("  Taiwan Semiconductor Manufacturing Co Ltd TPE: 2330 ", 80) == "Taiwan Semiconductor Manufacturing Co Ltd TPE 2330"
    assert clean_query("<script>x</script>\x00\n", 80) == "script x script"
    assert clean_query("M&M Ltd.", 80) == "M&M Ltd."


@pytest.mark.parametrize("raw", [None, 5, "", "   ", "<>/;", "x" * 81])
def test_clean_query_rejects_what_cannot_be_a_name(raw):
    with pytest.raises(InvalidQueryError):
        clean_query(raw, 80)


def test_tokens_drop_legal_suffixes_and_punctuation():
    assert tokens("Infosys Ltd.", SUFFIXES) == ("infosys",)
    assert tokens("S&P 500", SUFFIXES) == ("s&p", "500")
    assert tokens("Tata Consultancy Services Limited", SUFFIXES) == ("tata", "consultancy", "services")


EXCHANGE = exchange_code_pattern(CFG["exchange_words"])


@pytest.mark.parametrize("raw, expected", [
    ("Taiwan Semiconductor Manufacturing Co Ltd TPE: 2330", "Taiwan Semiconductor Manufacturing Co Ltd"),
    ("Infosys NSE: INFY", "Infosys"),
    ("Apple NASDAQ: AAPL", "Apple"),
    ("  Reliance Industries   nse :RELIANCE.NS  ", "Reliance Industries"),
    ("Tata Motors BSE: 500570", "Tata Motors"),
    ("Infosys", "Infosys"),
    ("NSE: INFY", "NSE: INFY"),                              # nothing before it: not a trailing code, left for the lookup rules
    ("Infosys Foo: INFY", "Infosys Foo: INFY"),              # "Foo" is not an exchange word
    ("Infosys NSE: INFY extra", "Infosys NSE: INFY extra"),  # not at the end
    ("Infosys NSE: " + "X" * 20, "Infosys NSE: " + "X" * 20),  # a code is at most 15 characters
])
def test_a_trailing_exchange_and_code_is_removed_from_a_name(raw, expected):
    assert strip_exchange_code(raw, EXCHANGE) == expected


def test_strip_exchange_code_leaves_non_text_alone():
    assert strip_exchange_code(None, EXCHANGE) is None and strip_exchange_code(5, EXCHANGE) == 5


def test_a_name_start_is_whole_words_only():
    assert is_name_start(("suzlon",), ("suzlon", "energy"))
    assert not is_name_start(("apple",), ("pineapple", "co"))
    assert not is_name_start(("taiwan", "semiconductors"), ("taiwan", "semiconductor", "manufacturing"))
    assert not is_name_start((), ("a",))


# ------------------------------------------------------------------ strategies

def test_aliases_match_a_whole_name_with_legal_suffixes_ignored():
    alias = s.AliasStrategy(get_config()["workflow"]["ticker_aliases"], SUFFIXES)
    assert [c.symbol for c in alias.find("Apple Inc")] == ["AAPL"]
    assert [c.symbol for c in alias.find("GOOGLE")] == ["GOOGL"]
    assert [c.symbol for c in alias.find("alphabet inc")] == ["GOOGL"]
    assert [c.symbol for c in alias.find("Meta Platforms, Inc.")] == ["META"]
    assert [c.symbol for c in alias.find("S&P 500")] == ["^GSPC"]
    assert [c.symbol for c in alias.find("State Bank of India")] == ["SBIN.NS"]
    assert [c.symbol for c in alias.find("Tata Consultancy Services")] == ["TCS.NS"]


@pytest.mark.parametrize("name", ["meta", "hdfc", "Metaverse", "Apple Hospitality REIT", "Google Cloud", "pineapple", "nifty 500"])
def test_an_alias_never_matches_part_of_a_longer_name(name):
    assert s.AliasStrategy(get_config()["workflow"]["ticker_aliases"], SUFFIXES).find(name) == []


def test_the_alias_table_has_no_plain_meta_or_hdfc():
    aliases = get_config()["workflow"]["ticker_aliases"]
    assert "meta" not in aliases and "hdfc" not in aliases
    assert aliases["google"] == aliases["alphabet"] == "GOOGL" and aliases["meta platforms"] == "META" and aliases["hdfc bank"] == "HDFCBANK.NS"


def test_parse_quotes_keeps_plain_data_and_skips_odd_entries():
    raw = [
        quote("suzlon.ns", "Suzlon Energy Limited", "NSI", "NSE"),
        {"symbol": "TSM", "shortname": "TSMC\x00 ADR  ", "exchange": "NYQ", "quoteType": "equity"},
        {"symbol": "<b>x</b>"}, {"symbol": 5}, "text", None, {"name": "no symbol"},
        {"symbol": "LONG", "longname": "x" * 500, "exchange": 7},
    ]
    found = s.parse_quotes(raw)
    assert [c.symbol for c in found] == ["SUZLON.NS", "TSM", "LONG"]
    assert found[0] == Candidate("SUZLON.NS", "Suzlon Energy Limited", "NSI", "NSE", "EQUITY")
    assert found[1].name == "TSMC ADR" and found[1].quote_type == "EQUITY"
    assert len(found[2].name) == s.TEXT_MAX_CHARS and found[2].exchange == ""


def test_parse_quotes_rejects_a_reply_that_is_not_a_list():
    with pytest.raises(PermanentProviderError):
        s.parse_quotes({"quotes": []})


def test_the_yahoo_call_passes_the_query_and_limits_and_returns_the_quotes(monkeypatch):
    seen = {}

    class FakeSearch:
        def __init__(self, query, **kwargs):
            seen.update(query=query, **kwargs)
            self.quotes = (quote("A", "A", "X"),)

    monkeypatch.setattr(s, "load_yfinance", lambda: type("YF", (), {"Search": FakeSearch}))
    assert s.yahoo_search("suzlon", 8, 5.0) == [quote("A", "A", "X")]
    assert seen == {"query": "suzlon", "max_results": 8, "news_count": 0, "lists_count": 0, "include_research": False, "timeout": 5.0, "raise_errors": True}


def test_run_with_deadline_returns_the_value_and_turns_failures_and_slowness_into_transient_errors():
    assert s.run_with_deadline(lambda: 7, 1) == 7
    with pytest.raises(TransientProviderError, match="ValueError"):
        s.run_with_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), 1)
    release = threading.Event()
    with pytest.raises(TransientProviderError, match="timed out"):
        s.run_with_deadline(lambda: release.wait(5), 0.05)
    release.set()


def test_the_yahoo_strategy_parses_what_the_search_returns_and_passes_the_limits():
    calls = []

    def search(query, max_results, timeout):
        calls.append((query, max_results, timeout))
        return [quote("SUZLON.NS", "Suzlon Energy Limited", "NSI")]

    found = s.YahooSearchStrategy(5, 8, search).find("suzlon")
    assert [c.symbol for c in found] == ["SUZLON.NS"] and calls == [("suzlon", 8, 5)]


def test_the_yahoo_strategy_reports_a_slow_or_failing_search_as_an_error():
    release = threading.Event()
    with pytest.raises(TransientProviderError):
        s.YahooSearchStrategy(0.05, 8, lambda *a: release.wait(5)).find("x")
    release.set()
    with pytest.raises(TransientProviderError):
        s.YahooSearchStrategy(1, 8, lambda *a: (_ for _ in ()).throw(ConnectionError())).find("x")
    with pytest.raises(PermanentProviderError):
        s.YahooSearchStrategy(1, 8, lambda *a: {"not": "a list"}).find("x")


def test_fake_lookup_returns_scripted_answers_records_calls_and_raises_scripted_errors():
    fake = FakeLookup({"a": [cand("A")], "down": TransientProviderError("x")})
    assert fake.find("A") == [cand("A")] and fake.find("zzz") == []
    with pytest.raises(TransientProviderError):
        fake.find("down")
    assert fake.calls == ["A", "zzz", "down"] and FakeLookup().table == {} and not fake.trusted


# ------------------------------------------------------------------ price check and cache

class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def get_history(self, ticker, period):
        self.calls.append((ticker, period))
        if self.error:
            raise self.error
        return self.result


def history(**kw):
    values = dict(data=[{"close": 1.0}], currency="INR", as_of=None, source="live")
    values.update(kw)
    return MarketResult(**values)


@pytest.mark.parametrize("result, error, expected", [
    (history(), None, Validation(True, "INR")),
    (history(source="cache"), None, Validation(True, "INR")),
    (history(source="stale cache", is_stale=True), None, Validation(False)),
    (history(source="sample data"), None, Validation(False)),
    (history(error="no data"), None, Validation(False)),
    (history(data=[]), None, Validation(False)),
    (history(data=None), None, Validation(False)),
    (None, RuntimeError("boom"), Validation(False)),
])
def test_the_price_check_needs_live_or_cached_prices(result, error, expected):
    provider = FakeProvider(result, error)
    assert r.provider_validator(provider, "5d")("SUZLON.NS") == expected
    assert provider.calls == [("SUZLON.NS", "5d")]


def test_the_cache_keeps_entries_for_the_time_limit_and_drops_the_oldest_when_full():
    now = [0.0]
    cache = r.ResolutionCache(10, 2, lambda: now[0])
    one, two, three = (Resolution("resolved", q) for q in "abc")
    assert cache.get("a") is None
    cache.put("a", one), cache.put("b", two)
    assert cache.get("a") is one
    cache.put("a", one)                                   # replacing does not evict
    assert cache.get("b") is two
    cache.put("c", three)                                 # full: the oldest ("b", since "a" was just put again) goes
    assert cache.get("b") is None and cache.get("a") is one and cache.get("c") is three
    now[0] = 10.0
    assert cache.get("a") is None


# ------------------------------------------------------------------ resolver

def sample(symbol, name, exchange, exchange_name="", quote_type="EQUITY"):
    return Candidate(symbol, name, exchange, exchange_name, quote_type)


SUZLON_NSE = sample("SUZLON.NS", "Suzlon Energy Limited", "NSI", "NSE")
SUZLON_BSE = sample("SUZLON.BO", "Suzlon Energy Limited", "BSE", "Bombay")
INFY_NYSE = sample("INFY", "Infosys Limited", "NYQ", "NYSE")
INFY_NSE = sample("INFY.NS", "Infosys Limited", "NSI", "NSE")
INFY_BSE = sample("INFY.BO", "Infosys Limited", "BSE", "Bombay")
HCL = sample("HCL-INSYS.BO", "HCL Infosystems Limited", "BSE", "Bombay")
GOOG = sample("GOOG", "Alphabet Inc.", "NMS", "NASDAQ")
GOOGL = sample("GOOGL", "Alphabet Inc.", "NMS", "NASDAQ")
TSM = sample("TSM", "Taiwan Semiconductor Manufacturing Company Limited", "NYQ", "NYSE")
TSMC = sample("2330.TW", "Taiwan Semiconductor Manufacturing Company Limited", "TAI", "Taiwan")


def test_an_alias_is_price_checked_and_resolved_without_a_search():
    res, lookup, check = make({})
    result = res.resolve("Tata Consultancy Services")
    assert (result.status, result.symbols, result.source, result.note) == ("resolved", ("TCS.NS",), "alias", "") and check.asked == ["TCS.NS"] and lookup.calls == []


@pytest.mark.parametrize("name", ["tsmc", "Taiwan Semiconductor", "taiwan semiconductors", "TSMC Ltd"])
def test_the_taiwan_semiconductor_names_are_aliases_for_the_nyse_adr_and_say_so_with_the_checked_currency(name):
    res, lookup, _ = make({}, currencies={"TSM": "USD"})
    result = res.resolve(name)
    assert result.symbols == ("TSM",) and result.source == "alias" and lookup.calls == []
    assert "NYSE-listed ADR" in result.note and "priced in USD" in result.note


def test_an_alias_note_without_a_reported_currency_does_not_invent_one():
    res, *_ = make({})
    assert "priced in its trading currency" in res.resolve("tsmc").note


def test_an_alias_that_fails_the_price_check_is_not_accepted_and_the_search_is_tried():
    res, lookup, check = make({"infosys": [INFY_NYSE]}, ok={"INFY"}, currencies={"INFY": "USD"})
    result = res.resolve("Infosys")
    assert result.status == "resolved" and result.symbols == ("INFY",) and result.source == "fake lookup" and lookup.calls == ["Infosys"]
    assert check.asked == ["INFY.NS", "INFY"]
    res, lookup, check = make({}, ok=set())
    assert res.resolve("Infosys").status == "not_found"


def test_the_best_ranked_listing_wins_is_the_only_one_checked_and_the_note_names_the_others_without_a_currency():
    res, _, check = make({"infosys": [INFY_NYSE, HCL, INFY_BSE, INFY_NSE]}, currencies={"INFY.NS": "INR"}, alias=False)
    result = res.resolve("Infosys")
    assert result.status == "resolved" and result.symbols == ("INFY.NS",) and result.candidates[0].currency == "INR"
    assert check.asked == ["INFY.NS"]                                              # lazy: nothing else is checked
    assert result.note == "Showing INFY.NS (Infosys Limited, NSE, INR). The same company is also listed as INFY.BO (Bombay); INFY (NYSE); ask for that symbol if you meant it."
    assert "USD" not in result.note and "HCL" not in result.note


def test_a_plain_exact_match_with_one_listing_has_no_note():
    res, *_ = make({"infosys": [INFY_NSE]}, alias=False)
    assert res.resolve("Infosys Ltd").note == ""


def test_the_next_listing_is_used_when_the_best_one_has_no_prices_and_at_most_max_validated_are_tried():
    res, _, check = make({"suzlon": [SUZLON_NSE, SUZLON_BSE]}, ok={"SUZLON.BO"}, alias=False)
    assert res.resolve("Suzlon").symbols == ("SUZLON.BO",) and check.asked == ["SUZLON.NS", "SUZLON.BO"]
    four = [sample(f"X{i}", "X Corp", "NYQ") for i in range(1)] + [sample("X.NS", "X Corp", "NSI"), sample("X.BO", "X Corp", "BSE"), sample("XN", "X Corp", "NMS")]
    res, _, check = make({"x": four}, ok=set(), alias=False)
    assert res.resolve("x").status == "not_found" and check.asked == ["X.NS", "X.BO", "X0"]


def test_a_name_that_is_the_start_of_a_company_name_resolves_and_the_note_shows_the_full_name():
    res, *_ = make({"suzlon": [SUZLON_BSE, SUZLON_NSE]}, currencies={"SUZLON.NS": "INR"}, alias=False)
    result = res.resolve("SUZLON")
    assert result.symbols == ("SUZLON.NS",)
    assert result.note.startswith("Showing SUZLON.NS (Suzlon Energy Limited, NSE, INR). The same company is also listed as SUZLON.BO (Bombay)")
    res, *_ = make({"tata consultancy": [sample("TCS.NS", "Tata Consultancy Services Limited", "NSI", "NSE")]}, alias=False, currencies={"TCS.NS": "INR"})
    assert res.resolve("tata consultancy").note == 'Showing TCS.NS (Tata Consultancy Services Limited, NSE, INR) for "tata consultancy".'


def test_an_exact_company_name_beats_longer_names_that_merely_start_with_it():
    res, *_ = make({"apple": [sample("APLE", "Apple Hospitality REIT, Inc.", "NYQ"), sample("AAPL", "Apple Inc.", "NMS")]}, alias=False)
    result = res.resolve("apple")
    assert result.symbols == ("AAPL",) and result.note == ""


def test_a_short_name_must_match_exactly_and_no_spelling_matches_are_offered():
    res, *_ = make({"abb": [sample("ABB.NS", "ABB India Limited", "NSI")]}, alias=False)
    assert res.resolve("abb").status == "not_found"
    res, *_ = make({"suzlon": [SUZLON_NSE]}, cfg={**CFG, "min_prefix_chars": 10}, alias=False)
    assert res.resolve("suzlon").status == "not_found"
    res, *_ = make({"taiwan semiconductors": [TSM], "pine": [sample("PINE", "Pineapple Co", "NYQ")]}, alias=False)
    assert res.resolve("taiwan semiconductors").status == "not_found" and res.resolve("pine").status == "not_found"


def test_several_companies_that_start_with_the_name_are_listed_with_their_best_listing_and_checked_currency():
    table = {"tata": [sample("TATASTEEL.BO", "Tata Steel Limited", "BSE", "Bombay"), sample("TATAMOTORS.NS", "Tata Motors Limited", "NSI", "NSE"),
                      sample("TATASTEEL.NS", "Tata Steel Limited", "NSI", "NSE"), sample("TTM", "Tata Motors Limited", "NYQ", "NYSE")]}
    res, _, check = make(table, currencies={"TATAMOTORS.NS": "INR", "TATASTEEL.NS": "INR"}, alias=False)
    result = res.resolve("tata")
    assert result.status == "ambiguous" and result.symbols == () and [c.symbol for c in result.candidates] == ["TATASTEEL.NS", "TATAMOTORS.NS"]     # equal rank: the search's own order
    assert "TATASTEEL.NS (Tata Steel Limited, NSE, INR); TATAMOTORS.NS (Tata Motors Limited, NSE, INR)" in result.message
    assert "Please type the ticker symbol you mean, for example TATASTEEL.NS." in result.message and check.asked == ["TATASTEEL.NS", "TATAMOTORS.NS"]


def test_two_listings_on_the_same_best_exchange_are_ambiguous_with_exchange_and_currency():
    res, *_ = make({"alphabet": [GOOG, GOOGL]}, currencies={"GOOG": "USD", "GOOGL": "USD"}, alias=False)
    result = res.resolve("Alphabet")
    assert result.status == "ambiguous" and [c.symbol for c in result.candidates] == ["GOOG", "GOOGL"]
    assert "GOOG (Alphabet Inc., NASDAQ, USD); GOOGL (Alphabet Inc., NASDAQ, USD)" in result.message


def test_an_option_that_fails_its_price_check_is_dropped_and_one_left_means_no_choice():
    res, *_ = make({"alphabet": [GOOG, GOOGL]}, ok={"GOOG"}, alias=False)
    result = res.resolve("Alphabet")
    assert result.status == "resolved" and result.symbols == ("GOOG",)
    res, *_ = make({"alphabet": [GOOG, GOOGL]}, ok=set(), alias=False)
    assert res.resolve("Alphabet").status == "not_found"


def test_options_beyond_max_validated_are_listed_unchecked_without_a_currency():
    many = [sample(f"A{i}", "Alpha Co", "NYQ", "NYSE") for i in range(5)]
    res, _, check = make({"alpha": many}, currencies={f"A{i}": "USD" for i in range(5)}, alias=False)
    result = res.resolve("alpha")
    assert check.asked == ["A0", "A1", "A2"] and len(result.candidates) == 5
    assert [c.currency for c in result.candidates] == ["USD", "USD", "USD", None, None]
    assert "A3 (Alpha Co, NYSE); A4 (Alpha Co, NYSE)" in result.message


def test_several_listings_with_none_on_a_preferred_market_are_ambiguous_but_a_single_listing_is_accepted():
    res, *_ = make({"z": [sample("Z.TW", "Z Corp", "TAI", "Taiwan"), sample("Z.L", "Z Corp", "LSE", "London")]}, alias=False)
    assert res.resolve("z").status == "ambiguous"
    res, *_ = make({"z": [sample("Z.L", "Z Corp", "LSE", "London")]}, alias=False)
    assert res.resolve("z").symbols == ("Z.L",)


def test_only_equities_and_indexes_are_considered_and_etfs_only_when_the_name_says_etf_fund_or_trust():
    table = {"gold": [sample("GLD", "Gold", "MF", quote_type="MUTUALFUND"), sample("FUTX", "Gold", "CMX", quote_type="FUTURE"), sample("NOTYPE", "Gold", "X", quote_type="")],
             "gold etf": [sample("GLDE", "Gold ETF", "NYQ", quote_type="ETF"), sample("GLDM", "Gold ETF", "NYQ", quote_type="MUTUALFUND")],
             "nifty 50": [sample("^NSEI", "NIFTY 50", "NSI", quote_type="INDEX"), sample("NFTY", "NIFTY 50", "NYQ", quote_type="ETF")],
             "global x-trust": [sample("GXT", "Global X-Trust", "NYQ", quote_type="ETF")]}
    res, *_ = make(table, alias=False)
    assert res.resolve("gold").status == "not_found"
    assert res.resolve("Gold ETF").symbols == ("GLDE",)
    assert res.resolve("Nifty 50").symbols == ("^NSEI",)
    assert res.resolve("Global X-Trust").symbols == ("GXT",)


def test_a_name_with_no_candidates_or_no_valid_prices_is_not_found_with_the_name_in_the_message():
    res, *_ = make({})
    result = res.resolve("Qzxv Wplk")
    assert result.status == "not_found" and "couldn't find a listed company called \"Qzxv Wplk\"" in result.message and "AAPL" in result.message
    res, *_ = make({"suzlon": [SUZLON_NSE]}, ok=set(), alias=False)
    assert res.resolve("suzlon").status == "not_found"


def test_a_lookup_that_is_down_or_slow_is_unavailable_never_a_guess():
    for error in (TransientProviderError("timed out"), PermanentProviderError("odd reply")):
        res, *_ = make({"suzlon": error})
        result = res.resolve("Suzlon")
        assert result.status == "unavailable" and result.symbols == () and "can't look up company names" in result.message and "RELIANCE.NS" in result.message


def test_unavailable_wins_over_not_found_but_a_later_strategy_can_still_resolve():
    down, up = FakeLookup({"x": TransientProviderError("t")}), FakeLookup({"x": [sample("X.NS", "X Ltd", "NSI")]})
    assert r.TickerResolver([down, FakeLookup()], validator()).resolve("x").status == "unavailable"
    assert r.TickerResolver([down, up], validator()).resolve("x").symbols == ("X.NS",)


def test_price_checks_that_run_out_of_time_make_the_answer_unavailable_not_a_guess():
    now = [0.0]

    def slow(symbol):
        now[0] += 6
        return Validation(False)

    cfg = {**CFG, "validation_budget_seconds": 10}
    infosys = r.TickerResolver([FakeLookup({"infosys": [INFY_NSE, INFY_BSE, INFY_NYSE]})], slow, cfg, clock=lambda: now[0])
    assert infosys.resolve("Infosys").status == "unavailable"
    now[0] = 0.0
    tie = r.TickerResolver([FakeLookup({"alphabet": [GOOG, GOOGL, sample("GOOGX", "Alphabet Inc.", "NMS")]})], slow, cfg, clock=lambda: now[0])
    assert tie.resolve("Alphabet").status == "unavailable"
    now[0] = 0.0
    alias = r.TickerResolver([s.AliasStrategy({"x": "X"}, SUFFIXES)], slow, {**cfg, "validation_budget_seconds": -1}, clock=lambda: now[0])
    assert alias.resolve("x").status == "unavailable"


@pytest.mark.parametrize("name", [None, 12, "", "<>", "x" * 200])
def test_a_name_that_cannot_be_cleaned_is_invalid_and_no_lookup_runs(name):
    res, lookup, check = make({})
    result = res.resolve(name)
    assert result.status == "invalid" and "couldn't read that company name" in result.message and lookup.calls == [] and check.asked == []


def test_the_exchange_code_is_dropped_before_matching_and_never_becomes_a_ticker_and_messages_keep_the_original_text():
    res, lookup, check = make({"suzlon": [SUZLON_NSE]}, alias=False)
    result = res.resolve("Suzlon NSE: SUZLON")
    assert result.symbols == ("SUZLON.NS",) and lookup.calls == ["Suzlon"] and check.asked == ["SUZLON.NS"]
    assert result.query == "Suzlon NSE SUZLON" and result.note.endswith('for "Suzlon NSE SUZLON".')
    res, lookup, _ = make({})
    missing = res.resolve("Zomato NSE: ZOMATO")
    assert missing.status == "not_found" and lookup.calls == ["Zomato"] and 'called "Zomato NSE ZOMATO"' in missing.message
    res, *_ = make({"x": [sample("X1", "X Corp", "NYQ"), sample("X2", "X Corp", "NYQ")]}, alias=False)
    assert 'more than one match for "X NYSE X1"' in res.resolve("X NYSE: X1").message


def test_an_alias_found_through_a_name_with_an_exchange_code_is_still_price_checked():
    res, lookup, check = make({})
    for name, symbol in (("Infosys NSE: INFY", "INFY.NS"), ("Apple NASDAQ: AAPL", "AAPL"), ("Tata Consultancy Services NSE: TCS", "TCS.NS")):
        assert res.resolve(name).symbols == (symbol,)
    assert check.asked == ["INFY.NS", "AAPL", "TCS.NS"] and lookup.calls == []


def test_the_query_sent_to_the_lookup_is_the_cleaned_one():
    res, lookup, _ = make({})
    res.resolve("  Taiwan <b>Semiconductor</b>: 2330 ")
    assert lookup.calls == ["Taiwan b Semiconductor b 2330"]


def test_successful_lookups_are_cached_by_name_and_failures_are_not():
    res, lookup, check = make({"suzlon": [SUZLON_NSE]}, alias=False)
    first = res.resolve("Suzlon")
    assert res.resolve("  SUZLON ") is first and lookup.calls == ["Suzlon"] and check.asked == ["SUZLON.NS"]
    asking, lookup2, _ = make({"alphabet": [GOOG, GOOGL]}, alias=False)
    assert asking.resolve("alphabet") is asking.resolve("ALPHABET") and lookup2.calls == ["alphabet"]
    lookup3 = FakeLookup({"x": TransientProviderError("t")})
    res3 = r.TickerResolver([lookup3], validator())
    res3.resolve("x"), res3.resolve("x")
    assert lookup3.calls == ["x", "x"]
    lookup3.table = {}
    assert res3.resolve("y").status == "not_found" and res3.resolve("y").status == "not_found" and lookup3.calls.count("y") == 2


def test_the_log_has_the_length_and_status_but_not_the_name(caplog):
    res, *_ = make({"suzlon": [SUZLON_NSE]}, alias=False)
    with caplog.at_level(logging.INFO):
        res.resolve("Suzlon")
        res.resolve("Suzlon")
        res.resolve("<>")
    text = caplog.text
    assert "6 chars, status=resolved, source=fake lookup" in text and "(cached)" in text and "status=invalid" in text
    assert "Suzlon" not in text and "SUZLON.NS" not in text


# ------------------------------------------------------------------ the factory and the queries that failed in the UI

def quote_of(candidate):
    return quote(candidate.symbol, candidate.name, candidate.exchange, candidate.exchange_name)


SEARCH_TABLE = {
    "taiwan semiconductor manufacturing": [quote_of(TSMC), quote_of(TSM)],
    "taiwan semiconductor manufacturing co ltd": [quote_of(TSMC), quote_of(TSM)],
    "suzlon": [quote_of(SUZLON_BSE), quote_of(SUZLON_NSE)],
    "infosys": [quote_of(INFY_NYSE), quote_of(INFY_NSE), quote_of(INFY_BSE), quote_of(HCL)],
}


def fake_search(query, max_results, timeout):
    return SEARCH_TABLE.get(query.casefold(), [])


def test_the_queries_that_failed_in_the_ui_with_a_fake_search():
    provider = FakeProvider(history())
    resolver = r.build_resolver(provider, search=fake_search)
    suzlon = resolver.resolve("SUZLON")
    assert suzlon.status == "resolved" and suzlon.symbols == ("SUZLON.NS",)
    for name in ("taiwan semiconductors", "TSMC"):
        assert resolver.resolve(name).symbols == ("TSM",)                       # the alias, with its note
    full = resolver.resolve("Taiwan Semiconductor Manufacturing")
    assert full.symbols == ("TSM",) and "also listed as 2330.TW (Taiwan)" in full.note
    assert resolver.resolve("Infosys").symbols == ("INFY.NS",) and resolver.resolve("nonsense qzx").status == "not_found"
    legal = resolver.resolve("Taiwan Semiconductor Manufacturing Co Ltd TPE: 2330")        # the exchange code is dropped, not read as a ticker
    assert legal.status == "resolved" and legal.symbols == ("TSM",) and "also listed as 2330.TW (Taiwan)" in legal.note
    assert all(period == CFG["validation_period"] for _, period in provider.calls)


def test_build_resolver_tries_the_alias_table_first_then_the_search():
    calls = []
    resolver = r.build_resolver(FakeProvider(history()), search=lambda *a: calls.append(a) or [])
    resolver.resolve("Infosys")
    assert calls == []
    resolver.resolve("Some Unlisted Name")
    assert calls == [("Some Unlisted Name", CFG["max_results"], CFG["timeout_seconds"])]


# ------------------------------------------------------------------ config

def _config(tmp_path, change):
    cfg = {"ticker_lookup": copy.deepcopy(get_config()["ticker_lookup"]), "market_data": {"history_periods": ["1d", "5d"]}}
    get_config.cache_clear()
    change(cfg["ticker_lookup"])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_the_real_block_is_valid_and_has_every_message():
    assert CFG["validation_period"] in get_config()["market_data"]["history_periods"] and set(CFG["messages"]) >= set(TICKER_LOOKUP_MESSAGES)
    assert CFG["preferred_markets"] == ["NSI", "BSE", "NYQ", "NMS", "NGM"] and CFG["min_prefix_chars"] == 4 and CFG["max_validated"] == 3
    assert CFG["allowed_quote_types"] == ["EQUITY", "INDEX"] and "same_market_groups" not in CFG


def test_a_valid_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda b: None))["ticker_lookup"]["timeout_seconds"] > 0


@pytest.mark.parametrize("change, text", [
    (lambda b: b.update(timeout_seconds=0), "timeout_seconds"),
    (lambda b: b.update(validation_budget_seconds=0), "validation_budget_seconds"),
    (lambda b: b.update(min_prefix_chars=0), "min_prefix_chars"),
    (lambda b: b.update(max_validated=-1), "max_validated"),
    (lambda b: b.update(cache_ttl_minutes=True), "cache_ttl_minutes"),
    (lambda b: b.update(max_results="8"), "max_results"),
    (lambda b: b.update(validation_period="10y"), "validation_period"),
    (lambda b: b.update(allowed_quote_types=[]), "allowed_quote_types"),
    (lambda b: b.update(etf_words="etf"), "etf_words"),
    (lambda b: b.update(exchange_words=[]), "exchange_words"),
    (lambda b: b.update(legal_suffixes=[""]), "legal_suffixes"),
    (lambda b: b.update(preferred_markets=[]), "preferred_markets"),
    (lambda b: b["alias_notes"].update(TSM=" "), "alias_notes.TSM"),
    (lambda b: b["messages"].update(not_found=" "), "messages.not_found"),
    (lambda b: b["messages"].pop("disclosure_with_others"), "messages.disclosure_with_others"),
])
def test_a_bad_block_is_rejected_with_the_key_named(tmp_path, change, text):
    with pytest.raises(ConfigError, match=text):
        get_config(_config(tmp_path, change))

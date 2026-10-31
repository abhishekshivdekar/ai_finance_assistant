"""TickerResolver: a company name -> one price-checked ticker, or a question for the user. It never guesses.

Design patterns:
- STRATEGY: candidates come from injected TickerStrategy objects (alias table, Yahoo search, a fake in tests).
- CHAIN of strategies: the first one that gives a decision wins.
Example: "Infosys" -> Yahoo lists INFY (NYSE), INFY.NS (NSE), INFY.BO, HCL-INSYS.BO ...; only names equal to "infosys"
count (HCL Infosystems does not); the best exchange in `preferred_markets` is NSI, so INFY.NS is price-checked and
accepted, with a note: "Showing INFY.NS (Infosys Limited, NSE, INR). The same company is also listed as INFY (NYSE) ...".

Rules (all numbers and lists are in config `ticker_lookup`):
1. The language model is never a source. A symbol comes from the alias table or from Yahoo's search.
2. Filter: only EQUITY and INDEX quotes (ETF only when the name typed has "etf", "fund" or "trust" in it).
3. Name match: the words typed equal the company name (legal suffixes such as Ltd ignored). Only when no company matches
   exactly, a name that STARTS with the words typed (whole words, at least `min_prefix_chars` characters) counts. No
   spelling or fuzzy matches. No match is `not_found`.
4. Selection among the listings of one company: the best exchange in `preferred_markets` wins. Two listings on the same
   best exchange (GOOG and GOOGL), or several listings with none on a preferred market, are `ambiguous`; several
   companies matching by prefix are `ambiguous` too (up to `max_candidates_shown`, each with its best listing). The user
   then types the symbol, which takes the ordinary literal path. A company with one listing is accepted.
5. Price check, lazily: only the best-ranked candidates (`max_validated`), in rank order, inside
   `validation_budget_seconds`; stale or sample data does not count. The first that passes is accepted. A currency is
   named only for a listing that was checked.
6. A lookup that is down, slow or out of time is `unavailable`, never a guess. Only resolved and ambiguous answers
   are cached. Log lines carry the length of the name and the status, never the text.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Callable

from src.core.portfolio_valuation import NOT_LIVE_SOURCES
from src.data.errors import PermanentProviderError, TransientProviderError
from src.data.market.provider import MarketDataProvider
from src.data.tickers.models import Candidate, Resolution, Status, Validation
from src.data.tickers.names import InvalidQueryError, clean_query, is_name_start, tokens
from src.data.tickers.strategies import AliasStrategy, Search, TickerStrategy, YahooSearchStrategy, yahoo_search
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

Validator = Callable[[str], Validation]


def provider_validator(provider: MarketDataProvider, period: str) -> Validator:
    """A price check made with the market provider: recent history that is live or cached, and the currency it reports."""

    def check(symbol: str) -> Validation:
        try:
            result = provider.get_history(symbol, period)
            ok = result.error is None and result.source not in NOT_LIVE_SOURCES and result.data is not None and len(result.data) > 0
        except Exception:                                    # any failure means "no usable prices"
            return Validation(False)
        return Validation(True, result.currency) if ok else Validation(False)

    return check


class ResolutionCache:
    """Thread-safe cache of successful resolutions with a time limit and a size limit (oldest entry goes first)."""

    def __init__(self, ttl_seconds: float, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl, self._max, self._clock = ttl_seconds, max_entries, clock
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, Resolution]] = {}

    def get(self, key: str) -> Resolution | None:
        """The entry if it is younger than the time limit."""
        with self._lock:
            entry = self._store.get(key)
        return entry[1] if entry is not None and self._clock() - entry[0] < self._ttl else None

    def put(self, key: str, resolution: Resolution) -> None:
        """Store an entry, dropping the oldest when full."""
        with self._lock:
            self._store.pop(key, None)
            while len(self._store) >= self._max:
                self._store.pop(next(iter(self._store)))
            self._store[key] = (self._clock(), resolution)


class TickerResolver:
    """Turns a company name into a Resolution using injected strategies and an injected price check."""

    def __init__(self, strategies: list[TickerStrategy], validator: Validator, cfg: dict[str, Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg if cfg is not None else get_config()["ticker_lookup"]
        self._strategies, self._validate, self._clock = strategies, validator, clock
        self._cache = ResolutionCache(self._cfg["cache_ttl_minutes"] * 60, self._cfg["cache_max_entries"], clock)

    def resolve(self, name: Any) -> Resolution:
        """The decision for one name; never raises."""
        try:
            query = clean_query(name, self._cfg["max_query_chars"])
        except InvalidQueryError:
            return self._done(Resolution("invalid", message=self._say("invalid")), 0)
        key = " ".join(query.casefold().split())
        cached = self._cache.get(key)
        if cached is not None:
            return self._done(cached, len(query), cached=True)
        unavailable = False
        for strategy in self._strategies:
            try:
                decision = self._decide(query, strategy.find(query), strategy)
            except (TransientProviderError, PermanentProviderError):
                unavailable = True
                continue
            if decision is not None:
                self._cache.put(key, decision)
                return self._done(decision, len(query))
        status: Status = "unavailable" if unavailable else "not_found"
        return self._done(Resolution(status, query, message=self._say(status, name=query)), len(query))

    def _say(self, key: str, **values: str) -> str:
        """A message from config with its placeholders filled."""
        return " ".join(self._cfg["messages"][key].split()).format(**values)

    @staticmethod
    def _done(resolution: Resolution, length: int, cached: bool = False) -> Resolution:
        """Log the length and the status (never the text) and return the resolution."""
        log.info("Ticker lookup: %d chars, status=%s, source=%s%s", length, resolution.status, resolution.source or "-", " (cached)" if cached else "")
        return resolution

    def _rank(self, candidate: Candidate) -> int:
        """Position of the candidate's exchange in preferred_markets; unlisted exchanges rank after all of them."""
        markets = self._cfg["preferred_markets"]
        return markets.index(candidate.exchange) if candidate.exchange in markets else len(markets)

    def _filtered(self, query: str, found: list[Candidate]) -> list[Candidate]:
        """Candidates of an allowed quote type (ETFs only when the name typed says etf, fund or trust), one per symbol."""
        allowed = set(self._cfg["allowed_quote_types"])
        words = {word.casefold() for word in query.replace("-", " ").split()}
        if words & {word.casefold() for word in self._cfg["etf_words"]}:
            allowed |= set(self._cfg["etf_quote_types"])
        unique: dict[str, Candidate] = {}
        for candidate in found:
            if candidate.quote_type in allowed:
                unique.setdefault(candidate.symbol, candidate)
        return list(unique.values())

    def _matching(self, query: str, found: list[Candidate]) -> list[list[Candidate]]:
        """Companies whose name equals the words typed, else (long enough query) starts with them.

        Each company is its listings, best-ranked first; the companies are ordered by their best listing.
        """
        suffixes = self._cfg["legal_suffixes"]
        typed = tokens(query, suffixes)
        named = [(tokens(candidate.name, suffixes), candidate) for candidate in found]
        matched = [(name, c) for name, c in named if name == typed]
        if not matched and len(" ".join(typed)) >= self._cfg["min_prefix_chars"]:
            matched = [(name, c) for name, c in named if is_name_start(typed, name)]
        companies: dict[tuple[str, ...], list[Candidate]] = {}
        for name, candidate in matched:
            companies.setdefault(name, []).append(candidate)
        listings = [sorted(members, key=self._rank) for members in companies.values()]
        return sorted(listings, key=lambda members: self._rank(members[0]))

    def _check(self, candidates: list[Candidate], first_only: bool) -> list[tuple[Candidate, Validation]]:
        """Price-check the first max_validated candidates in order; stop at the first pass when `first_only`.

        Raises TransientProviderError when the time budget is used up before the next check.
        """
        started, checked = self._clock(), []
        for candidate in candidates[: self._cfg["max_validated"]]:
            if self._clock() - started > self._cfg["validation_budget_seconds"]:
                raise TransientProviderError("The price checks ran out of time.")
            result = self._validate(candidate.symbol)
            checked.append((candidate, result))
            if first_only and result.ok:
                break
        return checked

    def _listing_text(self, candidate: Candidate) -> str:
        """Company name, exchange and, only for a checked listing, currency, as text."""
        parts = [candidate.name, candidate.exchange_name or candidate.exchange, candidate.currency or ""]
        return ", ".join(part for part in parts if part)

    def _accepted(self, query: str, chosen: Candidate, others: list[Candidate], source: str, whole_name: bool) -> Resolution:
        """A resolved answer. The note names the company chosen and other listings; it is empty for a plain exact match."""
        shown = [self._say("other_listing", symbol=o.symbol, details=o.exchange_name or o.exchange) for o in others[: self._cfg["max_others_shown"]]]
        note = ""
        if shown:
            note = self._say("disclosure_with_others", symbol=chosen.symbol, details=self._listing_text(chosen), others="; ".join(shown))
        elif not whole_name:
            note = self._say("disclosure_name_only", symbol=chosen.symbol, details=self._listing_text(chosen), name=query)
        return Resolution("resolved", query, (chosen.symbol,), (chosen,), source, note=note)

    def _ask(self, query: str, options: list[Candidate], source: str) -> Resolution | None:
        """The question for the user: the options with a currency where one could be checked; failed options are dropped.

        One option left means there was no real choice (it is accepted); none left means nothing usable.
        """
        kept = [replace(c, currency=v.currency) for c, v in self._check(options, first_only=False) if v.ok]
        kept += options[self._cfg["max_validated"]:]
        if not kept:
            return None
        if len(kept) == 1:
            return self._accepted(query, kept[0], [], source, whole_name=False)
        shown = kept[: self._cfg["max_candidates_shown"]]
        options_text = "; ".join(self._say("option", symbol=c.symbol, details=self._listing_text(c)) for c in shown)
        return Resolution("ambiguous", query, (), tuple(shown), source, self._say("ambiguous", name=query, options=options_text, example=shown[0].symbol))

    def _decide(self, query: str, found: list[Candidate], strategy: TickerStrategy) -> Resolution | None:
        """resolved or ambiguous, or None when this strategy has nothing usable."""
        if strategy.trusted:
            passed = [(c, v) for c, v in self._check(found[:1], first_only=True) if v.ok]
            if not passed:
                return None
            chosen = replace(passed[0][0], currency=passed[0][1].currency)
            note = self._cfg["alias_notes"].get(chosen.symbol, "")
            return Resolution("resolved", query, (chosen.symbol,), (chosen,), strategy.name, note=note.format(currency=chosen.currency or "its trading currency"))
        companies = self._matching(query, self._filtered(query, found))
        if not companies:
            return None
        if len(companies) > 1:
            return self._ask(query, [listings[0] for listings in companies], strategy.name)
        listings = companies[0]
        best = [c for c in listings if self._rank(c) == self._rank(listings[0])]
        if len(listings) > 1 and (len(best) > 1 or self._rank(listings[0]) >= len(self._cfg["preferred_markets"])):
            return self._ask(query, listings, strategy.name)
        passed = [(c, v) for c, v in self._check(listings, first_only=True) if v.ok]
        if not passed:
            return None
        chosen = replace(passed[0][0], currency=passed[0][1].currency)
        suffixes = self._cfg["legal_suffixes"]
        whole_name = tokens(chosen.name, suffixes) == tokens(query, suffixes)
        return self._accepted(query, chosen, [c for c in listings if c.symbol != chosen.symbol], strategy.name, whole_name)


def build_resolver(provider: MarketDataProvider, search: Search = yahoo_search, clock: Callable[[], float] = time.monotonic) -> TickerResolver:
    """The resolver as configured: the alias table first, then Yahoo's search; prices are checked with `provider`.

    Pass the shared CachedProvider (get_market_provider()), so a price check is reused by the agent that follows.
    """
    cfg = get_config()["ticker_lookup"]
    strategies: list[TickerStrategy] = [
        AliasStrategy(get_config()["workflow"]["ticker_aliases"], cfg["legal_suffixes"]),
        YahooSearchStrategy(cfg["timeout_seconds"], cfg["max_results"], search),
    ]
    return TickerResolver(strategies, provider_validator(provider, cfg["validation_period"]), cfg, clock)

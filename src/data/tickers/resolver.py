"""TickerResolver: a company name -> one price-checked ticker, or a question for the user. It never guesses.

Design patterns:
- STRATEGY: candidates come from injected TickerStrategy objects (alias table, Yahoo search, a fake in tests).
- CHAIN of strategies: the first one whose price-checked candidates give a decision wins.
Example: "Suzlon" -> the Yahoo strategy finds SUZLON.NS and SUZLON.BO (the same company, one market); both pass the price
check; the config order [NSI, BSE] picks SUZLON.NS. "Taiwan Semiconductor Manufacturing" finds TSM (NYSE) and 2330.TW
(Taiwan): the same company in two markets, so the answer is `ambiguous` and the user types the symbol they mean.

Rules (all numbers and lists are in config `ticker_lookup`):
1. The language model is never a source. A symbol comes from the alias table or from Yahoo's search.
2. Every accepted symbol passes a price check first (recent history from the market provider, live or cached; stale or
   sample data does not count). Candidates that fail are dropped.
3. A name is accepted only when exactly one company remains, it has one listing (listings of one company in the same
   market, such as NSE and BSE, collapse to the most preferred exchange) and the words typed are the first words of its
   name. Anything else is `ambiguous`: the user types the symbol, which then takes the ordinary literal path.
4. A lookup that is down or slow is `unavailable`, never a guess. Only resolved and ambiguous answers are cached.
Log lines carry the length of the name and the status, never the text.
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
        self._strategies, self._validate = strategies, validator
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
                found = strategy.find(query)
            except (TransientProviderError, PermanentProviderError):
                unavailable = True
                continue
            decision = self._decide(query, found, strategy)
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

    def _checked(self, candidates: list[Candidate]) -> list[Candidate]:
        """The first max_validated distinct symbols that pass the price check, with their currency."""
        seen: list[str] = []
        valid: list[Candidate] = []
        for candidate in candidates:
            if candidate.symbol in seen or len(seen) >= self._cfg["max_validated"]:
                continue
            seen.append(candidate.symbol)
            check = self._validate(candidate.symbol)
            if check.ok:
                valid.append(replace(candidate, currency=check.currency))
        return valid

    def _one_listing(self, members: list[Candidate]) -> list[Candidate]:
        """Listings of one company: if all are in one market group (NSE and BSE) keep the most preferred, else keep all."""
        for group in self._cfg["same_market_groups"]:
            if all(member.exchange in group for member in members):
                return [min(members, key=lambda member: group.index(member.exchange))]
        return members

    def _decide(self, query: str, found: list[Candidate], strategy: TickerStrategy) -> Resolution | None:
        """resolved / ambiguous from the price-checked candidates of one strategy, or None when nothing usable is left."""
        suffixes = self._cfg["legal_suffixes"]
        if not strategy.trusted:
            found = [c for c in found if c.quote_type in self._cfg["allowed_quote_types"]]
        valid = self._checked(found)
        if not valid:
            return None
        if strategy.trusted:
            return Resolution("resolved", query, (valid[0].symbol,), (valid[0],), strategy.name)
        companies: dict[tuple[str, ...], list[Candidate]] = {}
        for candidate in valid:
            companies.setdefault(tokens(candidate.name, suffixes), []).append(candidate)
        options = [option for members in companies.values() for option in self._one_listing(members)]
        if len(options) == 1 and is_name_start(tokens(query, suffixes), next(iter(companies))):
            return Resolution("resolved", query, (options[0].symbol,), (options[0],), strategy.name)
        shown = tuple(options[: self._cfg["max_candidates_shown"]])
        text = self._say("ambiguous", name=query, options="; ".join(self._option(c) for c in shown), example=shown[0].symbol)
        return Resolution("ambiguous", query, (), shown, strategy.name, text)

    def _option(self, candidate: Candidate) -> str:
        """One candidate as text: symbol, then name, exchange and currency when known."""
        details = [candidate.name, candidate.exchange_name or candidate.exchange, candidate.currency or ""]
        return self._say("option", symbol=candidate.symbol, details=", ".join(part for part in details if part))


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

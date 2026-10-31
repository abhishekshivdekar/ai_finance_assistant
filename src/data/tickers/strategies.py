"""Where candidate listings come from.

Design pattern: STRATEGY. `TickerStrategy` is the interface; `AliasStrategy` (a table in config) and
`YahooSearchStrategy` (Yahoo's search) are two implementations. Example: TickerResolver tries them in order and never
knows which one answered, so a test injects a fake and a later commit could add another lookup without touching it.

DEPENDENCY INJECTION: the Yahoo call is a constructor argument (`search`), so tests pass a fake and use no network.
The language model is not one of the strategies and never supplies a symbol.
"""
from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

from src.data.errors import PermanentProviderError, TransientProviderError
from src.data.market.yfinance_provider import load_yfinance
from src.data.tickers.models import Candidate
from src.data.tickers.names import tokens

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SYMBOL = re.compile(r"^[A-Za-z0-9^.=&-]{1,20}$")
TEXT_MAX_CHARS = 120                          # a name or exchange text longer than this is cut (it is shown, never interpreted)

Search = Callable[[str, int, float], Any]     # (query, max_results, timeout_seconds) -> the lookup's list of quotes


class TickerStrategy(ABC):
    """Strategy interface: candidate listings for a cleaned company name."""

    name: str = ""
    trusted: bool = False          # True: the candidates come from our own table, so no type or name check applies

    @abstractmethod
    def find(self, query: str) -> list[Candidate]:
        """Candidates for the name. May raise TransientProviderError / PermanentProviderError."""


class AliasStrategy(TickerStrategy):
    """The config table `workflow.ticker_aliases`: a whole name (legal suffixes ignored) -> a symbol.

    Whole-name matching: "Apple Inc" finds "apple", but "Apple Hospitality REIT" and "Metaverse" find nothing.
    """

    name = "alias"
    trusted = True

    def __init__(self, aliases: dict[str, str], legal_suffixes: list[str]) -> None:
        self._suffixes = legal_suffixes
        self._table = {tokens(str(key), legal_suffixes): str(symbol) for key, symbol in aliases.items()}

    def find(self, query: str) -> list[Candidate]:
        """The alias's symbol as a candidate, or []."""
        symbol = self._table.get(tokens(query, self._suffixes))
        return [Candidate(symbol=symbol, name=query)] if symbol else []


def _plain(value: Any) -> str:
    """Untrusted text as plain data: control characters removed, spaces collapsed, length capped."""
    return " ".join(_CONTROL.sub(" ", value).split())[:TEXT_MAX_CHARS] if isinstance(value, str) else ""


def parse_quotes(raw: Any) -> list[Candidate]:
    """Candidates from the lookup's quote dicts; entries that are not dicts or have an odd symbol are skipped."""
    if not isinstance(raw, list):
        raise PermanentProviderError("The lookup returned an unexpected shape.")
    found: list[Candidate] = []
    for quote in raw:
        symbol = quote.get("symbol") if isinstance(quote, dict) else None
        if not isinstance(symbol, str) or not _SYMBOL.match(symbol.strip()):
            continue
        found.append(Candidate(
            symbol=symbol.strip().upper(),
            name=_plain(quote.get("longname") or quote.get("shortname")),
            exchange=_plain(quote.get("exchange")),
            exchange_name=_plain(quote.get("exchDisp")),
            quote_type=_plain(quote.get("quoteType")).upper(),
        ))
    return found


def yahoo_search(query: str, max_results: int, timeout: float) -> Any:
    """The ONLY place that calls Yahoo's search (through yfinance). Returns its list of quote dicts."""
    yf = load_yfinance()
    return list(yf.Search(
        query, max_results=max_results, news_count=0, lists_count=0, include_research=False, timeout=timeout, raise_errors=True,
    ).quotes)


def run_with_deadline(call: Callable[[], Any], seconds: float) -> Any:
    """Run `call` in a daemon thread and give up after `seconds` (TransientProviderError); its own errors become Transient too."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = call()
        except Exception as exc:                  # any failure of the lookup means "unavailable"
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TransientProviderError("The lookup timed out.")
    if "error" in box:
        raise TransientProviderError(type(box["error"]).__name__) from box["error"]
    return box["value"]


class YahooSearchStrategy(TickerStrategy):
    """Yahoo's search by company name (through yfinance), with a hard time limit."""

    name = "yahoo search"

    def __init__(self, timeout_seconds: float, max_results: int, search: Search = yahoo_search) -> None:
        self._timeout, self._max_results, self._search = timeout_seconds, max_results, search

    def find(self, query: str) -> list[Candidate]:
        """The raw candidates for the name (no filtering, no price check)."""
        raw = run_with_deadline(lambda: self._search(query, self._max_results, self._timeout), self._timeout)
        return parse_quotes(raw)

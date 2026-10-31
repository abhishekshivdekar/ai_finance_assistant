"""The ticker box of a request, sorted into symbols and names, and the names resolved.

Design pattern: FACADE over the resolver for agents that take tickers (Market today, Portfolio later). Example:
["AAPL", "Suzlon", "zzzz qq"] -> symbols ["AAPL", "SUZLON.NS"], a note saying which company "Suzlon" became, and one
message for "zzzz qq" ("I couldn't find a listed company called ..."). An entry is a symbol only if `validate_symbol`
accepts it; anything else is a name and goes to the resolver, so no symbol enters an analysis without a price check.
`names` remembers the text typed for every entry that looked like a symbol, because a symbol-looking word can still be a
name ("apple" reads as the symbol APPLE, which has no prices): the agent looks such a name up when the prices are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.data.market.provider import InvalidSymbolError, validate_symbol
from src.data.tickers.resolver import TickerResolver


@dataclass
class Entries:
    """The result of sorting and resolving the entries of one request."""

    tickers: list[Any] = field(default_factory=list)       # symbols (and untouched non-text entries), ready for parse_request
    names: dict[str, str] = field(default_factory=dict)    # symbol -> the text typed, for entries that looked like symbols
    notes: list[str] = field(default_factory=list)         # disclosures of resolved names
    asks: list[str] = field(default_factory=list)          # messages for names that need the user (ambiguous, not found, ...)
    warnings: list[str] = field(default_factory=list)


def split_entries(raw: Any) -> list[Any]:
    """The entries of a ticker box: a list as it is; text split on commas, or on spaces when every word is a symbol.

    Example: "AAPL MSFT" -> ["AAPL", "MSFT"]; "Taiwan Semiconductor, Suzlon" -> two names; "tata consultancy" -> one name.
    """
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if not isinstance(raw, str):
        return []
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    words = raw.split()
    try:
        for word in words:
            validate_symbol(word)
    except InvalidSymbolError:
        return [" ".join(words)] if words else []
    return words


def resolve_entries(raw: Any, resolver: TickerResolver, max_tickers: int) -> Entries:
    """Symbols stay, names go to the resolver; only the first `max_tickers` entries are looked at."""
    entries, found = Entries(), split_entries(raw)
    for extra in found[max_tickers:]:
        entries.warnings.append(f"Only the first {max_tickers} tickers are analysed; {str(extra).strip()!r} was left out.")
    for item in found[:max_tickers]:
        if not isinstance(item, str):
            entries.tickers.append(item)                     # parse_request reports it as not a valid ticker
            continue
        try:
            symbol = validate_symbol(item)
        except InvalidSymbolError:
            resolution = resolver.resolve(item)
            if resolution.status == "resolved":
                entries.tickers.append(resolution.symbols[0])
                entries.notes += [resolution.note] if resolution.note else []
            else:
                entries.asks.append(resolution.message)
            continue
        entries.tickers.append(symbol)
        entries.names[symbol] = item.strip()
    return entries

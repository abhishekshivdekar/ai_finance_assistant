"""Plain data for the ticker resolver: a candidate listing, a price check and the outcome of one lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["resolved", "ambiguous", "not_found", "unavailable", "invalid"]


@dataclass(frozen=True)
class Candidate:
    """One listing found for a name. All text is plain data (control characters removed, length capped)."""

    symbol: str
    name: str = ""
    exchange: str = ""            # the lookup's exchange code, for example "NSI"
    exchange_name: str = ""       # the lookup's display name of the exchange
    quote_type: str = ""          # EQUITY, ETF, INDEX, ...
    currency: str | None = None   # filled by the price check, not by the lookup


@dataclass(frozen=True)
class Validation:
    """Outcome of the price check of one symbol."""

    ok: bool
    currency: str | None = None


@dataclass(frozen=True)
class Resolution:
    """What the resolver decided for one name.

    resolved: `symbols` holds the one price-checked symbol. ambiguous: `candidates` lists the price-checked options and
    `message` asks the user to type the symbol. not_found / unavailable / invalid: no symbol, `message` says why.
    `source` is "alias" or "yahoo search" for a resolved or ambiguous answer, else "".
    """

    status: Status
    query: str = ""
    symbols: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    source: str = ""
    message: str = ""

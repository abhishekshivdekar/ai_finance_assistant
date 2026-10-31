"""A network-free TickerStrategy for tests (and for trying the resolver by hand)."""
from __future__ import annotations

from src.data.tickers.models import Candidate
from src.data.tickers.strategies import TickerStrategy


class FakeLookup(TickerStrategy):
    """Returns scripted candidates by lower-case name; a scripted exception is raised instead; every call is recorded.

    Example: FakeLookup({"suzlon": [Candidate("SUZLON.NS", "Suzlon Energy Limited", "NSI", quote_type="EQUITY")]}).
    """

    name = "fake lookup"

    def __init__(self, table: dict[str, list[Candidate] | Exception] | None = None, trusted: bool = False) -> None:
        self.table = table or {}
        self.trusted = trusted
        self.calls: list[str] = []

    def find(self, query: str) -> list[Candidate]:
        """The scripted answer for the name, or [] when none is scripted."""
        self.calls.append(query)
        answer = self.table.get(query.casefold(), [])
        if isinstance(answer, Exception):
            raise answer
        return list(answer)

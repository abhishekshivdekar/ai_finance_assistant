"""NewsSource interface.

Design pattern: STRATEGY. NewsSource is the interface; GoogleNewsRssSource is the
implementation used today. Example: YFinanceProvider is given a NewsSource and
never knows whether headlines come from Google RSS or another feed, so a
different source could replace it without changing the provider or the agent.

Failure contract (same as the market providers): problems worth retrying raise
TransientProviderError, problems that retrying cannot fix raise
PermanentProviderError. An empty feed is not a failure: it returns [].
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NewsSource(ABC):
    """Strategy interface for headline sources."""

    @abstractmethod
    def fetch(self, query: str) -> list[dict[str, Any]]:
        """Headlines for a search phrase, newest first.

        Each item is {"title", "publisher", "link", "published"} with `published` as ISO text in UTC.
        """

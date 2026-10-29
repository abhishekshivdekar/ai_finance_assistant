"""Failure types shared by every data provider (market and news).

They live in their own module so the news layer can raise them without importing
the market package (which imports the news layer back).
"""
from __future__ import annotations


class TransientProviderError(Exception):
    """Temporary failure (network, timeout, rate limit): worth retrying."""


class PermanentProviderError(Exception):
    """Failure that retrying cannot fix (unknown ticker, no data, forbidden host, oversize response)."""

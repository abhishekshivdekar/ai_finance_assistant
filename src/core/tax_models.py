"""Immutable data models for the Tax Education calculators (architecture section 2.6).

Design pattern: VALUE OBJECT. Every model is frozen, so once a TaxCalculation is built
nothing can change it by accident. Example: capital_gains() returns a TaxCalculation and
the agent only reads it to build the payload.

Amounts are in INR and are never converted. No rate, limit or threshold lives here: they
come from config `tax.figures`.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Frozen(BaseModel):
    """Base: immutable and strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CapitalGainsInputs(_Frozen):
    """A validated capital gains request."""

    asset_type: str
    buy_value: float
    sell_value: float
    expenses: float = 0.0
    holding_months: int
    ltcg_exemption_already_used: float = 0.0
    slab_rate_pct: float | None = None


class SmallSavingsInputs(_Frozen):
    """A validated small-savings projection request."""

    scheme: str
    mode: str          # "annual" (deposit every year) or "one_time"
    amount: float
    years: int


class TaxCalculation(_Frozen):
    """Everything a calculator computed; the LLM only ever sees a formatted copy."""

    calculation: str
    inputs: dict[str, Any]
    results: dict[str, Any]
    steps: list[str]
    notes: list[str]
    assumptions: list[str]
    exclusions: list[str]
    config_used: dict[str, Any]

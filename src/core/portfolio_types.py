"""The two Literal types of a portfolio holding, in a module with no imports from the app.

They live here (and are re-exported by portfolio_models.py) so that the config loader can check the lists in
config.yaml against them without importing the market layer, which itself needs the config (an import cycle).
"""
from __future__ import annotations

from typing import Literal

InstrumentType = Literal["stock", "etf", "fund", "bond", "cash", "other"]   # what the holding IS
AssetClass = Literal["equity", "debt", "gold", "cash", "other"]            # what it behaves like (drives risk)

"""The Portfolio table choices in config.yaml: they must match the Literals and have UI labels."""
from __future__ import annotations

import copy
from typing import get_args

import pytest
import yaml

from src.core import portfolio_models
from src.core.portfolio_types import AssetClass, InstrumentType
from src.utils.config import ConfigError, get_config


def test_the_models_still_export_the_two_literals_from_their_own_module():
    assert portfolio_models.InstrumentType is InstrumentType and portfolio_models.AssetClass is AssetClass


def test_the_real_choices_equal_the_literals_and_every_choice_has_a_label():
    cfg = get_config()
    assert sorted(cfg["portfolio"]["instrument_types"]) == sorted(get_args(InstrumentType))
    assert sorted(cfg["portfolio"]["asset_classes"]) == sorted(get_args(AssetClass))
    assert sorted(cfg["ui"]["portfolio"]["instrument_labels"]) == sorted(get_args(InstrumentType))
    assert sorted(cfg["ui"]["portfolio"]["asset_class_labels"]) == sorted(get_args(AssetClass))
    assert cfg["portfolio"]["defaults"] == {"instrument_type": "stock", "asset_class": "equity"}


def _config(tmp_path, change):
    full = get_config()
    cfg = {"portfolio": copy.deepcopy(full["portfolio"]), "ui": copy.deepcopy(full["ui"])}
    get_config.cache_clear()
    change(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_a_valid_block_loads(tmp_path):
    assert get_config(_config(tmp_path, lambda c: None))["portfolio"]["defaults"]["asset_class"] == "equity"


def test_a_config_without_the_new_keys_is_not_checked(tmp_path):
    def old_style(cfg):
        cfg["portfolio"] = {"max_holdings": 5}

    assert get_config(_config(tmp_path, old_style))["portfolio"] == {"max_holdings": 5}


@pytest.mark.parametrize("change, text", [
    (lambda c: c["portfolio"]["instrument_types"].append("crypto"), "instrument_types"),
    (lambda c: c["portfolio"]["instrument_types"].remove("etf"), "instrument_types"),
    (lambda c: c["portfolio"].update(asset_classes="equity"), "asset_classes"),
    (lambda c: c["portfolio"]["asset_classes"].remove("gold"), "asset_classes"),
    (lambda c: c["portfolio"]["defaults"].update(instrument_type="crypto"), "defaults"),
    (lambda c: c["portfolio"]["defaults"].update(asset_class="stock"), "defaults"),
    (lambda c: c["portfolio"].update(resolved_note=" "), "resolved_note"),
    (lambda c: c["portfolio"].pop("row_problem_message"), "row_problem_message"),
    (lambda c: c["ui"]["portfolio"]["instrument_labels"].pop("etf"), "instrument_labels"),
    (lambda c: c["ui"]["portfolio"]["asset_class_labels"].update(gold=""), "asset_class_labels"),
    (lambda c: c["ui"]["portfolio"]["asset_class_labels"].update(crypto="Crypto"), "asset_class_labels"),
    (lambda c: c["ui"]["portfolio"]["columns"].pop("ticker"), "ui.portfolio.columns.ticker"),
])
def test_a_bad_portfolio_block_is_rejected_with_the_key_named(tmp_path, change, text):
    with pytest.raises(ConfigError, match=text):
        get_config(_config(tmp_path, change))


def test_the_ui_without_a_portfolio_section_is_still_rejected_by_the_text_check(tmp_path):
    with pytest.raises(ConfigError, match="ui.portfolio"):
        get_config(_config(tmp_path, lambda c: c["ui"].pop("portfolio")))

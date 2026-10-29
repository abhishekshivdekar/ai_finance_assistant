"""A synthetic `tax` config for tests: round made-up figures, so hand-computed expectations stay valid
when the real (to-be-verified) figures in config.yaml are corrected. The real block is checked in test_tax_config.py.

With these figures: equity short-term 15 percent, equity long-term 10 percent above an exemption of
100000, equity long-term after more than 12 months, other assets long-term after more than 24 months
at 20 percent, cess 10 percent, PPF 10 percent, NSC 5 percent, SSY 8 percent.
"""
import copy

from src.utils.config import get_config

FIGURES = {
    "equity_stcg_pct": 15, "equity_ltcg_pct": 10, "equity_ltcg_exemption": 100000,
    "holding_months_equity": 12, "holding_months_other": 24, "other_ltcg_pct": 20, "cess_pct": 10,
    "ppf_pct": 10, "nsc_pct": 5, "ssy_pct": 8,
}


DISPLAYS = {"pct": "{value}%", "inr": "INR {value:,}", "months": "{value} months"}


def tax_cfg(**changes):
    """A deep copy of the real tax block with the test figures (and any other top-level changes) applied.

    The display text is rebuilt from the test value, so a fact line never shows the real figure.
    """
    cfg = copy.deepcopy(get_config()["tax"])
    for key, value in FIGURES.items():
        fig = cfg["figures"][key]
        fig["value"] = value
        fig["display"] = DISPLAYS[fig["unit"]].format(value=value)
    cfg.update(changes)
    return cfg


def full_config(cfg):
    """The whole config with `tax` replaced, for patching get_config in the agent module."""
    full = copy.deepcopy(get_config())
    full["tax"] = cfg
    return full

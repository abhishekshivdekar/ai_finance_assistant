import ast
import copy
import re
from pathlib import Path

import pytest
import yaml

from src.utils.config import TAX_FIGURE_KEYS, ConfigError, get_config

ROOT = Path(__file__).resolve().parent.parent
TAX_SOURCES = ["src/core/tax_models.py", "src/core/tax_inputs.py", "src/core/tax_metrics.py", "src/agents/tax.py"]
ALLOWED_LITERALS = {0, 1, 2, 100}
# Parts of the tax block that may hold digits: the figures, our own input caps and the asset-type identifiers
# (debt_fund_post_2023 is a name, not a figure).
NUMERIC_PARTS = {"figures", "input_limits", "asset_types", "slab_asset_types"}


def _tax_config(tmp_path, change):
    """A config file holding only a (modified) copy of the real tax block."""
    tax = copy.deepcopy(get_config()["tax"])
    get_config.cache_clear()
    change(tax)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"tax": tax}), encoding="utf-8")
    return str(path)


def test_the_real_tax_block_has_every_figure_with_a_date_a_source_and_verify_true():
    tax = get_config()["tax"]
    assert set(TAX_FIGURE_KEYS) <= set(tax["figures"])
    for key, fig in tax["figures"].items():
        assert fig["as_of"] == "2026-10-29" and fig["source"] and fig["label"] and fig["verify"] is True, key
    assert tax["topic"] == "accounts_and_taxes_india" and tax["label"] == "illustrative, verify before relying"
    assert get_config()["llm"]["roles"]["tax_summary"]["model"]


def test_only_the_equity_figures_carry_the_tax_paid_condition():
    figures = get_config()["tax"]["figures"]
    assert [k for k, v in figures.items() if v.get("stt_condition")] == ["equity_stcg_pct", "equity_ltcg_pct", "equity_ltcg_exemption"]
    assert list(get_config()["tax"]["schemes"]) == ["ppf", "nsc", "ssy"]


def test_exclusions_and_the_tax_paid_assumption_are_in_config():
    tax = get_config()["tax"]
    assert tax["stt_assumption"] == "assumes the securities transaction tax was paid on the sale of listed equity and equity funds"
    joined = " ".join(tax["exclusions"])
    for word in ("surcharge", "indexation", "set-off", "slabs"):
        assert word in joined
    assert any("interest simplified to annual compounding; actual scheme interest calculation differs; check the official scheme rules" == a
               for a in tax["small_savings_assumptions"])


def test_valid_tax_config_loads(tmp_path):
    assert get_config(_tax_config(tmp_path, lambda t: None))["tax"]["topic"] == "accounts_and_taxes_india"


@pytest.mark.parametrize("field", ["as_of", "source", "verify", "label", "value", "unit"])
def test_every_figure_needs_all_its_fields(tmp_path, field):
    def drop(tax):
        del tax["figures"]["cess_pct"][field]

    with pytest.raises(ConfigError, match="cess_pct"):
        get_config(_tax_config(tmp_path, drop))


@pytest.mark.parametrize("field, bad", [
    ("as_of", "29/10/2026"), ("as_of", "soon"), ("source", "  "), ("label", ""), ("verify", "yes"), ("verify", 1),
    ("value", -1), ("value", "20"), ("value", True), ("unit", "dollars"),
])
def test_bad_figure_fields_are_rejected(tmp_path, field, bad):
    def change(tax):
        tax["figures"]["equity_stcg_pct"][field] = bad

    with pytest.raises(ConfigError, match="equity_stcg_pct"):
        get_config(_tax_config(tmp_path, change))


def test_a_missing_figure_is_rejected(tmp_path):
    def drop(tax):
        del tax["figures"]["ssy_pct"]

    with pytest.raises(ConfigError, match="ssy_pct is missing"):
        get_config(_tax_config(tmp_path, drop))


def test_a_scheme_must_point_at_a_figure(tmp_path):
    def change(tax):
        tax["schemes"]["ppf"] = "nope_pct"

    with pytest.raises(ConfigError, match="tax.schemes.ppf"):
        get_config(_tax_config(tmp_path, change))


def test_an_unquoted_yaml_date_is_accepted(tmp_path):
    path = _tax_config(tmp_path, lambda t: None)
    text = Path(path).read_text(encoding="utf-8").replace("'2026-10-29'", "2026-10-29")
    Path(path).write_text(text, encoding="utf-8")
    assert get_config(path)["tax"]["figures"]["cess_pct"]["as_of"]


# ---- no rate, limit or threshold outside the figures ----

def _strings(value, skip=()):
    """Every string in a nested config value (keys of `skip` are not entered)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, inner in value.items():
            if key not in skip:
                yield from _strings(inner, skip)
    elif isinstance(value, list):
        for inner in value:
            yield from _strings(inner, skip)


def test_tax_prompts_texts_and_labels_contain_no_digits():
    tax = get_config()["tax"]
    markers = re.compile(r"\[F?\d+\]")
    placeholders = re.compile(r"\{[^}]*\}")
    offenders = []
    for text in _strings(tax, skip=NUMERIC_PARTS):
        cleaned = placeholders.sub("", markers.sub("", text))
        if re.search(r"\d", cleaned):
            offenders.append(text[:60])
    assert offenders == []


def test_tax_code_has_no_numeric_literals_except_structural_ones():
    problems = []
    for relative in TAX_SOURCES:
        path = ROOT / relative
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=relative)):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if node.value not in ALLOWED_LITERALS:
                    problems.append(f"{relative}:{node.lineno} numeric literal {node.value!r} (allowed: {sorted(ALLOWED_LITERALS)})")
    assert problems == [], "\n".join(problems)


def test_the_literal_check_really_catches_a_rate(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text("RATE = 12.5\nOK = 100\n", encoding="utf-8")
    found = [n.value for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Constant) and n.value not in ALLOWED_LITERALS]
    assert found == [12.5]

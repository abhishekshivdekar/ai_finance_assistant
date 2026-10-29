import json

import pytest

from src.agents import base_agent
from src.agents import tax as tax_module
from src.agents.tax import TaxAgent
from src.rag.retriever import RetrievedChunk, retrieve
from src.utils.config import get_config
from tax_cfg import full_config, tax_cfg

CFG = tax_cfg()
TOPIC = CFG["topic"]
REFS = [{"name": "Income Tax Department", "url": "https://www.incometax.gov.in"}]
QUESTION = "Which rate applies to equity fund gains held for 30 months?"
JSON_MARKER = "JSON:"


def chunk(n, text="Capital gains are the profit made when an asset is sold for more than it cost."):
    cid = f"{TOPIC}/article-{n}#0"
    return RetrievedChunk(text, f"Article {n}", TOPIC, REFS, cid, 0.6)


class FakeLLM:
    """Records every prompt; returns `reply` or raises `error`."""

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return type("Msg", (), {"content": self.reply})()


class FakeRetrieve:
    """retrieve_fn stand-in: fixed chunks, calls recorded."""

    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def __call__(self, query, topic=None):
        self.calls.append((query, topic))
        return list(self.chunks)


@pytest.fixture(autouse=True)
def _synthetic_figures(monkeypatch):
    """The agent reads the test figures, and no real LLM can be created."""
    monkeypatch.setattr(tax_module, "get_config", lambda: full_config(CFG))

    def fail(**kwargs):
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr(base_agent, "get_llm", fail)


def ask(reply, chunks=None, **state):
    llm, retrieve_fn = FakeLLM(reply), FakeRetrieve([chunk(1), chunk(2)] if chunks is None else chunks)
    result = TaxAgent(llm=llm, retrieve_fn=retrieve_fn).run({"turn_id": "t1", "query": QUESTION, **state})
    return llm, retrieve_fn, result


def prompt_text(llm):
    return "\n".join(m.content for m in llm.calls[0])


# ---- question path ----

def test_retrieval_is_locked_to_the_tax_topic_whatever_the_dropdown_says():
    _, retrieve_fn, _ = ask("Gains are taxed [1].", topic_filter="stocks")
    assert retrieve_fn.calls == [(QUESTION, TOPIC)] and TOPIC == "accounts_and_taxes_india"


def test_the_prompt_carries_numbered_chunks_numbered_facts_and_the_rules():
    llm, _, _ = ask("Gains are taxed [1].")
    system, human = llm.calls[0][0].content, llm.calls[0][1].content
    assert "[1] Article 1" in human and "[2] Article 2" in human and f"Question: {QUESTION}" in human
    for number, (key, fig) in enumerate(CFG["figures"].items(), start=1):
        assert f"F{number}. {fig['label']}: {fig['display']}. As of" in human
        assert f"As of {fig['as_of']}. Source: {fig['source']}" in human
    assert human.count("Verify: yes") == len(CFG["figures"])
    assert human.count(f"Condition: {CFG['stt_assumption']}") == 3          # the three equity figures
    assert "[F1]" in system and "Never state an equity rate" in system and CFG["stt_assumption"] in system
    assert "chartered accountant" in system and "Never give personal tax advice" in system
    assert "illustrative, verify before relying" in system and "Do not write dates" in system


def test_a_chunk_citation_gives_an_ok_answer_with_only_the_cited_source():
    _, _, result = ask("Gains are taxed when you sell [2].")
    assert result["status"] == "ok" and result["answer"] == "Gains are taxed when you sell [1]."
    assert [s["title"] for s in result["sources"]] == ["Article 2"]
    assert result["data"]["fact_citations"] == [] and result["data"]["config_used"] == {}


def test_a_fact_citation_adds_the_fixed_label_as_of_date_and_tax_paid_note():
    _, _, result = ask("Equity fund gains held longer are taxed at 10% [F2] after an exemption of 100000 [F3].")
    assert result["status"] == "ok"
    data = result["data"]
    # the test figures repeat 10 (equity long-term, cess, PPF): cited figures plus every figure whose number is stated
    assert data["fact_citations"] == [2, 3]
    assert set(data["config_used"]) == {"equity_ltcg_pct", "equity_ltcg_exemption", "cess_pct", "ppf_pct"}
    assert data["config_used"]["equity_ltcg_pct"]["as_of"] == "2026-10-29" and data["config_used"]["equity_ltcg_pct"]["source"]
    assert "illustrative, verify before relying (as of 2026-10-29)" in result["answer"]
    assert result["answer"].endswith("Assumes the securities transaction tax was paid on the sale of listed equity and equity funds.")
    assert [f["id"] for f in data["facts"]][:2] == ["F1", "F2"] and data["label"] == CFG["label"]
    assert data["exclusions"] == [CFG["stt_assumption"], *CFG["exclusions"]["capital_gains"]]
    assert result["sources"] == []


def test_a_non_equity_fact_citation_has_no_tax_paid_note():
    _, _, result = ask("The Sukanya Samriddhi rate is 8% [F10].")
    assert result["status"] == "ok" and "transaction tax" not in result["answer"] and "(as of 2026-10-29)" in result["answer"]


def test_chunk_and_fact_citations_can_be_mixed():
    _, _, result = ask("A capital gain is a profit [1]. The cess is 10% [F7].")
    assert result["status"] == "ok" and result["data"]["fact_citations"] == [7] and len(result["sources"]) == 1


def test_invalid_fact_citations_are_dropped_and_valid_ones_keep_their_numbers():
    n = len(CFG["figures"])
    _, _, result = ask(f"The cess is 10% [F7][F{n + 1}][F0].")
    assert result["status"] == "ok" and "[F7]" in result["answer"] and f"F{n + 1}" not in result["answer"]
    assert result["data"]["fact_citations"] == [7]


def test_an_answer_with_only_invalid_citations_is_discarded():
    _, _, result = ask("The cess is 10% [F99]. Gains [9].")
    assert result["status"] == "no_answer" and result["answer"] == " ".join(CFG["discarded_message"].split())
    assert result["sources"] == []


def test_an_answer_without_any_citation_is_discarded():
    assert ask("Gains are taxed when you sell.")[2]["status"] == "no_answer"


def test_an_invented_number_is_rejected():
    _, _, result = ask("Equity fund gains are taxed at 18% [F2].")
    assert result["status"] == "no_answer" and result["answer"] == " ".join(CFG["discarded_message"].split())


def test_a_calculated_number_is_rejected_even_when_its_parts_are_allowed():
    assert ask("Ten percent of 100000 is 10000 [F2].")[2]["status"] == "no_answer"


def test_numbers_from_the_facts_the_chunks_and_the_question_are_allowed():
    text = "Held for 30 months [F2]: 10% [F2] above INR 100,000 [F3], with 5 years of context [1]."
    _, _, result = ask(text, chunks=[chunk(1, "A holding of 5 years is long.")])
    assert result["status"] == "ok"


def test_citation_digits_alone_are_not_mistaken_for_figures():
    assert ask("See the notes [1][2] and the fact [F7].")[2]["status"] == "ok"


def test_no_answer_sentinel_gives_no_answer_with_the_fixed_message():
    for reply in ("NO_ANSWER", " `no_answer`. "):
        _, _, result = ask(reply)
        assert result["status"] == "no_answer" and result["answer"] == " ".join(CFG["no_answer_message"].split())
        assert [f["id"] for f in result["data"]["facts"]][0] == "F1"


def test_no_chunks_means_no_answer_without_an_llm_call():
    llm, _, result = ask("never used", chunks=[])
    assert result["status"] == "no_answer" and llm.calls == []


def test_an_empty_question_is_needs_input_without_any_call():
    for state in ({"query": "   "}, {"query": None}):
        llm, retrieve_fn, result = ask("x", **state)
        assert result["status"] == "needs_input" and result["answer"] == CFG["question_needs_input"]
        assert llm.calls == [] and retrieve_fn.calls == []


def test_an_llm_failure_becomes_a_clean_error_result():
    agent = TaxAgent(llm=FakeLLM(error=RuntimeError("boom")), retrieve_fn=FakeRetrieve([chunk(1)]))
    result = agent.run({"turn_id": "t1", "query": QUESTION})
    assert result["status"] == "error" and result["data"] is None


def test_tax_params_without_a_calculation_is_still_a_question():
    for params in (None, {}, {"calculation": None}, "capital_gains", ["x"]):
        _, retrieve_fn, result = ask("Gains are taxed [1].", tax_params=params)
        assert result["status"] == "ok" and len(retrieve_fn.calls) == 1


# ---- calculator path ----

EQUITY = {"calculation": "capital_gains", "asset_type": "equity_fund", "buy_value": 100000, "sell_value": 250000,
          "expenses": 0, "holding_months": 13}
SAVINGS = {"calculation": "small_savings_growth", "scheme": "ppf", "mode": "annual", "amount": 1000, "years": 2}
GOOD = "This is an illustrative result, so please verify the rates on the official site."


def calc(params, reply=GOOD, error=None):
    llm = FakeLLM(reply, error)
    result = TaxAgent(llm=llm, retrieve_fn=FakeRetrieve([])).run({"turn_id": "t1", "tax_params": params})
    return llm, result


def test_capital_gains_payload_is_complete_and_deterministic():
    llm, result = calc({**EQUITY, "ltcg_exemption_already_used": 40000})
    data = result["data"]
    assert result["status"] == "ok" and data["path"] == "calculation" and data["calculation"] == "capital_gains"
    assert data["results"]["total_tax"] == 9900.0 and data["results"]["taxable_gain"] == 90000.0
    assert data["inputs"]["holding_months"] == 13 and len(data["steps"]) == 7 and data["label"] == CFG["label"]
    assert set(data["config_used"]) == {"cess_pct", "holding_months_equity", "equity_ltcg_pct", "equity_ltcg_exemption"}
    assert all(e["as_of"] == "2026-10-29" and e["source"] and e["verify"] for e in data["config_used"].values())
    assert data["exclusions"] == [CFG["stt_assumption"], *CFG["exclusions"]["capital_gains"]]
    assert data["warnings"] == [] and result["answer"].startswith(GOOD) and result["sources"] == []
    assert "illustrative, verify before relying (as of 2026-10-29)" in result["answer"]
    assert result["answer"].endswith("equity funds.")                        # equity figures used: tax-paid assumption shown


def test_the_commentary_prompt_holds_only_computed_data():
    llm, _ = calc(EQUITY)
    prompt = llm.calls[0]
    payload = json.loads(prompt.split(JSON_MARKER, 1)[1])
    assert set(payload) == {"calculation", "inputs", "results", "rates", "caveats"} and payload["results"]["total_tax"] == "₹5,500.00"
    assert payload["inputs"]["buy_value"] == "₹100,000.00" and payload["inputs"]["holding_months"] == 13
    assert payload["results"]["rate_pct"] == "10%" and payload["rates"]["equity_ltcg_pct"] == "10%"
    assert "copy any `display` text exactly as given" in " ".join(prompt.split()).lower()
    assert "\\u20b9" not in prompt and "₹5,500.00" in prompt                    # the sign is sent as a character, not a JSON escape
    assert "2026" not in prompt and "incometax" not in prompt and CFG["stt_assumption"] in payload["caveats"]
    assert "never tell the reader what to buy, sell or hold" in prompt.replace("\n", " ").replace("  ", " ")


def test_a_debt_fund_calculation_uses_the_slab_rate_and_no_equity_note():
    _, result = calc({"calculation": "capital_gains", "asset_type": "debt_fund_post_2023", "buy_value": 100000,
                      "sell_value": 150000, "holding_months": 36, "slab_rate_pct": 30}, reply="The total is 16500.")
    assert result["status"] == "ok" and result["data"]["results"]["total_tax"] == 16500.0
    assert set(result["data"]["config_used"]) == {"cess_pct"} and "transaction tax" not in result["answer"]
    assert result["data"]["exclusions"] == CFG["exclusions"]["capital_gains"]       # no tax-paid line for a slab-rate asset
    assert CFG["stt_assumption"] not in json.dumps(result["data"]["exclusions"])


def test_a_loss_reports_zero_tax_and_the_note():
    _, result = calc({**EQUITY, "sell_value": 50000}, reply="There is no tax to pay.")
    assert result["data"]["results"]["total_tax"] == 0.0 and result["data"]["notes"] == [CFG["texts"]["loss_note"]]


def test_small_savings_payload_has_the_projection_the_rate_and_the_assumptions():
    _, result = calc(SAVINGS, reply="Your 2000 grows to 2310.")
    data = result["data"]
    assert result["status"] == "ok" and data["results"]["final_value"] == 2310.0 and len(data["results"]["yearly"]) == 2
    assert data["exclusions"] == CFG["exclusions"]["small_savings"] and data["assumptions"] == [] and data["notes"] == [CFG["texts"]["approximate_note"]]
    assert set(data["config_used"]) == {"ppf_pct"} and data["config_used"]["ppf_pct"]["source"]


@pytest.mark.parametrize("reply, error, key", [
    ("", None, "commentary_empty"),
    ("   \n", None, "commentary_empty"),
    ("x", RuntimeError("down"), "commentary_unavailable"),
    ("The tax is 18000 rupees.", None, "commentary_discarded"),
    ("It will take 5 years.", None, "commentary_discarded"),
    ("As of 2026-10-29 the total is 9900.", None, "commentary_discarded"),
])
def test_commentary_failures_use_the_fixed_line_and_keep_the_payload(reply, error, key):
    _, result = calc(EQUITY, reply=reply, error=error)
    data = result["data"]
    assert result["status"] == "ok" and data["warnings"] == [CFG["texts"][key]]
    assert result["answer"].startswith(" ".join(CFG["fallback_commentary"].split()))
    assert data["results"]["total_tax"] == 5500.0                                  # the payload is untouched by the LLM
    assert data["steps"] and data["config_used"] and data["exclusions"] and data["label"]


def test_numbers_from_the_inputs_results_and_rates_are_allowed_in_the_commentary():
    _, result = calc(EQUITY, reply="You held for 13 months; the 10% rate and cess of 10% give a total of 5500.")
    assert result["data"]["warnings"] == [] and result["answer"].startswith("You held for 13 months")


@pytest.mark.parametrize("params", [
    {"calculation": "gold_loan"},
    {"calculation": 5},
])
def test_an_unknown_calculation_is_needs_input_listing_the_choices(params):
    llm, result = calc(params)
    assert result["status"] == "needs_input" and llm.calls == []
    assert "capital_gains, small_savings_growth" in result["answer"] and result["data"]["warnings"]
    assert result["data"]["label"] == CFG["label"] and result["data"]["exclusions"] == []


def test_missing_slab_rate_for_a_debt_fund_is_needs_input():
    llm, result = calc({"calculation": "capital_gains", "asset_type": "debt_fund_post_2023", "buy_value": 1,
                        "sell_value": 2, "holding_months": 3})
    assert result["status"] == "needs_input" and llm.calls == []
    assert CFG["errors"]["slab_rate_needed"] in result["answer"]


@pytest.mark.parametrize("params", [
    {**EQUITY, "buy_value": 10**400},
    {**EQUITY, "buy_value": float("inf")},
    {**EQUITY, "sell_value": float("nan")},
    {**EQUITY, "holding_months": -(10**400)},
    {**EQUITY, "asset_type": ["x"]},
    {"calculation": "capital_gains"},
    {**SAVINGS, "years": 10**400},
    {**SAVINGS, "amount": "lots"},
    {"calculation": "small_savings_growth"},
])
def test_huge_and_invalid_inputs_are_needs_input_not_crashes(params):
    llm, result = calc(params)
    assert result["status"] == "needs_input" and llm.calls == [] and result["data"]["warnings"]


# ---- wiring ----

def test_default_llm_comes_from_the_factory_with_the_tax_role(monkeypatch):
    roles = []

    def fake_get_llm(role=None, **kwargs):
        roles.append(role)
        return FakeLLM("x")

    monkeypatch.setattr(base_agent, "get_llm", fake_get_llm)
    agent = TaxAgent(retrieve_fn=FakeRetrieve([]))
    assert roles == ["tax_summary"] and agent.name == "tax_education"
    assert get_config()["llm"]["roles"]["tax_summary"]["model"]


def test_default_retrieve_function_is_the_shared_retriever():
    assert TaxAgent(llm=FakeLLM()).retrieve_fn is retrieve

def test_a_quoted_fact_number_with_only_a_chunk_citation_still_gets_the_footer():
    _, _, result = ask("Equity fund gains held long are taxed at 15% [1].")
    assert result["status"] == "ok" and result["data"]["fact_citations"] == []
    assert "equity_stcg_pct" in result["data"]["config_used"]                      # matched by number: listed as used
    assert result["data"]["exclusions"] == [CFG["stt_assumption"], *CFG["exclusions"]["capital_gains"]]
    assert "illustrative, verify before relying (as of 2026-10-29)" in result["answer"]
    assert result["answer"].endswith("equity funds.")


def test_an_answer_with_no_figure_and_no_fact_citation_gets_no_footer():
    _, _, result = ask("Gains are taxed when you sell [1].")
    assert result["answer"] == "Gains are taxed when you sell [1]." and result["data"]["config_used"] == {}


@pytest.mark.parametrize("text", [
    "Held for 7 years the gain is long-term [F1].",      # 7 is only a fact id (F7); 10 would clash with a test figure
    "The figures are as of 29 October [F1].",            # 29 is only part of the as_of date
    "The rate was set in 2026 [F1].",                    # the year is only in as_of / source
])
def test_numbers_that_only_appear_in_fact_ids_dates_or_sources_are_rejected(text):
    assert ask(text)[2]["status"] == "no_answer"


def test_a_figure_display_text_and_lakh_grouping_can_be_quoted():
    cfg = tax_cfg()
    cfg["figures"]["equity_ltcg_exemption"]["display"] = "INR 1,00,000 (INR 1 lakh) per financial year"
    cfg["figures"]["equity_ltcg_exemption"]["value"] = 100000
    original = tax_module.get_config
    tax_module.get_config = lambda: full_config(cfg)
    try:
        result = TaxAgent(llm=FakeLLM("The exemption is 1,00,000 a year [F3]."), retrieve_fn=FakeRetrieve([chunk(1)])).run(
            {"turn_id": "t1", "query": QUESTION})
    finally:
        tax_module.get_config = original
    assert result["status"] == "ok"


def test_the_ppf_commentary_input_has_no_tax_paid_surcharge_or_indexation_text():
    llm, result = calc(SAVINGS, reply="Your 2000 grows to 2310.")
    sent = llm.calls[0]
    for word in ("transaction tax", "surcharge", "indexation", "set-off"):
        assert word not in sent and word not in json.dumps(result["data"]["exclusions"])
    assert "annual compounding" in sent and "lock-in" in sent


def test_a_debt_fund_commentary_input_has_no_tax_paid_line_but_an_equity_one_has_it():
    debt = {"calculation": "capital_gains", "asset_type": "debt_fund_post_2023", "buy_value": 100000,
            "sell_value": 150000, "holding_months": 36, "slab_rate_pct": 30}
    assert CFG["stt_assumption"] not in calc(debt, reply="The total is 16500.")[0].calls[0]
    assert CFG["stt_assumption"] in calc(EQUITY)[0].calls[0]


def test_an_acronym_question_is_searched_as_written_and_in_plain_words_on_the_locked_topic():
    chunks = [chunk(1)]
    llm = FakeLLM("Gains are taxed [1].")
    retrieve_fn = FakeRetrieve(chunks)
    result = TaxAgent(llm=llm, retrieve_fn=retrieve_fn).run({"turn_id": "t1", "query": "What is STT?", "topic_filter": "stocks"})
    assert retrieve_fn.calls == [("What is STT?", TOPIC), ("What is securities transaction tax?", TOPIC)]
    assert result["status"] == "ok"


@pytest.mark.parametrize("params, expected", [
    ({"calculation": "capital_gains", "asset_type": "equity_fund"}, [CFG["stt_assumption"], *CFG["exclusions"]["capital_gains"]]),
    ({"calculation": "capital_gains", "asset_type": "debt_fund_post_2023"}, CFG["exclusions"]["capital_gains"]),
    ({"calculation": "small_savings_growth"}, CFG["exclusions"]["small_savings"]),
])
def test_a_needs_input_result_already_carries_the_exclusions_of_the_requested_calculation(params, expected):
    _, result = calc(params)
    assert result["status"] == "needs_input" and result["data"]["exclusions"] == expected


def test_an_answer_quoting_a_rate_lists_the_cited_fact_and_every_figure_with_that_number():
    cfg = tax_cfg()
    cfg["figures"]["other_ltcg_pct"]["value"], cfg["figures"]["other_ltcg_pct"]["display"] = 10, "10%"   # like 12.5 in two figures
    original = tax_module.get_config
    tax_module.get_config = lambda: full_config(cfg)
    try:
        result = TaxAgent(llm=FakeLLM("Long-term equity gains are taxed at 10% [F2]."), retrieve_fn=FakeRetrieve([chunk(1)])).run(
            {"turn_id": "t1", "query": QUESTION})
    finally:
        tax_module.get_config = original
    assert result["status"] == "ok" and result["data"]["fact_citations"] == [2]
    assert set(result["data"]["config_used"]) == {"equity_ltcg_pct", "other_ltcg_pct", "cess_pct", "ppf_pct"}   # the union, as accepted
    assert "illustrative, verify before relying" in result["answer"] and result["answer"].endswith("equity funds.")


def test_a_definition_answer_without_a_figure_or_a_fact_citation_has_no_exclusions_and_no_footer():
    _, _, result = ask("Capital gains are the profit made when an asset is sold for more than it cost [1].")
    data = result["data"]
    assert result["status"] == "ok" and data["exclusions"] == [] and data["config_used"] == {} and data["label"] == CFG["label"]
    assert "illustrative, verify" not in result["answer"]


def test_a_figure_matched_by_number_gets_the_footer_the_source_and_the_tax_paid_exclusion():
    cfg = tax_cfg()
    for key in ("equity_ltcg_pct", "other_ltcg_pct"):                     # 12.5 in two figures, as in the real config
        cfg["figures"][key]["value"], cfg["figures"][key]["display"] = 12.5, "12.5%"
    original = tax_module.get_config
    tax_module.get_config = lambda: full_config(cfg)
    try:
        result = TaxAgent(llm=FakeLLM("Long-term gains are taxed at 12.5% [1]."), retrieve_fn=FakeRetrieve([chunk(1)])).run(
            {"turn_id": "t1", "query": QUESTION})
    finally:
        tax_module.get_config = original
    data = result["data"]
    assert result["status"] == "ok" and result["answer"].startswith("Long-term gains are taxed at 12.5% [1].")
    assert f"(as of {cfg['figures']['equity_ltcg_pct']['as_of']})" in result["answer"] and cfg["stt_assumption"][1:] in result["answer"]
    assert set(data["config_used"]) == {"equity_ltcg_pct", "other_ltcg_pct"} and data["fact_citations"] == []   # both matched figures, none cited
    assert all(entry["source"] and entry["as_of"] for entry in data["config_used"].values())
    assert data["exclusions"] == [cfg["stt_assumption"], *cfg["exclusions"]["capital_gains"]]


def test_a_cited_equity_figure_adds_the_footer_the_figure_and_the_tax_paid_exclusion():
    _, _, result = ask("Long-term equity gains are taxed at 10% [F2].")
    data = result["data"]
    assert {"equity_ltcg_pct"} <= set(data["config_used"]) and data["fact_citations"] == [2]
    assert data["exclusions"] == [CFG["stt_assumption"], *CFG["exclusions"]["capital_gains"]]
    assert "illustrative, verify before relying" in result["answer"]


def test_a_cited_non_equity_figure_gets_the_footer_and_the_capital_gains_list_without_the_tax_paid_line():
    _, _, result = ask("Gains on other assets are taxed at 20% after 24 months [F6][F5].")
    data = result["data"]
    assert set(data["config_used"]) == {"other_ltcg_pct", "holding_months_other"}
    assert "illustrative, verify before relying" in result["answer"]
    assert data["exclusions"] == CFG["exclusions"]["capital_gains"] and "transaction tax" not in result["answer"]


@pytest.mark.parametrize("reply, ok", [
    ("The total tax is ₹5,500.00 on a gain of ₹150,000.00.", True),
    ("The total tax is 5,500 on a gain of 150000.", True),
    ("The total tax is ₹5,600.00.", False),
    ("A holding of 13 months at 10% gives ₹5,500.00 in total, 1.5 lakh gain.", False),     # 1.5 is not a computed number
])
def test_the_commentary_guard_compares_with_the_pre_formatted_amounts(reply, ok):
    _, result = calc(EQUITY, reply=reply)
    assert (result["data"]["warnings"] == []) is ok and result["answer"].startswith(reply) is ok


def test_a_lakh_style_display_goes_to_the_commentary_as_given_and_repeating_it_is_accepted(monkeypatch):
    shown = "₹1,25,000 (₹1.25 lakh) per financial year"
    cfg = tax_cfg()
    cfg["figures"]["equity_ltcg_exemption"]["display"] = shown
    monkeypatch.setattr(tax_module, "get_config", lambda: full_config(cfg))
    llm, result = calc(EQUITY, reply=f"The yearly exemption is {shown}, and the total tax is ₹5,500.00; this is illustrative.")
    assert f'"equity_ltcg_exemption": "{shown}"' in llm.calls[0]                  # sent exactly as in config
    assert "(no \"lakh\"" not in llm.calls[0] and "no lakh" not in llm.calls[0].lower()
    assert result["data"]["warnings"] == [] and shown in result["answer"]          # the guard accepts the repeated display text

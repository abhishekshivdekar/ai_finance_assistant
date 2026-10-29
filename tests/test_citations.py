import pytest

from src.core.citations import apply_citations, drop_invalid_citations, strip_citations


@pytest.mark.parametrize("text, count, expected, cited", [
    ("A [2]. B [1].", 3, "A [2]. B [1].", [1, 2]),
    ("A [2]. B [7].", 3, "A [2]. B.", [2]),                      # [7] does not exist; [2] is NOT renumbered
    ("A [2, 9, 3].", 3, "A [2, 3].", [2, 3]),
    ("A [1][1][3].", 3, "A [1][1][3].", [1, 3]),
    ("A [0]. B [4].", 3, "A. B.", []),
    ("No citations here.", 3, "No citations here.", []),
    ("A [3,1] and  [9]  tail", 3, "A [3, 1] and tail", [1, 3]),
])
def test_drop_invalid_citations_keeps_original_numbers(text, count, expected, cited):
    assert drop_invalid_citations(text, count) == (expected, cited)


def test_strip_citations_removes_every_marker():
    assert strip_citations("A [1] and [2, 3].") == "A  and ."


def test_apply_citations_renumbers_any_kind_of_item():
    answer, cited = apply_citations("A [3]. B [1][9]. C [3, 1].", ["x", "y", "z", "w"])
    assert answer == "A [1]. B [2]. C [1, 2]." and cited == ["z", "x"]


# ---- fact citations ([F1]) ----

from src.core.citations import drop_invalid_fact_citations, strip_fact_citations


@pytest.mark.parametrize("text, count, expected, cited", [
    ("Rate [F2]. Other [F9].", 3, "Rate [F2]. Other.", [2]),
    ("A [F3, F1] and [F2][F2].", 3, "A [F3, F1] and [F2][F2].", [1, 2, 3]),
    ("A [F0]. B [F4].", 3, "A. B.", []),
    ("A [F2, F9, F3].", 3, "A [F2, F3].", [2, 3]),
    ("Plain [1] chunk citations stay for the other helper.", 3, "Plain [1] chunk citations stay for the other helper.", []),
    ("No markers.", 3, "No markers.", []),
])
def test_drop_invalid_fact_citations_keeps_original_numbers(text, count, expected, cited):
    assert drop_invalid_fact_citations(text, count) == (expected, cited)


def test_strip_fact_citations_removes_only_fact_markers():
    assert strip_fact_citations("A [F1] and [F2, F3] but [4].") == "A  and  but [4]."


def test_fact_and_chunk_citations_do_not_interfere():
    text, facts = drop_invalid_fact_citations("Claim [1][F2] and [F1].", 2)
    answer, cited = apply_citations(text, ["x", "y"])
    assert facts == [1, 2] and answer == "Claim [1][F2] and [F1]." and cited == ["x"]

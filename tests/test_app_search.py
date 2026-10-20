"""Closest-match suggestions for the typed planta and SKU."""

from src.app.search import normalize, resolve, suggest

SKUS = [
    "HP2/01/125gsm/2440mm/1200-1450",
    "HP2/01/125gsm/2440mm/1200-1450/Subst.",
    "HP2/01/105gsm/2440mm/1200-1400",
    "TSL/01/100gsm/2450mm/1200-1400",
]
PLANTAS = ["SALC", "SCAN", "SCOC", "SPAL"]


def test_normalize_drops_case_accents_and_punctuation():
    assert normalize("  Péñón/01-125GSM ") == "penon 01 125gsm"
    assert normalize("") == ""


def test_exact_name_comes_first_then_longer_names_that_start_with_it():
    assert suggest("hp2/01/125gsm/2440mm/1200-1450", SKUS)[:2] == [SKUS[0], SKUS[1]]


def test_every_typed_word_inside_the_name_matches_in_any_order():
    assert suggest("2440 hp2 125", SKUS) == [SKUS[0], SKUS[1]]
    assert suggest("tsl", SKUS) == [SKUS[3]]


def test_a_typo_still_finds_the_planta_and_hopeless_text_finds_nothing():
    assert suggest("scn", PLANTAS) == ["SCAN"]
    assert suggest("qqqq", PLANTAS) == []


def test_empty_query_returns_the_first_options_and_limit_is_respected():
    assert suggest("", PLANTAS, limit=2) == ["SALC", "SCAN"]
    assert len(suggest("hp2", SKUS, limit=1)) == 1


def test_resolve_needs_the_full_name_but_ignores_case_and_punctuation():
    assert resolve("scan", PLANTAS) == "SCAN"
    assert resolve("hp2 01 125gsm 2440mm 1200 1450", SKUS) == SKUS[0]
    assert resolve("sca", PLANTAS) is None

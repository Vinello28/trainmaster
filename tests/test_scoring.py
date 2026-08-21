from __future__ import annotations

import pytest

from trainmaster.scoring import (
    ExactFieldScorer,
    RecursiveFieldScorer,
    SampleScore,
    aggregate,
    get_scorer,
    parse_prediction,
    score_sample,
)

# --- RecursiveFieldScorer -----------------------------------------------------------

_NESTED_GROUND_TRUTH = {
    "a": 1,
    "b": {"c": "x"},
    "items": [{"n": 1}, {"n": 2}],
}


def test_recursive_scorer_exact_match_is_perfect() -> None:
    scorer = RecursiveFieldScorer(tolerance=0.01)
    precision, recall, f1 = scorer.score(_NESTED_GROUND_TRUTH, _NESTED_GROUND_TRUTH)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_recursive_scorer_missing_key_lowers_recall_only() -> None:
    predicted = {"a": 1, "items": [{"n": 1}, {"n": 2}]}  # manca "b.c"
    scorer = RecursiveFieldScorer(tolerance=0.01)
    precision, recall, f1 = scorer.score(predicted, _NESTED_GROUND_TRUTH)

    assert precision == 1.0  # tutto ciò che è stato predetto è corretto
    assert recall == pytest.approx(3 / 4)
    assert 0 < f1 < 1


def test_recursive_scorer_hallucinated_key_lowers_precision_only() -> None:
    predicted = {**_NESTED_GROUND_TRUTH, "extra": {"ghost": True}}
    scorer = RecursiveFieldScorer(tolerance=0.01)
    precision, recall, f1 = scorer.score(predicted, _NESTED_GROUND_TRUTH)

    assert recall == 1.0
    assert precision < 1.0


def test_recursive_scorer_numeric_tolerance() -> None:
    scorer = RecursiveFieldScorer(tolerance=0.01)

    within = scorer.score({"weight": 100.5}, {"weight": 100.0})
    outside = scorer.score({"weight": 102.0}, {"weight": 100.0})

    assert within == (1.0, 1.0, 1.0)
    assert outside != (1.0, 1.0, 1.0)


def test_recursive_scorer_list_missing_row_lowers_recall() -> None:
    ground_truth = {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}
    predicted = {"items": [{"n": 1}, {"n": 2}]}
    scorer = RecursiveFieldScorer()
    precision, recall, _ = scorer.score(predicted, ground_truth)

    assert precision == 1.0
    assert recall == pytest.approx(2 / 3)


def test_recursive_scorer_list_extra_row_lowers_precision() -> None:
    ground_truth = {"items": [{"n": 1}, {"n": 2}]}
    predicted = {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}
    scorer = RecursiveFieldScorer()
    precision, recall, _ = scorer.score(predicted, ground_truth)

    assert recall == 1.0
    assert precision == pytest.approx(2 / 3)


def test_recursive_scorer_both_empty_is_perfect() -> None:
    assert RecursiveFieldScorer().score({}, {}) == (1.0, 1.0, 1.0)


def test_recursive_scorer_none_predicted_scores_zero() -> None:
    precision, recall, f1 = RecursiveFieldScorer().score(None, {"a": 1})
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


# --- ExactFieldScorer -----------------------------------------------------------------


def test_exact_field_scorer_match() -> None:
    scorer = ExactFieldScorer()
    result = scorer.score({"codice_fiscale": "RSSMRA80A01H501U"}, {"codice_fiscale": "RSSMRA80A01H501U"})
    assert result == (1.0, 1.0, 1.0)


def test_exact_field_scorer_mismatch() -> None:
    scorer = ExactFieldScorer()
    result = scorer.score({"codice_fiscale": "WRONG"}, {"codice_fiscale": "RSSMRA80A01H501U"})
    assert result == (0.0, 0.0, 0.0)


def test_exact_field_scorer_missing_field() -> None:
    scorer = ExactFieldScorer()
    result = scorer.score({}, {"codice_fiscale": "RSSMRA80A01H501U"})
    assert result == (0.0, 0.0, 0.0)


# --- parse_prediction -------------------------------------------------------------------


def test_parse_prediction_valid_json() -> None:
    value, parse_error = parse_prediction('{"a": 1}')
    assert value == {"a": 1}
    assert parse_error is False


def test_parse_prediction_markdown_fence() -> None:
    value, parse_error = parse_prediction('```json\n{"a": 1}\n```')
    assert value == {"a": 1}
    assert parse_error is False


def test_parse_prediction_extracts_json_from_surrounding_text() -> None:
    value, parse_error = parse_prediction('Sure! Here is the JSON: {"a": 1} Hope that helps!')
    assert value == {"a": 1}
    assert parse_error is False


def test_parse_prediction_garbage_never_raises() -> None:
    value, parse_error = parse_prediction("questo non e' assolutamente JSON")
    assert value is None
    assert parse_error is True


def test_parse_prediction_empty_string_never_raises() -> None:
    value, parse_error = parse_prediction("")
    assert value is None
    assert parse_error is True


# --- registry / get_scorer -----------------------------------------------------------


@pytest.mark.parametrize(
    "document_type", ["cargo_manifest", "camion_list", "veicols_list", "passenger_list"]
)
def test_get_scorer_page_documents_use_recursive_scorer(document_type: str) -> None:
    assert isinstance(get_scorer(document_type), RecursiveFieldScorer)


@pytest.mark.parametrize("document_type", ["carta_identita", "patente", "tessera_sanitaria"])
def test_get_scorer_identity_documents_use_exact_scorer(document_type: str) -> None:
    assert isinstance(get_scorer(document_type), ExactFieldScorer)


def test_get_scorer_falls_back_to_recursive_for_unknown_type() -> None:
    assert isinstance(get_scorer("some_future_document_type"), RecursiveFieldScorer)


# --- score_sample ---------------------------------------------------------------------


def test_score_sample_parse_error_scores_zero_without_raising() -> None:
    score = score_sample(
        id="s1",
        document_type="cargo_manifest",
        language="en",
        predicted_text="non e' JSON",
        ground_truth_text='{"a": 1}',
    )
    assert score.parse_error is True
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


def test_score_sample_valid_prediction() -> None:
    score = score_sample(
        id="s1",
        document_type="carta_identita",
        language="it",
        predicted_text='{"codice_fiscale": "RSSMRA80A01H501U"}',
        ground_truth_text='{"codice_fiscale": "RSSMRA80A01H501U"}',
    )
    assert score.parse_error is False
    assert score.f1 == 1.0


# --- aggregate --------------------------------------------------------------------------


def test_aggregate_computes_means_and_parse_error_rate_per_group() -> None:
    scores = [
        SampleScore("a", "cargo_manifest", "en", 1.0, 1.0, 1.0, parse_error=False),
        SampleScore("b", "cargo_manifest", "en", 0.0, 0.0, 0.0, parse_error=True),
        SampleScore("c", "carta_identita", "it", 1.0, 1.0, 1.0, parse_error=False),
    ]
    result = aggregate(scores)

    assert result["overall"]["n"] == 3
    assert result["overall"]["f1"] == pytest.approx((1.0 + 0.0 + 1.0) / 3)
    assert result["overall"]["parse_error_rate"] == pytest.approx(1 / 3)

    manifest_stats = result["by_document_type"]["cargo_manifest"]
    assert manifest_stats["n"] == 2
    assert manifest_stats["f1"] == pytest.approx(0.5)
    assert manifest_stats["parse_error_rate"] == pytest.approx(0.5)

    identity_stats = result["by_document_type"]["carta_identita"]
    assert identity_stats["n"] == 1
    assert identity_stats["parse_error_rate"] == 0.0

    en_stats = result["by_language"]["en"]
    assert en_stats["n"] == 2
    it_stats = result["by_language"]["it"]
    assert it_stats["n"] == 1


def test_aggregate_empty_input_does_not_crash() -> None:
    result = aggregate([])
    assert result["overall"]["n"] == 0
    assert result["by_document_type"] == {}
    assert result["by_language"] == {}

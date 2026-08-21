from __future__ import annotations

import json

from PIL import Image as PILImage

from trainmaster.inference import Prediction
from trainmaster.report import render_report
from trainmaster.scoring import SampleScore


def _prediction(
    id_: str,
    predicted_text: str,
    ground_truth: dict,
    *,
    document_type: str = "cargo_manifest",
    language: str = "en",
) -> Prediction:
    return Prediction(
        id=id_,
        document_type=document_type,
        language=language,
        instruction="Extract the structured data and return it as JSON.",
        ground_truth=json.dumps(ground_truth, ensure_ascii=False),
        predicted_text=predicted_text,
        image=PILImage.new("RGB", (4, 4), (10, 20, 30)),
    )


def test_render_report_includes_summary_and_examples() -> None:
    prediction = _prediction("sample_0001", '{"a": 1}', {"a": 1})
    score = SampleScore("sample_0001", "cargo_manifest", "en", 1.0, 1.0, 1.0, parse_error=False)

    html = render_report([prediction], [score], max_examples=10)

    assert "sample_0001" in html
    assert "cargo_manifest" in html
    assert "data:image/png;base64," in html


def test_render_report_handles_parse_error_without_raising() -> None:
    prediction = _prediction("bad_0001", "questo non e' JSON", {"a": 1})
    score = SampleScore("bad_0001", "cargo_manifest", "en", 0.0, 0.0, 0.0, parse_error=True)

    html = render_report([prediction], [score], max_examples=10)

    assert "bad_0001" in html
    assert "JSON malformato" in html


def test_render_report_escapes_html_in_predicted_text() -> None:
    prediction = _prediction("xss_0001", "<script>alert(1)</script>", {"a": 1})
    score = SampleScore("xss_0001", "cargo_manifest", "en", 0.0, 0.0, 0.0, parse_error=True)

    html = render_report([prediction], [score], max_examples=10)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_report_orders_worst_first() -> None:
    good = _prediction("good", '{"a": 1}', {"a": 1})
    bad = _prediction("bad", "not json at all", {"a": 1})
    good_score = SampleScore("good", "cargo_manifest", "en", 1.0, 1.0, 1.0, parse_error=False)
    bad_score = SampleScore("bad", "cargo_manifest", "en", 0.0, 0.0, 0.0, parse_error=True)

    html = render_report([good, bad], [good_score, bad_score], max_examples=10)

    assert html.index("<strong>bad</strong>") < html.index("<strong>good</strong>")


def test_render_report_respects_max_examples() -> None:
    predictions = [_prediction(f"s{i}", '{"a": 1}', {"a": 1}) for i in range(5)]
    scores = [SampleScore(f"s{i}", "cargo_manifest", "en", 1.0, 1.0, 1.0, parse_error=False) for i in range(5)]

    html = render_report(predictions, scores, max_examples=2)

    assert sum(html.count(f"s{i}") for i in range(5)) >= 2
    assert html.count('class="example"') == 2

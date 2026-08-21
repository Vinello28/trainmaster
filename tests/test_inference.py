"""Solo le parti pure di inference.py (dataclass Prediction, round-trip Parquet):
run_inference stesso richiede torch/unsloth ed è esercitato indirettamente in
test_pipeline.py tramite dependency injection."""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from trainmaster.inference import Prediction, load_predictions


def test_prediction_is_a_plain_dataclass() -> None:
    prediction = Prediction(
        id="s1",
        document_type="cargo_manifest",
        language="en",
        instruction="Extract the structured data and return it as JSON.",
        ground_truth='{"a": 1}',
        predicted_text='{"a": 1}',
        image=PILImage.new("RGB", (2, 2)),
    )
    assert prediction.id == "s1"
    assert prediction.document_type == "cargo_manifest"


def test_load_predictions_roundtrip(tmp_path: Path) -> None:
    from datasets import Dataset, Features, Image, Value

    features = Features(
        {
            "id": Value("string"),
            "document_type": Value("string"),
            "language": Value("string"),
            "instruction": Value("string"),
            "ground_truth": Value("string"),
            "predicted_text": Value("string"),
            "image": Image(),
        }
    )
    rows = [
        {
            "id": "s1",
            "document_type": "cargo_manifest",
            "language": "en",
            "instruction": "Extract the structured data and return it as JSON.",
            "ground_truth": '{"a": 1}',
            "predicted_text": '{"a": 1}',
            "image": PILImage.new("RGB", (2, 2), (5, 5, 5)),
        }
    ]
    path = tmp_path / "predictions.parquet"
    Dataset.from_list(rows, features=features).to_parquet(str(path))

    predictions = load_predictions(str(path))

    assert len(predictions) == 1
    assert predictions[0].id == "s1"
    assert predictions[0].predicted_text == '{"a": 1}'
    assert predictions[0].image.size == (2, 2)

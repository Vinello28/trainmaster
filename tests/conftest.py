"""Fixture condivise: dataset Parquet in-memory con lo schema reale di synThor
(verificato leggendo per intero ``synthor/export.py``), niente GPU/torch richiesti."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image as PILImage

def _features():
    """Stesso schema dichiarato in synthor/export.py::_write_parquet — riprodotto qui
    apposta (niente import da synThor, solo contratto dati) per costruire fixture
    Parquet realistiche."""
    from datasets import Features, Image, Value

    return Features(
        {
            "id": Value("string"),
            "language": Value("string"),
            "document_type": Value("string"),
            "image": Image(),
            "instruction": Value("string"),
            "ground_truth": Value("string"),
            "messages": [
                {
                    "role": Value("string"),
                    "content": [
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                        }
                    ],
                }
            ],
        }
    )


def make_row(index: int, document_type: str = "cargo_manifest", language: str = "en") -> dict[str, Any]:
    """Una riga con lo schema Parquet reale, ground truth deterministico dall'indice."""
    ground_truth: dict[str, Any] = {"document_type": document_type, "value": index, "label": f"item-{index}"}
    ground_truth_text = json.dumps(ground_truth, ensure_ascii=False)
    instruction = f"Extract the structured data from this {document_type} and return it as JSON."
    return {
        "id": f"sample_{index:04d}",
        "language": language,
        "document_type": document_type,
        "image": PILImage.new("RGB", (8, 8), ((index * 37) % 255, 60, 90)),
        "instruction": instruction,
        "ground_truth": ground_truth_text,
        # Placeholder immagine deliberatamente vuoto, come in synthor/export.py::_parquet_row:
        # to_conversation() deve ignorarlo e ricostruire da 'image'.
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "text": ""},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ground_truth_text}],
            },
        ],
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    from datasets import Dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=_features()).to_parquet(str(path))
    return path


@pytest.fixture
def sample_rows() -> list[dict[str, Any]]:
    doc_types = ["cargo_manifest", "camion_list", "carta_identita", "cargo_manifest", "passenger_list"]
    languages = ["en", "it", "el", "tr", "en"]
    return [make_row(i, doc_types[i], languages[i]) for i in range(len(doc_types))]


@pytest.fixture
def dataset_dir(tmp_path: Path, sample_rows: list[dict[str, Any]]) -> Path:
    """Un file singolo per split, come ``export_dataset.py --format parquet``."""
    data_dir = tmp_path / "data"
    write_parquet(data_dir / "run_train.parquet", sample_rows[:3])
    write_parquet(data_dir / "run_validation.parquet", sample_rows[3:])
    return data_dir


@pytest.fixture
def sharded_dataset_dir(tmp_path: Path, sample_rows: list[dict[str, Any]]) -> Path:
    """Split di training spezzato in due shard, come ``export_dataset.py --format unsloth``."""
    data_dir = tmp_path / "shards"
    write_parquet(data_dir / "run_train-00001-of-00002.parquet", sample_rows[:2])
    write_parquet(data_dir / "run_train-00002-of-00002.parquet", sample_rows[2:])
    return data_dir


@pytest.fixture
def run_config_yaml(tmp_path: Path, dataset_dir: Path) -> Path:
    """Un file YAML RunConfig minimo, valido, che punta a ``dataset_dir``."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: test-run
data:
  train_files: "{(dataset_dir / '*_train*.parquet').as_posix()}"
  validation_files: "{(dataset_dir / '*_validation*.parquet').as_posix()}"
training:
  output_dir: "{(tmp_path / 'runs').as_posix()}"
""",
        encoding="utf-8",
    )
    return config_path

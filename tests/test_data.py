from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trainmaster.data import build_conversations, load_split, to_conversation


def test_load_split_single_file(dataset_dir: Path) -> None:
    dataset = load_split(str(dataset_dir / "*_train*.parquet"))
    assert len(dataset) == 3
    expected_columns = {"id", "language", "document_type", "image", "instruction", "ground_truth"}
    assert expected_columns <= set(dataset.column_names)


def test_load_split_respects_max_samples(dataset_dir: Path) -> None:
    dataset = load_split(str(dataset_dir / "*_train*.parquet"), max_samples=1)
    assert len(dataset) == 1


def test_load_split_multi_shard(sharded_dataset_dir: Path) -> None:
    # sample_rows ha 5 righe totali, spezzate 2+3 fra i due shard.
    dataset = load_split(str(sharded_dataset_dir / "*_train*.parquet"))
    assert len(dataset) == 5


def test_load_split_missing_pattern_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split(str(tmp_path / "nope-*.parquet"))


def test_to_conversation_ignores_empty_messages_placeholder(sample_rows: list[dict[str, Any]]) -> None:
    sample = sample_rows[0]
    # Precondizione sulla fixture: il placeholder immagine dentro 'messages' è vuoto,
    # esattamente come lo produce synthor/export.py::_parquet_row.
    assert sample["messages"][0]["content"][1]["text"] == ""

    conversation = to_conversation(sample)
    user_content = conversation["messages"][0]["content"]
    image_entry = next(c for c in user_content if c["type"] == "image")

    assert "text" not in image_entry
    assert image_entry["image"] is sample["image"]


def test_to_conversation_uses_instruction_and_ground_truth(sample_rows: list[dict[str, Any]]) -> None:
    sample = sample_rows[0]
    conversation = to_conversation(sample)

    assert conversation["messages"][0]["content"][0] == {"type": "text", "text": sample["instruction"]}
    assert conversation["messages"][1]["content"][0] == {"type": "text", "text": sample["ground_truth"]}


def test_build_conversations_from_loaded_dataset(dataset_dir: Path) -> None:
    dataset = load_split(str(dataset_dir / "*_train*.parquet"))
    conversations = build_conversations(dataset)

    assert len(conversations) == len(dataset)
    assert all(set(c.keys()) == {"messages"} for c in conversations)


class _CountingDataset:
    """Sorgente fake per accertare che ``build_conversations`` non decodifichi/acceda a
    un campione finché non viene letto per indice (RAM regression del 2026-08-22:
    build_conversations era eager e materializzava l'intero split in anticipo)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.getitem_calls = 0

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.getitem_calls += 1
        return self._rows[index]


def test_build_conversations_is_lazy(sample_rows: list[dict[str, Any]]) -> None:
    source = _CountingDataset(sample_rows)
    conversations = build_conversations(source)

    assert len(conversations) == len(sample_rows)
    assert source.getitem_calls == 0

    conversations[0]
    assert source.getitem_calls == 1

    conversations[1]
    assert source.getitem_calls == 2

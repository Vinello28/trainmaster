from __future__ import annotations

from pathlib import Path

from trainmaster.config import load_config
from trainmaster.model_card import render_model_card, write_model_card


def test_render_model_card_includes_frontmatter_and_base_model(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    card = render_model_card(config, metrics=None, has_adapter=True)

    assert card.startswith("---\n")
    assert f"base_model: {config.model.model_id}" in card
    assert "library_name: peft" in card
    assert config.name in card


def test_render_model_card_without_adapter_uses_transformers_library(run_config_yaml: Path) -> None:
    """has_adapter=False copre sia il full fine-tuning sia un checkpoint LoRA fuso con
    trainmaster-export: in entrambi i casi non c'è più un adapter separato da dichiarare."""
    config = load_config(run_config_yaml)
    card = render_model_card(config, metrics=None, has_adapter=False)

    assert "library_name: transformers" in card
    assert "nessun adapter separato" in card


def test_render_model_card_omits_metrics_section_when_none(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    card = render_model_card(config, metrics=None, has_adapter=True)

    assert "Metriche di validazione" not in card


def test_render_model_card_includes_metrics_table_when_present(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    metrics = {
        "overall": {"n": 3, "precision": 1.0, "recall": 1.0, "f1": 1.0, "parse_error_rate": 0.0},
        "by_document_type": {
            "cargo_manifest": {"n": 3, "precision": 1.0, "recall": 1.0, "f1": 1.0, "parse_error_rate": 0.0}
        },
        "by_language": {
            "en": {"n": 3, "precision": 1.0, "recall": 1.0, "f1": 1.0, "parse_error_rate": 0.0}
        },
    }

    card = render_model_card(config, metrics, has_adapter=True)

    assert "Metriche di validazione" in card
    assert "cargo_manifest" in card
    assert "| en |" in card


def test_write_model_card_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint" / "README.md"
    write_model_card(path, "# hello")

    assert path.read_text(encoding="utf-8") == "# hello"

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from trainmaster.config import (
    DataConfig,
    ModelConfig,
    RunConfig,
    dump_config,
    load_config,
    parse_cli_overrides,
)


def test_load_minimal_config_applies_defaults(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)

    assert config.name == "test-run"
    assert isinstance(config.data, DataConfig)
    assert config.model == ModelConfig()  # sezione omessa -> tutti i default
    assert config.lora.r == 16
    assert isinstance(config.training.output_dir, Path)


def test_missing_required_field_raises_clear_error(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("name: broken\n", encoding="utf-8")  # manca 'data'

    with pytest.raises(ValueError, match="train_files"):
        load_config(config_path)


def test_cli_overrides_coerce_types(run_config_yaml: Path) -> None:
    overrides = parse_cli_overrides(
        [
            "training.learning_rate=0.0001",
            "data.max_samples=8",
            "training.bf16=false",
        ]
    )
    config = load_config(run_config_yaml, overrides)

    assert config.training.learning_rate == pytest.approx(0.0001)
    assert isinstance(config.data.max_samples, int)
    assert config.data.max_samples == 8
    assert config.training.bf16 is False


def test_parse_cli_overrides_rejects_missing_equals() -> None:
    with pytest.raises(ValueError):
        parse_cli_overrides(["training.learning_rate"])


def test_run_config_is_frozen(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "mutated"  # type: ignore[misc]


def test_dump_and_reload_roundtrip(run_config_yaml: Path, tmp_path: Path) -> None:
    config = load_config(run_config_yaml)
    dumped_path = tmp_path / "dumped.yaml"
    dump_config(config, dumped_path)

    reloaded = load_config(dumped_path)

    assert reloaded == config

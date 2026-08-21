"""Verifica l'orchestrazione con model_loader/trainer_builder/inference_runner fake
iniettati via dependency injection: mai import di torch/unsloth."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

from trainmaster.config import load_config
from trainmaster.inference import Prediction
from trainmaster.pipeline import run_evaluation, run_training


class FakeHandle:
    def __init__(self) -> None:
        self.saved_to: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_to = Path(path)
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeTrainer:
    def __init__(self, handle: FakeHandle, train_conversations, eval_conversations, training_config) -> None:
        self.handle = handle
        self.train_conversations = train_conversations
        self.eval_conversations = eval_conversations
        self.trained = False

    def train(self) -> None:
        self.trained = True


def fake_model_loader(model_config, lora, *, for_training, checkpoint=None, seed=0) -> FakeHandle:
    return FakeHandle()


def fake_trainer_builder(handle, train_conversations, eval_conversations, training_config) -> FakeTrainer:
    return FakeTrainer(handle, train_conversations, eval_conversations, training_config)


def fake_inference_runner(handle, dataset, evaluation) -> Iterator[Prediction]:
    for sample in dataset:
        # "predizione" = ground truth: rende gli score prevedibili (F1 == 1.0) senza
        # bisogno di un modello vero.
        yield Prediction(
            id=sample["id"],
            document_type=sample["document_type"],
            language=sample["language"],
            instruction=sample["instruction"],
            ground_truth=sample["ground_truth"],
            predicted_text=sample["ground_truth"],
            image=sample["image"],
        )


def test_run_training_writes_checkpoint_and_config(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    checkpoint_dir = run_training(
        config, model_loader=fake_model_loader, trainer_builder=fake_trainer_builder
    )

    assert checkpoint_dir.exists()
    assert (config.training.output_dir / "config.yaml").exists()


def test_run_evaluation_writes_predictions_metrics_and_report(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    checkpoint_dir = config.training.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = run_evaluation(
        config,
        checkpoint_dir,
        model_loader=fake_model_loader,
        inference_runner=fake_inference_runner,
    )

    assert result.predictions_path.exists()
    assert (config.training.output_dir / "eval" / "metrics.json").exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.aggregate["overall"]["f1"] == 1.0


def test_run_training_then_evaluation_end_to_end(run_config_yaml: Path) -> None:
    config = load_config(run_config_yaml)
    checkpoint_dir = run_training(
        config, model_loader=fake_model_loader, trainer_builder=fake_trainer_builder
    )
    result = run_evaluation(
        config,
        checkpoint_dir,
        model_loader=fake_model_loader,
        inference_runner=fake_inference_runner,
    )

    assert result.aggregate["overall"]["n"] > 0


def test_run_evaluation_predictions_are_reloadable(run_config_yaml: Path) -> None:
    from trainmaster.inference import load_predictions

    config = load_config(run_config_yaml)
    checkpoint_dir = config.training.output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = run_evaluation(
        config,
        checkpoint_dir,
        model_loader=fake_model_loader,
        inference_runner=fake_inference_runner,
    )

    reloaded = load_predictions(str(result.predictions_path))
    assert {p.id for p in reloaded} == {s.id for s in result.scores}


def test_pipeline_never_imports_torch(run_config_yaml: Path) -> None:
    """Prova diretta del vincolo architetturale: model.py/inference.py non devono
    importare torch a livello modulo, altrimenti anche solo `import trainmaster.pipeline`
    romperebbe la testabilità senza GPU."""
    assert "torch" not in sys.modules

    config = load_config(run_config_yaml)
    checkpoint_dir = run_training(
        config, model_loader=fake_model_loader, trainer_builder=fake_trainer_builder
    )
    run_evaluation(
        config,
        checkpoint_dir,
        model_loader=fake_model_loader,
        inference_runner=fake_inference_runner,
    )

    assert "torch" not in sys.modules

"""Orchestrazione: prepara dati -> addestra -> salva -> valuta -> report.

Composizione di funzioni con dependency injection sui punti che toccano la GPU
(``model_loader``/``trainer_builder``/``inference_runner``), non una gerarchia Template
Method: esiste un'unica forma dell'algoritmo, non più varianti che devono fare override
di singoli step — una classe qui sarebbe astrazione prematura contro "Simplicity First"
(CLAUDE.md del progetto). I parametri iniettabili sono il punto in cui
``tests/test_pipeline.py`` sostituisce ``model.py``/``inference.py`` con dei fake, senza
mai importare ``torch``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from trainmaster import data, model, report, scoring
from trainmaster.config import RunConfig, dump_config
from trainmaster.inference import Prediction, run_inference

__all__ = ["EvaluationResult", "run_training", "run_evaluation", "run"]


@dataclass(frozen=True)
class EvaluationResult:
    scores: tuple[scoring.SampleScore, ...]
    aggregate: dict[str, Any]
    predictions_path: Path
    report_path: Path | None


def run_training(
    config: RunConfig,
    *,
    model_loader: Callable[..., Any] = model.load_model,
    trainer_builder: Callable[..., Any] = model.build_trainer,
) -> Path:
    """Prepara i dati, addestra e salva il checkpoint. Ritorna la directory del checkpoint."""
    train_dataset = data.load_split(config.data.train_files, max_samples=config.data.max_samples)
    train_conversations = data.build_conversations(train_dataset)

    eval_conversations = None
    if config.training.eval_strategy != "no":
        eval_dataset = data.load_split(
            config.data.validation_files, max_samples=config.data.max_samples
        )
        eval_conversations = data.build_conversations(eval_dataset)

    handle = model_loader(config.model, config.lora, for_training=True, seed=config.training.seed)
    trainer = trainer_builder(handle, train_conversations, eval_conversations, config.training)
    trainer.train()

    checkpoint_dir = Path(config.training.output_dir) / "checkpoint"
    handle.save(checkpoint_dir)
    dump_config(config, Path(config.training.output_dir) / "config.yaml")
    return checkpoint_dir


def run_evaluation(
    config: RunConfig,
    checkpoint_dir: Path,
    *,
    model_loader: Callable[..., Any] = model.load_model,
    inference_runner: Callable[..., Any] = run_inference,
) -> EvaluationResult:
    """Carica il checkpoint, genera predizioni sulla validazione, punteggia e produce
    il report. Le predizioni grezze vengono persistite **prima** dello scoring: separa
    "genera predizioni" (serve GPU) da "punteggia e produci report" (CPU puro), il che
    rende possibile iterare sulla logica di scoring via ``score.py`` senza mai
    ricaricare il modello."""
    eval_dir = Path(config.training.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    validation_dataset = data.load_split(
        config.data.validation_files, max_samples=config.data.max_samples
    )
    handle = model_loader(config.model, None, for_training=False, checkpoint=checkpoint_dir)
    predictions = list(inference_runner(handle, validation_dataset, config.evaluation))

    predictions_path = eval_dir / "predictions.parquet"
    _write_predictions(predictions, predictions_path)

    scores = [
        scoring.score_sample(
            id=p.id,
            document_type=p.document_type,
            language=p.language,
            predicted_text=p.predicted_text,
            ground_truth_text=p.ground_truth,
            tolerance=config.evaluation.numeric_tolerance,
        )
        for p in predictions
    ]
    aggregated = scoring.aggregate(scores)
    (eval_dir / "metrics.json").write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    report_path = None
    if config.report.enabled:
        html = report.render_report(predictions, scores, max_examples=config.report.max_examples)
        report_path = eval_dir / "report.html"
        report.write_report(report_path, html)

    return EvaluationResult(
        scores=tuple(scores),
        aggregate=aggregated,
        predictions_path=predictions_path,
        report_path=report_path,
    )


def _write_predictions(predictions: list[Prediction], path: Path) -> None:
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
            "id": p.id,
            "document_type": p.document_type,
            "language": p.language,
            "instruction": p.instruction,
            "ground_truth": p.ground_truth,
            "predicted_text": p.predicted_text,
            "image": p.image,
        }
        for p in predictions
    ]
    Dataset.from_list(rows, features=features).to_parquet(str(path))


def run(config: RunConfig) -> EvaluationResult | None:
    """Pipeline completa: training, poi valutazione se abilitata in config."""
    checkpoint_dir = run_training(config)
    if not config.evaluation.enabled:
        return None
    return run_evaluation(config, checkpoint_dir)

"""Generazione delle predizioni su uno split (tipicamente la validazione).

Frontiera GPU insieme a ``model.py``: ``torch`` viene importato solo dentro il corpo
delle funzioni, mai a livello modulo, così ``import trainmaster.inference`` resta legale
in un ambiente senza lo stack di training (serve a ``pipeline.py``/``report.py``, che lo
importano per il tipo ``Prediction`` e per iniettarlo via dependency injection senza mai
toccare ``torch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:
    from PIL.Image import Image

    from trainmaster.config import EvaluationConfig
    from trainmaster.model import UnslothModelHandle

__all__ = ["Prediction", "run_inference", "load_predictions"]


@dataclass(frozen=True)
class Prediction:
    id: str
    document_type: str
    language: str
    instruction: str
    ground_truth: str
    predicted_text: str
    image: "Image"


def run_inference(
    handle: "UnslothModelHandle",
    dataset: Iterable[dict[str, Any]],
    evaluation: "EvaluationConfig",
) -> Iterator[Prediction]:
    """Genera una predizione per ciascun campione dello split.

    Loop semplice (batch_size=1, ``torch.no_grad()``): batching reale è un miglioramento
    futuro fuori scope, non richiesto per la validazione periodica di un piccolo VLM.
    Rispetta ``evaluation.sample_limit`` se impostato.
    """
    import torch  # import locale: vedi nota di modulo

    with torch.no_grad():
        for count, sample in enumerate(dataset):
            if evaluation.sample_limit is not None and count >= evaluation.sample_limit:
                break
            predicted_text = handle.generate(
                sample["image"],
                sample["instruction"],
                max_new_tokens=evaluation.max_new_tokens,
            )
            yield Prediction(
                id=sample["id"],
                document_type=sample.get("document_type", ""),
                language=sample.get("language", ""),
                instruction=sample["instruction"],
                ground_truth=sample["ground_truth"],
                predicted_text=predicted_text,
                image=sample["image"],
            )


def load_predictions(path: str) -> list[Prediction]:
    """Ricarica le predizioni salvate da ``pipeline.run_evaluation`` in
    ``<output_dir>/eval/predictions.parquet``. Nessun bisogno di ``torch``: usata da
    ``score.py`` per iterare su scoring/report senza mai ricaricare il modello."""
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(path), split="train")
    return [
        Prediction(
            id=row["id"],
            document_type=row["document_type"],
            language=row["language"],
            instruction=row["instruction"],
            ground_truth=row["ground_truth"],
            predicted_text=row["predicted_text"],
            image=row["image"],
        )
        for row in dataset
    ]

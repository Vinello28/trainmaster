"""Lettura dei Parquet esportati da synThor e ricostruzione delle conversazioni.

Nessuna dipendenza da ``torch``/``unsloth``: questo modulo gira ovunque sia installato
``datasets`` (il core CPU-safe del progetto), incluso in ``pytest`` senza GPU.

Contratto dati (verificato leggendo ``synThor/synthor/export.py`` per intero): ogni riga
ha le colonne ``id``, ``language``, ``document_type``, ``image`` (già ``PIL.Image``
decodificato da ``datasets.Image()``), ``instruction``, ``ground_truth`` (JSON compatto)
e ``messages`` — quest'ultima con un placeholder immagine **vuoto**
(``{"type": "image", "text": ""}``), perché la colonna ``image`` di primo livello è la
fonte autorevole. ``to_conversation`` ricostruisce quindi i messaggi da zero e ignora
deliberatamente la sotto-struttura immagine di ``sample["messages"]``.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Iterable

__all__ = ["load_split", "to_conversation", "build_conversations"]


def load_split(file_pattern: str, *, max_samples: int | None = None) -> Any:
    """Carica uno split come ``datasets.Dataset`` da uno o più file Parquet.

    ``file_pattern`` è un glob (es. ``"data/*_train*.parquet"``): copre sia il caso
    singolo file (``--format parquet``) sia gli shard multipli
    (``-00001-of-00003.parquet``) prodotti da ``--format unsloth``.
    """
    from datasets import load_dataset  # import locale: pesante, non serve altrove

    files = sorted(glob.glob(str(file_pattern)))
    if not files:
        raise FileNotFoundError(f"nessun file Parquet trovato per il pattern {file_pattern!r}")

    dataset = load_dataset("parquet", data_files=files, split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def to_conversation(sample: dict[str, Any]) -> dict[str, Any]:
    """Un campione nel formato *messages* atteso dai notebook vision di Unsloth.

    Costruita da ``image``/``instruction``/``ground_truth``, non da ``messages``: vedi
    la nota di modulo sul placeholder immagine vuoto.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["instruction"]},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["ground_truth"]}],
            },
        ]
    }


def build_conversations(dataset: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converte l'intero split in una lista Python di conversazioni.

    Deliberatamente una lista e non un ``Dataset.map()``: la struttura ``content`` è
    eterogenea fra i turni (``{"type": "text", "text": ...}`` vs
    ``{"type": "image", "image": ...}``), e Arrow richiede uno schema uniforme per
    colonna — materializzarla in un Dataset romperebbe l'inferenza di tipo. È lo stesso
    pattern (lista di dict passata direttamente a ``SFTTrainer``) documentato nel README
    di synThor per l'esportazione ``imagefolder``.
    """
    return [to_conversation(sample) for sample in dataset]

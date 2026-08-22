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
from typing import Any, Sequence

__all__ = ["load_split", "to_conversation", "build_conversations", "ConversationView"]


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


class ConversationView(Sequence[dict[str, Any]]):
    """Vista lazy su uno split: converte un campione in conversazione solo quando il suo
    indice viene letto, invece di decodificare in anticipo tutte le immagini dello split
    (train *e* validation restano entrambi residenti in RAM per l'intero training se
    costruiti eager — con dataset reali a piena risoluzione questo satura la RAM di
    sistema anche prima che il training inizi, si veda l'incidente del 2026-08-22).

    Espone solo ``__len__``/``__getitem__``: è il contratto minimo richiesto da
    ``SFTTrainer`` (passato come ``train_dataset``/``eval_dataset`` con
    ``dataset_kwargs={"skip_prepare_dataset": True}``, vedi ``model.build_trainer``),
    quindi sostituisce una ``list`` senza richiedere modifiche al trainer.
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        return to_conversation(self._dataset[index])


def build_conversations(dataset: Any) -> Sequence[dict[str, Any]]:
    """Vista lazy sull'intero split, una conversazione per campione.

    Deliberatamente non un ``Dataset.map()`` eager: la struttura ``content`` è
    eterogenea fra i turni (``{"type": "text", "text": ...}`` vs
    ``{"type": "image", "image": ...}``), e Arrow richiede uno schema uniforme per
    colonna — materializzarla in un Dataset romperebbe l'inferenza di tipo. Ma neanche
    una lista Python eager: forzerebbe la decodifica di ogni immagine dello split in
    anticipo, tenendola in RAM per tutta la durata del training (vedi
    ``ConversationView``). ``dataset`` resta il ``datasets.Dataset`` restituito da
    ``load_split`` (memory-mapped, decodifica l'immagine solo all'accesso).
    """
    return ConversationView(dataset)

"""Valutazione semantica: confronto JSON predetto vs ground truth, per document_type.

Nessuna dipendenza da ``torch``/``unsloth``: logica pura, testabile senza GPU.

I 7 ``document_type`` di synThor richiedono due famiglie di confronto (vedi il README di
synThor, sezione "Schemi del ground truth"): ``cargo_manifest`` (annidato) e le tre liste
d'imbarco (piatte, liste di dict) si prestano a un confronto ricorsivo campo-per-campo;
``carta_identita``/``patente``/``tessera_sanitaria`` hanno un solo campo
(``codice_fiscale``) e vogliono un match esatto stringa. Il registry sotto rispecchia lo
stesso pattern già usato da synThor per l'``INSTRUCTIONS`` per-document_type in
``synthor/export.py`` (dict con fallback), qui applicato allo scoring.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Iterable, Protocol

__all__ = [
    "SampleScore",
    "DocumentScorer",
    "RecursiveFieldScorer",
    "ExactFieldScorer",
    "get_scorer",
    "parse_prediction",
    "score_sample",
    "aggregate",
]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class SampleScore:
    id: str
    document_type: str
    language: str
    precision: float
    recall: float
    f1: float
    parse_error: bool


class DocumentScorer(Protocol):
    def score(self, predicted: Any, ground_truth: Any) -> tuple[float, float, float]:
        """Restituisce (precision, recall, f1) per una coppia predetto/ground_truth già
        parsati (non stringhe)."""
        ...


def parse_prediction(raw_text: str) -> tuple[Any | None, bool]:
    """Prova a estrarre JSON dal testo grezzo generato dal modello.

    Non solleva mai un'eccezione: un output malformato produce ``(None, True)`` invece
    di far crashare l'intera valutazione su un singolo campione. Prova, in ordine: il
    testo intero, il contenuto di un fence ```` ```json ... ``` ````, e la sottostringa
    fra la prima ``{`` e l'ultima ``}`` (copre preamboli/commenti chiacchierati intorno
    al JSON).
    """
    candidates = [raw_text.strip()]

    fence_match = _FENCE_RE.search(raw_text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate), False
        except (json.JSONDecodeError, ValueError):
            continue
    return None, True


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Appiattisce dict annidati e liste in ``{"a.b[0].c": leaf, ...}``."""
    if isinstance(obj, dict):
        flat: dict[str, Any] = {}
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten(value, child))
        return flat
    if isinstance(obj, list):
        flat = {}
        for index, value in enumerate(obj):
            flat.update(_flatten(value, f"{prefix}[{index}]"))
        return flat
    return {prefix: obj}


def _leaves_equal(predicted: Any, expected: Any, tolerance: float) -> bool:
    # bool prima del check numerico: in Python bool è sottoclasse di int.
    if isinstance(predicted, bool) or isinstance(expected, bool):
        return predicted == expected
    if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(predicted, expected, rel_tol=tolerance, abs_tol=1e-6)
    return predicted == expected


class RecursiveFieldScorer:
    """Confronto campo-per-campo su strutture annidate/liste, con tolleranza numerica.

    Limite noto, accettato per v1: il confronto delle liste (``cargo_items``, righe delle
    liste d'imbarco) è posizionale per indice — un riordino valido da parte del modello
    penalizza lo score. Non risolto qui per restare aderenti a YAGNI; se le metriche
    sembrano ingiustamente basse rispetto al report visivo, è il primo sospetto.
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def score(self, predicted: Any, ground_truth: Any) -> tuple[float, float, float]:
        predicted_flat = _flatten(predicted) if predicted is not None else {}
        ground_truth_flat = _flatten(ground_truth)

        if not ground_truth_flat and not predicted_flat:
            return 1.0, 1.0, 1.0

        correct = sum(
            1
            for path, expected in ground_truth_flat.items()
            if path in predicted_flat and _leaves_equal(predicted_flat[path], expected, self.tolerance)
        )
        precision = (
            correct / len(predicted_flat) if predicted_flat else (1.0 if not ground_truth_flat else 0.0)
        )
        recall = correct / len(ground_truth_flat) if ground_truth_flat else 1.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1


class ExactFieldScorer:
    """Match esatto su un singolo campo: i tre documenti d'identità italiani, il cui
    unico ground truth è ``codice_fiscale``."""

    def __init__(self, field_name: str = "codice_fiscale") -> None:
        self.field_name = field_name

    def score(self, predicted: Any, ground_truth: Any) -> tuple[float, float, float]:
        predicted_value = predicted.get(self.field_name) if isinstance(predicted, dict) else None
        expected_value = ground_truth.get(self.field_name) if isinstance(ground_truth, dict) else None
        matched = predicted_value is not None and predicted_value == expected_value
        outcome = 1.0 if matched else 0.0
        return outcome, outcome, outcome


#: Registry document_type -> factory(tolerance). Stesso pattern (dict + fallback) di
#: ``INSTRUCTIONS`` in synthor/export.py, qui per lo scoring invece che per il prompt.
_SCORER_FACTORIES: dict[str, Callable[[float], DocumentScorer]] = {
    "carta_identita": lambda _tolerance: ExactFieldScorer(),
    "patente": lambda _tolerance: ExactFieldScorer(),
    "tessera_sanitaria": lambda _tolerance: ExactFieldScorer(),
}


def get_scorer(document_type: str, *, tolerance: float = 0.01) -> DocumentScorer:
    """Ritorna la strategia di scoring per ``document_type``, con fallback al
    comparatore ricorsivo generico per qualunque tipo non mappato (incluso, per
    costruzione, ``cargo_manifest``/``camion_list``/``veicols_list``/``passenger_list``
    e ogni tipo futuro non ancora registrato)."""
    factory = _SCORER_FACTORIES.get(document_type, RecursiveFieldScorer)
    return factory(tolerance)


def score_sample(
    *,
    id: str,
    document_type: str,
    language: str,
    predicted_text: str,
    ground_truth_text: str,
    tolerance: float = 0.01,
) -> SampleScore:
    predicted, parse_error = parse_prediction(predicted_text)
    if parse_error:
        return SampleScore(
            id=id,
            document_type=document_type,
            language=language,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            parse_error=True,
        )
    ground_truth = json.loads(ground_truth_text)
    precision, recall, f1 = get_scorer(document_type, tolerance=tolerance).score(predicted, ground_truth)
    return SampleScore(
        id=id,
        document_type=document_type,
        language=language,
        precision=precision,
        recall=recall,
        f1=f1,
        parse_error=False,
    )


def _group_stats(items: list[SampleScore]) -> dict[str, float | int]:
    n = len(items)
    if n == 0:
        return {"n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "parse_error_rate": 0.0}
    return {
        "n": n,
        "precision": fmean(s.precision for s in items),
        "recall": fmean(s.recall for s in items),
        "f1": fmean(s.f1 for s in items),
        "parse_error_rate": sum(1 for s in items if s.parse_error) / n,
    }


def aggregate(scores: Iterable[SampleScore]) -> dict[str, Any]:
    """Aggrega per document_type e per language: n, precision/recall/f1 medi, tasso di
    parse_error. Usata sia dal report che dal comando ``score.py``/``evaluate.py``."""
    scores = list(scores)
    by_document_type: dict[str, list[SampleScore]] = {}
    by_language: dict[str, list[SampleScore]] = {}
    for s in scores:
        by_document_type.setdefault(s.document_type, []).append(s)
        by_language.setdefault(s.language, []).append(s)
    return {
        "overall": _group_stats(scores),
        "by_document_type": {k: _group_stats(v) for k, v in by_document_type.items()},
        "by_language": {k: _group_stats(v) for k, v in by_language.items()},
    }

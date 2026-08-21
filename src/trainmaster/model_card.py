"""Genera la model card (README.md) per la pubblicazione su Hugging Face Hub.

Nessuna dipendenza da ``torch``/``unsloth``/rete: legge solo ``RunConfig`` e
l'eventuale ``eval/metrics.json`` già scritti su disco da ``pipeline.py``. Usato da
``cli/publish_model_card.py`` nel flusso di pubblicazione manuale (vedi
README.md#pubblicazione-su-hugging-face): genera ``checkpoint/README.md`` prima che
l'utente lanci ``hf upload`` da terminale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trainmaster.config import RunConfig

__all__ = ["render_model_card", "write_model_card"]


def _metrics_table(aggregated: dict[str, Any]) -> str:
    rows_by_group: dict[str, dict[str, dict[str, Any]]] = {
        "document_type": aggregated.get("by_document_type", {}),
        "language": aggregated.get("by_language", {}),
    }

    def _table(title: str, rows: dict[str, dict[str, Any]]) -> str:
        if not rows:
            return ""
        header = f"### Per {title}\n\n| {title} | n | precision | recall | f1 | parse_error |\n"
        header += "|---|---|---|---|---|---|\n"
        body = "".join(
            f"| {key} | {stats['n']} | {stats['precision']:.3f} | {stats['recall']:.3f} "
            f"| {stats['f1']:.3f} | {stats['parse_error_rate']:.3f} |\n"
            for key, stats in sorted(rows.items())
        )
        return header + body + "\n"

    overall = aggregated.get("overall", {})
    overall_section = ""
    if overall:
        overall_section = (
            "### Totale\n\n| n | precision | recall | f1 | parse_error |\n|---|---|---|---|---|\n"
            f"| {overall['n']} | {overall['precision']:.3f} | {overall['recall']:.3f} "
            f"| {overall['f1']:.3f} | {overall['parse_error_rate']:.3f} |\n\n"
        )

    return (
        overall_section
        + _table("document_type", rows_by_group["document_type"])
        + _table("language", rows_by_group["language"])
    )


def render_model_card(config: RunConfig, metrics: dict[str, Any] | None) -> str:
    """Model card Markdown con frontmatter YAML, per un run ``config`` e le metriche
    opzionali di ``eval/metrics.json`` (``None`` se la valutazione non è ancora stata
    eseguita: la sezione metriche viene semplicemente omessa)."""
    library_name = "peft" if config.lora.enabled else "transformers"
    finetune_note = (
        f"LoRA (r={config.lora.r}, alpha={config.lora.alpha}, dropout={config.lora.dropout})"
        if config.lora.enabled
        else "full fine-tuning (nessun adapter, tutti i pesi allenati)"
    )

    frontmatter = (
        "---\n"
        f"base_model: {config.model.model_id}\n"
        f"library_name: {library_name}\n"
        "tags:\n"
        "  - unsloth\n"
        "  - vision-language\n"
        "  - document-extraction\n"
        "  - trainmaster\n"
        "---\n\n"
    )

    body = (
        f"# {config.name}\n\n"
        f"Fine-tuning di [`{config.model.model_id}`](https://huggingface.co/{config.model.model_id}) "
        "per estrazione documentale immagine → JSON strutturato, con "
        "[trainmaster](https://github.com/Vinello28/trainmaster) "
        "(dati sintetici synThor).\n\n"
        f"**Metodo**: {finetune_note}.\n\n"
    )

    metrics_section = ""
    if metrics is not None:
        metrics_section = "## Metriche di validazione\n\n" + _metrics_table(metrics)

    return frontmatter + body + metrics_section


def write_model_card(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

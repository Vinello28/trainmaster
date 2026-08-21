"""Report HTML delle predizioni: immagine, ground truth e predizione affiancati.

Nessuna dipendenza da ``torch``/``unsloth`` (solo ``Pillow``, già nel core CPU-safe).
Ispirato a ``tools/preview.py`` di synThor (pagina self-contained via data-URI, per la
revisione visiva manuale) ma riscritto da zero: qui affianca anche la predizione del
modello e ordina i casi per F1 **crescente**, così i difetti peggiori emergono in cima
invece di dover scorrere l'intero report — le metriche aggregate da sole nascondono
esattamente il tipo di difetto che synThor ha imparato a cercare guardando le immagini.
"""

from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from trainmaster import scoring

if TYPE_CHECKING:
    from trainmaster.inference import Prediction
    from trainmaster.scoring import SampleScore

__all__ = ["render_report", "write_report"]

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
h1, h2 { margin-bottom: 0.5rem; }
table.summary { border-collapse: collapse; margin-bottom: 2rem; }
table.summary th, table.summary td { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: right; }
table.summary th:first-child, table.summary td:first-child { text-align: left; }
.example { display: flex; gap: 1rem; border-top: 1px solid #ddd; padding: 1rem 0; align-items: flex-start; }
.example img { max-width: 320px; max-height: 420px; border: 1px solid #ccc; }
.example pre { background: #f6f6f6; padding: 0.5rem; max-width: 480px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 0.3rem; font-weight: 600; color: white; }
.badge.good { background: #2e7d32; }
.badge.bad { background: #c62828; }
.badge.mid { background: #ef6c00; }
"""


def _image_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return text


def _f1_badge_class(f1: float) -> str:
    if f1 >= 0.9:
        return "good"
    if f1 >= 0.5:
        return "mid"
    return "bad"


def _render_summary(aggregated: dict) -> str:
    def _table(title: str, rows: dict[str, dict]) -> str:
        body = "".join(
            f"<tr><td>{escape(str(key))}</td><td>{stats['n']}</td>"
            f"<td>{stats['precision']:.3f}</td><td>{stats['recall']:.3f}</td>"
            f"<td>{stats['f1']:.3f}</td><td>{stats['parse_error_rate']:.3f}</td></tr>"
            for key, stats in sorted(rows.items())
        )
        return (
            f"<h2>{escape(title)}</h2>"
            "<table class=\"summary\"><tr><th>chiave</th><th>n</th><th>precision</th>"
            f"<th>recall</th><th>f1</th><th>parse_error</th></tr>{body}</table>"
        )

    overall = aggregated["overall"]
    overall_html = (
        "<h2>Totale</h2>"
        "<table class=\"summary\"><tr><th>n</th><th>precision</th><th>recall</th>"
        "<th>f1</th><th>parse_error</th></tr>"
        f"<tr><td>{overall['n']}</td><td>{overall['precision']:.3f}</td>"
        f"<td>{overall['recall']:.3f}</td><td>{overall['f1']:.3f}</td>"
        f"<td>{overall['parse_error_rate']:.3f}</td></tr></table>"
    )
    return (
        overall_html
        + _table("Per document_type", aggregated["by_document_type"])
        + _table("Per language", aggregated["by_language"])
    )


def _render_example(prediction: "Prediction", score: "SampleScore") -> str:
    badge_class = _f1_badge_class(score.f1)
    predicted_display = "[JSON malformato]\n" + prediction.predicted_text if score.parse_error else _pretty_json(
        prediction.predicted_text
    )
    return f"""
<div class="example">
  <div>
    <img src="{_image_data_uri(prediction.image)}" alt="{escape(prediction.id)}">
    <p><strong>{escape(prediction.id)}</strong> — {escape(prediction.document_type)} / {escape(prediction.language)}
      <span class="badge {badge_class}">F1 {score.f1:.2f}</span></p>
  </div>
  <div>
    <h3>Ground truth</h3>
    <pre>{escape(_pretty_json(prediction.ground_truth))}</pre>
  </div>
  <div>
    <h3>Predizione</h3>
    <pre>{escape(predicted_display)}</pre>
  </div>
</div>"""


def render_report(
    predictions: list["Prediction"],
    scores: list["SampleScore"],
    *,
    max_examples: int = 40,
) -> str:
    scores_by_id = {s.id: s for s in scores}
    aggregated = scoring.aggregate(scores)
    ranked = sorted(predictions, key=lambda p: scores_by_id[p.id].f1)
    examples_html = "".join(_render_example(p, scores_by_id[p.id]) for p in ranked[:max_examples])
    return f"""<!doctype html>
<html lang="it">
<head><meta charset="utf-8"><title>trainmaster — report predizioni</title><style>{_STYLE}</style></head>
<body>
<h1>Report predizioni</h1>
{_render_summary(aggregated)}
<h2>Esempi (F1 crescente, i peggiori {min(max_examples, len(ranked))} in cima)</h2>
{examples_html}
</body>
</html>"""


def write_report(path: Path, html: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")

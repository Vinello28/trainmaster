"""CLI sottile: punteggia un ``predictions.parquet`` già generato e (ri)produce il
report, senza mai toccare il modello. Niente GPU/torch/unsloth richiesti: utile per
iterare sulla logica di scoring/report senza rieseguire l'inferenza.

Esempio::

    uv run trainmaster-score --predictions runs/default/eval/predictions.parquet -c configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainmaster import report, scoring
from trainmaster.config import load_config, parse_cli_overrides
from trainmaster.inference import load_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-score",
        description="Punteggia un predictions.parquet e (ri)produce il report, senza GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="predictions.parquet da punteggiare."
    )
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="RunConfig (per tolerance/report)."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="chiave.punto=valore",
        help="Override puntuale (ripetibile).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Cartella di output. Default: la cartella di --predictions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, parse_cli_overrides(args.overrides))
    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(str(args.predictions))
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
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(f"Metriche: {metrics_path}")

    if config.report.enabled:
        html = report.render_report(predictions, scores, max_examples=config.report.max_examples)
        report_path = output_dir / "report.html"
        report.write_report(report_path, html)
        print(f"Report: {report_path}")

    print(f"F1 medio: {aggregated['overall']['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

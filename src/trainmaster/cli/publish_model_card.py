"""CLI sottile: genera la model card in ``<run_dir>/checkpoint/README.md`` da
``<run_dir>/config.yaml`` e, se presente, ``<run_dir>/eval/metrics.json``.

Passo che precede ``hf upload`` nella pubblicazione manuale su Hugging Face (vedi
README.md#pubblicazione-su-hugging-face): nessuna chiamata di rete qui, solo file locali.

Esempio::

    uv run trainmaster-model-card --run-dir runs/default
    hf upload <namespace>/<nome-modello> runs/default/checkpoint .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainmaster.config import load_config
from trainmaster.model_card import render_model_card, write_model_card


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-model-card",
        description="Genera la model card per la pubblicazione su Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Cartella del run (contiene config.yaml, checkpoint/, eval/metrics.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir: Path = args.run_dir

    config = load_config(run_dir / "config.yaml")

    metrics_path = run_dir / "eval" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

    card = render_model_card(config, metrics)
    card_path = run_dir / "checkpoint" / "README.md"
    write_model_card(card_path, card)
    print(f"Model card scritta in {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

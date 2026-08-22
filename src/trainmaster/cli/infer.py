"""CLI sottile: inferenza su una singola immagine con un checkpoint già addestrato.

Esempio::

    uv run trainmaster-infer --config configs/codex.yaml \\
        --checkpoint models/codex/checkpoint-300 \\
        --image documento.jpg \\
        --instruction "Estrai il codice fiscale da questa patente e restituiscilo in JSON."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainmaster.config import load_config, parse_cli_overrides
from trainmaster.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-infer",
        description="Genera una predizione per una singola immagine con un checkpoint già addestrato.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="File YAML RunConfig (parametri modello)."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Directory del checkpoint da usare."
    )
    parser.add_argument("--image", type=Path, required=True, help="Immagine da processare.")
    parser.add_argument(
        "--instruction", type=str, required=True, help="Istruzione/prompt per il modello."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override di evaluation.max_new_tokens dalla config.",
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
        "-o", "--output", type=Path, default=None, help="Se passato, scrive il testo generato anche qui."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from PIL import Image

    args = build_parser().parse_args(argv)
    config = load_config(args.config, parse_cli_overrides(args.overrides))
    max_new_tokens = args.max_new_tokens or config.evaluation.max_new_tokens

    handle = load_model(config.model, for_training=False, checkpoint=args.checkpoint)
    image = Image.open(args.image).convert("RGB")
    predicted_text = handle.generate(image, args.instruction, max_new_tokens=max_new_tokens)

    print(predicted_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(predicted_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI sottile: fonde un checkpoint LoRA nei pesi base e salva un modello standalone
(16-bit, nessun adapter separato), usabile con stack esterni a Unsloth
(``transformers``/``vLLM`` puri) senza dover fondere l'adapter a runtime.

Esempio::

    uv run trainmaster-export --config configs/codex.yaml \\
        --checkpoint models/codex/checkpoint-300 \\
        --output models/codex/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainmaster.config import load_config, parse_cli_overrides
from trainmaster.model import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-export",
        description="Fonde un checkpoint LoRA nei pesi base e salva un modello standalone (16-bit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, required=True, help="File YAML RunConfig (parametri modello)."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Directory del checkpoint LoRA da fondere."
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Directory di destinazione del modello fuso."
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="chiave.punto=valore",
        help="Override puntuale (ripetibile).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, parse_cli_overrides(args.overrides))

    handle = load_model(config.model, for_training=False, checkpoint=args.checkpoint)
    handle.save_merged(args.output)
    print(f"Modello fuso salvato in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

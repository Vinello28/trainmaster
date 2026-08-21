"""CLI sottile: valuta un checkpoint già addestrato, senza rieseguire il training.

Esempio::

    uv run trainmaster-evaluate --config configs/default.yaml --checkpoint runs/default/checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainmaster.config import load_config, parse_cli_overrides
from trainmaster.pipeline import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-evaluate",
        description=(
            "Valuta un checkpoint già addestrato: genera predizioni sulla validazione, "
            "punteggia e produce il report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=Path, required=True, help="File YAML RunConfig.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Directory del checkpoint da valutare."
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
    result = run_evaluation(config, args.checkpoint)
    print(f"Predizioni: {result.predictions_path}")
    if result.report_path:
        print(f"Report: {result.report_path}")
    print(f"F1 medio: {result.aggregate['overall']['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

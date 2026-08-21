"""CLI sottile: addestra (ed eventualmente valuta) un Vision-LLM da una RunConfig YAML.

Esempi::

    uv run trainmaster-train --config configs/default.yaml
    uv run trainmaster-train --config configs/default.yaml --set training.learning_rate=1e-4
    uv run trainmaster-train --config configs/smoke_test.yaml --skip-eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trainmaster.config import load_config, parse_cli_overrides
from trainmaster.pipeline import run, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trainmaster-train",
        description="Addestra un Vision-LLM con Unsloth secondo una RunConfig YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=Path, required=True, help="File YAML RunConfig.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="chiave.punto=valore",
        help="Override puntuale (ripetibile), es. --set training.learning_rate=1e-4",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Salta la valutazione semantica post-training anche se abilitata in config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, parse_cli_overrides(args.overrides))

    if args.skip_eval:
        checkpoint_dir = run_training(config)
        print(f"Checkpoint salvato in {checkpoint_dir}")
        return 0

    result = run(config)
    if result is not None:
        print(f"Predizioni: {result.predictions_path}")
        if result.report_path:
            print(f"Report: {result.report_path}")
        print(f"F1 medio: {result.aggregate['overall']['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

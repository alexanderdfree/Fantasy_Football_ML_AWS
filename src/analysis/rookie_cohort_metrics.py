"""Compatibility wrapper for the consolidated rookie cohort metric.

New usage:
    python -m src.analysis.cohort_analysis rookie --no-model
    python -m src.analysis.cohort_analysis rookie --with-model-error
"""

from __future__ import annotations

import argparse

from src.analysis.cohort_analysis import (
    ACTUAL,
    EARLY_GAMES,
    MODELS,
    ROOKIE_BUCKET,
    ROOKIE_EARLY,
    ROOKIE_REST,
    UNKNOWN,
    VETERAN,
    available_models,
    best_model,
    bias_corrected_mae,
    label_rookie_rows,
    player_min_season,
    rookie_gap,
)
from src.analysis.cohort_analysis import (
    DEFAULT_ROOKIE_POSITIONS as DEFAULT_POSITIONS,
)
from src.analysis.cohort_analysis import (
    main as _cohort_main,
)
from src.analysis.cohort_analysis import (
    rookie_cohort_model_table as cohort_model_table,
)

__all__ = [
    "ACTUAL",
    "DEFAULT_POSITIONS",
    "EARLY_GAMES",
    "MODELS",
    "ROOKIE_BUCKET",
    "ROOKIE_EARLY",
    "ROOKIE_REST",
    "UNKNOWN",
    "VETERAN",
    "available_models",
    "best_model",
    "bias_corrected_mae",
    "cohort_model_table",
    "label_rookie_rows",
    "main",
    "player_min_season",
    "rookie_gap",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rookie cold-start subgroup metric")
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-games", type=int, default=EARLY_GAMES)
    args = parser.parse_args()

    argv = [
        "rookie",
        "--positions",
        *[p.upper() for p in args.positions],
        "--early-games",
        str(args.early_games),
    ]
    if args.no_model:
        argv.append("--no-model")
    else:
        argv.extend(["--with-model-error", "--seed", str(args.seed)])
    _cohort_main(argv)


if __name__ == "__main__":
    main()

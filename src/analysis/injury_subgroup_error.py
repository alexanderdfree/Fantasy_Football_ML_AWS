"""Compatibility wrapper for the consolidated injury/return cohort metric.

New usage:
    python -m src.analysis.cohort_analysis injury_return --with-model-error --deep-dive
"""

from __future__ import annotations

import argparse

from src.analysis.cohort_analysis import (
    ACTUAL,
    ANALYSIS_NAME,
    DAYS_REST,
    DEFAULT_POSITIONS,
    GAME_STATUS,
    HISTORY_DIR,
    RETURN_FLAG,
    SMALL_N,
    SUBGROUP_SPECS,
    _flag,
    analyze_position,
)
from src.analysis.cohort_analysis import (
    main as _cohort_main,
)

__all__ = [
    "ACTUAL",
    "ANALYSIS_NAME",
    "DAYS_REST",
    "DEFAULT_POSITIONS",
    "GAME_STATUS",
    "HISTORY_DIR",
    "RETURN_FLAG",
    "SMALL_N",
    "SUBGROUP_SPECS",
    "_flag",
    "analyze_position",
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Injury / return-from-absence subgroup error")
    parser.add_argument("positions", nargs="*", default=None)
    args = parser.parse_args()

    positions = [p.upper() for p in (args.positions or DEFAULT_POSITIONS)]
    _cohort_main(["injury_return", "--positions", *positions, "--with-model-error", "--deep-dive"])


if __name__ == "__main__":
    main()

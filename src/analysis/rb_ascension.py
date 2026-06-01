"""Compatibility wrapper for the consolidated cohort analysis CLI.

New usage:
    python -m src.analysis.cohort_analysis ascension
    python -m src.analysis.cohort_analysis ascension --with-model-error
"""

from __future__ import annotations

import sys

from src.analysis.cohort_analysis import (
    ACTUAL,
    ASCENSION,
    BACKUP_OPP,
    ESTABLISHED,
    MIN_PRIOR_GAMES,
    MODELS,
    UNKNOWN,
    WORKHORSE_OPP,
    _depth_chart_ranks,
    _offense_depth_ranks,
    add_injury_attribution,
    convergence_table,
    find_ascension_events,
    label_ascension_rows,
    prepare_weekly,
)
from src.analysis.cohort_analysis import (
    ascension_cohort_model_table as cohort_model_table,
)
from src.analysis.cohort_analysis import (
    main as _cohort_main,
)

__all__ = [
    "ACTUAL",
    "ASCENSION",
    "BACKUP_OPP",
    "ESTABLISHED",
    "MIN_PRIOR_GAMES",
    "MODELS",
    "UNKNOWN",
    "WORKHORSE_OPP",
    "_depth_chart_ranks",
    "_offense_depth_ranks",
    "add_injury_attribution",
    "cohort_model_table",
    "convergence_table",
    "find_ascension_events",
    "label_ascension_rows",
    "main",
    "prepare_weekly",
]


def main() -> None:
    _cohort_main(["ascension", *sys.argv[1:]])


if __name__ == "__main__":
    main()

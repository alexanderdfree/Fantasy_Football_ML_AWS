"""Compatibility wrapper for the consolidated cohort analysis CLI.

New usage:
    python -m src.analysis.cohort_analysis sparse_history --positions RB --with-model-error --deep-dive
"""

from __future__ import annotations

import sys

from src.analysis.cohort_analysis import (
    ACTUAL,
    CALIB_BINS,
    DISAGREEMENT_THRESHOLD,
    HISTORY_DEPTH_BUCKETS,
    HOT_RECENT_FP,
    LGBM,
    MODELS,
    PEERS,
    RECENT_MAX_COL,
    add_history_depth,
    available_models,
    calibration_table,
    gap_decomposition,
    history_depth_table,
    peer_gap,
    per_model_metrics,
)
from src.analysis.cohort_analysis import (
    main as _cohort_main,
)

__all__ = [
    "ACTUAL",
    "CALIB_BINS",
    "DISAGREEMENT_THRESHOLD",
    "HISTORY_DEPTH_BUCKETS",
    "HOT_RECENT_FP",
    "LGBM",
    "MODELS",
    "PEERS",
    "RECENT_MAX_COL",
    "add_history_depth",
    "available_models",
    "calibration_table",
    "gap_decomposition",
    "history_depth_table",
    "main",
    "peer_gap",
    "per_model_metrics",
]


def main() -> None:
    args = ["sparse_history", "--positions", "RB", "--with-model-error", "--deep-dive"]
    if "--no-plots" in sys.argv[1:]:
        args.append("--no-plots")
    _cohort_main(args)


if __name__ == "__main__":
    main()

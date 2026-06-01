"""Compatibility wrapper for the consolidated cohort analysis CLI.

New usage:
    python -m src.analysis.cohort_analysis late_week --deep-dive
    python -m src.analysis.cohort_analysis late_week --with-model-error
"""

from __future__ import annotations

import argparse

from src.analysis.cohort_analysis import (
    BUCKET_ORDER,
    DEFAULT_ESTABLISHED_GAMES,
    DEFAULT_TOP_K,
    EARLY,
    FINAL,
    PENULT,
    SKILL_POSITIONS,
    TRUE_COL,
    _avg_weekly_ranking,
    _drop_final_week,
    _week_bucket_eval,
    assign_final_week_buckets,
    run_ablation,
    stage1_label_anomaly,
    stage2_prediction_degradation,
)
from src.config import POSITIONS, SPLITS_DIR

__all__ = [
    "BUCKET_ORDER",
    "DEFAULT_ESTABLISHED_GAMES",
    "DEFAULT_TOP_K",
    "EARLY",
    "FINAL",
    "PENULT",
    "SKILL_POSITIONS",
    "TRUE_COL",
    "_avg_weekly_ranking",
    "_drop_final_week",
    "_week_bucket_eval",
    "assign_final_week_buckets",
    "main",
    "run_ablation",
    "stage1_label_anomaly",
    "stage2_prediction_degradation",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", action="store_true", help="label anomaly (cheap, no models)")
    ap.add_argument("--stage2", action="store_true", help="prediction degradation (trains models)")
    ap.add_argument("--ablation", action="store_true", help="KEEP vs CUT final-week ablation")
    ap.add_argument("--positions", nargs="*", default=POSITIONS)
    ap.add_argument("--splits-dir", default=SPLITS_DIR)
    ap.add_argument("--established-games", type=int, default=DEFAULT_ESTABLISHED_GAMES)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--eval-max-week", type=int, default=17)
    args = ap.parse_args()

    positions = [p.upper() for p in args.positions]
    run_s1 = args.stage1 or not (args.stage2 or args.ablation)
    if run_s1:
        stage1_label_anomaly(args.splits_dir, args.established_games)
    if args.stage2:
        stage2_prediction_degradation(positions, args.top_k)
    if args.ablation:
        run_ablation(positions, args.splits_dir, args.top_k, args.eval_max_week)


if __name__ == "__main__":
    main()

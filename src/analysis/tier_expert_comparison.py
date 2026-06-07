"""Elite-vs-field head-to-head: every one of our models vs each expert, per tier.

:mod:`src.analysis.analysis_expert_comparison` adjudicates our **attention NN** vs
each expert (NFL.com, Sleeper/RotoWire) on matched player-weeks. This script answers
a different question raised by the scoring-tier diagnostic
(:mod:`src.analysis.cohort_analysis` ``scoring_tier``): on the **a-priori elite /
top-drafted slice**, how do **all** of our models (Ridge / NN / Attention NN /
LightGBM) and each expert compare — and does the expert also under-predict elites?

For each (position, expert) it restricts to the matched player-weeks, labels each row
with the same a-priori tier (prior-season FP rank), and reports per tier
{elite_top_drafted, field}: per-model and expert **MAE** and signed **bias**
(``mean(pred - actual)``; negative = under-prediction). "All models matter" — we do
not collapse to the served best model.

Read-only: runs each position pipeline once for held-out predictions; adds no model
feature and triggers no retrain.

Usage:
    python -m src.analysis.tier_expert_comparison --positions QB RB WR TE --tier-topn 24
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd

from src.analysis.analysis_expert_comparison import (
    _EXPERT_PRED_COL,
    _KEY_COLS,
    _build_experts,
)
from src.analysis.cohort_analysis import (
    ACTUAL,
    TIER_BUCKET,
    TIER_ELITE,
    TIER_FIELD,
    _load_splits,
    available_models,
    label_scoring_tier_rows,
    player_prior_season_fp,
)
from src.config import TEST_SEASONS

DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]
EVAL_SEASONS_DEFAULT = tuple(TEST_SEASONS) if TEST_SEASONS else (2025,)


def _mae_bias(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    if len(actual) == 0:
        return float("nan"), float("nan"), 0
    err = pred - actual
    return float(np.abs(err).mean()), float(err.mean()), int(len(actual))


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["player_id"] = out["player_id"].astype(str)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


def compare_position(
    pos: str,
    test_df: pd.DataFrame,
    prior_fp: pd.Series,
    experts,
    expert_raws: dict,
    *,
    tier_topn: int,
) -> None:
    """Print per-tier model-vs-expert MAE/bias tables for one position."""
    models = available_models(test_df)
    if not models or ACTUAL not in test_df.columns:
        print(f"  ! {pos}: no prediction/actual columns; skipping.")
        return

    labeled = _normalize_keys(test_df)
    labeled[TIER_BUCKET] = label_scoring_tier_rows(labeled, prior_fp, top_n=tier_topn)

    model_cols = list(models.values())
    base = labeled[[*_KEY_COLS, ACTUAL, TIER_BUCKET, *model_cols]]

    for src in experts:
        if pos in src.skipped or expert_raws.get(src.name) is None:
            continue
        proj = src.project(expert_raws[src.name], pos, "ppr")
        if proj is None or proj.empty:
            print(f"\n  {pos} vs {src.label}: no projections; skipping.")
            continue
        expert = _normalize_keys(proj[[*_KEY_COLS, _EXPERT_PRED_COL]])
        joined = base.merge(expert, on=_KEY_COLS, how="inner")
        if joined.empty:
            print(f"\n  {pos} vs {src.label}: no matched player-weeks; skipping.")
            continue

        print(f"\n{'=' * 84}")
        print(f"{pos} vs {src.label}  (matched player-weeks; MAE / bias by a-priori tier)")
        print("  bias = mean(pred - actual); negative on elite = under-prediction.")
        print("=" * 84)
        for bucket in (TIER_ELITE, TIER_FIELD):
            sub = joined[joined[TIER_BUCKET] == bucket]
            actual = sub[ACTUAL].to_numpy(dtype=float)
            print(f"\n  {bucket}  (n={len(sub)})")
            print(f"    {'forecaster':<18}{'MAE':>9}{'bias':>9}")
            for name, col in models.items():
                mae, bias, _ = _mae_bias(actual, sub[col].to_numpy(dtype=float))
                print(f"    {name:<18}{mae:>9.3f}{bias:>+9.3f}")
            e_mae, e_bias, _ = _mae_bias(actual, sub[_EXPERT_PRED_COL].to_numpy(dtype=float))
            print(f"    {src.label:<18}{e_mae:>9.3f}{e_bias:>+9.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", nargs="*", default=DEFAULT_POSITIONS)
    parser.add_argument("--seasons", nargs="*", type=int, default=list(EVAL_SEASONS_DEFAULT))
    parser.add_argument("--tier-topn", type=int, default=24)
    parser.add_argument(
        "--from-artifacts",
        action="store_true",
        help="Evaluate the most-recent SAVED model artifacts on the test split instead of "
        "retraining via run() — faster and reflects the served models "
        "(see src.analysis.artifact_eval).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="With --from-artifacts: pull the latest artifacts from S3 before evaluating.",
    )
    args = parser.parse_args(argv)

    positions = [p.upper() for p in args.positions]
    eval_set = {int(s) for s in args.seasons}

    train_df, val_df, test_df_all = _load_splits()
    prior_fp = player_prior_season_fp([train_df, val_df, test_df_all])

    if args.from_artifacts and args.sync:
        from src.shared.model_sync import sync_models_from_s3

        print("Syncing latest model artifacts from S3 ...", flush=True)
        sync_models_from_s3()

    experts = _build_experts(None, None)
    expert_raws: dict = {}
    for src in experts:
        print(f"Loading {src.label} projections for {sorted(eval_set)} ...", flush=True)
        try:
            expert_raws[src.name] = src.load(sorted(eval_set))
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            print(f"  WARN: {src.label} load failed ({type(e).__name__}: {e}); skipping expert")
            expert_raws[src.name] = None

    for pos in positions:
        if args.from_artifacts:
            print(f"\nScoring {pos} from saved artifacts ...", flush=True)
            from src.analysis.artifact_eval import build_test_df_from_artifacts

            test_df = build_test_df_from_artifacts(pos, train_df, val_df, test_df_all)
        else:
            print(f"\nRunning {pos} pipeline ...", flush=True)
            result = importlib.import_module(f"src.{pos.lower()}.run_pipeline").run()
            test_df = result["test_df"]
        test_df = test_df[test_df["season"].astype(int).isin(eval_set)]
        compare_position(pos, test_df, prior_fp, experts, expert_raws, tier_topn=args.tier_topn)


if __name__ == "__main__":
    main()

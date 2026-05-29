import os

import pandas as pd

from src.config import CV_VAL_SEASONS, SPLITS_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.data.preprocessing import impute_snap_pct


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list[int] | None = None,
    val_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
) -> tuple:
    """Split data by season into train/val/test sets.

    ``snap_pct`` (if present) is imputed POST-SPLIT using train-only
    medians per ``(position, week)`` — val/test rows are filled with the
    train-only statistic to avoid leaking holdout distribution back into
    the model's imputation. Empty ``(position, week)`` groups fall back
    to 0, matching the pre-split behaviour.
    """
    if train_seasons is None:
        train_seasons = TRAIN_SEASONS
    if val_seasons is None:
        val_seasons = VAL_SEASONS
    if test_seasons is None:
        test_seasons = TEST_SEASONS

    # Drop playoff rows — fantasy leagues end with the regular season, and
    # the schedule lookup used for Vegas/weather features only covers REG
    # games. ``season_type`` is part of the nflverse weekly schema and so is
    # guaranteed present by ``src/data/loader.py``; fail loudly on absence
    # instead of silently keeping playoff rows.
    assert "season_type" in df.columns, (
        "temporal_split() requires 'season_type'; upstream loader "
        "(src/data/loader.py) must emit it. Absent column would silently "
        "include POST rows in train/val/test."
    )
    df = df[df["season_type"] == "REG"].copy()

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"].isin(val_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    # Assert no overlap
    all_seasons = set(train_seasons) | set(val_seasons) | set(test_seasons)
    assert len(all_seasons) == len(train_seasons) + len(val_seasons) + len(test_seasons), (
        "Season overlap detected between splits"
    )

    # Split-aware snap_pct imputation: fit medians on train only, apply to
    # all three splits. Prevents val/test snap_pct distribution leaking into
    # train-row imputation (the pre-split groupby-transform did the opposite
    # — every split influenced every other split's medians, a subtle
    # cross-fold leak that biased the model's reaction to NaN snap_pct
    # toward holdout-era patterns).
    train_df = impute_snap_pct(train_df, fit_on=train_df)
    val_df = impute_snap_pct(val_df, fit_on=train_df)
    test_df = impute_snap_pct(test_df, fit_on=train_df)

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save to disk
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_parquet(f"{SPLITS_DIR}/train.parquet", index=False)
    val_df.to_parquet(f"{SPLITS_DIR}/val.parquet", index=False)
    test_df.to_parquet(f"{SPLITS_DIR}/test.parquet", index=False)

    return train_df, val_df, test_df


def expanding_window_folds(
    df: pd.DataFrame,
    val_seasons: list[int] | None = None,
    min_train_season: int = 2012,
) -> list[tuple]:
    """Generate expanding-window CV folds.

    For each val season, training data includes all seasons from
    min_train_season up to (but not including) the val season.

    Returns:
        List of (fold_idx, train_df, val_df) tuples.
    """
    if val_seasons is None:
        val_seasons = CV_VAL_SEASONS

    folds = []
    for i, val_season in enumerate(val_seasons):
        train_seasons = list(range(min_train_season, val_season))
        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"] == val_season].copy()
        # Per-fold train-only snap_pct imputation, mirroring temporal_split.
        # Each fold's val rows must be filled with that fold's TRAIN-only
        # medians — never medians computed over the val season — so the CV
        # estimate doesn't leak the holdout snap_pct distribution back into
        # the imputation (the same cross-split leak temporal_split closes).
        # No-op when snap_pct is already absent or fully observed.
        train_df = impute_snap_pct(train_df, fit_on=train_df)
        val_df = impute_snap_pct(val_df, fit_on=train_df)
        print(
            f"  Fold {i + 1}: train seasons {train_seasons[0]}-{train_seasons[-1]} "
            f"({len(train_df)} rows), val season {val_season} ({len(val_df)} rows)"
        )
        folds.append((i, train_df, val_df))

    return folds

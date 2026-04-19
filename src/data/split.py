import os
import pandas as pd
from src.config import TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, SPLITS_DIR, CV_VAL_SEASONS


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list[int] = None,
    val_seasons: list[int] = None,
    test_seasons: list[int] = None,
) -> tuple:
    """Split data by season into train/val/test sets."""
    if train_seasons is None:
        train_seasons = TRAIN_SEASONS
    if val_seasons is None:
        val_seasons = VAL_SEASONS
    if test_seasons is None:
        test_seasons = TEST_SEASONS

    # Drop playoff rows — fantasy leagues end with the regular season, and
    # the schedule lookup used for Vegas/weather features only covers REG games.
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"].isin(val_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    # Assert no overlap
    all_seasons = set(train_seasons) | set(val_seasons) | set(test_seasons)
    assert len(all_seasons) == len(train_seasons) + len(val_seasons) + len(test_seasons), \
        "Season overlap detected between splits"

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save to disk
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_parquet(f"{SPLITS_DIR}/train.parquet", index=False)
    val_df.to_parquet(f"{SPLITS_DIR}/val.parquet", index=False)
    test_df.to_parquet(f"{SPLITS_DIR}/test.parquet", index=False)

    return train_df, val_df, test_df


def expanding_window_folds(
    df: pd.DataFrame,
    val_seasons: list[int] = None,
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
        print(f"  Fold {i+1}: train seasons {train_seasons[0]}-{train_seasons[-1]} "
              f"({len(train_df)} rows), val season {val_season} ({len(val_df)} rows)")
        folds.append((i, train_df, val_df))

    return folds

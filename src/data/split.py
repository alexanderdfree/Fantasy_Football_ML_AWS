import os
import pandas as pd
from src.config import TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, SPLITS_DIR


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

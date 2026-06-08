import os

import pandas as pd

from src.config import (
    CV_VAL_SEASONS,
    ROLLING_ORIGIN_TEST_SEASONS,
    SPLITS_DIR,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)
from src.data.cache_io import atomic_write_parquet
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
    atomic_write_parquet(train_df, f"{SPLITS_DIR}/train.parquet", index=False)
    atomic_write_parquet(val_df, f"{SPLITS_DIR}/val.parquet", index=False)
    atomic_write_parquet(test_df, f"{SPLITS_DIR}/test.parquet", index=False)

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
    # Drop playoff rows, mirroring temporal_split / rolling_origin_folds — the
    # CV folds must train/eval on the same REG-only population as production so
    # the cross-validated estimate stays comparable. ``season_type`` is
    # guaranteed by src/data/loader.py; fail loudly on absence rather than fold
    # POST rows into the expanding windows (#773 / #363 F4).
    assert "season_type" in df.columns, (
        "expanding_window_folds() requires 'season_type'; upstream loader "
        "(src/data/loader.py) must emit it. Absent column would silently "
        "include POST rows in the CV folds."
    )
    df = df[df["season_type"] == "REG"].copy()
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


def rolling_origin_folds(
    df: pd.DataFrame,
    test_seasons: list[int] | None = None,
    min_train_season: int = 2012,
) -> list[tuple]:
    """Generate rolling-origin (walk-forward) train/val/test folds.

    For each test season ``T``: train = ``[min_train_season .. T-2]``, val =
    ``T-1``, test = ``T``. Unlike ``expanding_window_folds`` (where val == eval,
    a mildly optimistic in-the-loop estimate), each origin's *test* season is a
    clean forward holdout never used for tuning or early stopping — so
    aggregating per-origin test metrics mean±std turns the single-season point
    estimate into a distribution. The final origin (``test == TEST_SEASONS[0]``)
    reproduces the production ``temporal_split`` season-for-season, so a
    rolling-origin run is directly comparable to the headline single split.

    Distinct from the K-fold-over-seasons rejected in D1: every origin trains
    strictly on the past, preserving the deployment-mirror (no leakage).

    Leakage note: every engineered column in ``df`` is split-independent by
    construction (within-(player, season) or prior-season aggregates), and
    ``snap_pct`` is pre-lagged in ``src/features/engineer.py`` *before* any
    split — so re-slicing an already-featured frame by season is leakage-free,
    and the per-origin ``impute_snap_pct`` below is a no-op in the canonical
    ``preprocess -> build_features -> split`` flow (it stays for frames that
    still carry raw NaN ``snap_pct``). The only genuinely split-sensitive steps
    — ``depth_chart_rank`` imputation and the ``StandardScaler`` — run *inside*
    ``run_pipeline`` per origin, post-split, so they refit on each origin's
    train. If a future change bakes a train-fit transform into the split
    parquets, this invariant breaks and the re-slice would leak.

    Returns:
        List of ``(origin_idx, train_df, val_df, test_df)`` tuples.
    """
    if test_seasons is None:
        test_seasons = ROLLING_ORIGIN_TEST_SEASONS

    assert "season_type" in df.columns, (
        "rolling_origin_folds() requires 'season_type'; upstream loader "
        "(src/data/loader.py) must emit it."
    )
    df = df[df["season_type"] == "REG"].copy()

    folds = []
    for i, test_season in enumerate(test_seasons):
        val_season = test_season - 1
        train_seasons = list(range(min_train_season, val_season))
        if not train_seasons:
            raise ValueError(
                f"rolling_origin_folds: test_season={test_season} leaves no train "
                f"seasons above min_train_season={min_train_season}."
            )
        train_df = df[df["season"].isin(train_seasons)].copy()
        val_df = df[df["season"] == val_season].copy()
        test_df = df[df["season"] == test_season].copy()
        # Per-origin train-only snap_pct imputation, mirroring temporal_split /
        # expanding_window_folds (no-op when snap_pct is already lag-filled).
        train_df = impute_snap_pct(train_df, fit_on=train_df)
        val_df = impute_snap_pct(val_df, fit_on=train_df)
        test_df = impute_snap_pct(test_df, fit_on=train_df)
        print(
            f"  Origin {chr(65 + i)}: train {train_seasons[0]}-{train_seasons[-1]} "
            f"({len(train_df)} rows), val {val_season} ({len(val_df)} rows), "
            f"test {test_season} ({len(test_df)} rows)"
        )
        folds.append((i, train_df, val_df, test_df))

    return folds

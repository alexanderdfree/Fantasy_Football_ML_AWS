"""Shared feature-building for training and inference paths.

Both ``shared/pipeline.py::_prepare_position_data`` (training) and
``app.py::_apply_position_models`` (serving) need the same per-position
feature-engineering pipeline. They drifted in the past — TODO.md archive
entry "Weather/Vegas features missing at inference in ``app.py``" was the
most recent recurrence: a training-side feature wasn't mirrored at serving
time, so 12 weather/Vegas features were silently zeroed at inference.
Centralizing the shared block here makes that drift class impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.shared.weather_features import merge_schedule_features

# Clip bounds applied after every ``StandardScaler`` op. Guards against
# extreme z-scores when test-distribution features wander well outside the
# training distribution — see TODO.md archive "No feature clipping after
# StandardScaler". Kept at (-4, 4): catches ~0.3-0.4% of values under a
# Gaussian assumption, well below the extreme tails that cause catastrophic
# NN extrapolation.
FEATURE_CLIP: tuple[float, float] = (-4.0, 4.0)


def build_position_features(
    pos_train: pd.DataFrame,
    pos_val: pd.DataFrame,
    pos_test: pd.DataFrame | None,
    cfg: dict,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Merge schedule features, add position-specific features, backfill missing
    whitelist columns, and clean inf/NaN values.

    Callers are responsible for ``filter_fn`` and ``compute_targets_fn`` up to
    this point; this helper starts from already-filtered, target-computed
    splits so both training and serving can route through the same block.
    """
    dfs = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    split_labels = ["train", "val", "test"][: len(dfs)]

    # Schedule merge first — downstream ``add_features_fn`` may read the
    # merged weather/Vegas columns.
    for label, df in zip(split_labels, dfs, strict=True):
        merge_schedule_features(df, label=label)

    # Position-specific feature engineering + fill_nans. Both take three
    # splits; when test is None, pass an empty stub so the signatures line up.
    if pos_test is not None:
        pos_train, pos_val, pos_test = cfg["add_features_fn"](pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = cfg["fill_nans_fn"](
            pos_train, pos_val, pos_test, cfg["specific_features"]
        )
    else:
        empty = pos_val.iloc[:0].copy()
        pos_train, pos_val, _ = cfg["add_features_fn"](pos_train, pos_val, empty)
        pos_train, pos_val, _ = cfg["fill_nans_fn"](
            pos_train, pos_val, empty, cfg["specific_features"]
        )

    # Backfill any whitelist columns the pipeline didn't produce. One concat
    # per df avoids pandas' fragmented-DataFrame PerformanceWarning.
    dfs = [pos_train, pos_val] + ([pos_test] if pos_test is not None else [])
    missing = [c for c in feature_cols if c not in pos_train.columns]
    if missing:
        print(f"  WARNING: {len(missing)} feature columns missing, filling with 0")
        filled = [
            pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)
            for df in dfs
        ]
        pos_train = filled[0]
        pos_val = filled[1]
        if pos_test is not None:
            pos_test = filled[2]
        dfs = filled

    for df in dfs:
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return pos_train, pos_val, pos_test


def scale_and_clip(
    scaler: StandardScaler,
    X: np.ndarray,
    *,
    fit: bool = False,
) -> np.ndarray:
    """Scale X with ``scaler`` (fit first if ``fit=True``) and clip to ``FEATURE_CLIP``."""
    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    return np.clip(X, *FEATURE_CLIP)


def safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Return ``num / denom`` with all ill-defined results replaced by 0.

    Handles the three pathological cases inline so callers don't need the
    ``(a/b).fillna(0); df.loc[b == 0, col] = 0`` pair:

    - ``0 / 0``  → NaN   → 0
    - ``x / 0``  → ±inf  → 0  (``fillna`` alone wouldn't catch this)
    - ``x / NaN`` or ``NaN / x`` → NaN → 0
    """
    return (num / denom).replace([np.inf, -np.inf], 0).fillna(0)


def rolling_agg(
    df: pd.DataFrame,
    col: str,
    groupby: str | list[str],
    window: int,
    *,
    agg: str = "sum",
    shift: int = 1,
    min_periods: int = 1,
    fill: float | None = None,
) -> pd.Series:
    """Grouped, shifted rolling aggregation.

    ``shift`` defaults to 1 to prevent current-week leakage into rolling
    features — the same convention every position's feature code follows.
    ``groupby`` is explicit because K deliberately uses ``["player_id"]``
    only (cross-season windows) while skill positions use
    ``["player_id", "season"]``. See TODO.md "K features use cross-season
    rolling windows".

    ``fill`` optionally replaces the leading NaN produced by ``shift`` (and
    any NaN the input itself carries). K and DST pass ``fill=0`` because
    their features are used directly — skill positions usually leave
    ``fill=None`` because their rolling outputs feed into ``safe_divide``,
    which already maps NaN to 0.
    """
    result = df.groupby(groupby)[col].transform(
        lambda x: getattr(x.shift(shift).rolling(window, min_periods=min_periods), agg)()
    )
    if fill is not None:
        result = result.fillna(fill)
    return result


def fill_nans_with_train_means(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace inf → NaN in ``cols`` across all splits, then backfill NaNs
    with the training-set column means.

    Using train-set statistics for every split is the leakage-safe contract
    every position's ``fill_*_nans`` already follows; lifting the loop here
    keeps the ``cfg["fill_nans_fn"]`` signature intact.

    Two failure modes that used to be silent are now explicit:

    - A column listed in ``cols`` but missing from ``train_df`` raises
      ``KeyError`` with the offending columns named (used to surface as a
      cryptic pandas KeyError on ``train_df[cols]`` access).
    - A column entirely NaN in ``train_df`` has ``train_means[col] = NaN``,
      so the per-column ``fillna`` would be a no-op and the NaN would only
      get caught by ``build_position_features``'s catch-all ``.fillna(0)``
      with no signal that anything went wrong. We now log a warning and
      substitute 0 for those columns, matching the catch-all's behavior
      but making the silent zero-feature visible.
    """
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise KeyError(f"fill_nans_with_train_means: cols not in train_df: {missing}")

    for split_df in (train_df, val_df, test_df):
        split_df[cols] = split_df[cols].replace([np.inf, -np.inf], np.nan)

    train_means = train_df[cols].mean()
    all_nan_cols = [c for c in cols if pd.isna(train_means[c])]
    if all_nan_cols:
        print(
            f"  WARNING: {len(all_nan_cols)} feature(s) entirely NaN in training "
            f"set; filling with 0: {all_nan_cols}"
        )
        train_means[all_nan_cols] = 0.0

    for split_df in (train_df, val_df, test_df):
        for col in cols:
            split_df[col] = split_df[col].fillna(train_means[col])
    return train_df, val_df, test_df

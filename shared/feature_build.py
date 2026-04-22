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

from shared.weather_features import merge_schedule_features

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

"""Feature contract tests for K (Kicker) position.

Uses `K.k_features.get_feature_columns()` as the source of truth: after the
full K feature pipeline (compute_targets -> compute_features), every
advertised feature column must exist, be numeric, and satisfy the documented
NaN and range ceilings.

Catches the "accidentally dropped a feature column" and "silently infinite"
bugs that leakage tests can't detect.
"""

import numpy as np
import pandas as pd
import pytest

from src.k.config import CONTEXTUAL_FEATURES, SPECIFIC_FEATURES
from src.k.features import (
    add_specific_features,
    compute_features,
    fill_nans,
    get_feature_columns,
)
from src.k.targets import compute_targets


@pytest.fixture(scope="module")
def k_feature_frame(tiny_dataset):
    """Run the K feature pipeline end-to-end and return the resulting frame."""
    df = tiny_dataset.copy()
    df = compute_targets(df)
    compute_features(df)
    # Mimic the pipeline's pre-split fill on the full frame so the contract
    # check reflects what the downstream models actually consume.
    train, val, test = (
        df[df["season"] <= 2023].copy(),
        df[df["season"] == 2024].copy(),
        df[df["season"] == 2025].copy(),
    )
    feature_cols = get_feature_columns()
    # add_features_fn is a no-op for K (computed before split); call it anyway
    # so the test exercises the real entry points.
    train, val, test = add_specific_features(train, val, test)
    # Fill only over columns that the K feature pipeline actually produces
    # (specific features are always there; contextual ones come from schedule).
    fillable = [c for c in feature_cols if c in train.columns]
    train, val, test = fill_nans(train, val, test, fillable)
    return pd.concat([train, val, test], ignore_index=True)


# ---------------------------------------------------------------------------
# Column presence & source-of-truth
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feature_columns_is_source_of_truth():
    """get_feature_columns() = specific + contextual, in order."""
    cols = get_feature_columns()
    assert cols == list(SPECIFIC_FEATURES) + list(CONTEXTUAL_FEATURES)


@pytest.mark.unit
def test_fg_yards_made_column_present(k_feature_frame):
    """`fg_yards_made` (the raw sum-of-kick-distances source of fg_yard_points)
    must be present after the feature pipeline runs: it feeds compute_targets
    and must be non-null, non-negative across the frame."""
    assert "fg_yards_made" in k_feature_frame.columns, "fg_yards_made missing after pipeline"
    series = k_feature_frame["fg_yards_made"]
    assert series.notna().all(), "fg_yards_made has NaNs after pipeline"
    assert (series >= 0).all(), "fg_yards_made has negative values"


@pytest.mark.unit
def test_all_specific_features_present_after_compute(k_feature_frame):
    """Every K-specific feature column is produced by compute_features."""
    for col in SPECIFIC_FEATURES:
        assert col in k_feature_frame.columns, f"Missing specific feature: {col}"


@pytest.mark.unit
def test_all_contextual_features_present_in_fixture(k_feature_frame):
    """The synthetic tiny_dataset pre-fills all contextual features.

    (Real pipeline sources these from the schedule merge; the fixture ships
    them pre-merged so this test can assert the contract end-to-end.)
    """
    for col in CONTEXTUAL_FEATURES:
        assert col in k_feature_frame.columns, f"Missing contextual feature: {col}"


# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_specific_features_are_numeric(k_feature_frame):
    for col in SPECIFIC_FEATURES:
        dtype = k_feature_frame[col].dtype
        assert np.issubdtype(dtype, np.number), f"{col} dtype {dtype} is not numeric"


@pytest.mark.unit
def test_contextual_features_are_numeric(k_feature_frame):
    for col in CONTEXTUAL_FEATURES:
        dtype = k_feature_frame[col].dtype
        assert np.issubdtype(dtype, np.number), f"{col} dtype {dtype} is not numeric"


# ---------------------------------------------------------------------------
# NaN ceilings & finiteness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_inf_in_features(k_feature_frame):
    """No feature column contains +/-Inf (would poison training)."""
    for col in get_feature_columns():
        if col in k_feature_frame.columns:
            assert not np.isinf(k_feature_frame[col]).any(), (
                f"{col} contains Inf after feature pipeline"
            )


@pytest.mark.unit
def test_specific_features_have_no_nans(k_feature_frame):
    """Specific features must be NaN-free after compute_features (uses fillna)."""
    for col in SPECIFIC_FEATURES:
        n_nan = int(k_feature_frame[col].isna().sum())
        assert n_nan == 0, f"{col} has {n_nan} NaNs after compute"


@pytest.mark.unit
def test_contextual_features_nan_ceiling(k_feature_frame):
    """Contextual feature NaN rate is bounded (fixture pre-fills them)."""
    ceiling = 0.01  # 1% — allow for the odd unmatched row if any slips in
    n_rows = len(k_feature_frame)
    for col in CONTEXTUAL_FEATURES:
        if col in k_feature_frame.columns:
            nan_rate = float(k_feature_frame[col].isna().sum()) / n_rows
            assert nan_rate <= ceiling, f"{col} NaN rate {nan_rate:.3f} exceeds ceiling {ceiling}"


# ---------------------------------------------------------------------------
# Value ranges (loose sanity — anything out here is likely a scaling bug)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rate_features_in_unit_interval(k_feature_frame):
    """Rate/accuracy features must live in [0, 1]."""
    rate_cols = [
        "fg_accuracy_L5",
        "long_fg_rate_L3",
        "fg_pct_40plus_L5",
        "q4_fg_rate_L5",
        "xp_accuracy_L5",
    ]
    for col in rate_cols:
        series = k_feature_frame[col]
        assert series.min() >= 0.0, f"{col} has negative values (min={series.min()})"
        assert series.max() <= 1.0 + 1e-6, f"{col} exceeds 1.0 (max={series.max()})"


@pytest.mark.unit
def test_volume_features_non_negative(k_feature_frame):
    """Attempt/volume rolling features cannot be negative."""
    for col in ["fg_attempts_L3", "pat_volume_L3"]:
        series = k_feature_frame[col]
        assert series.min() >= 0.0, f"{col} has negative value (min={series.min()})"


@pytest.mark.unit
def test_avg_fg_distance_plausible(k_feature_frame):
    """Avg FG distance should land in 0-65 yards even averaged."""
    col = k_feature_frame["avg_fg_distance_L3"]
    assert col.min() >= 0.0
    assert col.max() <= 65.0, f"avg_fg_distance_L3 max={col.max()} > 65yd"

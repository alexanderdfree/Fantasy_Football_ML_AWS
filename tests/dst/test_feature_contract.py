"""Columnar feature contract for DST.

The pipeline expects a specific set of feature columns with specific dtypes
and NaN ceilings after DST feature computation.  A silently-dropped column
degrades the model invisibly; a dtype regression (e.g. ``object`` instead of
``float``) blows up scaler fit.  These tests codify the contract so either
regression fails loud.

Notes specific to DST:
* ``get_dst_feature_columns()`` is the source of truth — it names the
  ``DST_SPECIFIC_FEATURES + DST_CONTEXTUAL_FEATURES`` that the NN and
  Ridge model consume.
* ``compute_dst_features()`` is called on the FULL dataset (not per-split),
  so all rolling/EWMA/trend columns should be populated by the time the
  pipeline hands data to the model.
* ``add_dst_specific_features`` is a no-op for DST (a legacy interface kept
  for symmetry with the player-level positions).  We still exercise it to
  pin the identity behaviour.
* Prior-season features require >=2 seasons of history — for first-season
  teams, some NaNs are acceptable.  We set a generous NaN ceiling per
  feature to avoid flakes while still catching "broke the merge" bugs.
"""

import numpy as np
import pandas as pd
import pytest

from src.dst.config import (
    DST_ALL_FEATURES,
    DST_CONTEXTUAL_FEATURES,
    DST_SPECIFIC_FEATURES,
)
from src.dst.features import (
    add_dst_specific_features,
    compute_dst_features,
    fill_dst_nans,
    get_dst_feature_columns,
)
from src.dst.targets import compute_dst_targets

# Prior-season features are NaN for the earliest season (no prior history).
# The NaN-ceiling test raises the threshold for these columns to account
# for that; all other DST-specific features come out of .fillna(0) inside
# compute_dst_features and should have zero NaNs.
_PRIOR_SEASON_COLS = {
    "prior_season_dst_pts_avg",
    "prior_season_pts_allowed_avg",
}


def _build_fixture(tiny_dst_dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute targets + features on the session dataset and return a copy."""
    df = tiny_dst_dataset.copy()
    df = compute_dst_targets(df)
    compute_dst_features(df)
    return df


@pytest.mark.unit
class TestDSTFeatureContract:
    """Columnar + dtype + NaN-ceiling assertions on DST features."""

    def test_get_feature_columns_is_ordered_and_stable(self):
        """Feature order must be deterministic — scaler depends on index."""
        cols1 = get_dst_feature_columns()
        cols2 = get_dst_feature_columns()
        assert cols1 == cols2
        assert cols1 == list(DST_ALL_FEATURES)
        # No duplicates
        assert len(cols1) == len(set(cols1))

    def test_specific_features_subset_of_all(self):
        """Every specific feature appears in the full feature list."""
        assert set(DST_SPECIFIC_FEATURES).issubset(set(DST_ALL_FEATURES))
        assert set(DST_CONTEXTUAL_FEATURES).issubset(set(DST_ALL_FEATURES))

    def test_specific_features_populated_after_compute(self, tiny_dst_dataset):
        """All rolling/EWMA/trend features land on the frame."""
        df = _build_fixture(tiny_dst_dataset)
        for col in DST_SPECIFIC_FEATURES:
            assert col in df.columns, (
                f"compute_dst_features did not emit '{col}' — "
                "either the feature was renamed or the computation was dropped"
            )

    def test_specific_features_are_numeric(self, tiny_dst_dataset):
        """Scaler requires float-castable columns."""
        df = _build_fixture(tiny_dst_dataset)
        for col in DST_SPECIFIC_FEATURES:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"'{col}' dtype={df[col].dtype} — expected numeric"
            )

    def test_specific_features_finite_and_within_nan_ceiling(self, tiny_dst_dataset):
        """Rolling/EWMA features must be finite and respect NaN ceilings.

        Prior-season features are allowed NaN for the earliest season
        (no prior history) — their ceiling is raised to account for that.
        All other features come out of .fillna(0) inside compute_dst_features
        and should have zero NaNs.
        """
        df = _build_fixture(tiny_dst_dataset)
        first_season = df["season"].min()
        first_season_frac = (df["season"] == first_season).mean()

        for col in DST_SPECIFIC_FEATURES:
            series = df[col]
            # No infinities
            assert not np.isinf(series.astype(float)).any(), (
                f"'{col}' contains +/-inf — indicates unguarded division"
            )
            # NaN ceiling — prior-season features exempted for the earliest season
            nan_frac = series.isna().mean()
            if col in _PRIOR_SEASON_COLS:
                ceiling = first_season_frac + 0.01  # +1 % slack for rounding
            else:
                ceiling = 0.0
            assert nan_frac <= ceiling, (
                f"'{col}' NaN fraction {nan_frac:.3f} exceeds ceiling {ceiling:.3f}"
            )

    def test_prior_season_features_present_after_compute(self, tiny_dst_dataset):
        """Prior-season features should be emitted for all seasons >= second."""
        df = _build_fixture(tiny_dst_dataset)
        for col in _PRIOR_SEASON_COLS:
            assert col in df.columns
        # Second season onward should be populated (merge worked)
        second_plus = df[df["season"] > df["season"].min()]
        for col in _PRIOR_SEASON_COLS:
            missing_frac = second_plus[col].isna().mean()
            assert missing_frac < 0.1, (
                f"'{col}' NaN fraction {missing_frac:.3f} in seasons>=2 — merge likely broken"
            )

    def test_compute_drops_temp_columns(self, tiny_dst_dataset):
        """``_dst_total_pts`` / ``_turnovers`` are scratch — must not leak."""
        df = _build_fixture(tiny_dst_dataset)
        leaked = [c for c in df.columns if c.startswith("_")]
        assert leaked == [], f"Temp columns leaked out of compute_dst_features: {leaked}"

    @pytest.mark.parametrize(
        "col",
        [
            "yards_allowed",
            "def_fumbles_forced",
            "def_blocked_kicks",
            "def_tds",
            "def_safeties",
        ],
    )
    def test_raw_defensive_columns_nonnegative(self, tiny_dst_dataset, col):
        """Upstream columns must be non-null and >= 0 across all seasons."""
        df = _build_fixture(tiny_dst_dataset)
        assert col in df.columns, f"expected {col} on the DST frame"
        series = df[col]
        assert series.isna().sum() == 0, f"{col} has nulls"
        assert (series >= 0).all(), f"{col} has negative values"

    @pytest.mark.parametrize(
        "col",
        [
            "forced_fumbles_L3",
            "blocked_kicks_L5",
            "yards_allowed_L3",
            "yards_allowed_L5",
            "yards_allowed_ewma",
        ],
    )
    def test_new_rolling_features_finite(self, tiny_dst_dataset, col):
        """New rolling features (Unit 2) must be finite and non-null."""
        df = _build_fixture(tiny_dst_dataset)
        assert col in df.columns, f"{col} missing from DST frame"
        series = df[col].astype(float)
        assert series.isna().sum() == 0, f"{col} has nulls"
        assert np.isfinite(series.to_numpy()).all(), f"{col} has non-finite values"

    def test_add_features_is_identity(self, tiny_dst_dataset):
        """``add_dst_specific_features`` is a no-op by design — assert invariant."""
        df = _build_fixture(tiny_dst_dataset)
        train = df[df["season"] == 2022].copy()
        val = df[df["season"] == 2023].copy()
        test = df[df["season"] == 2024].copy()
        out_train, out_val, out_test = add_dst_specific_features(train, val, test)
        # Identity — same object references
        assert out_train is train
        assert out_val is val
        assert out_test is test
        # Same columns, same row counts
        assert list(out_train.columns) == list(train.columns)
        assert len(out_train) == len(train)

    def test_fill_dst_nans_produces_finite_columns(self, tiny_dst_dataset):
        """After fill_dst_nans, every feature column must be finite."""
        df = _build_fixture(tiny_dst_dataset)
        # Fake a pipeline split
        train = df[df["season"] == 2022].copy()
        val = df[df["season"] == 2023].copy()
        test = df[df["season"] == 2024].copy()
        # Inject some NaNs on purpose, in a column that is produced by
        # compute_dst_features so fill_dst_nans has something to work with.
        for split in (val, test):
            split.loc[split.index[:3], "sacks_L3"] = np.nan
            split.loc[split.index[:2], "dst_pts_L5"] = np.inf

        train, val, test = fill_dst_nans(train, val, test, DST_SPECIFIC_FEATURES)
        for split_name, split in [("train", train), ("val", val), ("test", test)]:
            for col in DST_SPECIFIC_FEATURES:
                arr = split[col].astype(float).to_numpy()
                assert np.isfinite(arr).all(), (
                    f"{split_name}.{col} still has non-finite values after fill_dst_nans"
                )

    def test_feature_count_matches_documented_list(self):
        """Fencepost against accidental duplicates/omissions in config."""
        all_cols = get_dst_feature_columns()
        # 21 specific + 17 contextual = 38 total after the Unit 2 refactor
        # (added forced_fumbles_L3, blocked_kicks_L5, yards_allowed_L3/L5/ewma).
        assert len(DST_SPECIFIC_FEATURES) == 21
        assert len(DST_CONTEXTUAL_FEATURES) == 17
        assert len(all_cols) == 38

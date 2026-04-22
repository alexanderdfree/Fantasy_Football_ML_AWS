"""Tests for shared.feature_build — FEATURE_CLIP, scale_and_clip,
and build_position_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from shared.feature_build import FEATURE_CLIP, build_position_features, scale_and_clip

# ---------------------------------------------------------------------------
# scale_and_clip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScaleAndClip:
    def test_clips_to_feature_clip_bounds(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5)) * 10  # force extremes well past +/-4
        scaler = StandardScaler()
        out = scale_and_clip(scaler, X, fit=True)
        lo, hi = FEATURE_CLIP
        assert out.min() >= lo
        assert out.max() <= hi

    def test_fit_vs_transform(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 3))
        X_test = rng.standard_normal((50, 3))
        fit_scaler = StandardScaler()
        scale_and_clip(fit_scaler, X_train, fit=True)
        # After fit, the scaler's mean_ should be the train mean.
        np.testing.assert_allclose(fit_scaler.mean_, X_train.mean(axis=0), atol=1e-6)
        # A transform-only call uses the fitted stats without refitting.
        pre_mean = fit_scaler.mean_.copy()
        scale_and_clip(fit_scaler, X_test)
        np.testing.assert_array_equal(fit_scaler.mean_, pre_mean)

    def test_matches_manual_pipeline(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        scaler = StandardScaler()
        lo, hi = FEATURE_CLIP
        expected = np.clip(scaler.fit_transform(X), lo, hi)
        got = scale_and_clip(StandardScaler(), X, fit=True)
        np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# build_position_features
# ---------------------------------------------------------------------------


def _make_cfg(features: list[str]):
    """Minimal config stubbing add_features_fn / fill_nans_fn."""

    def _add(train, val, test):
        for df in (train, val, test):
            for f in features:
                if f not in df.columns:
                    df[f] = 1.0
        return train, val, test

    def _fill(train, val, test, _feats):
        for df in (train, val, test):
            for f in features:
                if f in df.columns:
                    df[f] = df[f].fillna(0)
        return train, val, test

    return {"add_features_fn": _add, "fill_nans_fn": _fill, "specific_features": features}


def _minimal_df(n: int = 3) -> pd.DataFrame:
    # Schedule merge expects season/week/recent_team; merge_schedule_features
    # short-circuits if _schedule_merged is already present, so set that flag
    # to isolate these tests from the real lookup.
    return pd.DataFrame(
        {
            "season": np.full(n, 2024),
            "week": np.arange(1, n + 1),
            "recent_team": ["KC"] * n,
            "_schedule_merged": True,
        }
    )


@pytest.mark.unit
class TestBuildPositionFeatures:
    def test_three_splits_populated(self):
        cfg = _make_cfg(["feat_a"])
        pos_train, pos_val, pos_test = build_position_features(
            _minimal_df(3), _minimal_df(2), _minimal_df(2), cfg, ["feat_a"]
        )
        assert "feat_a" in pos_train.columns
        assert "feat_a" in pos_val.columns
        assert "feat_a" in pos_test.columns

    def test_test_split_none_returns_none(self):
        cfg = _make_cfg(["feat_a"])
        pos_train, pos_val, pos_test = build_position_features(
            _minimal_df(3), _minimal_df(2), None, cfg, ["feat_a"]
        )
        assert pos_test is None
        assert "feat_a" in pos_train.columns

    def test_missing_columns_are_backfilled_with_zero(self):
        # _make_cfg creates "feat_a" but not "feat_missing"; the helper should
        # backfill the missing column with 0.0 across all splits.
        cfg = _make_cfg(["feat_a"])
        pos_train, pos_val, pos_test = build_position_features(
            _minimal_df(3), _minimal_df(2), _minimal_df(2), cfg, ["feat_a", "feat_missing"]
        )
        assert (pos_train["feat_missing"] == 0.0).all()
        assert (pos_val["feat_missing"] == 0.0).all()
        assert (pos_test["feat_missing"] == 0.0).all()

    def test_inf_and_nan_replaced_with_zero_in_feature_cols(self):
        def _add(train, val, test):
            train = train.copy()
            train["feat_a"] = [1.0, np.inf, -np.inf]
            val = val.copy()
            val["feat_a"] = [np.nan, 2.0]
            test = test.copy()
            test["feat_a"] = [3.0, 4.0]
            return train, val, test

        def _fill(train, val, test, _feats):
            return train, val, test

        cfg = {"add_features_fn": _add, "fill_nans_fn": _fill, "specific_features": ["feat_a"]}
        pos_train, pos_val, pos_test = build_position_features(
            _minimal_df(3), _minimal_df(2), _minimal_df(2), cfg, ["feat_a"]
        )
        # inf -> nan -> 0
        assert list(pos_train["feat_a"]) == [1.0, 0.0, 0.0]
        # nan -> 0
        assert list(pos_val["feat_a"]) == [0.0, 2.0]
        # unchanged
        assert list(pos_test["feat_a"]) == [3.0, 4.0]

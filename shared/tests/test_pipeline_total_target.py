"""Regression tests for the aux-loss ``total`` target in ``_prepare_position_data``.

Three regimes:

* K/DST with a non-None ``compute_adjustment_fn`` (today's main-branch state) —
  fall back to ``sum(targets)`` so the adjustment isn't double-counted.
* K/DST with ``compute_adjustment_fn is None`` (post Unit 1/2 refactors) —
  supervise on ``fantasy_points`` so heads learn the full scoring semantics.
* QB/RB/WR/TE — always ``sum(targets)``, regardless of adjustment fn, because
  their post-migration raw-stat heads don't sum to ``fantasy_points``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.pipeline import _prepare_position_data

# Feature columns the pipeline feeds into the NN. Kept small so the fixture
# DataFrames stay tiny and fast.
_FEATURE_COLS = ["feat_a", "feat_b"]


def _identity(df):
    """Filter/feature passthrough used by the minimal cfg."""
    return df


def _passthrough_features(train, val, test):
    return train, val, test


def _passthrough_fill(train, val, test, _features):
    return train, val, test


def _get_feature_cols():
    return list(_FEATURE_COLS)


def _build_min_df(n: int, targets: list[str], fp_scale: float, seed: int) -> pd.DataFrame:
    """Return a minimal per-position frame with feature cols, targets, and fantasy_points.

    ``fantasy_points`` is intentionally *not* equal to the sum of targets so we
    can tell the two aux-target paths apart.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {
        "season": np.full(n, 2023, dtype=np.int64),
        "week": np.arange(1, n + 1, dtype=np.int64) % 17 + 1,
        "player_id": np.array([f"P{i:03d}" for i in range(n)]),
        "recent_team": np.full(n, "KC"),
        "opponent_team": np.full(n, "DEN"),
    }
    for col in _FEATURE_COLS:
        cols[col] = rng.standard_normal(n).astype(np.float64)
    for t in targets:
        cols[t] = rng.random(n).astype(np.float64) * 5.0
    cols["fantasy_points"] = rng.random(n).astype(np.float64) * fp_scale + 100.0
    return pd.DataFrame(cols)


def _build_min_cfg(targets: list[str], compute_adjustment_fn=None) -> dict:
    cfg = {
        "targets": targets,
        "specific_features": list(_FEATURE_COLS),
        "filter_fn": _identity,
        "compute_targets_fn": _identity,  # targets already present in fixture
        "add_features_fn": _passthrough_features,
        "fill_nans_fn": _passthrough_fill,
        "get_feature_columns_fn": _get_feature_cols,
    }
    if compute_adjustment_fn is not None:
        cfg["compute_adjustment_fn"] = compute_adjustment_fn
    return cfg


@pytest.fixture(autouse=True)
def _stub_weather_merge(monkeypatch):
    """Skip the schedule-parquet fetch — our tiny frames don't need weather."""
    monkeypatch.setattr("shared.pipeline.merge_schedule_features", lambda df, label: df)


@pytest.fixture
def min_splits_factory():
    """Factory producing ``(train, val, test)`` frames with enough rows per player
    that the MIN_GAMES_PER_SEASON filter in ``_prepare_position_data`` doesn't
    drop all of training. Using one player with N rows is the simplest shape
    that survives that filter.
    """

    def _make(targets: list[str], fp_scale: float = 20.0, n: int = 20):
        train = _build_min_df(n, targets, fp_scale, seed=1)
        val = _build_min_df(10, targets, fp_scale, seed=2)
        test = _build_min_df(10, targets, fp_scale, seed=3)
        # Collapse to a single player so every row passes min-games filter.
        for df in (train, val, test):
            df["player_id"] = "P001"
        return train, val, test

    return _make


# ---------------------------------------------------------------------------
# Gate: compute_adjustment_fn set → fall back to sum(targets)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_k_with_adjustment_fn_uses_sum_of_targets(min_splits_factory):
    """K with a non-None ``compute_adjustment_fn`` must use ``sum(targets)``.

    This is today's main-branch state — the gate must be a no-op.
    """
    targets = ["fg_points", "pat_points"]
    train, val, test = min_splits_factory(targets)

    def adjustment_fn(df):
        return pd.Series(np.zeros(len(df)), index=df.index)

    cfg = _build_min_cfg(targets, compute_adjustment_fn=adjustment_fn)

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        "K", cfg, train, val, test
    )

    expected_train = np.sum([pos_train[t].values for t in targets], axis=0).astype(np.float32)
    expected_val = np.sum([pos_val[t].values for t in targets], axis=0).astype(np.float32)
    expected_test = np.sum([pos_test[t].values for t in targets], axis=0).astype(np.float32)

    np.testing.assert_array_equal(y_train["total"], expected_train)
    np.testing.assert_array_equal(y_val["total"], expected_val)
    np.testing.assert_array_equal(y_test["total"], expected_test)
    # Sanity: fantasy_points must not match — otherwise the test is vacuous.
    assert not np.allclose(y_train["total"], pos_train["fantasy_points"].values.astype(np.float32))


# ---------------------------------------------------------------------------
# Gate: K with adjustment_fn=None AND fantasy_points column → use fantasy_points
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_k_without_adjustment_fn_uses_fantasy_points(min_splits_factory):
    """Post-Unit-1 K (no ``compute_adjustment_fn``) must supervise on ``fantasy_points``."""
    targets = ["fg_points", "pat_points"]
    train, val, test = min_splits_factory(targets)

    cfg = _build_min_cfg(targets)  # no compute_adjustment_fn

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        "K", cfg, train, val, test
    )

    np.testing.assert_array_equal(
        y_train["total"], pos_train["fantasy_points"].values.astype(np.float32)
    )
    np.testing.assert_array_equal(
        y_val["total"], pos_val["fantasy_points"].values.astype(np.float32)
    )
    np.testing.assert_array_equal(
        y_test["total"], pos_test["fantasy_points"].values.astype(np.float32)
    )


@pytest.mark.unit
def test_dst_without_adjustment_fn_uses_fantasy_points(min_splits_factory):
    """Post-Unit-2 DST (no ``compute_adjustment_fn``) must supervise on ``fantasy_points``."""
    targets = ["defensive_scoring", "td_points", "pts_allowed_bonus"]
    train, val, test = min_splits_factory(targets)

    cfg = _build_min_cfg(targets)

    _, _, _, y_train, _, _, pos_train, _, _, _ = _prepare_position_data(
        "DST", cfg, train, val, test
    )

    np.testing.assert_array_equal(
        y_train["total"], pos_train["fantasy_points"].values.astype(np.float32)
    )


# ---------------------------------------------------------------------------
# Safety: QB/RB/WR/TE NEVER switch, even with adjustment_fn=None
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_player_positions_always_use_sum_of_targets(position, min_splits_factory):
    """QB/RB/WR/TE keep ``sum(targets)`` regardless of ``compute_adjustment_fn``.

    Post-target-migration, these positions emit RAW stats (yards, TDs,
    receptions) whose ``sum(heads)`` lives on a different scale than
    ``fantasy_points``. Switching supervision to ``fantasy_points`` while the
    NN still emits ``sum(heads)`` as ``preds["total"]`` would create a
    systematic mismatch the model cannot minimize. The explicit position gate
    in ``_prepare_position_data`` prevents this.
    """
    targets = ["t_yards", "t_tds", "t_receptions"]
    train, val, test = min_splits_factory(targets, fp_scale=500.0)

    cfg = _build_min_cfg(targets)  # no compute_adjustment_fn

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        position, cfg, train, val, test
    )

    expected_train = np.sum([pos_train[t].values for t in targets], axis=0).astype(np.float32)
    expected_val = np.sum([pos_val[t].values for t in targets], axis=0).astype(np.float32)
    expected_test = np.sum([pos_test[t].values for t in targets], axis=0).astype(np.float32)

    np.testing.assert_array_equal(y_train["total"], expected_train)
    np.testing.assert_array_equal(y_val["total"], expected_val)
    np.testing.assert_array_equal(y_test["total"], expected_test)
    # Sanity: fantasy_points *is* present but we're ignoring it for these positions.
    assert "fantasy_points" in pos_train.columns
    assert not np.allclose(y_train["total"], pos_train["fantasy_points"].values.astype(np.float32))

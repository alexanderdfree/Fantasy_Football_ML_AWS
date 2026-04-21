"""Regression tests for the aux-loss ``total`` target in ``_prepare_position_data``.

The target is chosen via a triple gate: the position must be in the
scoring-aware whitelist (K/DST), ``cfg["compute_adjustment_fn"]`` must be
``None`` (post-hoc adjustment folded into heads), and ``cfg["aggregate_fn"]``
must be set (NN is wired to emit ``preds["total"] = aggregate_fn(preds)``,
so ``sum(heads) == fantasy_points`` by construction).

Regimes exercised:

* K/DST with ``compute_adjustment_fn`` still set — fall back to ``sum(targets)``.
* K/DST with ``compute_adjustment_fn is None`` but no ``aggregate_fn`` — fall
  back to ``sum(targets)`` so ``sum(raw_heads)`` and ``targets["total"]`` stay
  consistent (the NN still emits ``sum(heads)``).
* K/DST with both ``compute_adjustment_fn is None`` and an ``aggregate_fn`` —
  supervise on ``fantasy_points`` since the NN aggregates heads into fantasy
  points.
* QB/RB/WR/TE — always ``sum(targets)``, regardless of the other flags,
  because they aren't in the whitelist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.pipeline import _nn_aggregate_fn, _prepare_position_data

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


def _noop_aggregate(preds):
    """Placeholder aggregate_fn — identity is fine; the gate only checks presence."""
    return sum(preds.values())


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


def _build_min_cfg(
    targets: list[str],
    compute_adjustment_fn=None,
    aggregate_fn=None,
) -> dict:
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
    if aggregate_fn is not None:
        cfg["aggregate_fn"] = aggregate_fn
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


def _assert_total_equals_sum(y_dict, pos_df, targets):
    expected = np.sum([pos_df[t].values for t in targets], axis=0).astype(np.float32)
    np.testing.assert_array_equal(y_dict["total"], expected)
    # Sanity: fantasy_points must not match — otherwise the test is vacuous.
    assert not np.allclose(y_dict["total"], pos_df["fantasy_points"].values.astype(np.float32))


def _assert_total_equals_fantasy_points(y_dict, pos_df):
    np.testing.assert_array_equal(
        y_dict["total"], pos_df["fantasy_points"].values.astype(np.float32)
    )


# ---------------------------------------------------------------------------
# Gate: compute_adjustment_fn set → fall back to sum(targets) even if
# aggregate_fn would otherwise be present.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_k_with_adjustment_fn_uses_sum_of_targets(min_splits_factory):
    """K with ``compute_adjustment_fn`` set must use ``sum(targets)``."""
    targets = ["fg_points", "pat_points"]
    train, val, test = min_splits_factory(targets)

    def adjustment_fn(df):
        return pd.Series(np.zeros(len(df)), index=df.index)

    # aggregate_fn present but adjustment still active → must fall back.
    cfg = _build_min_cfg(
        targets,
        compute_adjustment_fn=adjustment_fn,
        aggregate_fn=_noop_aggregate,
    )

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        "K", cfg, train, val, test
    )

    _assert_total_equals_sum(y_train, pos_train, targets)
    _assert_total_equals_sum(y_val, pos_val, targets)
    _assert_total_equals_sum(y_test, pos_test, targets)


# ---------------------------------------------------------------------------
# Gate: adjustment_fn is None but aggregate_fn is missing → fall back.
# Without aggregate_fn the NN emits ``sum(raw_heads)`` so ``fantasy_points``
# supervision would be systematically unreachable.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("position", ["K", "DST"])
def test_without_aggregate_fn_uses_sum_of_targets(position, min_splits_factory):
    """K/DST without ``aggregate_fn`` keep ``sum(targets)`` supervision."""
    targets = ["scoring_a", "scoring_b", "scoring_c"]
    train, val, test = min_splits_factory(targets)

    cfg = _build_min_cfg(targets)  # neither compute_adjustment_fn nor aggregate_fn

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        position, cfg, train, val, test
    )

    _assert_total_equals_sum(y_train, pos_train, targets)
    _assert_total_equals_sum(y_val, pos_val, targets)
    _assert_total_equals_sum(y_test, pos_test, targets)


# ---------------------------------------------------------------------------
# Gate: K/DST with adjustment_fn=None, aggregate_fn wired, fantasy_points
# present → supervise on fantasy_points.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_k_with_aggregate_fn_uses_fantasy_points(min_splits_factory):
    """K with ``aggregate_fn`` set must supervise on ``fantasy_points``."""
    targets = ["fg_points", "pat_points"]
    train, val, test = min_splits_factory(targets)

    cfg = _build_min_cfg(targets, aggregate_fn=_noop_aggregate)

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        "K", cfg, train, val, test
    )

    _assert_total_equals_fantasy_points(y_train, pos_train)
    _assert_total_equals_fantasy_points(y_val, pos_val)
    _assert_total_equals_fantasy_points(y_test, pos_test)


@pytest.mark.unit
def test_dst_with_aggregate_fn_uses_fantasy_points(min_splits_factory):
    """DST with ``aggregate_fn`` set must supervise on ``fantasy_points``."""
    targets = ["defensive_scoring", "td_points", "pts_allowed_bonus"]
    train, val, test = min_splits_factory(targets)

    cfg = _build_min_cfg(targets, aggregate_fn=_noop_aggregate)

    _, _, _, y_train, _, _, pos_train, _, _, _ = _prepare_position_data(
        "DST", cfg, train, val, test
    )

    _assert_total_equals_fantasy_points(y_train, pos_train)


# ---------------------------------------------------------------------------
# Safety: QB/RB/WR/TE NEVER switch, even when adjustment_fn=None and
# aggregate_fn is wired. They're not in the scoring-aware whitelist.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: _nn_aggregate_fn must return None for QB/RB/WR/TE even when cfg has
# an ``aggregate_fn`` (they set it for *inference*, not for NN training — the
# NN aux target for them is ``sum(raw targets)``, not ``fantasy_points``).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_nn_aggregate_fn_returns_none_for_raw_stat_positions(position):
    """QB/RB/WR/TE: inference-time ``aggregate_fn`` in cfg must NOT be wired
    into the NN training graph. Regression against the scale mismatch where
    ``aggregate_fn(preds)`` (fantasy-points scale) would be compared to
    ``sum(targets)`` (raw-stat scale) in the aux loss.
    """
    cfg = {"aggregate_fn": _noop_aggregate, "compute_adjustment_fn": None}
    assert _nn_aggregate_fn(position, cfg) is None


@pytest.mark.unit
@pytest.mark.parametrize("position", ["K", "DST"])
def test_nn_aggregate_fn_forwards_for_whitelisted_positions(position):
    """K/DST with ``compute_adjustment_fn is None`` and an ``aggregate_fn``
    set must get that callable wired into the NN so ``preds["total"]`` lives
    in fantasy-points space."""
    cfg = {"aggregate_fn": _noop_aggregate, "compute_adjustment_fn": None}
    assert _nn_aggregate_fn(position, cfg) is _noop_aggregate


@pytest.mark.unit
@pytest.mark.parametrize("position", ["K", "DST"])
def test_nn_aggregate_fn_blocked_when_adjustment_fn_set(position):
    """K/DST with a non-None ``compute_adjustment_fn`` must NOT wire
    aggregate_fn — the adjustment would be applied twice (once inside
    aggregate_fn, once post-hoc)."""
    cfg = {
        "aggregate_fn": _noop_aggregate,
        "compute_adjustment_fn": lambda df: df,
    }
    assert _nn_aggregate_fn(position, cfg) is None


@pytest.mark.unit
@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE"])
def test_player_positions_always_use_sum_of_targets(position, min_splits_factory):
    """QB/RB/WR/TE keep ``sum(targets)`` regardless of the other gate inputs.

    Post-target-migration, these positions emit RAW stats (yards, TDs,
    receptions) whose ``sum(heads)`` lives on a different scale than
    ``fantasy_points``, and the aux loss compares ``sum(heads) ≈ sum(targets)``
    by construction. The whitelist gate prevents an accidental ``fantasy_points``
    supervision for them even if a misconfigured cfg sets ``aggregate_fn``.
    """
    targets = ["t_yards", "t_tds", "t_receptions"]
    train, val, test = min_splits_factory(targets, fp_scale=500.0)

    # Both adjustment-free AND aggregate_fn present — whitelist still blocks.
    cfg = _build_min_cfg(targets, aggregate_fn=_noop_aggregate)

    _, _, _, y_train, y_val, y_test, pos_train, pos_val, pos_test, _ = _prepare_position_data(
        position, cfg, train, val, test
    )

    _assert_total_equals_sum(y_train, pos_train, targets)
    _assert_total_equals_sum(y_val, pos_val, targets)
    _assert_total_equals_sum(y_test, pos_test, targets)
    assert "fantasy_points" in pos_train.columns

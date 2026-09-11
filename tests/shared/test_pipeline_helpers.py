"""Coverage tests for small helpers inside ``src/shared/pipeline.py``.

These functions aren't exercised by the position-level E2E tests because
E2E goes through ``run_pipeline`` which only hits the onecycle+ridge path.
Direct tests target the uncovered branches: onecycle/cosine/plateau
scheduler dispatch + the unknown-scheduler ValueError, ElasticNet
alpha/l1_ratio CV tuner (coarse + fine pass), the ``_read_split``
parquet helper, and the ``_attn_saved_static_cols`` attention-artifact
metadata resolver (#1432).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# --------------------------------------------------------------------------
# _read_split
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_read_split_loads_parquet(tmp_path):
    from src.shared.pipeline import _read_split

    path = tmp_path / "train.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)
    out = _read_split(str(path))
    assert len(out) == 3


# --------------------------------------------------------------------------
# _build_scheduler — all four branches
# --------------------------------------------------------------------------


def _dummy_optim_and_loader():
    """Minimal optimizer + single-batch DataLoader; _build_scheduler reads
    len(train_loader) for OneCycleLR's steps_per_epoch calculation."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1e-3)

    class _Loader:
        def __len__(self):
            return 10

    return opt, _Loader()


@pytest.mark.unit
def test_build_scheduler_onecycle():
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        "scheduler_type": "onecycle",
        "onecycle_max_lr": 0.01,
        "nn_epochs": 2,
        "onecycle_pct_start": 0.3,
    }
    sched, per_batch = _build_scheduler(opt, cfg, loader)
    assert per_batch is True
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)


@pytest.mark.unit
def test_build_scheduler_cosine_warm_restarts():
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 5,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
    }
    sched, per_batch = _build_scheduler(opt, cfg, loader)
    assert per_batch is False
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


@pytest.mark.unit
def test_build_scheduler_prefers_attention_overrides():
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 5,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
        "attn_cosine_eta_min": 2e-5,
    }
    sched, per_batch = _build_scheduler(opt, cfg, loader, scheduler_prefix="attn_")
    assert per_batch is False
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
    assert sched.eta_min == pytest.approx(2e-5)


@pytest.mark.unit
def test_build_scheduler_prefers_attention_scheduler_type():
    """#792: the attention path (scheduler_prefix='attn_') uses attn_scheduler_type
    over the shared scheduler_type, so a tuned attention scheduler can no longer
    re-schedule the regular NN. The regular path (no prefix) is unchanged."""
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        # Shared (regular NN) schedule = cosine.
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 5,
        "cosine_t_mult": 2,
        "cosine_eta_min": 1e-5,
        "nn_epochs": 2,
        # Attention-only overrides = onecycle with its own shape.
        "attn_scheduler_type": "onecycle",
        "attn_onecycle_max_lr": 0.02,
        "attn_onecycle_pct_start": 0.25,
    }
    sched, per_batch = _build_scheduler(opt, cfg, loader, scheduler_prefix="attn_")
    assert per_batch is True
    assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
    # Regular NN path still uses the shared type (cosine) — Δ0 vs pre-#792.
    sched2, per_batch2 = _build_scheduler(opt, cfg, loader)
    assert per_batch2 is False
    assert isinstance(sched2, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


@pytest.mark.unit
def test_build_scheduler_prefers_attention_cosine_shape():
    """#792: attn-prefixed cosine T_0/T_mult override the shared values on the
    attention path; the shared values still drive the regular NN."""
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        "scheduler_type": "cosine_warm_restarts",
        "cosine_t0": 5,
        "cosine_t_mult": 1,
        "cosine_eta_min": 1e-5,
        "attn_cosine_t0": 30,
        "attn_cosine_t_mult": 2,
    }
    sched, _ = _build_scheduler(opt, cfg, loader, scheduler_prefix="attn_")
    assert sched.T_0 == 30
    assert sched.T_mult == 2
    sched2, _ = _build_scheduler(opt, cfg, loader)
    assert sched2.T_0 == 5
    assert sched2.T_mult == 1


@pytest.mark.unit
def test_build_scheduler_plateau():
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    cfg = {
        "scheduler_type": "plateau",
        "plateau_factor": 0.5,
        "plateau_patience": 3,
    }
    sched, per_batch = _build_scheduler(opt, cfg, loader)
    assert per_batch is False
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


@pytest.mark.unit
def test_build_scheduler_unknown_raises():
    from src.shared.pipeline import _build_scheduler

    opt, loader = _dummy_optim_and_loader()
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        _build_scheduler(opt, {"scheduler_type": "bogus"}, loader)


# --------------------------------------------------------------------------
# _eval_enet_cv / _tune_enet_cv
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_eval_enet_cv_returns_positive_mae():
    """Single (alpha, l1_ratio) eval over a tiny fold set returns a finite
    non-negative MAE."""
    from src.shared.pipeline import _eval_enet_cv

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    # Linear target with noise: y = 2*x0 + 0.5*x1 + noise (non-negative range).
    y = np.clip(2 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, n), 0, None)
    folds = [(np.arange(0, 40), np.arange(40, n))]
    mae = _eval_enet_cv(X, y, folds, alpha=1.0, l1_ratio=0.5)
    assert np.isfinite(mae)
    assert mae >= 0


@pytest.mark.unit
def test_tune_enet_cv_returns_best_alpha_and_l1_ratio():
    """Tuner runs the coarse + fine search across two targets and returns
    ``{target: {alpha, l1_ratio}}``."""
    from src.shared.pipeline import _tune_enet_cv

    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 4))
    y_dict = {
        "target_a": np.clip(X[:, 0] + rng.normal(0, 0.2, n), 0, None),
        "target_b": np.clip(0.5 * X[:, 1] + rng.normal(0, 0.2, n), 0, None),
    }
    split_values = np.concatenate(
        [np.full(25, 2020), np.full(25, 2021), np.full(25, 2022), np.full(25, 2023)]
    )
    alpha_grids = {"target_a": [0.1, 1.0, 10.0], "target_b": [0.1, 1.0, 10.0]}
    best = _tune_enet_cv(
        X_train=X,
        y_train_dict=y_dict,
        split_values=split_values,
        targets=["target_a", "target_b"],
        alpha_grids=alpha_grids,
        l1_ratios=[0.1, 0.5, 0.9],
        n_cv_folds=2,
        refine_points=3,
    )
    assert set(best) == {"target_a", "target_b"}
    for t in best:
        assert "alpha" in best[t]
        assert "l1_ratio" in best[t]
        assert 0 <= best[t]["l1_ratio"] <= 1


@pytest.mark.unit
def test_tune_enet_cv_no_refinement_when_refine_points_zero():
    """refine_points=0 → only the coarse-grid pass fires (no logspace search)."""
    from src.shared.pipeline import _tune_enet_cv

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    y_dict = {"t": np.abs(X[:, 0] + rng.normal(0, 0.2, n))}
    split_values = np.concatenate([np.full(20, 2020), np.full(20, 2021), np.full(20, 2022)])
    best = _tune_enet_cv(
        X_train=X,
        y_train_dict=y_dict,
        split_values=split_values,
        targets=["t"],
        alpha_grids={"t": [0.1, 1.0]},
        l1_ratios=[0.5],
        n_cv_folds=2,
        refine_points=0,
    )
    # alpha must be one of the coarse-grid values (no refinement).
    assert best["t"]["alpha"] in {0.1, 1.0}


# --------------------------------------------------------------------------
# _build_expanding_cv_folds
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# _attn_saved_static_cols — attention checkpoint/scaler metadata (#1432)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_attn_saved_static_cols_filters_flat_path():
    """Flat path (``attn_static_from_df`` unset/False — QB/RB/WR/TE/DST):
    ``_train_attention_nn`` fit the scaler on the get_attn_static_columns-
    filtered subset, so the saved metadata must be that subset (in base
    feature order), NOT the full base feature list ``_train_attention_holdout``
    returns — otherwise ``n_features`` over-reports and
    ``assert_scaler_matches`` fails at serving load. This is the resolver
    ``run_pipeline`` AND ``run_cv_pipeline`` both save through."""
    from src.shared.pipeline import _attn_saved_static_cols

    feature_cols = ["prior_a", "rolling_b", "matchup_c", "ewma_d"]
    cfg = {"attn_static_features": ["matchup_c", "prior_a"]}
    # Filtered to the whitelist, preserving feature_cols order.
    assert _attn_saved_static_cols(cfg, feature_cols) == ["prior_a", "matchup_c"]


@pytest.mark.unit
def test_attn_saved_static_cols_from_df_passthrough():
    """from-df path (``attn_static_from_df=True`` — K): the nested trainer fit
    on ``attn_feature_cols`` as-is (already ``cfg["attn_static_features"]``),
    so the resolver must NOT re-filter."""
    from src.shared.pipeline import _attn_saved_static_cols

    cols = ["kick_dist_a", "kick_dist_b"]
    cfg = {"attn_static_from_df": True, "attn_static_features": ["kick_dist_a"]}
    assert _attn_saved_static_cols(cfg, cols) == cols


@pytest.mark.unit
def test_build_expanding_cv_folds_contiguous_splits():
    """3 distinct split_values → 2 folds: [v0] train → [v1] val,
    [v0, v1] train → [v2] val."""
    from src.shared.pipeline import _build_expanding_cv_folds

    split_values = np.array([2020] * 10 + [2021] * 10 + [2022] * 10)
    folds = _build_expanding_cv_folds(split_values, n_folds=2)
    assert len(folds) == 2
    # First fold: train on 2020, val on 2021
    tr0, va0 = folds[0]
    assert set(split_values[tr0]) == {2020}
    assert set(split_values[va0]) == {2021}
    # Second fold: train on {2020, 2021}, val on 2022
    tr1, va1 = folds[1]
    assert set(split_values[tr1]) == {2020, 2021}
    assert set(split_values[va1]) == {2022}


# --------------------------------------------------------------------------
# _scale_xs — bounded-flag scaling wiring (nn_bounded_flag_scaling)
# --------------------------------------------------------------------------


def _flag_arrays(seed: int = 0):
    """(X_train, X_test, cols) with a realistic ~4%-prevalence game_status."""
    rng = np.random.default_rng(seed)
    gs = np.where(rng.random(500) < 0.04, 0.5, 1.0)
    X_train = np.column_stack([rng.standard_normal(500), gs, rng.standard_normal(500)])
    X_test = np.column_stack([rng.standard_normal(50), np.full(50, 0.5), rng.standard_normal(50)])
    return X_train, X_test, ["other", "game_status", "another"]


@pytest.mark.unit
def test_scale_xs_off_matches_legacy_fit_transform():
    """Regression guard: the knob-off path must stay byte-identical to the
    pre-change ``scale_and_clip(..., fit=True)`` implementation, so production
    (nn_bounded_flag_scaling=False) is provably untouched."""
    from sklearn.preprocessing import StandardScaler

    from src.shared.feature_build import scale_and_clip
    from src.shared.pipeline import _scale_xs

    X_train, X_test, cols = _flag_arrays()
    legacy_scaler = StandardScaler()
    legacy = [
        scale_and_clip(legacy_scaler, X_train, fit=True),
        scale_and_clip(legacy_scaler, X_test),
    ]

    for cfg in (None, {}, {"nn_bounded_flag_range": None}):
        scaler, scaled = _scale_xs(X_train, X_test, cfg=cfg, feature_cols=cols)
        np.testing.assert_array_equal(scaled[0], legacy[0])
        np.testing.assert_array_equal(scaled[1], legacy[1])
        np.testing.assert_array_equal(scaler.mean_, legacy_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, legacy_scaler.scale_)


@pytest.mark.unit
@pytest.mark.parametrize("flag_range,expected_scale", [(1.0, 0.5), (4.0, 0.125)])
def test_scale_xs_on_applies_the_flag_override(flag_range, expected_scale):
    from src.shared.pipeline import _scale_xs

    X_train, X_test, cols = _flag_arrays()
    scaler, scaled = _scale_xs(
        X_train, X_test, cfg={"nn_bounded_flag_range": flag_range}, feature_cols=cols
    )
    assert scaler.mean_[1] == 0.5
    assert scaler.scale_[1] == expected_scale
    # Every test row is Questionable (0.5) -> 0.0 under the override, whereas
    # the plain scaler pins them all to the -4 clip floor.
    np.testing.assert_allclose(scaled[1][:, 1], 0.0, atol=1e-12)
    # Non-flag columns still get ordinary standardization.
    np.testing.assert_allclose(scaler.mean_[0], X_train[:, 0].mean(), atol=1e-12)


@pytest.mark.unit
def test_scale_xs_on_without_columns_raises():
    """Degrading to plain scaling would be invisible in a fleet run's table."""
    from src.shared.pipeline import _scale_xs

    X_train, X_test, _ = _flag_arrays()
    with pytest.raises(ValueError, match="no column list reached"):
        _scale_xs(X_train, X_test, cfg={"nn_bounded_flag_range": 1.0}, feature_cols=None)


@pytest.mark.unit
def test_scale_xs_on_with_no_matching_column_raises():
    """The silent-no-op trap: the knob is on, but nothing it targets is being
    scaled. The Ridge sentinel cannot catch this (Ridge never sees NN config
    either way), so it must fail here or a never-applied arm reads as
    'no effect' in the aggregate table."""
    from src.shared.pipeline import _scale_xs

    X_train, X_test, _ = _flag_arrays()
    with pytest.raises(ValueError, match="silent no-op"):
        _scale_xs(X_train, X_test, cfg={"nn_bounded_flag_range": 1.0}, feature_cols=["a", "b", "c"])


@pytest.mark.unit
@pytest.mark.parametrize("flag_range", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_xs_rejects_invalid_configured_flag_ranges(flag_range):
    from src.shared.pipeline import _scale_xs

    X_train, X_test, cols = _flag_arrays()
    with pytest.raises(ValueError, match="must be positive|exceeds FEATURE_CLIP"):
        _scale_xs(X_train, X_test, cfg={"nn_bounded_flag_range": flag_range}, feature_cols=cols)

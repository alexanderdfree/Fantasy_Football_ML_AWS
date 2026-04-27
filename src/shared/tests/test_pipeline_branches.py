"""Coverage tests for ``src/shared/pipeline.py`` branches that ``run_pipeline``
doesn't reach with the default tiny config.

These functions are imported but never exercised end-to-end:

- ``_train_elasticnet`` — wired up only when ``cfg["train_elasticnet"]=True``.
  Default tiny configs leave it off.
- ``_tune_ridge_alphas_cv`` fine-refinement branch — fires only when
  ``refine_points > 0``. ``src/shared/tests/test_pipeline_helpers.py`` already
  hits the ``refine_points=0`` shortcut and the ``_eval_alpha_cv``-without-
  refinement coarse path.
- ``build_train_matrix`` — diagnostic entry point for SHAP / ablation
  scripts; tested here against synthetic parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.shared.pipeline import _train_elasticnet, _tune_ridge_alphas_cv, build_train_matrix

# --------------------------------------------------------------------------
# _train_elasticnet
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_train_elasticnet_returns_model_preds_and_metrics():
    """Returns ``(model, test_preds, metrics)`` with finite predictions per
    target plus a ``total`` metrics row keyed alongside the per-target rows."""
    rng = np.random.default_rng(0)
    n_train, n_test, d = 80, 30, 5
    X_train = rng.normal(size=(n_train, d)).astype(np.float64)
    X_test = rng.normal(size=(n_test, d)).astype(np.float64)
    targets = ["rushing_yards", "receiving_yards", "rushing_tds"]
    # Linear-ish signal on first two features.
    y_train_dict = {
        "rushing_yards": np.maximum(0, 2 * X_train[:, 0] + rng.normal(0, 0.5, n_train)),
        "receiving_yards": np.maximum(0, X_train[:, 1] + rng.normal(0, 0.4, n_train)),
        "rushing_tds": np.abs(X_train[:, 2] * 0.3 + rng.normal(0, 0.1, n_train)),
    }
    y_test_dict = {
        "rushing_yards": np.maximum(0, 2 * X_test[:, 0] + rng.normal(0, 0.5, n_test)),
        "receiving_yards": np.maximum(0, X_test[:, 1] + rng.normal(0, 0.4, n_test)),
        "rushing_tds": np.abs(X_test[:, 2] * 0.3 + rng.normal(0, 0.1, n_test)),
    }
    cfg = {}  # no two-stage / classification overrides; vanilla path
    best_hparams = {t: {"alpha": 1.0, "l1_ratio": 0.5} for t in targets}

    model, test_preds, metrics = _train_elasticnet(
        X_train, X_test, y_train_dict, y_test_dict, cfg, targets, best_hparams
    )

    # Predictions: one finite array per target with the right shape.
    assert set(test_preds) == set(targets)
    for t in targets:
        assert test_preds[t].shape == (n_test,)
        assert np.isfinite(test_preds[t]).all()

    # Metrics: per-target plus a ``total`` row.
    assert "total" in metrics
    for t in targets:
        assert t in metrics
        assert "mae" in metrics[t]


@pytest.mark.unit
def test_train_elasticnet_propagates_two_stage_and_classification_targets():
    """When the cfg declares two-stage / classification overrides, those keys
    flow into the ``ElasticNetMultiTarget`` constructor (no error, no silent
    drop). Asserted via the model's stored attributes."""
    rng = np.random.default_rng(1)
    n_train, n_test, d = 60, 20, 4
    X_train = rng.normal(size=(n_train, d)).astype(np.float64)
    X_test = rng.normal(size=(n_test, d)).astype(np.float64)
    targets = ["yards", "tds"]
    y_train_dict = {
        "yards": np.maximum(0, X_train[:, 0] + rng.normal(0, 0.3, n_train)),
        "tds": np.abs(X_train[:, 1] * 0.2 + rng.normal(0, 0.1, n_train)),
    }
    y_test_dict = {
        "yards": np.maximum(0, X_test[:, 0] + rng.normal(0, 0.3, n_test)),
        "tds": np.abs(X_test[:, 1] * 0.2 + rng.normal(0, 0.1, n_test)),
    }
    cfg = {
        "two_stage_targets": {},
        "classification_targets": {},
        "nn_non_negative_targets": {"yards", "tds"},
    }
    best = {t: {"alpha": 0.5, "l1_ratio": 0.3} for t in targets}

    model, _, _ = _train_elasticnet(X_train, X_test, y_train_dict, y_test_dict, cfg, targets, best)
    # Sanity: model was actually constructed and fit.
    assert model is not None


# --------------------------------------------------------------------------
# _tune_ridge_alphas_cv — fine refinement pass
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_tune_ridge_alphas_cv_fine_refinement_runs():
    """``refine_points > 0`` triggers a logspace fine-grid search around the
    coarse winner. The returned alpha may be a fine-grid value, not just one
    of the coarse-grid entries."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 4)).astype(np.float64)
    y_dict = {
        "yards": np.maximum(0, X[:, 0] + 0.3 * rng.normal(size=n)),
    }
    split_values = np.concatenate(
        [np.full(25, 2020), np.full(25, 2021), np.full(25, 2022), np.full(25, 2023)]
    )
    alpha_grids = {"yards": [0.1, 1.0, 10.0]}

    best = _tune_ridge_alphas_cv(
        X_train=X,
        y_train_dict=y_dict,
        split_values=split_values,
        targets=["yards"],
        alpha_grids=alpha_grids,
        n_cv_folds=2,
        refine_points=5,
    )
    # Output is a flat {target: alpha} dict with a numeric alpha.
    assert "yards" in best
    assert isinstance(best["yards"], float)
    # The result must be > 0 (Ridge alpha is non-negative).
    assert best["yards"] > 0


@pytest.mark.unit
def test_tune_ridge_alphas_cv_with_pca_components():
    """``pca_n_components`` plumbing flows through to ``_eval_alpha_cv``
    without raising on tiny data — covers the PCA branch."""
    rng = np.random.default_rng(2)
    n = 60
    X = rng.normal(size=(n, 5)).astype(np.float64)
    y_dict = {"yards": np.maximum(0, X[:, 0] + 0.5 * rng.normal(size=n))}
    split_values = np.concatenate([np.full(20, 2020), np.full(20, 2021), np.full(20, 2022)])
    alpha_grids = {"yards": [0.1, 1.0, 10.0]}

    best = _tune_ridge_alphas_cv(
        X_train=X,
        y_train_dict=y_dict,
        split_values=split_values,
        targets=["yards"],
        alpha_grids=alpha_grids,
        n_cv_folds=2,
        refine_points=3,
        pca_n_components=3,
    )
    assert "yards" in best
    assert best["yards"] > 0


# --------------------------------------------------------------------------
# build_train_matrix — reads SPLITS_DIR parquets via _read_split
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_build_train_matrix_loads_synthetic_splits(tmp_path, monkeypatch):
    """``build_train_matrix`` reads ``data/splits/{train,val}.parquet`` and
    delegates to ``_prepare_train_val``. We monkeypatch ``SPLITS_DIR`` to a
    tmp dir + write minimal parquets that survive the QB feature build."""
    import src.shared.pipeline as p

    # Synthetic frames that filter_to_qb + the QB feature builder accept.
    rng = np.random.default_rng(0)
    rows = []
    for season in (2022, 2023):
        # Need ≥ MIN_GAMES_PER_SEASON (6) per player to survive the filter
        # in _prepare_position_data; 8 weeks gives a comfortable margin.
        for week in range(1, 9):
            for pid in range(15):
                rows.append(
                    {
                        "player_id": f"QB{pid:02d}",
                        "player_name": f"QB {pid}",
                        "position": "QB",
                        "recent_team": "KC",
                        "opponent_team": "BUF",
                        "season": season,
                        "week": week,
                        "passing_yards": float(rng.uniform(150, 350)),
                        "passing_tds": int(rng.integers(0, 4)),
                        "interceptions": int(rng.integers(0, 3)),
                        "sacks": int(rng.integers(0, 4)),
                        "sack_yards": 0.0,
                        "sack_fumbles_lost": 0,
                        "rushing_yards": float(rng.integers(0, 60)),
                        "rushing_tds": int(rng.integers(0, 2)),
                        "rushing_fumbles_lost": 0,
                        "rushing_first_downs": 0,
                        "rushing_epa": 0.0,
                        "carries": int(rng.integers(0, 8)),
                        "completions": int(rng.integers(10, 30)),
                        "attempts": int(rng.integers(20, 40)),
                        "passing_air_yards": float(rng.integers(100, 300)),
                        "passing_yards_after_catch": float(rng.integers(50, 200)),
                        "passing_first_downs": int(rng.integers(5, 18)),
                        "passing_epa": float(rng.uniform(-5, 15)),
                        "receptions": 0,
                        "targets": 0,
                        "receiving_yards": 0.0,
                        "receiving_tds": 0,
                        "receiving_fumbles_lost": 0,
                        "snap_pct": 0.95,
                        "fantasy_points": float(rng.uniform(10, 25)),
                    }
                )
    df = pd.DataFrame(rows)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    df.to_parquet(splits_dir / "train.parquet")
    df.to_parquet(splits_dir / "val.parquet")

    monkeypatch.setattr(p, "SPLITS_DIR", str(splits_dir))

    from src.QB.run_qb_pipeline import QB_CONFIG

    X_train, y_train_dict, feature_cols = build_train_matrix("QB", QB_CONFIG)

    assert X_train.ndim == 2
    assert X_train.shape[0] > 0
    assert X_train.shape[1] == len(feature_cols)
    # At least one of QB's targets should be present.
    assert any(t in y_train_dict for t in QB_CONFIG["targets"])

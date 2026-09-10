"""Joint raw-stat model selection must score the final fantasy prediction."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.shared import model_selection as selection
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.pipeline import _tune_ridge_alphas_cv
from src.shared.registry import get_config

pytestmark = pytest.mark.unit


def test_ridge_scores_joint_ppr_instead_of_independent_head_rmse(monkeypatch):
    cfg = get_config("K")
    targets = cfg["targets"]

    def candidate(X, y, folds, target, alpha, config, pca):
        if target == "fg_yard_points":
            pred = np.array([0.0, 4.0]) if alpha == 1.0 else np.array([2.0, 2.0])
        else:
            pred = np.array([0.0, 4.0]) if target == "fg_misses" else np.zeros(2)
        return target, alpha, [pred.copy() for _ in folds]

    monkeypatch.setattr(selection, "_ridge_candidate_predictions", candidate)
    info = {}
    alphas = _tune_ridge_alphas_cv(
        np.zeros((4, 2)),
        {t: np.zeros(4) for t in targets},
        np.array([1, 1, 2, 2]),
        targets,
        {t: [1.0, 2.0] if t == "fg_yard_points" else [1.0] for t in targets},
        n_cv_folds=1,
        refine_points=0,
        n_jobs=1,
        cfg=cfg,
        selection_info=info,
    )
    # alpha=2 has lower individual yard-point RMSE; alpha=1 cancels the miss
    # contribution and has zero total error with K's signed scoring formula.
    assert alphas["fg_yard_points"] == 1.0
    assert info["score"] == 0.0
    assert info["metric"] == "mean_cv_fantasy_rmse_ppr"
    assert all(
        a >= b for a, b in zip(info["score_history"][:-1], info["score_history"][1:], strict=True)
    )


def test_ridge_keeps_special_heads_in_every_candidate_score(monkeypatch):
    cfg = get_config("RB")
    cfg["classification_targets"] = {"fumbles_lost": {"type": "ordinal"}}
    targets = [t for t in cfg["targets"] if t != "fumbles_lost"]
    calls = []

    def candidate(X, y, folds, target, alpha, config, pca):
        calls.append((target, alpha))
        value = (
            3.0
            if target == "fumbles_lost"
            else 60.0
            if target == "rushing_yards" and alpha == 2
            else 0.0
        )
        return target, alpha, [np.full(len(v), value) for _, v in folds]

    monkeypatch.setattr(selection, "_ridge_candidate_predictions", candidate)
    best = selection.tune_ridge_ppr(
        np.zeros((4, 2)),
        {t: np.zeros(4) for t in cfg["targets"]},
        [(np.array([0, 1]), np.array([2, 3]))],
        targets,
        {t: [1.0, 2.0] if t == "rushing_yards" else [1.0] for t in targets},
        cfg,
        refine_points=0,
        n_jobs=1,
    )
    assert best["rushing_yards"] == 2.0  # +6 yards points offset -6 fumble points.
    assert calls.count(("fumbles_lost", 1.0)) == 1


def test_ridge_candidates_fit_only_their_training_fold(monkeypatch):
    seen = []

    class FakeModel:
        def __init__(self, targets, **kwargs):
            self.target = targets[0]
            seen.append(kwargs)

        def fit(self, X, y):
            np.testing.assert_array_equal(X[:, 0], [0, 1])
            np.testing.assert_array_equal(y[self.target], [10, 11])

        def predict(self, X):
            np.testing.assert_array_equal(X[:, 0], [2, 3])
            return {self.target: np.ones(2)}

    monkeypatch.setattr("src.shared.models.RidgeMultiTarget", FakeModel)
    selection._ridge_candidate_predictions(
        np.arange(4).reshape(-1, 1),
        {"receptions": np.arange(10, 14)},
        [(np.array([0, 1]), np.array([2, 3]))],
        "receptions",
        2.0,
        {"classification_targets": {}, "nn_non_negative_targets": {"receptions"}},
        1,
    )
    assert seen[0]["pca_n_components"] == 1
    assert seen[0]["non_negative_targets"] == {"receptions"}


def test_lightgbm_selects_joint_prefix_even_when_head_rmse_prefers_another():
    cfg = get_config("K")
    model = LightGBMMultiTarget(cfg["targets"], selection_metric="fantasy_rmse_ppr")

    class FakeHead:
        def __init__(self, prefixes, rmse):
            self.prefixes = np.array(prefixes, dtype=float)
            self.evals_result_ = {"valid_0": {"rmse": rmse}}
            self.booster_ = SimpleNamespace(
                params={"num_threads": 1},
                current_iteration=lambda: 2,
                predict=lambda X, start_iteration, **kw: (
                    self.prefixes[start_iteration]
                    - (self.prefixes[start_iteration - 1] if start_iteration else 0)
                ),
            )

        def predict(self, X, num_iteration=2):
            return self.prefixes[num_iteration - 1].copy()

    model._models = {
        t: FakeHead([[0, 0], [4, 4]], [0, 4])
        if t == "fg_yard_points"
        else FakeHead([[4, 4], [4, 4]], [4, 4])
        if t == "fg_misses"
        else FakeHead([[0, 0], [0, 0]], [0, 0])
        for t in cfg["targets"]
    }
    model._select_ppr_iterations(np.zeros((2, 1)), {t: np.zeros(2) for t in cfg["targets"]})
    assert model.selected_iterations["fg_yard_points"] == 2
    assert model.selection_info["score"] == 0.0
    assert model.selection_info["initial_score"] == 4.0


def test_real_lightgbm_prefixes_and_roundtrip_match_selected_predictions(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(48, 3))
    targets = get_config("K")["targets"]
    y = {t: np.maximum(0, 10 + (i + 1) * X[:36, 0]) for i, t in enumerate(targets)}
    names = ["a", "b", "c"]
    params = dict(
        n_estimators=8,
        num_leaves=4,
        min_child_samples=2,
        learning_rate=0.2,
        objective="regression",
        n_jobs=1,
    )
    reference = LightGBMMultiTarget(targets, **params)
    reference.fit(X[:36], y, feature_names=names)
    val = pd.DataFrame(X[36:], columns=names)
    # Make round 1 exactly optimal so the persistence test must honor a
    # non-default tree prefix rather than coincidentally selecting all trees.
    truth = {t: reference._models[t].predict(val, num_iteration=1) for t in targets}
    model = LightGBMMultiTarget(targets, selection_metric="fantasy_rmse_ppr", **params)
    model.fit(X[:36], y, X[36:], truth, feature_names=names)
    assert model.selection_info["score"] == pytest.approx(0.0, abs=1e-10)
    assert set(model.selected_iterations.values()) == {1}
    for target in targets:
        for iteration, predicted in model._prefix_predictions(target, val):
            expected = np.maximum(model._models[target].predict(val, num_iteration=iteration), 0)
            np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-12)
    expected = model.predict(X[36:])
    model.save(tmp_path)
    loaded = LightGBMMultiTarget(targets)
    loaded.load(tmp_path)
    assert loaded.selection_info == model.selection_info
    assert all(m.booster_.current_iteration() == 1 for m in loaded._models.values())
    for target, values in loaded.predict(X[36:]).items():
        np.testing.assert_array_equal(values, expected[target])
    assert set(loaded.get_feature_importance(names)) == set(targets)


def test_ridge_selection_metadata_roundtrip(tmp_path):
    model = RidgeMultiTarget(["x"])
    model.fit(np.arange(6).reshape(-1, 1), {"x": np.arange(6)})
    model.selection_info = {"metric": "mean_cv_fantasy_rmse_ppr", "score": 1.2}
    model.save(tmp_path)
    loaded = RidgeMultiTarget(["x"])
    loaded.load(tmp_path)
    assert loaded.selection_info == model.selection_info
    loaded.fit(np.arange(6).reshape(-1, 1), {"x": np.arange(6)})
    assert loaded.selection_info is None
    model.selection_info = None
    model.save(tmp_path)
    assert not (tmp_path / "ridge_selection.json").exists()


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_all_positions_enable_joint_selection(position):
    cfg = get_config(position)
    assert cfg["ridge_selection_metric"] == "fantasy_rmse_ppr"
    assert cfg["lgbm_selection_metric"] == "fantasy_rmse_ppr"
    assert cfg["lgbm_objective"] == "regression"

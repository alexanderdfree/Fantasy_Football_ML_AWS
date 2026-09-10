"""Validation-only selection of raw-stat models using canonical PPR scoring."""

import numpy as np
from joblib import Parallel, delayed

from src.shared.aggregate_targets import infer_position, predictions_to_fantasy_points


def selection_position(targets):
    position = infer_position(targets)
    if position is None:
        raise ValueError("PPR selection requires a complete known position target set")
    return position


def ppr_rmse(position, truth, predictions):
    actual = predictions_to_fantasy_points(position, truth, "ppr")
    predicted = predictions_to_fantasy_points(position, predictions, "ppr")
    score = float(np.sqrt(np.mean(np.square(predicted - actual))))
    if not np.isfinite(score):
        raise ValueError("Non-finite PPR validation RMSE")
    return score


def _ridge_candidate_predictions(X, y, folds, target, alpha, cfg, pca_n_components):
    # The production wrapper supplies identical special heads, preprocessing,
    # PCA and per-head output constraints within each training-only CV fold.
    from src.shared.models import RidgeMultiTarget

    predictions = []
    for train, val in folds:
        model = RidgeMultiTarget(
            [target],
            alpha={target: alpha},
            two_stage_targets=cfg.get("two_stage_targets"),
            classification_targets=cfg.get("classification_targets"),
            pca_n_components=pca_n_components,
            non_negative_targets=cfg.get("nn_non_negative_targets"),
        )
        model.fit(X[train], {target: y[target][train]})
        predictions.append(model.predict(X[val])[target])
    return target, alpha, predictions


def tune_ridge_ppr(
    X,
    y,
    folds,
    targets,
    alpha_grids,
    cfg,
    *,
    refine_points=5,
    pca_n_components=None,
    n_jobs=-1,
    selection_info=None,
):
    """Bounded coordinate grid search over joint out-of-fold PPR predictions.

    Fits each candidate only once, parallelizing the candidate grid. Special
    heads are fitted once per fold and remain in the combined score. Two
    coordinate sweeps per coarse/refined grid retain target-specific alphas;
    this is a bounded search, not an exhaustive Cartesian-product optimum.
    """
    if not folds:
        raise ValueError("Ridge PPR selection requires at least one nonempty CV fold")
    all_targets = list(cfg["targets"])
    position = selection_position(all_targets)
    truth = [{t: y[t][val] for t in all_targets} for _, val in folds]
    candidates = {t: {} for t in all_targets}
    grids = {t: list(dict.fromkeys(float(a) for a in alpha_grids[t])) for t in targets}

    def add_candidates(requests):
        pending = [(t, a) for t, a in requests if a not in candidates[t]]
        fitted = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_ridge_candidate_predictions)(X, y, folds, t, a, cfg, pca_n_components)
            for t, a in pending
        )
        for t, a, predictions in fitted:
            candidates[t][a] = predictions

    add_candidates(
        [(t, a) for t in targets for a in grids[t]]
        + [(t, 1.0) for t in all_targets if t not in targets]
    )
    # Per-head RMSE supplies a deterministic starting point only. All final
    # candidate decisions below use the combined fantasy-point objective.
    selected = {
        t: min(
            candidates[t],
            key=lambda a: np.mean(
                [
                    np.sqrt(np.mean(np.square(pred - true[t])))
                    for pred, true in zip(candidates[t][a], truth, strict=True)
                ]
            ),
        )
        for t in all_targets
    }

    def score():
        return float(
            np.mean(
                [
                    ppr_rmse(
                        position, true, {t: candidates[t][selected[t]][i] for t in all_targets}
                    )
                    for i, true in enumerate(truth)
                ]
            )
        )

    best_score = score()
    score_history = [best_score]

    def sweep():
        nonlocal best_score
        for _ in range(2):
            changed = False
            for t in targets:
                best_alpha = selected[t]
                for alpha in candidates[t]:
                    selected[t] = alpha
                    value = score()
                    if value < best_score - 1e-12:
                        best_score, best_alpha = value, alpha
                        changed = True
                selected[t] = best_alpha
            score_history.append(best_score)
            if not changed:
                break

    sweep()
    fine = []
    for t, grid in grids.items():
        if refine_points and len(grid) >= 2 and min(grid[:2]) > 0 and selected[t] > 0:
            step = np.log10(grid[1]) - np.log10(grid[0])
            center = np.log10(selected[t])
            fine.extend(
                (t, float(a)) for a in np.logspace(center - step, center + step, refine_points)
            )
    if fine:
        add_candidates(fine)
        sweep()
    alphas = {t: float(selected[t]) for t in targets}
    if selection_info is not None:
        selection_info.update(
            metric="mean_cv_fantasy_rmse_ppr",
            scoring_format="ppr",
            score=best_score,
            alphas=alphas,
            n_folds=len(folds),
            search="two_coordinate_sweeps_per_coarse_and_fine_grid",
            score_history=score_history,
        )
    print(f"  Joint Ridge CV PPR RMSE={best_score:.4f}; alphas={alphas}")
    return alphas

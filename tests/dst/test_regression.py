"""Model-quality regression thresholds for DST.

These tests guard against silent accuracy regressions by asserting that
our trained models beat a naive baseline (and sit within a reasonable
band of each other) on a deterministic synthetic dataset.

Design
------
* Dataset: ``tiny_dataset`` (32 teams x 4 seasons x 17 weeks, fixed seed).
* Split:   earliest 3 seasons -> train, last season -> test.
* Models:  Ridge (per-target, PCA off), LightGBM (per-target), small NN.
* Baseline: season-average per team (``SeasonAverageBaseline``).

Thresholds are chosen from the null hypothesis that the targets are
correlated with the features in the synthetic data.  With Poisson-ish
``def_sacks`` / ``def_ints`` directly predicted as raw targets, a model
that uses the rolling mean should trivially outperform "mean of all weeks".

* ``MAE_baseline > MAE_ridge``               (Ridge beats naive mean)
* ``MAE_lgbm <= MAE_ridge * 1.10``           (LightGBM at least on par)
* ``|MAE_nn - MAE_lgbm| / MAE_lgbm <= 0.35`` (NN within +/-35 % of LightGBM)

Band is wide enough that CPU-vs-GPU / BLAS-version numerical noise
doesn't flake it, tight enough that flipping "mean" to "sum" in a feature
(or forgetting to fit a head) fails the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.dst.config import (
    ALL_FEATURES,
    HUBER_DELTAS,
    LOSS_WEIGHTS,
    NN_NON_NEGATIVE_TARGETS,
    TARGETS,
)
from src.dst.features import compute_features
from src.dst.targets import compute_targets
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline
from src.shared.aggregate_targets import aggregate_fn_for
from src.shared.feature_build import scale_and_clip
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import MultiHeadNet
from src.shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders

SEED = 42

# Features the tiny synthetic dataset supplies.  This is a subset of
# ALL_FEATURES — some opponent-scoring rolling features are built in
# dst_data.build_data (from opponent merges) rather than
# compute_features, so we mirror that logic here by feeding the
# tiny-dataset values directly.
_TINY_FEATURE_COLS = list(ALL_FEATURES)


def _prepare_tiny(tiny_dataset: pd.DataFrame):
    """Compute targets + features, then split into (train, test) tensors."""
    df = tiny_dataset.copy()
    df = compute_targets(df)
    compute_features(df)

    # 0-fill any feature the tiny dataset didn't materialise.  In prod the
    # pipeline's merge_schedule_features and the dst_data opponent merges
    # provide these; here we inject synthetic values up-stream in the
    # fixture and 0-fill anything else (e.g. prior_season_* for season 0).
    for col in _TINY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # ``fantasy_points`` is already written by compute_targets (linear
    # raw-stat combo + PA/YA tier bonuses). Don't overwrite it — we want
    # the real scoring, which is what baseline / NN total supervision uses.

    # Temporal split: earliest 3 seasons train, last season test
    seasons = sorted(df["season"].unique())
    train_seasons = seasons[:-1]
    test_seasons = seasons[-1:]
    train = df[df["season"].isin(train_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()

    X_train = train[_TINY_FEATURE_COLS].values.astype(np.float32)
    X_test = test[_TINY_FEATURE_COLS].values.astype(np.float32)
    y_train = {t: train[t].values.astype(np.float32) for t in TARGETS}
    y_test = {t: test[t].values.astype(np.float32) for t in TARGETS}
    # Fantasy-point truth used only for baseline / model MAE comparison — the
    # NN itself trains on raw-stat heads, no aux total supervision.
    fp_test = test["fantasy_points"].values.astype(np.float32)
    return X_train, X_test, y_train, y_test, fp_test, train, test


# ---------------------------------------------------------------------------
# Cached per-session result — training costs dominate, keep it to one pass
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_results(tiny_dataset):
    """Train Ridge + NN on the tiny dataset and return summary MAEs.

    Scope is ``module`` — expensive enough to warrant reuse across the
    four regression asserts, but not so stateful that session reuse is
    worth the extra coupling.
    """
    X_train, X_test, y_train, y_test, fp_test, train_df, test_df = _prepare_tiny(tiny_dataset)
    dst_agg = aggregate_fn_for("DST")

    # --- Baseline (season-average of fantasy_points per team) ---
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(test_df)
    baseline_mae = compute_metrics(fp_test, baseline_preds)["mae"]

    # --- Ridge multi-target ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ridge = RidgeMultiTarget(
        target_names=TARGETS,
        alpha=1.0,
        non_negative_targets=NN_NON_NEGATIVE_TARGETS,
        pca_n_components=None,  # PCA unused on tiny data
    )
    ridge.fit(X_train, y_train)
    # Aggregate raw-stat preds into fantasy-point totals for MAE comparison;
    # the tier-mapped PA/YA means ``sum(raw)`` is NOT the truth we want here.
    ridge_mae = compute_metrics(fp_test, dst_agg(ridge.predict(X_test)))["mae"]

    # --- LightGBM multi-target ---
    lgbm = LightGBMMultiTarget(
        target_names=TARGETS,
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        seed=SEED,
    )
    lgbm.fit(X_train, y_train, feature_names=_TINY_FEATURE_COLS)
    lgbm_mae = compute_metrics(fp_test, dst_agg(lgbm.predict(X_test)))["mae"]

    # --- NN (shrunk) ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaler = StandardScaler()
    X_tr_s = scale_and_clip(scaler, X_train, fit=True).astype(np.float32)
    X_te_s = scale_and_clip(scaler, X_test).astype(np.float32)

    train_loader, val_loader = make_dataloaders(
        X_tr_s,
        y_train,
        X_te_s,
        y_test,
        batch_size=64,
    )

    model = MultiHeadNet(
        input_dim=X_tr_s.shape[1],
        target_names=TARGETS,
        backbone_layers=[16, 8],
        head_hidden=8,
        dropout=0.0,
        non_negative_targets=NN_NON_NEGATIVE_TARGETS,
    )
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)
    criterion = MultiTargetLoss(
        target_names=TARGETS,
        loss_weights=LOSS_WEIGHTS,
        huber_deltas=HUBER_DELTAS,
    )
    trainer = MultiHeadTrainer(
        model,
        optim,
        sched,
        criterion,
        torch.device("cpu"),
        target_names=TARGETS,
        patience=5,
        log_every=100,
    )
    trainer.train(train_loader, val_loader, n_epochs=40)

    nn_preds = model.predict_numpy(X_te_s, torch.device("cpu"))
    nn_mae = compute_metrics(fp_test, dst_agg(nn_preds))["mae"]

    return {
        "baseline": baseline_mae,
        "ridge": ridge_mae,
        "lgbm": lgbm_mae,
        "nn": nn_mae,
    }


@pytest.mark.regression
@pytest.mark.integration
class TestDSTRegression:
    """MAE regression thresholds on the tiny DST dataset."""

    def test_ridge_beats_season_average_baseline(self, regression_results):
        baseline = regression_results["baseline"]
        ridge = regression_results["ridge"]
        assert ridge < baseline, (
            f"Ridge MAE {ridge:.3f} did not beat season-average baseline "
            f"{baseline:.3f} — a trained Ridge model must outperform naive "
            "historical averaging on features that carry real signal."
        )

    def test_nn_roughly_matches_baseline(self, regression_results):
        """NN total MAE within 10% of the season-average baseline.

        DST's fantasy points include nonlinear PA/YA tier bonuses that the
        per-head regression can't directly learn (the aggregation is
        piecewise-constant). Without an aux total-loss the NN has no direct
        pull toward fantasy-point accuracy on this tiny synthetic set, so
        we only require it to land near the naive mean — real breakage
        (e.g. heads not fitting at all) would push it far beyond this band.
        """
        baseline = regression_results["baseline"]
        nn = regression_results["nn"]
        assert nn <= baseline * 1.10, (
            f"NN MAE {nn:.3f} > 1.10 × baseline {baseline:.3f} — heads aren't fitting"
        )

    def test_lightgbm_not_much_worse_than_ridge(self, regression_results):
        """LightGBM should perform at least as well as Ridge (+/-10 %)."""
        lgbm = regression_results["lgbm"]
        ridge = regression_results["ridge"]
        assert lgbm <= ridge * 1.10, (
            f"LightGBM MAE {lgbm:.3f} is >10 % worse than Ridge {ridge:.3f} — "
            "tree model should exploit non-linearities at least as well as Ridge"
        )

    def test_nn_within_35pct_of_lightgbm(self, regression_results):
        """NN should land within +/-35 % of LightGBM's MAE.

        Catches "forgot to fit a head" (NN would be ~2x worse) and
        "over-optimised NN" (NN would be unreasonably better — possibly
        due to label leakage in feature engineering). Band widened for DST
        since the tier-mapped PA/YA nonlinearity is hard for per-head
        regression to match tree-based ensembles on this synthetic set.
        """
        lgbm = regression_results["lgbm"]
        nn = regression_results["nn"]
        rel_diff = abs(nn - lgbm) / lgbm
        assert rel_diff <= 0.35, (
            f"NN MAE {nn:.3f} differs from LightGBM {lgbm:.3f} by "
            f"{rel_diff:.1%} — expected within 35 %"
        )

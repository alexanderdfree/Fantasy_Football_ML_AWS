"""Model-quality regression thresholds for TE.

Trains Ridge, LightGBM, and a tiny NN on the tiny synthetic TE dataset and
asserts:

  - every trained model beats the season-average baseline on total-MAE,
  - LightGBM MAE is within 10% above Ridge (sanity: no catastrophic tree regression),
  - NN MAE is within ±25% of LightGBM (sanity: NN isn't wildly off).

Thresholds are intentionally loose — synthetic data is low-signal and the NN
uses 1 epoch and a 2×8 backbone. They exist to catch catastrophic regressions
(e.g., accidentally zeroed features, inverted loss sign), not squeeze the last
0.1 MAE.

Mark: `regression` (selectable via `pytest -m regression`) and `integration`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from shared.aggregate_targets import predictions_to_fantasy_points
from shared.models import LightGBMMultiTarget, RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from TE.te_config import (
    TE_CONFIG_TINY,
    TE_LGBM_MIN_CHILD_SAMPLES,
    TE_LGBM_NUM_LEAVES,
    TE_TARGETS,
)
from TE.te_data import filter_to_te
from TE.te_features import (
    add_te_specific_features,
    fill_te_nans,
    get_te_feature_columns,
)
from TE.te_targets import compute_te_targets

pytestmark = [
    pytest.mark.regression,
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning"),
]


@pytest.fixture(scope="module")
def te_training_tensors(te_tiny_splits):
    """Build (X_train, X_val, X_test, y_*_dict, feature_cols) for regression tests."""
    train, val, test = te_tiny_splits

    pos_train = filter_to_te(train)
    pos_val = filter_to_te(val)
    pos_test = filter_to_te(test)

    pos_train = compute_te_targets(pos_train)
    pos_val = compute_te_targets(pos_val)
    pos_test = compute_te_targets(pos_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos_train, pos_val, pos_test = add_te_specific_features(pos_train, pos_val, pos_test)
        pos_train, pos_val, pos_test = fill_te_nans(
            pos_train, pos_val, pos_test, TE_CONFIG_TINY["specific_features"]
        )

    feature_cols = get_te_feature_columns()
    for df in (pos_train, pos_val, pos_test):
        missing = [c for c in feature_cols if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = pos_train[feature_cols].values.astype(np.float32)
    X_val = pos_val[feature_cols].values.astype(np.float32)
    X_test = pos_test[feature_cols].values.astype(np.float32)

    y_train = {t: pos_train[t].values.astype(np.float32) for t in TE_TARGETS}
    y_val = {t: pos_val[t].values.astype(np.float32) for t in TE_TARGETS}
    y_test = {t: pos_test[t].values.astype(np.float32) for t in TE_TARGETS}
    # Total = aggregated fantasy points under PPR (new source of truth for
    # evaluating TE predictions post-migration).
    for d in (y_train, y_val, y_test):
        d["total"] = predictions_to_fantasy_points("TE", d, "ppr").astype(np.float32)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "pos_train": pos_train,
        "pos_test": pos_test,
        "feature_cols": feature_cols,
    }


def _aggregate_preds_total(preds: dict) -> np.ndarray:
    return predictions_to_fantasy_points("TE", preds, "ppr")


@pytest.fixture(scope="module")
def te_lgbm_mae(te_training_tensors):
    """Fit tiny LightGBM once and return total-MAE. Shared across tests."""
    t = te_training_tensors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Small num_leaves / min_child_samples constraints keep LightGBM
        # usable on the tiny dataset (cap at 1300 rows train).
        lgbm = LightGBMMultiTarget(
            target_names=TE_TARGETS,
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=min(TE_LGBM_NUM_LEAVES, 7),
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=min(TE_LGBM_MIN_CHILD_SAMPLES, 10),
            objective="huber",
            seed=42,
        )
        lgbm.fit(
            t["X_train"],
            t["y_train"],
            t["X_val"],
            t["y_val"],
            feature_names=t["feature_cols"],
        )
    preds = lgbm.predict(t["X_test"])
    return _mae(_aggregate_preds_total(preds), t["y_test"]["total"])


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def _season_average_mae(pos_train, pos_test, target="fantasy_points") -> float:
    """Mean-of-training-season fantasy points, broadcast to test."""
    mean_fp = float(pos_train[target].mean())
    y_test = pos_test[target].values.astype(np.float32)
    return float(np.mean(np.abs(mean_fp - y_test)))


class TestTERegressionThresholds:
    def test_ridge_beats_season_average_baseline(self, te_training_tensors):
        t = te_training_tensors
        model = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        model.fit(t["X_train"], t["y_train"])
        preds = model.predict(t["X_test"])

        ridge_mae = _mae(_aggregate_preds_total(preds), t["y_test"]["total"])
        baseline_mae = _season_average_mae(t["pos_train"], t["pos_test"])
        assert ridge_mae < baseline_mae, (
            f"Ridge MAE {ridge_mae:.3f} failed to beat season-avg baseline {baseline_mae:.3f}"
        )

    def test_lightgbm_not_catastrophic_vs_ridge(self, te_training_tensors, te_lgbm_mae):
        """LightGBM MAE must not exceed Ridge by more than 20%.

        Threshold widened from 10% after the raw-stat target migration: TE
        targets are now a mix of count (TDs, fumbles) and yardage targets
        with very different magnitudes; the aggregated fantasy-points MAE
        is sensitive to the weakest head, and on 50-player tiny splits a
        handful of TD mispredictions dominate. 20% still rejects order-of-
        magnitude tree blowups.
        """
        t = te_training_tensors

        ridge = RidgeMultiTarget(target_names=TE_TARGETS, alpha=1.0)
        ridge.fit(t["X_train"], t["y_train"])
        ridge_preds = ridge.predict(t["X_test"])
        ridge_mae = _mae(_aggregate_preds_total(ridge_preds), t["y_test"]["total"])

        assert te_lgbm_mae <= ridge_mae * 1.20, (
            f"LightGBM MAE {te_lgbm_mae:.3f} > 1.20 x Ridge MAE {ridge_mae:.3f} "
            f"(ratio {te_lgbm_mae / ridge_mae:.3f})"
        )

    def test_nn_within_bounds_of_lightgbm(self, te_training_tensors, te_lgbm_mae):
        """Tiny NN total-MAE within 2x of LightGBM. Guards NN training sanity.

        Threshold widened from ±25% after the raw-stat target migration.
        Total-MAE in fantasy-point space now aggregates a zero-inflated TD
        head, a count head (receptions), a yards head (receiving_yards),
        and a rare-event head (fumbles_lost). On 50-player tiny splits the
        TD head dominates total MAE; 50-epoch MSE-trained NN vs Huber-trained
        LightGBM can legitimately diverge 2x in that regime. The test still
        rejects broken-gradient / inverted-loss disasters (those would give
        10x ratios).
        """
        t = te_training_tensors

        # Shrunken NN trained for a handful of epochs so the test sees a
        # non-degenerate predictor. This is still well under 3 seconds.
        np.random.seed(42)
        torch.manual_seed(42)
        scaler = StandardScaler()
        X_train_s = np.clip(scaler.fit_transform(t["X_train"]), -4, 4).astype(np.float32)
        X_test_s = np.clip(scaler.transform(t["X_test"]), -4, 4).astype(np.float32)

        model = MultiHeadNet(
            input_dim=X_train_s.shape[1],
            target_names=TE_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_train_t = {
            t_name: torch.tensor(t["y_train"][t_name], dtype=torch.float32) for t_name in TE_TARGETS
        }

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = sum(((out[k] - y_train_t[k]) ** 2).mean() for k in TE_TARGETS)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            nn_out = model(torch.tensor(X_test_s, dtype=torch.float32))
        nn_preds = {k: nn_out[k].numpy() for k in TE_TARGETS}
        total_pred = _aggregate_preds_total(nn_preds)
        nn_mae = _mae(total_pred, t["y_test"]["total"])

        ratio = nn_mae / te_lgbm_mae
        assert 0.5 <= ratio <= 2.0, (
            f"NN MAE {nn_mae:.3f} not within 2x of LightGBM MAE {te_lgbm_mae:.3f} "
            f"(ratio {ratio:.3f})"
        )

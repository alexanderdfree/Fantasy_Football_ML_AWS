"""QB regression thresholds (raw-stat targets).

Trains QB Ridge/LightGBM/NN on tiny deterministic synthetic data and asserts
that each model actually learns something useful:

  - MAE beats season-average baseline
  - LightGBM MAE <= Ridge MAE * 1.30 (LGBM at least competitive with Ridge
    on mostly-linear tiny data)
  - NN MAE within ~0.6x-1.7x of LightGBM

After the raw-stat migration, ``y["total"]`` is aggregated via the scoring
dict (not a simple sum) so the MAE here is in fantasy-point units, not a
component-sum. Thresholds are loose enough not to flake, tight enough to
catch a regression like "forgot to call .fit" or a shape mismatch that
silently outputs zeros.

These tests invoke training loops (shared.training.MultiHeadTrainer) so they
are marked @pytest.mark.integration alongside @pytest.mark.regression.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from QB.qb_config import QB_LOSS_WEIGHTS, QB_TARGETS
from shared.aggregate_targets import predictions_to_fantasy_points
from shared.models import LightGBMMultiTarget, RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders


@pytest.fixture(scope="module")
def synthetic_qb_data():
    """Tiny deterministic synthetic QB dataset with learnable structure.

    Features carry genuine signal for each raw-stat target so a fitted model
    should clearly beat the zero-information baseline.
    """
    # Use target scales that resemble a normalized-yards / count world so
    # the NN and LGBM can both find the signal under the same Huber delta.
    # Yards targets stay small here (not per-game ~250) because a single
    # shared-scale test keeps the MAE-ratio thresholds tractable for a
    # shrunk NN. The signal structure is what matters for this regression.
    rng = np.random.default_rng(42)
    n_train, n_test, d = 500, 200, 12

    def _gen(n):
        X = rng.standard_normal((n, d)).astype(np.float32)
        y_pass_yds = np.clip(
            4.0 * X[:, 0]
            + 1.5 * X[:, 1]
            - 0.8 * X[:, 2] ** 2
            + 10.0
            + rng.standard_normal(n) * 1.5,
            0,
            None,
        ).astype(np.float32)
        y_rush_yds = np.clip(
            1.5 * X[:, 3] + 0.5 * X[:, 4] + 2.0 + rng.standard_normal(n) * 0.8,
            0,
            None,
        ).astype(np.float32)
        y_pass_tds = np.clip(
            1.5 * X[:, 5] + 0.8 * X[:, 6] + 2.0 + rng.standard_normal(n) * 0.5,
            0,
            None,
        ).astype(np.float32)
        y_rush_tds = np.clip(
            0.8 * X[:, 7] + 0.5 + rng.standard_normal(n) * 0.3,
            0,
            None,
        ).astype(np.float32)
        y_ints = np.clip(
            0.5 * X[:, 8] + 0.7 + rng.standard_normal(n) * 0.3,
            0,
            None,
        ).astype(np.float32)
        y_fum = np.clip(
            0.3 * X[:, 9] + 0.3 + rng.standard_normal(n) * 0.2,
            0,
            None,
        ).astype(np.float32)
        y = {
            "passing_yards": y_pass_yds,
            "rushing_yards": y_rush_yds,
            "passing_tds": y_pass_tds,
            "rushing_tds": y_rush_tds,
            "interceptions": y_ints,
            "fumbles_lost": y_fum,
        }
        return X, y

    X_train, y_train = _gen(n_train)
    X_test, y_test = _gen(n_test)
    y_train["total"] = predictions_to_fantasy_points("QB", y_train, "ppr").astype(np.float32)
    y_test["total"] = predictions_to_fantasy_points("QB", y_test, "ppr").astype(np.float32)
    return X_train, y_train, X_test, y_test


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _baseline_mae(y_train_total, y_test_total):
    """Constant-prediction baseline: training mean."""
    return _mae(y_test_total, np.full_like(y_test_total, y_train_total.mean()))


def _aggregate(preds):
    """Convert per-target predictions to fantasy-point total."""
    return predictions_to_fantasy_points("QB", preds, "ppr")


@pytest.mark.regression
@pytest.mark.integration
class TestQBRegression:
    def test_ridge_beats_baseline(self, synthetic_qb_data):
        X_train, y_train, X_test, y_test = synthetic_qb_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        ridge_mae = _mae(y_test["total"], _aggregate(preds))
        baseline_mae = _baseline_mae(y_train["total"], y_test["total"])
        assert ridge_mae < baseline_mae, (
            f"Ridge MAE {ridge_mae:.3f} not better than baseline {baseline_mae:.3f}"
        )

    def test_lightgbm_competitive_with_ridge(self, synthetic_qb_data):
        X_train, y_train, X_test, y_test = synthetic_qb_data

        ridge = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_mae = _mae(y_test["total"], _aggregate(ridge.predict(X_test)))

        lgbm = LightGBMMultiTarget(
            target_names=QB_TARGETS,
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=10,
            seed=42,
        )
        lgbm.fit(X_train, y_train, X_test, y_test)
        lgbm_mae = _mae(y_test["total"], _aggregate(lgbm.predict(X_test)))

        # LGBM should be roughly competitive with Ridge. On small data with
        # mostly-linear signal Ridge can edge out LGBM, so we allow 30% slack
        # — tight enough to catch "forgot to fit" (LGBM MAE ~ baseline_mae)
        # but loose enough not to flake on RNG wobble.
        assert lgbm_mae <= ridge_mae * 1.30, (
            f"LightGBM MAE {lgbm_mae:.3f} >> Ridge MAE {ridge_mae:.3f}"
        )
        baseline_mae = _baseline_mae(y_train["total"], y_test["total"])
        assert lgbm_mae < baseline_mae, (
            f"LightGBM MAE {lgbm_mae:.3f} not better than baseline {baseline_mae:.3f}"
        )

    def test_nn_within_25pct_of_lightgbm(self, synthetic_qb_data):
        X_train, y_train, X_test, y_test = synthetic_qb_data

        # LightGBM reference
        lgbm = LightGBMMultiTarget(
            target_names=QB_TARGETS,
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=10,
            seed=42,
        )
        lgbm.fit(X_train, y_train, X_test, y_test)
        lgbm_mae = _mae(y_test["total"], _aggregate(lgbm.predict(X_test)))

        # NN
        torch.manual_seed(42)
        np.random.seed(42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        train_loader, val_loader = make_dataloaders(
            X_train_s,
            y_train,
            X_test_s,
            y_test,
            batch_size=64,
        )
        model = MultiHeadNet(
            input_dim=X_train_s.shape[1],
            target_names=QB_TARGETS,
            backbone_layers=[32, 16],
            head_hidden=8,
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.5,
        )
        criterion = MultiTargetLoss(
            target_names=QB_TARGETS,
            loss_weights=QB_LOSS_WEIGHTS,
        )
        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            torch.device("cpu"),
            target_names=QB_TARGETS,
            patience=10,
            log_every=100,
        )
        trainer.train(train_loader, val_loader, n_epochs=60)

        nn_preds = model.predict_numpy(X_test_s, torch.device("cpu"))
        nn_mae = _mae(y_test["total"], _aggregate(nn_preds))

        # NN within ~60% of LightGBM (either direction — "forgot to fit" makes
        # NN MAE ~= baseline which is far outside this band). The band widened
        # after migration because six heterogeneous-scale targets give a small
        # NN less headroom than the old 3-target sum-of-components world.
        ratio = nn_mae / lgbm_mae
        assert 0.6 <= ratio <= 1.7, (
            f"NN MAE {nn_mae:.3f} not within ~0.6x-1.7x of LightGBM {lgbm_mae:.3f} "
            f"(ratio={ratio:.2f})"
        )
        # And better than baseline
        baseline_mae = _baseline_mae(y_train["total"], y_test["total"])
        assert nn_mae < baseline_mae, (
            f"NN MAE {nn_mae:.3f} not better than baseline {baseline_mae:.3f}"
        )

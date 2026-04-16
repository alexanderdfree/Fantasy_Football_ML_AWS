"""QB regression thresholds.

Trains QB Ridge/LightGBM/NN on tiny deterministic synthetic data and asserts
that each model actually learns something useful:

  - MAE beats season-average baseline
  - LightGBM MAE <= Ridge MAE * 1.30 (LGBM at least competitive with Ridge
    on mostly-linear tiny data)
  - NN MAE within +/-25% of LightGBM

Thresholds are loose enough not to flake, tight enough to catch a
regression like "forgot to call .fit" or a shape mismatch that silently
outputs zeros.

These tests invoke training loops (shared.training.MultiHeadTrainer) so they
are marked @pytest.mark.integration alongside @pytest.mark.regression.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from QB.qb_config import QB_TARGETS
from shared.models import RidgeMultiTarget, LightGBMMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiTargetLoss, MultiHeadTrainer, make_dataloaders

QB_LOSS_WEIGHTS = {t: 1.0 for t in QB_TARGETS}


@pytest.fixture(scope="module")
def synthetic_qb_data():
    """Tiny deterministic synthetic QB dataset with learnable structure.

    Features carry genuine signal for each target so a fitted model should
    clearly beat the zero-information season-average baseline.
    """
    rng = np.random.default_rng(42)
    n_train, n_test, d = 500, 200, 12

    def _gen(n):
        X = rng.standard_normal((n, d)).astype(np.float32)
        # Non-trivial signal + noise so models have room to beat baseline
        y_pass = np.clip(
            4.0 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] ** 2 + 10.0
            + rng.standard_normal(n) * 1.5, 0, None,
        ).astype(np.float32)
        y_rush = np.clip(
            1.5 * X[:, 3] + 0.5 * X[:, 4] + 2.0
            + rng.standard_normal(n) * 0.8, 0, None,
        ).astype(np.float32)
        y_td = np.clip(
            3.0 * X[:, 5] + 2.0 * X[:, 6] + 4.0
            + rng.standard_normal(n) * 1.0, 0, None,
        ).astype(np.float32)
        return X, {"passing_floor": y_pass, "rushing_floor": y_rush, "td_points": y_td}

    X_train, y_train = _gen(n_train)
    X_test, y_test = _gen(n_test)
    y_train["total"] = y_train["passing_floor"] + y_train["rushing_floor"] + y_train["td_points"]
    y_test["total"] = y_test["passing_floor"] + y_test["rushing_floor"] + y_test["td_points"]
    return X_train, y_train, X_test, y_test


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _baseline_mae(y_train_total, y_test_total):
    """Constant-prediction baseline: training mean."""
    return _mae(y_test_total, np.full_like(y_test_total, y_train_total.mean()))


@pytest.mark.regression
@pytest.mark.integration
class TestQBRegression:
    def test_ridge_beats_baseline(self, synthetic_qb_data):
        X_train, y_train, X_test, y_test = synthetic_qb_data
        model = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        ridge_mae = _mae(y_test["total"], preds["total"])
        baseline_mae = _baseline_mae(y_train["total"], y_test["total"])
        assert ridge_mae < baseline_mae, (
            f"Ridge MAE {ridge_mae:.3f} not better than baseline {baseline_mae:.3f}"
        )

    def test_lightgbm_competitive_with_ridge(self, synthetic_qb_data):
        X_train, y_train, X_test, y_test = synthetic_qb_data

        ridge = RidgeMultiTarget(target_names=QB_TARGETS, alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_mae = _mae(y_test["total"], ridge.predict(X_test)["total"])

        lgbm = LightGBMMultiTarget(
            target_names=QB_TARGETS,
            n_estimators=100, learning_rate=0.1, num_leaves=15,
            min_child_samples=10, seed=42,
        )
        lgbm.fit(X_train, y_train, X_test, y_test)
        lgbm_mae = _mae(y_test["total"], lgbm.predict(X_test)["total"])

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
            n_estimators=100, learning_rate=0.1, num_leaves=15,
            min_child_samples=10, seed=42,
        )
        lgbm.fit(X_train, y_train, X_test, y_test)
        lgbm_mae = _mae(y_test["total"], lgbm.predict(X_test)["total"])

        # NN
        torch.manual_seed(42)
        np.random.seed(42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)

        train_loader, val_loader = make_dataloaders(
            X_train_s, y_train, X_test_s, y_test, batch_size=64,
        )
        model = MultiHeadNet(
            input_dim=X_train_s.shape[1], target_names=QB_TARGETS,
            backbone_layers=[32, 16], head_hidden=8, dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )
        criterion = MultiTargetLoss(
            target_names=QB_TARGETS, loss_weights=QB_LOSS_WEIGHTS,
        )
        trainer = MultiHeadTrainer(
            model, optimizer, scheduler, criterion,
            torch.device("cpu"), target_names=QB_TARGETS, patience=10,
            log_every=100,
        )
        trainer.train(train_loader, val_loader, n_epochs=60)

        nn_preds = model.predict_numpy(X_test_s, torch.device("cpu"))
        nn_preds["total"] = sum(nn_preds[t] for t in QB_TARGETS)
        nn_mae = _mae(y_test["total"], nn_preds["total"])

        # NN within 25% of LightGBM (either direction — "forgot to fit" makes
        # NN MAE ~= baseline which is far outside 25%).
        ratio = nn_mae / lgbm_mae
        assert 0.75 <= ratio <= 1.25, (
            f"NN MAE {nn_mae:.3f} not within +/-25% of LightGBM {lgbm_mae:.3f} "
            f"(ratio={ratio:.2f})"
        )
        # And better than baseline
        baseline_mae = _baseline_mae(y_train["total"], y_test["total"])
        assert nn_mae < baseline_mae, (
            f"NN MAE {nn_mae:.3f} not better than baseline {baseline_mae:.3f}"
        )

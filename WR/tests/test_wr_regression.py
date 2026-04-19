"""Model quality regression thresholds for the WR pipeline.

Trains Ridge / LightGBM / Multi-Head NN on a tiny deterministic synthetic
dataset where the target is a known linear-plus-nonlinear function of the
features, and asserts:

  * every trained model beats the season-average baseline
  * LightGBM is not dramatically worse than Ridge (tuned loosely so a
    genuine regression catches it, not flaky CI noise)
  * Neural net MAE is within ±25% of LightGBM (sanity guard)

These thresholds are intentionally generous — they exist to catch silent
"forgot to fit" / "lost a feature" regressions, not to tune accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shared.models import LightGBMMultiTarget, RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders
from WR.wr_config import WR_LOSS_WEIGHTS, WR_TARGETS


def _synthetic_wr_dataset(n: int = 800, n_features: int = 10, seed: int = 42):
    """Generate a deterministic synthetic WR dataset with a learnable signal.

    Targets are simple linear-plus-interaction functions of the features,
    scaled to match WR ranges (receiving_floor 0-20, rushing_floor 0-4,
    td_points 0-12). Returns train/val splits and the baseline predictor
    output (per-target training-mean) for the baseline threshold check.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)

    # Construct targets with known signal + small noise
    recv = np.clip(
        6.0
        + 3.0 * X[:, 0]
        + 2.0 * X[:, 1]
        + 0.5 * X[:, 0] * X[:, 2]
        + rng.standard_normal(n) * 1.0,
        0.0,
        25.0,
    ).astype(np.float32)
    rush = np.clip(
        0.8 + 0.5 * X[:, 3] + 0.3 * X[:, 4] + rng.standard_normal(n) * 0.3,
        0.0,
        5.0,
    ).astype(np.float32)
    tds = np.clip(
        2.0 + 1.5 * X[:, 5] + 1.0 * X[:, 6] + rng.standard_normal(n) * 0.8,
        0.0,
        12.0,
    ).astype(np.float32)

    # Train/val split: 80/20
    n_train = int(0.8 * n)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train = {
        "receiving_floor": recv[:n_train],
        "rushing_floor": rush[:n_train],
        "td_points": tds[:n_train],
    }
    y_val = {
        "receiving_floor": recv[n_train:],
        "rushing_floor": rush[n_train:],
        "td_points": tds[n_train:],
    }
    y_train["total"] = sum(y_train[t] for t in WR_TARGETS)
    y_val["total"] = sum(y_val[t] for t in WR_TARGETS)
    return X_train, X_val, y_train, y_val


def _mae_total(preds_dict: dict, y_dict: dict) -> float:
    """Mean absolute error of the total-prediction vs. total-truth."""
    return float(np.mean(np.abs(preds_dict["total"] - y_dict["total"])))


def _baseline_mae(y_train: dict, y_val: dict) -> float:
    """Season-average baseline: predict train-set mean for every val row."""
    per_target_mean = {t: float(np.mean(y_train[t])) for t in WR_TARGETS}
    total_pred = sum(per_target_mean[t] for t in WR_TARGETS)
    total_true = y_val["total"]
    return float(np.mean(np.abs(total_pred - total_true)))


@pytest.mark.regression
@pytest.mark.integration
def test_all_models_beat_season_average_baseline():
    """Ridge, LightGBM, and NN each trained on learnable signal must beat the
    constant-mean baseline. Catches "forgot to fit" regressions."""
    X_train, X_val, y_train, y_val = _synthetic_wr_dataset()
    baseline = _baseline_mae(y_train, y_val)

    # Ridge
    ridge = RidgeMultiTarget(target_names=WR_TARGETS, alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_mae = _mae_total(ridge.predict(X_val), y_val)
    assert ridge_mae < baseline, f"Ridge {ridge_mae:.3f} >= baseline {baseline:.3f}"

    # LightGBM (tiny: 50 estimators, small leaves — learnable signal is simple)
    lgbm = LightGBMMultiTarget(
        target_names=WR_TARGETS,
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        objective="huber",
        seed=42,
    )
    lgbm.fit(X_train, y_train)
    lgbm_mae = _mae_total(lgbm.predict(X_val), y_val)
    assert lgbm_mae < baseline, f"LightGBM {lgbm_mae:.3f} >= baseline {baseline:.3f}"

    # Neural net
    nn_mae = _train_tiny_nn_mae(X_train, X_val, y_train, y_val)
    assert nn_mae < baseline, f"NN {nn_mae:.3f} >= baseline {baseline:.3f}"


@pytest.mark.regression
@pytest.mark.integration
def test_lightgbm_within_tolerance_of_ridge():
    """LightGBM MAE is not wildly worse than Ridge on the simple signal.

    LightGBM should be competitive with Ridge here (the signal is mostly
    linear). A 10% tolerance over Ridge catches catastrophic configuration
    breakage; a real LightGBM regression will blow past it.
    """
    X_train, X_val, y_train, y_val = _synthetic_wr_dataset()

    ridge = RidgeMultiTarget(target_names=WR_TARGETS, alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_mae = _mae_total(ridge.predict(X_val), y_val)

    lgbm = LightGBMMultiTarget(
        target_names=WR_TARGETS,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        objective="huber",
        seed=42,
    )
    lgbm.fit(X_train, y_train)
    lgbm_mae = _mae_total(lgbm.predict(X_val), y_val)

    assert lgbm_mae <= ridge_mae * 1.10, (
        f"LightGBM {lgbm_mae:.3f} exceeds Ridge {ridge_mae:.3f} * 1.10 = {ridge_mae * 1.10:.3f}"
    )


@pytest.mark.regression
@pytest.mark.integration
def test_nn_mae_within_25pct_of_lightgbm():
    """NN MAE within ±25% of LightGBM MAE.

    Tiny NN on 640 synthetic rows is noisier than LightGBM; the ±25% band is
    intentionally wide. A real NN regression (e.g. gradient explode, wrong
    loss) will escape this band.
    """
    X_train, X_val, y_train, y_val = _synthetic_wr_dataset()

    lgbm = LightGBMMultiTarget(
        target_names=WR_TARGETS,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=5,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        reg_alpha=0.0,
        objective="huber",
        seed=42,
    )
    lgbm.fit(X_train, y_train)
    lgbm_mae = _mae_total(lgbm.predict(X_val), y_val)

    nn_mae = _train_tiny_nn_mae(X_train, X_val, y_train, y_val)

    low, high = lgbm_mae * 0.75, lgbm_mae * 1.25
    assert low <= nn_mae <= high, (
        f"NN {nn_mae:.3f} outside ±25% of LightGBM {lgbm_mae:.3f} (band: [{low:.3f}, {high:.3f}])"
    )


def _train_tiny_nn_mae(X_train, X_val, y_train, y_val) -> float:
    """Train a small NN and return total-MAE on the validation set."""
    np.random.seed(42)
    torch.manual_seed(42)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)
    model = MultiHeadNet(
        input_dim=X_train.shape[1],
        target_names=WR_TARGETS,
        backbone_layers=[32, 16],
        head_hidden=8,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
    device = torch.device("cpu")
    trainer = MultiHeadTrainer(
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        target_names=WR_TARGETS,
        patience=10,
    )
    trainer.train(train_loader, val_loader, n_epochs=80)

    preds = model.predict_numpy(X_val, device)
    preds["total"] = sum(preds[t] for t in WR_TARGETS)
    return _mae_total(preds, y_val)

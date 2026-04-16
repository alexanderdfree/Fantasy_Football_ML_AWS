"""Model regression thresholds for the K (Kicker) position.

Trains Ridge + LightGBM + a tiny NN on the tiny synthetic kicker dataset and
asserts quality floor guards that catch silent accuracy regressions:

  - Each trained model beats the season-average baseline MAE.
  - LightGBM is not dramatically worse than Ridge (<= Ridge * 1.10).
  - NN MAE is within +/- 25% of LightGBM (sanity, not tight quality bar).

Thresholds are loose enough to not flake on the synthetic data; tight enough
that a real regression (e.g., feature accidentally zeroed, label swap, scaler
misuse) fails the suite.
"""
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from K.k_config import K_HUBER_DELTAS, K_LOSS_WEIGHTS, K_TARGETS
from K.k_features import compute_k_features, get_k_feature_columns
from K.k_targets import compute_k_targets
from shared.models import LightGBMMultiTarget, RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders


# ---------------------------------------------------------------------------
# Shared training data + trained models (module-scoped: trained once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def k_training_arrays(tiny_k_dataset):
    """Prepare (X_train, X_val, y_train, y_val, baseline_mae, feature_cols)."""
    feature_cols = get_k_feature_columns()

    df = tiny_k_dataset.copy()
    df = compute_k_targets(df)
    compute_k_features(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    train_df = df[df["season"] <= 2023].copy()
    val_df = df[df["season"] == 2024].copy()

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)

    y_train_dict = {t: train_df[t].values.astype(np.float32) for t in K_TARGETS}
    y_val_dict = {t: val_df[t].values.astype(np.float32) for t in K_TARGETS}
    y_train_dict["total"] = sum(y_train_dict[t] for t in K_TARGETS)
    y_val_dict["total"] = sum(y_val_dict[t] for t in K_TARGETS)

    train_mean = float(y_train_dict["total"].mean())
    baseline_preds = np.full(len(X_val), train_mean, dtype=np.float32)
    baseline_mae = float(np.mean(np.abs(baseline_preds - y_val_dict["total"])))

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train_dict": y_train_dict,
        "y_val_dict": y_val_dict,
        "baseline_mae": baseline_mae,
        "feature_cols": feature_cols,
    }


def _total_mae(preds: dict, y_true_total: np.ndarray) -> float:
    total = preds["fg_points"] + preds["pat_points"]
    return float(np.mean(np.abs(total - y_true_total)))


@pytest.fixture(scope="module")
def ridge_mae(k_training_arrays):
    """Train Ridge once, return its total-MAE on val."""
    d = k_training_arrays
    model = RidgeMultiTarget(target_names=K_TARGETS, alpha=1.0)
    model.fit(d["X_train"], d["y_train_dict"])
    return _total_mae(model.predict(d["X_val"]), d["y_val_dict"]["total"])


@pytest.fixture(scope="module")
def lightgbm_mae(k_training_arrays):
    """Train LightGBM once, return its total-MAE on val."""
    d = k_training_arrays
    model = LightGBMMultiTarget(
        target_names=K_TARGETS,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        seed=42,
    )
    model.fit(
        d["X_train"], d["y_train_dict"],
        d["X_val"], d["y_val_dict"],
        feature_names=d["feature_cols"],
    )
    return _total_mae(model.predict(d["X_val"]), d["y_val_dict"]["total"])


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

@pytest.mark.regression
@pytest.mark.integration
def test_ridge_beats_baseline(k_training_arrays, ridge_mae):
    """Ridge total MAE < season-average baseline (catches `forgot to fit`)."""
    baseline = k_training_arrays["baseline_mae"]
    assert ridge_mae < baseline, (
        f"Ridge MAE {ridge_mae:.3f} did not beat baseline {baseline:.3f}"
    )


@pytest.mark.regression
@pytest.mark.integration
def test_lightgbm_beats_baseline(k_training_arrays, lightgbm_mae):
    """LightGBM total MAE < season-average baseline."""
    baseline = k_training_arrays["baseline_mae"]
    assert lightgbm_mae < baseline, (
        f"LightGBM MAE {lightgbm_mae:.3f} did not beat baseline {baseline:.3f}"
    )


@pytest.mark.regression
@pytest.mark.integration
def test_lightgbm_within_110pct_of_ridge(ridge_mae, lightgbm_mae):
    """LightGBM MAE <= Ridge MAE * 1.10 (roughly as-good-as)."""
    assert lightgbm_mae <= ridge_mae * 1.10, (
        f"LightGBM MAE {lightgbm_mae:.3f} > Ridge MAE {ridge_mae:.3f} * 1.10 "
        f"({ridge_mae * 1.10:.3f})"
    )


@pytest.mark.regression
@pytest.mark.integration
def test_nn_within_25pct_of_lightgbm(k_training_arrays, lightgbm_mae):
    """NN MAE within +/- 25% of LightGBM MAE — sanity guard, not tight."""
    d = k_training_arrays

    np.random.seed(42)
    torch.manual_seed(42)
    scaler = StandardScaler()
    X_train_s = np.clip(scaler.fit_transform(d["X_train"]), -4, 4).astype(np.float32)
    X_val_s = np.clip(scaler.transform(d["X_val"]), -4, 4).astype(np.float32)

    train_loader, val_loader = make_dataloaders(
        X_train_s, d["y_train_dict"], X_val_s, d["y_val_dict"], batch_size=64
    )
    model = MultiHeadNet(
        input_dim=X_train_s.shape[1],
        target_names=K_TARGETS,
        backbone_layers=[16, 8],
        head_hidden=4,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = MultiTargetLoss(
        target_names=K_TARGETS, loss_weights=K_LOSS_WEIGHTS,
        huber_deltas=K_HUBER_DELTAS, w_total=1.0,
    )
    trainer = MultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=torch.device("cpu"),
        target_names=K_TARGETS, patience=5, log_every=50,
    )
    trainer.train(train_loader, val_loader, n_epochs=20)

    nn_mae = _total_mae(
        model.predict_numpy(X_val_s, torch.device("cpu")),
        d["y_val_dict"]["total"],
    )

    assert lightgbm_mae * 0.75 <= nn_mae <= lightgbm_mae * 1.25, (
        f"NN MAE {nn_mae:.3f} not within +/-25% of LightGBM MAE {lightgbm_mae:.3f}"
    )

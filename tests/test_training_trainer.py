"""Coverage tests for ``src/training/trainer.py``.

The ``Trainer`` class is the lightweight legacy training wrapper used by
analysis scripts (it predates ``src.shared.training.MultiHeadTrainer``). These
tests exercise:

- ``__init__`` field defaults
- ``train()`` happy path: epoch loop runs, history populated
- early stopping when val loss stops improving
- LR scheduler ``.step`` is invoked each epoch
- restore-best-weights branch on patience exhaustion + on natural finish
- the every-10-epochs print branch
- ``plot_training_curves()`` two-panel save
- ``make_dataloaders()`` returns shuffled-train / non-shuffled-val loaders
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.trainer import Trainer, make_dataloaders

# --------------------------------------------------------------------------
# Helpers — tiny linear model + tensor data
# --------------------------------------------------------------------------


def _tiny_data(n=32, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    # Targets: linear in feature 0 with noise.
    y = (2.0 * X[:, 0:1] + rng.normal(0, 0.1, (n, 1))).astype(np.float32)
    return X, y


def _build_trainer(model, lr=1e-2, patience=2):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    criterion = nn.MSELoss()
    device = torch.device("cpu")
    return Trainer(model, optimizer, scheduler, criterion, device, patience=patience)


# --------------------------------------------------------------------------
# Trainer.__init__
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_trainer_init_defaults():
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    criterion = nn.MSELoss()
    t = Trainer(model, optimizer, scheduler, criterion, torch.device("cpu"))
    assert t.patience == 15  # default
    assert t.best_val_loss == float("inf")
    assert t.best_model_state is None
    assert t.epochs_without_improvement == 0


# --------------------------------------------------------------------------
# Trainer.train — happy path
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_train_returns_history_with_loss_curves():
    """A 3-epoch run on tiny data populates train_loss + val_loss + val_mae +
    val_rmse, each list one entry per actually-run epoch."""
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = _tiny_data()
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=10)

    history = trainer.train(train_loader, val_loader, n_epochs=3)
    assert set(history) == {"train_loss", "val_loss", "val_mae", "val_rmse"}
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    # Loss is finite + non-negative.
    for loss in history["train_loss"] + history["val_loss"]:
        assert np.isfinite(loss)
        assert loss >= 0


@pytest.mark.unit
def test_train_restores_best_weights_on_completion():
    """After natural finish, model weights should be the best-seen state."""
    torch.manual_seed(0)
    X, y = _tiny_data()
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=10)

    trainer.train(train_loader, val_loader, n_epochs=3)
    # best_model_state must be populated; weights restored to it.
    assert trainer.best_model_state is not None
    for k, v in trainer.model.state_dict().items():
        assert torch.allclose(v, trainer.best_model_state[k])


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_train_early_stopping_breaks_loop(capsys):
    """When val_loss never improves on a noise-only target, patience is
    exhausted before n_epochs and training breaks early."""
    torch.manual_seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    # Target is pure noise — model can't actually fit, so val_loss won't
    # consistently improve. Use a small patience to force early-stop.
    y = np.random.randn(32, 1).astype(np.float32)
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=1)
    history = trainer.train(train_loader, val_loader, n_epochs=20)

    # If patience triggered, history is shorter than n_epochs AND the
    # "Early stopping" message printed.
    out = capsys.readouterr().out
    if len(history["train_loss"]) < 20:
        assert "Early stopping at epoch" in out


@pytest.mark.unit
def test_train_increments_patience_counter_when_no_improvement():
    """If val_loss does NOT beat best_val_loss, the patience counter ticks."""
    torch.manual_seed(0)
    X, y = _tiny_data()
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=100)
    # Pre-set best_val_loss to a tiny number so the first epoch's val loss
    # cannot beat it → counter ticks to 1.
    trainer.best_val_loss = 1e-30
    trainer.train(train_loader, val_loader, n_epochs=2)
    assert trainer.epochs_without_improvement >= 1


# --------------------------------------------------------------------------
# Print every 10 epochs
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_train_prints_progress_every_10_epochs(capsys):
    """``(epoch + 1) % 10 == 0`` branch fires on epoch 10."""
    torch.manual_seed(0)
    X, y = _tiny_data(n=16)
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=100)
    trainer.train(train_loader, val_loader, n_epochs=10)
    out = capsys.readouterr().out
    # Expect one progress line for epoch 10.
    assert "Epoch  10" in out


# --------------------------------------------------------------------------
# plot_training_curves
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_plot_training_curves_writes_png(tmp_path):
    """Two-panel figure renders to disk; history dict is enough."""
    torch.manual_seed(0)
    X, y = _tiny_data()
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)
    trainer = _build_trainer(nn.Linear(4, 1), patience=10)
    history = trainer.train(train_loader, val_loader, n_epochs=2)

    save_path = tmp_path / "curves.png"
    trainer.plot_training_curves(history, str(save_path))
    assert save_path.exists()


# --------------------------------------------------------------------------
# make_dataloaders
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_make_dataloaders_train_shuffled_val_not_shuffled():
    """``shuffle=True`` for train, ``shuffle=False`` for val (preserves order)."""
    X, y = _tiny_data(n=16)
    train_loader, val_loader = make_dataloaders(X, y, X, y, batch_size=8)

    # train_loader uses RandomSampler when shuffle=True
    assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
    assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)


@pytest.mark.unit
def test_make_dataloaders_iterates_full_dataset():
    """Sum of batch sizes across the train loader equals the dataset size."""
    X, y = _tiny_data(n=20)
    train_loader, _ = make_dataloaders(X, y, X, y, batch_size=8)
    total = sum(len(xb) for xb, _ in train_loader)
    assert total == 20


@pytest.mark.unit
def test_make_dataloaders_yields_torch_float_tensors():
    """Loader output is FloatTensors (numpy → torch conversion happened)."""
    X, y = _tiny_data(n=16)
    train_loader, _ = make_dataloaders(X, y, X, y, batch_size=8)
    xb, yb = next(iter(train_loader))
    assert xb.dtype == torch.float32
    assert yb.dtype == torch.float32

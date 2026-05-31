"""Guards the per-batch-scheduler branch of MultiHeadTrainer's step path.

The perf refactor reads ``GradScaler.get_scale()`` (a CPU-GPU sync) only when
``scheduler_per_batch`` is True. The per-batch (OneCycleLR) branch is exercised
in CI only implicitly via the K/TE e2e tests; this pins it directly: a per-batch
scheduler must advance once per optimizer step, not once per epoch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.shared.neural_net import MultiHeadNet
from src.shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders

TARGETS = ["a", "b"]
LOSS_WEIGHTS = {"a": 1.0, "b": 1.0}


def _tiny_loaders(n=8, d=3, batch_size=4):
    rng = np.random.RandomState(0)
    X_train = rng.randn(n, d).astype(np.float32)
    X_val = rng.randn(n, d).astype(np.float32)
    y_train = {t: rng.randn(n).astype(np.float32) for t in TARGETS}
    y_val = {t: rng.randn(n).astype(np.float32) for t in TARGETS}
    loaders = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=batch_size)
    return loaders, n // batch_size  # train uses drop_last=True → batches/epoch


@pytest.mark.unit
def test_onecycle_scheduler_advances_per_batch():
    """per_batch=True: OneCycleLR steps once per optimizer step (not per epoch)."""
    torch.manual_seed(0)
    (train_loader, val_loader), batches_per_epoch = _tiny_loaders()
    model = MultiHeadNet(
        input_dim=3, target_names=TARGETS, backbone_layers=[8], head_hidden=4, dropout=0.0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=100)
    crit = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
    trainer = MultiHeadTrainer(
        model,
        opt,
        sched,
        crit,
        torch.device("cpu"),
        target_names=TARGETS,
        patience=100,
        scheduler_per_batch=True,
    )

    n_epochs = 3
    trainer.train(train_loader, val_loader, n_epochs=n_epochs)

    # Per-batch stepping ⇒ ≥ epochs×batches steps; strictly more than per-epoch
    # (which would be n_epochs). On CPU the scaler is disabled so no step is
    # skipped, so every batch advances the schedule.
    assert sched.last_epoch >= n_epochs * batches_per_epoch
    assert sched.last_epoch > n_epochs


@pytest.mark.unit
def test_non_per_batch_scheduler_advances_per_epoch_only():
    """per_batch=False (cosine): scheduler steps per epoch, not per batch."""
    torch.manual_seed(0)
    (train_loader, val_loader), batches_per_epoch = _tiny_loaders()
    model = MultiHeadNet(
        input_dim=3, target_names=TARGETS, backbone_layers=[8], head_hidden=4, dropout=0.0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
    crit = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
    trainer = MultiHeadTrainer(
        model,
        opt,
        sched,
        crit,
        torch.device("cpu"),
        target_names=TARGETS,
        patience=100,
        scheduler_per_batch=False,
    )

    n_epochs = 3
    trainer.train(train_loader, val_loader, n_epochs=n_epochs)

    # One step per epoch — far fewer than epochs×batches.
    assert sched.last_epoch == n_epochs
    assert sched.last_epoch < n_epochs * batches_per_epoch

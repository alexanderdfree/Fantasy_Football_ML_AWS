"""Tests for shared.training — MultiTargetLoss, MultiTargetDataset, dataloaders, trainer."""

import numpy as np
import pytest
import torch

from RB.rb_config import RB_LOSS_WEIGHTS, RB_TARGETS
from shared.neural_net import MultiHeadNet
from shared.training import (
    MultiHeadTrainer,
    MultiTargetDataset,
    MultiTargetLoss,
    make_dataloaders,
)

# ---------------------------------------------------------------------------
# MultiTargetLoss
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTargetLoss:
    def test_output_types(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        combined, components = loss_fn(preds, targets)
        assert isinstance(combined, torch.Tensor)
        assert isinstance(components, dict)

    def test_component_keys(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        _, components = loss_fn(preds, targets)
        expected = {f"loss_{t}" for t in RB_TARGETS} | {"loss_combined"}
        assert set(components.keys()) == expected

    def test_components_are_scalars(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        _, components = loss_fn(preds, targets)
        for key, val in components.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_zero_loss_on_perfect_prediction(self):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        targets = {t: torch.tensor([1.0, 2.0]) for t in RB_TARGETS}
        combined, _ = loss_fn(targets, targets)
        assert pytest.approx(combined.item(), abs=1e-6) == 0.0

    def test_weights_affect_loss(self, make_tensors):
        preds, targets = make_tensors()
        equal = {t: 1.0 for t in RB_TARGETS}
        heavy = {t: 1.0 for t in RB_TARGETS}
        heavy["rushing_yards"] = 10.0
        loss_equal = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=equal)
        loss_heavy = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=heavy)
        c1, _ = loss_equal(preds, targets)
        c2, _ = loss_heavy(preds, targets)
        assert c1.item() != c2.item()

    def test_combined_loss_is_positive(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        combined, _ = loss_fn(preds, targets)
        assert combined.item() >= 0

    def test_backward_pass(self):
        loss_fn = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        preds = {t: torch.randn(5, requires_grad=True) for t in RB_TARGETS}
        targets = {k: torch.randn(5) for k in preds}
        combined, _ = loss_fn(preds, targets)
        combined.backward()
        for k in preds:
            assert preds[k].grad is not None

    def test_dual_gate_td_losses_emitted(self):
        """RB's two gated TD targets each emit a loss component."""
        loss_fn = MultiTargetLoss(
            target_names=RB_TARGETS,
            loss_weights=RB_LOSS_WEIGHTS,
            gated_td_targets=["rushing_tds", "receiving_tds"],
        )
        preds = {t: torch.randn(5, requires_grad=True) for t in RB_TARGETS}
        preds["rushing_tds_gate_logit"] = torch.randn(5, requires_grad=True)
        preds["receiving_tds_gate_logit"] = torch.randn(5, requires_grad=True)
        targets = {k: torch.randn(5).clamp_min(0) for k in RB_TARGETS}
        _, components = loss_fn(preds, targets)
        assert "loss_td_gate_rushing_tds" in components
        assert "loss_td_gate_receiving_tds" in components


# ---------------------------------------------------------------------------
# MultiTargetDataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTargetDataset:
    def test_length(self):
        X = np.random.randn(20, 5).astype(np.float32)
        y = {"rushing_yards": np.random.randn(20).astype(np.float32)}
        ds = MultiTargetDataset(X, y)
        assert len(ds) == 20

    def test_getitem_types(self):
        X = np.random.randn(10, 3).astype(np.float32)
        y = {
            "rushing_yards": np.random.randn(10).astype(np.float32),
            "receiving_yards": np.random.randn(10).astype(np.float32),
        }
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, dict)
        assert x_item.shape == (3,)

    def test_single_element(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        y = {"rushing_tds": np.array([1.0], dtype=np.float32)}
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert pytest.approx(x_item[0].item()) == 1.0
        assert pytest.approx(y_item["rushing_tds"].item()) == 1.0


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeDataloaders:
    def test_returns_two_loaders(self):
        X_train = np.random.randn(50, 5).astype(np.float32)
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_train = {"rushing_yards": np.random.randn(50).astype(np.float32)}
        y_val = {"rushing_yards": np.random.randn(20).astype(np.float32)}
        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=16)
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_size(self):
        X_train = np.random.randn(64, 5).astype(np.float32)
        y_train = {"rushing_yards": np.random.randn(64).astype(np.float32)}
        X_val = np.random.randn(16, 5).astype(np.float32)
        y_val = {"rushing_yards": np.random.randn(16).astype(np.float32)}
        train_loader, _ = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 32

    def test_iterate_all_batches(self):
        n = 128
        X = np.random.randn(n, 3).astype(np.float32)
        y = {"rushing_yards": np.random.randn(n).astype(np.float32)}
        loader, _ = make_dataloaders(X, y, X[:10], y, batch_size=32)
        total = sum(x.shape[0] for x, _ in loader)
        assert total == n


# ---------------------------------------------------------------------------
# MultiHeadTrainer (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiHeadTrainer:
    @pytest.fixture
    def setup_trainer(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n_train, n_val, d = 64, 16, 5
        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_val = np.random.randn(n_val, d).astype(np.float32)

        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in RB_TARGETS}
        y_val = {t: np.random.randn(n_val).astype(np.float32) for t in RB_TARGETS}

        y_train["total"] = sum(y_train[t] for t in RB_TARGETS)
        y_val["total"] = sum(y_val[t] for t in RB_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

        model = MultiHeadNet(
            input_dim=d,
            target_names=RB_TARGETS,
            backbone_layers=[16, 8],
            head_hidden=4,
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)
        device = torch.device("cpu")

        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            target_names=RB_TARGETS,
            patience=5,
        )
        return trainer, train_loader, val_loader

    def test_train_returns_history(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=10)
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) <= 10

    def test_history_has_all_keys(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=5)
        expected_keys = {"train_loss", "val_loss"}
        expected_keys |= {f"val_loss_{t}" for t in RB_TARGETS}
        expected_keys |= {f"val_mae_{t}" for t in RB_TARGETS}
        assert expected_keys.issubset(set(history.keys()))

    def test_losses_decrease(self, setup_trainer):
        """Training loss should generally decrease (not guaranteed but likely over 20 epochs)."""
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=20)
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_early_stopping(self):
        """Trainer should stop before n_epochs if val loss doesn't improve."""
        np.random.seed(0)
        torch.manual_seed(0)
        n_train, n_val, d = 32, 32, 3
        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_val = np.random.randn(n_val, d).astype(np.float32) * 5 + 10
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in RB_TARGETS}
        y_train["total"] = sum(y_train[t] for t in RB_TARGETS)
        y_val = {t: np.random.randn(n_val).astype(np.float32) * 10 for t in RB_TARGETS}
        y_val["total"] = sum(y_val[t] for t in RB_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        model = MultiHeadNet(
            input_dim=d,
            target_names=RB_TARGETS,
            backbone_layers=[256, 128],
            head_hidden=64,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = MultiTargetLoss(target_names=RB_TARGETS, loss_weights=RB_LOSS_WEIGHTS)

        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            torch.device("cpu"),
            target_names=RB_TARGETS,
            patience=3,
        )
        history = trainer.train(train_loader, val_loader, n_epochs=500)
        assert len(history["train_loss"]) < 500

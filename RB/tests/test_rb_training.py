"""Tests for RB.rb_training — MultiTargetLoss, RBMultiTargetDataset, dataloaders, trainer."""

import numpy as np
import torch
import torch.nn as nn
import pytest

from RB.rb_training import (
    MultiTargetLoss,
    RBMultiTargetDataset,
    make_rb_dataloaders,
    RBMultiHeadTrainer,
)
from RB.rb_neural_net import RBMultiHeadNet


# ---------------------------------------------------------------------------
# MultiTargetLoss
# ---------------------------------------------------------------------------

class TestMultiTargetLoss:
    def _make_tensors(self, n=10):
        preds = {
            "rushing_floor": torch.randn(n),
            "receiving_floor": torch.randn(n),
            "td_points": torch.randn(n),
            "total": torch.randn(n),
        }
        targets = {
            "rushing_floor": torch.randn(n),
            "receiving_floor": torch.randn(n),
            "td_points": torch.randn(n),
            "total": torch.randn(n),
        }
        return preds, targets

    def test_output_types(self):
        loss_fn = MultiTargetLoss()
        preds, targets = self._make_tensors()
        combined, components = loss_fn(preds, targets)
        assert isinstance(combined, torch.Tensor)
        assert isinstance(components, dict)

    def test_component_keys(self):
        loss_fn = MultiTargetLoss()
        preds, targets = self._make_tensors()
        _, components = loss_fn(preds, targets)
        assert set(components.keys()) == {
            "loss_rushing", "loss_receiving", "loss_td",
            "loss_total_aux", "loss_combined",
        }

    def test_components_are_scalars(self):
        loss_fn = MultiTargetLoss()
        preds, targets = self._make_tensors()
        _, components = loss_fn(preds, targets)
        for key, val in components.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_zero_loss_on_perfect_prediction(self):
        loss_fn = MultiTargetLoss()
        targets = {
            "rushing_floor": torch.tensor([1.0, 2.0]),
            "receiving_floor": torch.tensor([3.0, 4.0]),
            "td_points": torch.tensor([5.0, 6.0]),
            "total": torch.tensor([9.0, 12.0]),
        }
        combined, components = loss_fn(targets, targets)  # preds == targets
        assert pytest.approx(combined.item(), abs=1e-6) == 0.0

    def test_weights_affect_loss(self):
        preds, targets = self._make_tensors()
        loss_equal = MultiTargetLoss(w_rushing=1.0, w_receiving=1.0, w_td=1.0, w_total=1.0)
        loss_rush_heavy = MultiTargetLoss(w_rushing=10.0, w_receiving=1.0, w_td=1.0, w_total=1.0)
        c1, _ = loss_equal(preds, targets)
        c2, _ = loss_rush_heavy(preds, targets)
        assert c1.item() != c2.item()

    def test_combined_loss_is_positive(self):
        loss_fn = MultiTargetLoss()
        preds, targets = self._make_tensors()
        combined, _ = loss_fn(preds, targets)
        assert combined.item() >= 0

    def test_backward_pass(self):
        loss_fn = MultiTargetLoss()
        preds = {k: torch.randn(5, requires_grad=True) for k in
                 ["rushing_floor", "receiving_floor", "td_points", "total"]}
        targets = {k: torch.randn(5) for k in preds}
        combined, _ = loss_fn(preds, targets)
        combined.backward()
        for k in preds:
            assert preds[k].grad is not None


# ---------------------------------------------------------------------------
# RBMultiTargetDataset
# ---------------------------------------------------------------------------

class TestRBMultiTargetDataset:
    def test_length(self):
        X = np.random.randn(20, 5).astype(np.float32)
        y = {"rushing_floor": np.random.randn(20).astype(np.float32)}
        ds = RBMultiTargetDataset(X, y)
        assert len(ds) == 20

    def test_getitem_types(self):
        X = np.random.randn(10, 3).astype(np.float32)
        y = {
            "rushing_floor": np.random.randn(10).astype(np.float32),
            "receiving_floor": np.random.randn(10).astype(np.float32),
        }
        ds = RBMultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, dict)
        assert x_item.shape == (3,)

    def test_single_element(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        y = {"td_points": np.array([6.0], dtype=np.float32)}
        ds = RBMultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert pytest.approx(x_item[0].item()) == 1.0
        assert pytest.approx(y_item["td_points"].item()) == 6.0


# ---------------------------------------------------------------------------
# make_rb_dataloaders
# ---------------------------------------------------------------------------

class TestMakeRBDataloaders:
    def test_returns_two_loaders(self):
        X_train = np.random.randn(50, 5).astype(np.float32)
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_train = {"rushing_floor": np.random.randn(50).astype(np.float32)}
        y_val = {"rushing_floor": np.random.randn(20).astype(np.float32)}
        train_loader, val_loader = make_rb_dataloaders(X_train, y_train, X_val, y_val, batch_size=16)
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_size(self):
        X_train = np.random.randn(64, 5).astype(np.float32)
        y_train = {"rushing_floor": np.random.randn(64).astype(np.float32)}
        X_val = np.random.randn(16, 5).astype(np.float32)
        y_val = {"rushing_floor": np.random.randn(16).astype(np.float32)}
        train_loader, _ = make_rb_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 32

    def test_iterate_all_batches(self):
        n = 100
        X = np.random.randn(n, 3).astype(np.float32)
        y = {"rushing_floor": np.random.randn(n).astype(np.float32)}
        loader, _ = make_rb_dataloaders(X, y, X[:10], y, batch_size=32)
        total = sum(x.shape[0] for x, _ in loader)
        assert total == n


# ---------------------------------------------------------------------------
# RBMultiHeadTrainer (integration)
# ---------------------------------------------------------------------------

class TestRBMultiHeadTrainer:
    @pytest.fixture
    def setup_trainer(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n_train, n_val, d = 64, 16, 5
        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_val = np.random.randn(n_val, d).astype(np.float32)

        targets = ["rushing_floor", "receiving_floor", "td_points"]
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in targets}
        y_val = {t: np.random.randn(n_val).astype(np.float32) for t in targets}

        # Add total to targets for loss computation
        y_train["total"] = sum(y_train[t] for t in targets)
        y_val["total"] = sum(y_val[t] for t in targets)

        train_loader, val_loader = make_rb_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=32
        )

        model = RBMultiHeadNet(input_dim=d, backbone_layers=[16, 8], head_hidden=4, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss()
        device = torch.device("cpu")

        trainer = RBMultiHeadTrainer(model, optimizer, scheduler, criterion, device, patience=5)
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
        expected_keys = {
            "train_loss", "val_loss",
            "val_loss_rushing", "val_loss_receiving", "val_loss_td",
            "val_mae_total", "val_mae_rushing", "val_mae_receiving", "val_mae_td",
            "val_rmse_total",
        }
        assert expected_keys.issubset(set(history.keys()))

    def test_losses_decrease(self, setup_trainer):
        """Training loss should generally decrease (not guaranteed but likely over 20 epochs)."""
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=20)
        # Just check first loss > last loss as a sanity check
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_early_stopping(self):
        """Trainer should stop before n_epochs if val loss doesn't improve."""
        np.random.seed(0)
        torch.manual_seed(0)
        n_train, n_val, d = 32, 32, 3
        X_train = np.random.randn(n_train, d).astype(np.float32)
        # Val data drawn from a very different distribution so val loss stalls
        X_val = np.random.randn(n_val, d).astype(np.float32) * 5 + 10
        targets = ["rushing_floor", "receiving_floor", "td_points"]
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in targets}
        y_train["total"] = sum(y_train[t] for t in targets)
        y_val = {t: np.random.randn(n_val).astype(np.float32) * 10 for t in targets}
        y_val["total"] = sum(y_val[t] for t in targets)

        train_loader, val_loader = make_rb_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        model = RBMultiHeadNet(input_dim=d, backbone_layers=[256, 128], head_hidden=64, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = MultiTargetLoss()

        trainer = RBMultiHeadTrainer(model, optimizer, scheduler, criterion,
                                     torch.device("cpu"), patience=3)
        history = trainer.train(train_loader, val_loader, n_epochs=500)
        # Should stop well before 500 epochs due to val loss plateau
        assert len(history["train_loss"]) < 500

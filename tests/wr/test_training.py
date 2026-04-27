"""Tests for src.shared.training — MultiTargetLoss, dataloaders, trainer (WR targets)."""

import numpy as np
import pytest
import torch

from src.shared.neural_net import MultiHeadNet
from src.shared.training import (
    MultiHeadTrainer,
    MultiTargetDataset,
    MultiTargetLoss,
    make_dataloaders,
)
from src.wr.config import WR_LOSS_WEIGHTS, WR_TARGETS


@pytest.mark.unit
class TestMultiTargetLoss:
    def test_output_types(self, wr_nn_tensors):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        preds, targets = wr_nn_tensors
        combined, components = loss_fn(preds, targets)
        assert isinstance(combined, torch.Tensor)
        assert isinstance(components, dict)

    def test_component_keys(self, wr_nn_tensors):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        preds, targets = wr_nn_tensors
        _, components = loss_fn(preds, targets)
        expected = {f"loss_{t}" for t in WR_TARGETS} | {"loss_combined"}
        assert set(components.keys()) == expected

    def test_components_are_scalars(self, wr_nn_tensors):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        preds, targets = wr_nn_tensors
        _, components = loss_fn(preds, targets)
        for key, val in components.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_zero_loss_on_perfect_prediction(self):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        targets = {
            "receiving_tds": torch.tensor([1.0, 0.0]),
            "receiving_yards": torch.tensor([80.0, 40.0]),
            "receptions": torch.tensor([6.0, 3.0]),
            "fumbles_lost": torch.tensor([0.0, 0.0]),
        }
        combined, _ = loss_fn(targets, targets)
        assert pytest.approx(combined.item(), abs=1e-6) == 0.0

    def test_weights_affect_loss(self, wr_nn_tensors):
        preds, targets = wr_nn_tensors
        loss_equal = MultiTargetLoss(
            target_names=WR_TARGETS,
            loss_weights={t: 1.0 for t in WR_TARGETS},
        )
        heavy_weights = {t: 1.0 for t in WR_TARGETS}
        heavy_weights["receiving_yards"] = 10.0
        loss_heavy = MultiTargetLoss(
            target_names=WR_TARGETS,
            loss_weights=heavy_weights,
        )
        c1, _ = loss_equal(preds, targets)
        c2, _ = loss_heavy(preds, targets)
        assert c1.item() != c2.item()

    def test_combined_loss_is_positive(self, wr_nn_tensors):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        preds, targets = wr_nn_tensors
        combined, _ = loss_fn(preds, targets)
        assert combined.item() >= 0

    def test_backward_pass(self):
        loss_fn = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)
        preds = {k: torch.randn(5, requires_grad=True) for k in WR_TARGETS}
        targets = {k: torch.randn(5) for k in preds}
        combined, _ = loss_fn(preds, targets)
        combined.backward()
        for k in preds:
            assert preds[k].grad is not None


@pytest.mark.unit
class TestMultiTargetDataset:
    def test_length(self):
        X = np.random.randn(20, 5).astype(np.float32)
        y = {"receiving_yards": np.random.randn(20).astype(np.float32)}
        ds = MultiTargetDataset(X, y)
        assert len(ds) == 20

    def test_getitem_types(self):
        X = np.random.randn(10, 3).astype(np.float32)
        y = {
            "receiving_yards": np.random.randn(10).astype(np.float32),
            "receptions": np.random.randn(10).astype(np.float32),
        }
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, dict)
        assert x_item.shape == (3,)

    def test_single_element(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        y = {"receiving_tds": np.array([1.0], dtype=np.float32)}
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert pytest.approx(x_item[0].item()) == 1.0
        assert pytest.approx(y_item["receiving_tds"].item()) == 1.0


@pytest.mark.unit
class TestMakeDataloaders:
    def test_returns_two_loaders(self):
        X_train = np.random.randn(50, 5).astype(np.float32)
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_train = {"receiving_yards": np.random.randn(50).astype(np.float32)}
        y_val = {"receiving_yards": np.random.randn(20).astype(np.float32)}
        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=16)
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_size(self):
        X_train = np.random.randn(64, 5).astype(np.float32)
        y_train = {"receiving_yards": np.random.randn(64).astype(np.float32)}
        X_val = np.random.randn(16, 5).astype(np.float32)
        y_val = {"receiving_yards": np.random.randn(16).astype(np.float32)}
        train_loader, _ = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 32

    def test_iterate_all_batches(self):
        n = 128
        X = np.random.randn(n, 3).astype(np.float32)
        y = {"receiving_yards": np.random.randn(n).astype(np.float32)}
        loader, _ = make_dataloaders(X, y, X[:10], y, batch_size=32)
        total = sum(x.shape[0] for x, _ in loader)
        assert total == n


@pytest.mark.integration
class TestMultiHeadTrainer:
    @pytest.fixture
    def setup_trainer(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n_train, n_val, d = 64, 16, 5
        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_val = np.random.randn(n_val, d).astype(np.float32)

        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in WR_TARGETS}
        y_val = {t: np.random.randn(n_val).astype(np.float32) for t in WR_TARGETS}

        y_train["total"] = sum(y_train[t] for t in WR_TARGETS)
        y_val["total"] = sum(y_val[t] for t in WR_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

        model = MultiHeadNet(
            input_dim=d,
            target_names=WR_TARGETS,
            backbone_layers=[16, 8],
            head_hidden=4,
            dropout=0.1,
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
        expected_keys |= {f"val_loss_{t}" for t in WR_TARGETS}
        expected_keys |= {f"val_mae_{t}" for t in WR_TARGETS}
        assert expected_keys.issubset(set(history.keys()))

    def test_losses_decrease(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=20)
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_early_stopping(self):
        np.random.seed(0)
        torch.manual_seed(0)
        n_train, n_val, d = 32, 32, 3
        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_val = np.random.randn(n_val, d).astype(np.float32) * 5 + 10
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in WR_TARGETS}
        y_train["total"] = sum(y_train[t] for t in WR_TARGETS)
        y_val = {t: np.random.randn(n_val).astype(np.float32) * 10 for t in WR_TARGETS}
        y_val["total"] = sum(y_val[t] for t in WR_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        model = MultiHeadNet(
            input_dim=d,
            target_names=WR_TARGETS,
            backbone_layers=[256, 128],
            head_hidden=64,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = MultiTargetLoss(target_names=WR_TARGETS, loss_weights=WR_LOSS_WEIGHTS)

        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            torch.device("cpu"),
            target_names=WR_TARGETS,
            patience=3,
        )
        history = trainer.train(train_loader, val_loader, n_epochs=500)
        assert len(history["train_loss"]) < 500

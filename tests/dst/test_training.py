"""Tests for src.shared.training — MultiTargetLoss, dataloaders, trainer (DST targets)."""

import numpy as np
import pytest
import torch

from src.DST.dst_config import DST_LOSS_WEIGHTS, DST_POISSON_TARGETS, DST_TARGETS
from src.shared.neural_net import MultiHeadNet
from src.shared.training import (
    MultiHeadTrainer,
    MultiTargetDataset,
    MultiTargetLoss,
    make_dataloaders,
)

# Representative target name for dataset / dataloader tests that don't
# care which specific head they bind to.
_SAMPLE_TARGET = DST_TARGETS[0]


@pytest.mark.unit
class TestMultiTargetLoss:
    def test_output_types(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        combined, components = loss_fn(preds, targets)
        assert isinstance(combined, torch.Tensor)
        assert isinstance(components, dict)

    def test_component_keys(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        _, components = loss_fn(preds, targets)
        expected = {f"loss_{t}" for t in DST_TARGETS} | {"loss_combined"}
        assert set(components.keys()) == expected

    def test_components_are_scalars(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        _, components = loss_fn(preds, targets)
        for key, val in components.items():
            assert isinstance(val, float), f"{key} is not a float"

    def test_zero_loss_on_perfect_prediction(self):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        # Build a two-row per-target tensor with arbitrary (float) values; the
        # loss against itself must be zero regardless of which targets exist.
        targets = {t: torch.tensor([5.0, 7.0]) for t in DST_TARGETS}
        combined, _ = loss_fn(targets, targets)
        assert pytest.approx(combined.item(), abs=1e-6) == 0.0

    def test_weights_affect_loss(self, make_tensors):
        preds, targets = make_tensors()
        equal = {t: 1.0 for t in DST_TARGETS}
        heavy = dict(equal)
        heavy[DST_TARGETS[-1]] = 10.0
        loss_equal = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=equal,
        )
        loss_heavy = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=heavy,
        )
        c1, _ = loss_equal(preds, targets)
        c2, _ = loss_heavy(preds, targets)
        assert c1.item() != c2.item()

    def test_combined_loss_is_positive(self, make_tensors):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        preds, targets = make_tensors()
        combined, _ = loss_fn(preds, targets)
        assert combined.item() >= 0

    def test_backward_pass(self):
        loss_fn = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        preds = {k: torch.randn(5, requires_grad=True) for k in DST_TARGETS}
        targets = {k: torch.randn(5) for k in preds}
        combined, _ = loss_fn(preds, targets)
        combined.backward()
        for k in preds:
            assert preds[k].grad is not None


@pytest.mark.unit
class TestMultiTargetLossPoisson:
    """Poisson-NLL path for the four very-rare DST targets."""

    def test_poisson_targets_use_poisson_nll(self):
        loss_fn = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=DST_LOSS_WEIGHTS,
            poisson_targets=DST_POISSON_TARGETS,
        )
        for t in DST_POISSON_TARGETS:
            assert isinstance(loss_fn.loss_fns[t], torch.nn.PoissonNLLLoss)
        for t in set(DST_TARGETS) - set(DST_POISSON_TARGETS):
            assert isinstance(loss_fn.loss_fns[t], torch.nn.HuberLoss)

    def test_poisson_component_matches_reference(self):
        loss_fn = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights={t: 1.0 for t in DST_TARGETS},
            poisson_targets=DST_POISSON_TARGETS,
        )
        # Non-negative rate inputs (what the clamped head output looks like).
        preds = {t: torch.tensor([0.05, 0.12, 0.0, 0.2]) for t in DST_TARGETS}
        targets = {t: torch.tensor([0.0, 1.0, 0.0, 2.0]) for t in DST_TARGETS}
        _, components = loss_fn(preds, targets)
        ref = torch.nn.PoissonNLLLoss(log_input=False, full=False)
        expected = ref(preds["def_tds"], targets["def_tds"]).item()
        assert components["loss_def_tds"] == pytest.approx(expected, rel=1e-6)

    def test_poisson_nll_different_from_huber_on_same_inputs(self):
        """Same preds/targets, different loss family -> different combined loss."""
        preds = {t: torch.tensor([0.05, 0.12, 0.0, 0.2]) for t in DST_TARGETS}
        targets = {t: torch.tensor([0.0, 1.0, 0.0, 2.0]) for t in DST_TARGETS}
        huber_only = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=DST_LOSS_WEIGHTS,
        )
        mixed = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=DST_LOSS_WEIGHTS,
            poisson_targets=DST_POISSON_TARGETS,
        )
        c_huber, _ = huber_only(preds, targets)
        c_mixed, _ = mixed(preds, targets)
        assert c_huber.item() != c_mixed.item()

    def test_poisson_head_backward_pass(self):
        loss_fn = MultiTargetLoss(
            target_names=DST_TARGETS,
            loss_weights=DST_LOSS_WEIGHTS,
            poisson_targets=DST_POISSON_TARGETS,
        )
        # Rates must be non-negative; the tensor must also be a leaf so .grad
        # populates. ``uniform_(a, b)`` fills an empty tensor in-place, keeping
        # it a leaf, and ``requires_grad_(True)`` enables autograd tracking.
        preds = {k: torch.empty(8).uniform_(0.01, 1.0).requires_grad_(True) for k in DST_TARGETS}
        targets = {k: torch.randint(0, 3, (8,)).float() for k in DST_TARGETS}
        combined, _ = loss_fn(preds, targets)
        combined.backward()
        for t in DST_POISSON_TARGETS:
            assert preds[t].grad is not None
            assert torch.isfinite(preds[t].grad).all()


@pytest.mark.unit
class TestMultiTargetDataset:
    def test_length(self):
        X = np.random.randn(20, 5).astype(np.float32)
        y = {_SAMPLE_TARGET: np.random.randn(20).astype(np.float32)}
        ds = MultiTargetDataset(X, y)
        assert len(ds) == 20

    def test_getitem_types(self):
        X = np.random.randn(10, 3).astype(np.float32)
        y = {
            DST_TARGETS[0]: np.random.randn(10).astype(np.float32),
            DST_TARGETS[1]: np.random.randn(10).astype(np.float32),
        }
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, dict)
        assert x_item.shape == (3,)

    def test_single_element(self):
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        y = {_SAMPLE_TARGET: np.array([6.0], dtype=np.float32)}
        ds = MultiTargetDataset(X, y)
        x_item, y_item = ds[0]
        assert pytest.approx(x_item[0].item()) == 1.0
        assert pytest.approx(y_item[_SAMPLE_TARGET].item()) == 6.0


@pytest.mark.unit
class TestMakeDataloaders:
    def test_returns_two_loaders(self):
        X_train = np.random.randn(50, 5).astype(np.float32)
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_train = {_SAMPLE_TARGET: np.random.randn(50).astype(np.float32)}
        y_val = {_SAMPLE_TARGET: np.random.randn(20).astype(np.float32)}
        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=16)
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_size(self):
        X_train = np.random.randn(64, 5).astype(np.float32)
        y_train = {_SAMPLE_TARGET: np.random.randn(64).astype(np.float32)}
        X_val = np.random.randn(16, 5).astype(np.float32)
        y_val = {_SAMPLE_TARGET: np.random.randn(16).astype(np.float32)}
        train_loader, _ = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 32

    def test_iterate_all_batches(self):
        n = 128
        X = np.random.randn(n, 3).astype(np.float32)
        y = {_SAMPLE_TARGET: np.random.randn(n).astype(np.float32)}
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

        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in DST_TARGETS}
        y_val = {t: np.random.randn(n_val).astype(np.float32) for t in DST_TARGETS}

        y_train["total"] = sum(y_train[t] for t in DST_TARGETS)
        y_val["total"] = sum(y_val[t] for t in DST_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)

        model = MultiHeadNet(
            input_dim=d,
            target_names=DST_TARGETS,
            backbone_layers=[16, 8],
            head_hidden=4,
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)
        device = torch.device("cpu")

        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            target_names=DST_TARGETS,
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
        expected_keys.update(f"val_loss_{t}" for t in DST_TARGETS)
        expected_keys.update(f"val_mae_{t}" for t in DST_TARGETS)
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
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in DST_TARGETS}
        y_train["total"] = sum(y_train[t] for t in DST_TARGETS)
        y_val = {t: np.random.randn(n_val).astype(np.float32) * 10 for t in DST_TARGETS}
        y_val["total"] = sum(y_val[t] for t in DST_TARGETS)

        train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)
        model = MultiHeadNet(
            input_dim=d,
            target_names=DST_TARGETS,
            backbone_layers=[256, 128],
            head_hidden=64,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = MultiTargetLoss(target_names=DST_TARGETS, loss_weights=DST_LOSS_WEIGHTS)

        trainer = MultiHeadTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            torch.device("cpu"),
            target_names=DST_TARGETS,
            patience=3,
        )
        history = trainer.train(train_loader, val_loader, n_epochs=500)
        assert len(history["train_loss"]) < 500

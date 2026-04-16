"""Tests for shared.training — history-based components (dataset, collation, dataloaders, trainer)."""

import numpy as np
import torch
import pytest

from shared.training import (
    MultiTargetHistoryDataset,
    collate_with_history,
    make_history_dataloaders,
    MultiTargetLoss,
    MultiHeadHistoryTrainer,
)
from shared.neural_net import MultiHeadNetWithHistory

TARGETS = ["rushing_floor", "receiving_floor", "td_points"]
LOSS_WEIGHTS = {"rushing_floor": 1.0, "receiving_floor": 1.0, "td_points": 1.0}


# ---------------------------------------------------------------------------
# MultiTargetHistoryDataset
# ---------------------------------------------------------------------------

class TestMultiTargetHistoryDataset:
    def test_length(self):
        X_s = np.random.randn(10, 5).astype(np.float32)
        X_h = [np.random.randn(np.random.randint(1, 8), 3).astype(np.float32) for _ in range(10)]
        y = {"t1": np.random.randn(10).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, y)
        assert len(ds) == 10

    def test_getitem_types(self):
        X_s = np.random.randn(5, 4).astype(np.float32)
        X_h = [np.random.randn(3, 2).astype(np.float32) for _ in range(5)]
        y = {"t1": np.random.randn(5).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, y)
        static, history, targets = ds[0]
        assert isinstance(static, torch.Tensor)
        assert isinstance(history, torch.Tensor)
        assert isinstance(targets, dict)
        assert static.shape == (4,)
        assert history.shape == (3, 2)

    def test_variable_length_histories(self):
        X_s = np.random.randn(3, 4).astype(np.float32)
        X_h = [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(5, 3).astype(np.float32),
            np.random.randn(1, 3).astype(np.float32),
        ]
        y = {"t1": np.random.randn(3).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, y)
        _, h0, _ = ds[0]
        _, h1, _ = ds[1]
        assert h0.shape == (2, 3)
        assert h1.shape == (5, 3)


# ---------------------------------------------------------------------------
# collate_with_history
# ---------------------------------------------------------------------------

class TestCollateWithHistory:
    def _make_batch(self, seq_lens, static_dim=4, game_dim=3):
        batch = []
        for slen in seq_lens:
            static = torch.randn(static_dim)
            history = torch.randn(slen, game_dim)
            targets = {"t1": torch.tensor(1.0)}
            batch.append((static, history, targets))
        return batch

    def test_output_structure(self):
        batch = self._make_batch([3, 5, 2])
        statics, padded, masks, targets = collate_with_history(batch)
        assert isinstance(statics, torch.Tensor)
        assert isinstance(padded, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert isinstance(targets, dict)

    def test_padding_to_max_length(self):
        batch = self._make_batch([2, 5, 3])
        statics, padded, masks, targets = collate_with_history(batch)
        assert padded.shape == (3, 5, 3)  # max_len=5
        assert masks.shape == (3, 5)

    def test_mask_values(self):
        batch = self._make_batch([2, 5, 3])
        _, _, masks, _ = collate_with_history(batch)
        assert masks[0, :2].all()
        assert not masks[0, 2:].any()
        assert masks[1, :5].all()
        assert masks[2, :3].all()
        assert not masks[2, 3:].any()

    def test_padded_values_are_zero(self):
        batch = self._make_batch([2, 5])
        _, padded, _, _ = collate_with_history(batch)
        # Sample 0 has 2 real games; positions 2-4 should be zeros
        assert (padded[0, 2:] == 0).all()

    def test_single_sample_batch(self):
        batch = self._make_batch([4])
        statics, padded, masks, targets = collate_with_history(batch)
        assert statics.shape == (1, 4)
        assert padded.shape == (1, 4, 3)
        assert masks.shape == (1, 4)
        assert masks[0].all()


# ---------------------------------------------------------------------------
# make_history_dataloaders
# ---------------------------------------------------------------------------

class TestMakeHistoryDataloaders:
    def _make_data(self, n, static_dim=5, game_dim=3):
        X_s = np.random.randn(n, static_dim).astype(np.float32)
        X_h = [np.random.randn(np.random.randint(1, 8), game_dim).astype(np.float32) for _ in range(n)]
        y = {t: np.random.randn(n).astype(np.float32) for t in TARGETS}
        y["total"] = sum(y[t] for t in TARGETS)
        return X_s, X_h, y

    def test_returns_two_loaders(self):
        X_s, X_h, y = self._make_data(64)
        X_vs, X_vh, yv = self._make_data(16)
        train_loader, val_loader = make_history_dataloaders(
            X_s, X_h, y, X_vs, X_vh, yv, batch_size=32,
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_unpacks_correctly(self):
        X_s, X_h, y = self._make_data(64)
        X_vs, X_vh, yv = self._make_data(16)
        train_loader, _ = make_history_dataloaders(
            X_s, X_h, y, X_vs, X_vh, yv, batch_size=32,
        )
        statics, padded, masks, targets = next(iter(train_loader))
        assert statics.dim() == 2
        assert padded.dim() == 3
        assert masks.dim() == 2
        assert isinstance(targets, dict)

    def test_mask_dtype_is_bool(self):
        X_s, X_h, y = self._make_data(64)
        X_vs, X_vh, yv = self._make_data(16)
        train_loader, _ = make_history_dataloaders(
            X_s, X_h, y, X_vs, X_vh, yv, batch_size=32,
        )
        _, _, masks, _ = next(iter(train_loader))
        assert masks.dtype == torch.bool


# ---------------------------------------------------------------------------
# MultiHeadHistoryTrainer (integration)
# ---------------------------------------------------------------------------

class TestMultiHeadHistoryTrainer:
    @pytest.fixture
    def setup_trainer(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n_train, n_val = 64, 16
        static_dim, game_dim = 5, 3

        X_ts = np.random.randn(n_train, static_dim).astype(np.float32)
        X_th = [np.random.randn(np.random.randint(1, 8), game_dim).astype(np.float32) for _ in range(n_train)]
        y_train = {t: np.random.randn(n_train).astype(np.float32) for t in TARGETS}
        y_train["total"] = sum(y_train[t] for t in TARGETS)

        X_vs = np.random.randn(n_val, static_dim).astype(np.float32)
        X_vh = [np.random.randn(np.random.randint(1, 8), game_dim).astype(np.float32) for _ in range(n_val)]
        y_val = {t: np.random.randn(n_val).astype(np.float32) for t in TARGETS}
        y_val["total"] = sum(y_val[t] for t in TARGETS)

        train_loader, val_loader = make_history_dataloaders(
            X_ts, X_th, y_train, X_vs, X_vh, y_val, batch_size=32,
        )

        model = MultiHeadNetWithHistory(
            static_dim=static_dim, game_dim=game_dim, target_names=TARGETS,
            backbone_layers=[16, 8], d_model=8, n_attn_heads=2,
            head_hidden=4, dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
        device = torch.device("cpu")

        trainer = MultiHeadHistoryTrainer(
            model, optimizer, scheduler, criterion, device,
            target_names=TARGETS, patience=5,
        )
        return trainer, train_loader, val_loader

    def test_train_returns_history(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=10)
        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) <= 10

    def test_history_keys_complete(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=5)
        expected_keys = {
            "train_loss", "val_loss",
            "val_loss_rushing_floor", "val_loss_receiving_floor", "val_loss_td_points",
            "val_mae_total", "val_mae_rushing_floor", "val_mae_receiving_floor", "val_mae_td_points",
            "val_rmse_total",
        }
        assert expected_keys.issubset(set(history.keys()))

    def test_losses_decrease(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=20)
        assert history["train_loss"][0] > history["train_loss"][-1]

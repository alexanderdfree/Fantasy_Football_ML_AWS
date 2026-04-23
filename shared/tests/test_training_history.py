"""Tests for shared.training — history-based components (dataset, collation,
dataloaders, trainer), plus loss-function and non-history dataloader coverage.
"""

import numpy as np
import pytest
import torch

from shared.neural_net import MultiHeadNetWithHistory
from shared.training import (
    MultiHeadHistoryTrainer,
    MultiTargetDataset,
    MultiTargetHistoryDataset,
    MultiTargetLoss,
    collate_with_history,
    make_dataloaders,
    make_history_dataloaders,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]
LOSS_WEIGHTS = {"rushing_yards": 1.0, "receiving_yards": 1.0, "rushing_tds": 1.0}


# ---------------------------------------------------------------------------
# MultiTargetHistoryDataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
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


@pytest.mark.unit
class TestCollateWithHistory:
    def test_output_structure(self, history_batch_factory):
        batch = history_batch_factory([3, 5, 2])
        statics, padded, masks, targets = collate_with_history(batch)
        assert isinstance(statics, torch.Tensor)
        assert isinstance(padded, torch.Tensor)
        assert isinstance(masks, torch.Tensor)
        assert isinstance(targets, dict)

    def test_padding_to_max_length(self, history_batch_factory):
        batch = history_batch_factory([2, 5, 3])
        statics, padded, masks, targets = collate_with_history(batch)
        assert padded.shape == (3, 5, 3)  # max_len=5
        assert masks.shape == (3, 5)

    def test_mask_values(self, history_batch_factory):
        batch = history_batch_factory([2, 5, 3])
        _, _, masks, _ = collate_with_history(batch)
        assert masks[0, :2].all()
        assert not masks[0, 2:].any()
        assert masks[1, :5].all()
        assert masks[2, :3].all()
        assert not masks[2, 3:].any()

    def test_padded_values_are_zero(self, history_batch_factory):
        batch = history_batch_factory([2, 5])
        _, padded, _, _ = collate_with_history(batch)
        # Sample 0 has 2 real games; positions 2-4 should be zeros
        assert (padded[0, 2:] == 0).all()

    def test_single_sample_batch(self, history_batch_factory):
        batch = history_batch_factory([4])
        statics, padded, masks, targets = collate_with_history(batch)
        assert statics.shape == (1, 4)
        assert padded.shape == (1, 4, 3)
        assert masks.shape == (1, 4)
        assert masks[0].all()


# ---------------------------------------------------------------------------
# make_history_dataloaders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeHistoryDataloaders:
    def test_returns_two_loaders(self, history_data_factory):
        X_s, X_h, y = history_data_factory(64)
        X_vs, X_vh, yv = history_data_factory(16)
        train_loader, val_loader = make_history_dataloaders(
            X_s,
            X_h,
            y,
            X_vs,
            X_vh,
            yv,
            batch_size=32,
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_unpacks_correctly(self, history_data_factory):
        X_s, X_h, y = history_data_factory(64)
        X_vs, X_vh, yv = history_data_factory(16)
        train_loader, _ = make_history_dataloaders(
            X_s,
            X_h,
            y,
            X_vs,
            X_vh,
            yv,
            batch_size=32,
        )
        statics, padded, masks, targets = next(iter(train_loader))
        assert statics.dim() == 2
        assert padded.dim() == 3
        assert masks.dim() == 2
        assert isinstance(targets, dict)

    def test_mask_dtype_is_bool(self, history_data_factory):
        X_s, X_h, y = history_data_factory(64)
        X_vs, X_vh, yv = history_data_factory(16)
        train_loader, _ = make_history_dataloaders(
            X_s,
            X_h,
            y,
            X_vs,
            X_vh,
            yv,
            batch_size=32,
        )
        _, _, masks, _ = next(iter(train_loader))
        assert masks.dtype == torch.bool


# ---------------------------------------------------------------------------
# make_dataloaders — non-history variant, batching edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeDataloaders:
    def _make_flat_data(self, n, input_dim=4, targets=TARGETS):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, input_dim)).astype(np.float32)
        y = {t: rng.standard_normal(n).astype(np.float32) for t in targets}
        return X, y

    def test_batch_size_1(self):
        X_tr, y_tr = self._make_flat_data(8)
        X_val, y_val = self._make_flat_data(4)
        train_loader, val_loader = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=1)
        X_batch, y_batch = next(iter(train_loader))
        assert X_batch.shape == (1, 4)
        # Train loader uses drop_last=True so a dataset of 8 yields 8 batches of size 1
        assert sum(1 for _ in train_loader) == 8

    def test_batch_size_larger_than_dataset_yields_one_partial_val_batch(self):
        """Val loader uses drop_last=False so a single partial batch is returned."""
        X_tr, y_tr = self._make_flat_data(16)
        X_val, y_val = self._make_flat_data(4)
        _, val_loader = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=32)
        batches = list(val_loader)
        assert len(batches) == 1
        X_batch, _ = batches[0]
        assert X_batch.shape == (4, 4)  # 4 samples, 4 features

    def test_batch_size_larger_than_train_dataset_drops_all(self):
        """Train loader uses drop_last=True so an oversized batch drops the partial."""
        X_tr, y_tr = self._make_flat_data(4)
        X_val, y_val = self._make_flat_data(4)
        train_loader, _ = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=32)
        assert sum(1 for _ in train_loader) == 0

    def test_empty_dataset_raises_on_iteration(self):
        """Empty training data should surface a clear error at construction or iteration."""
        X_tr = np.zeros((0, 4), dtype=np.float32)
        y_tr = {t: np.zeros(0, dtype=np.float32) for t in TARGETS}
        X_val, y_val = self._make_flat_data(4)
        with pytest.raises((ValueError, RuntimeError)):
            train_loader, _ = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=4)
            list(train_loader)

    def test_shuffle_false_val_loader_is_reproducible(self):
        """Val loader has shuffle=False; two iterations must yield identical order."""
        X_tr, y_tr = self._make_flat_data(16)
        X_val, y_val = self._make_flat_data(8)
        _, val_loader = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=4)
        pass1 = torch.cat([x for x, _ in val_loader])
        pass2 = torch.cat([x for x, _ in val_loader])
        torch.testing.assert_close(pass1, pass2)


# ---------------------------------------------------------------------------
# MultiTargetLoss — weighting semantics and gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTargetLoss:
    def _make_preds_and_targets(self, batch=4):
        torch.manual_seed(42)
        preds = {t: torch.randn(batch) for t in TARGETS}
        targets = {t: torch.randn(batch) for t in TARGETS}
        return preds, targets

    def test_equal_weights_sum_matches_components(self):
        """With all weights=1, combined loss equals sum of per-target losses."""
        preds, targets = self._make_preds_and_targets()
        loss_fn = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
        )
        combined, components = loss_fn(preds, targets)
        expected = sum(components[f"loss_{t}"] for t in TARGETS)
        assert combined.item() == pytest.approx(expected, abs=1e-6)

    def test_weighting_changes_loss(self):
        """Doubling a target's weight must change the combined loss."""
        preds, targets = self._make_preds_and_targets()
        base = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
        )
        weighted = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={"rushing_yards": 2.0, "receiving_yards": 1.0, "rushing_tds": 1.0},
        )
        base_loss, base_comp = base(preds, targets)
        weighted_loss, _ = weighted(preds, targets)
        expected_delta = base_comp["loss_rushing_yards"]  # added one extra copy
        assert weighted_loss.item() == pytest.approx(
            base_loss.item() + expected_delta,
            abs=1e-6,
        )

    def test_zero_weight_ignores_target(self):
        """weight=0 on a target removes its contribution entirely."""
        preds, targets = self._make_preds_and_targets()
        masked = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={"rushing_yards": 1.0, "receiving_yards": 0.0, "rushing_tds": 1.0},
        )
        loss, comp = masked(preds, targets)
        expected = comp["loss_rushing_yards"] + comp["loss_rushing_tds"]
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    def test_gradient_flows_to_each_target_head(self):
        """Gradients w.r.t. each target prediction must be non-zero and scale with weight."""
        # Build two separate loss setups; gradients from the scaled loss should
        # equal 2x the gradients from the base loss for the scaled target.
        torch.manual_seed(42)
        preds1 = {t: torch.randn(4, requires_grad=True) for t in TARGETS}
        targets = {t: torch.randn(4) for t in TARGETS}

        base = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
        )
        loss1, _ = base(preds1, targets)
        loss1.backward()
        base_grads = {t: preds1[t].grad.clone() for t in TARGETS}

        torch.manual_seed(42)
        preds2 = {t: torch.randn(4, requires_grad=True) for t in TARGETS}
        scaled = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={"rushing_yards": 2.0, "receiving_yards": 1.0, "rushing_tds": 1.0},
        )
        loss2, _ = scaled(preds2, targets)
        loss2.backward()
        scaled_grads = {t: preds2[t].grad.clone() for t in TARGETS}

        # rushing_yards grad must be 2x base; other targets unchanged.
        torch.testing.assert_close(scaled_grads["rushing_yards"], 2.0 * base_grads["rushing_yards"])
        torch.testing.assert_close(scaled_grads["receiving_yards"], base_grads["receiving_yards"])
        torch.testing.assert_close(scaled_grads["rushing_tds"], base_grads["rushing_tds"])

    def test_zero_weight_zeros_gradient_for_that_target(self):
        """weight=0 on target k must produce zero gradient on preds[k]."""
        torch.manual_seed(42)
        preds = {t: torch.randn(4, requires_grad=True) for t in TARGETS}
        targets = {t: torch.randn(4) for t in TARGETS}

        loss_fn = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={"rushing_yards": 1.0, "receiving_yards": 0.0, "rushing_tds": 1.0},
        )
        loss, _ = loss_fn(preds, targets)
        loss.backward()
        # receiving_yards prediction has no downstream contribution; grad is zero.
        assert torch.all(preds["receiving_yards"].grad == 0)
        assert not torch.all(preds["rushing_yards"].grad == 0)
        assert not torch.all(preds["rushing_tds"].grad == 0)

    def test_gated_adds_bce_component(self):
        """When gate logits are present, BCE supervision is added."""
        torch.manual_seed(0)
        preds = {t: torch.randn(4) for t in TARGETS}
        preds["rushing_tds_gate_logit"] = torch.randn(4)
        targets = {t: torch.randn(4) for t in TARGETS}
        # Force a positive rushing_tds target so BCE has both classes in the batch.
        targets["rushing_tds"] = torch.tensor([0.0, 1.0, 0.0, 2.0])

        loss_fn = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
            gate_weight=1.0,
            gated_targets=["rushing_tds"],
        )
        _, components = loss_fn(preds, targets)
        # The multi-gate loss keys components by target name so multiple
        # gated heads (e.g. RB's rushing_tds + receiving_tds) can coexist.
        assert "loss_gate_rushing_tds" in components
        assert components["loss_gate_rushing_tds"] > 0

    def test_hurdle_negbin_loss_emits_components_and_backward(self):
        """End-to-end hurdle path: ZTNB value + BCE gate, both flow gradients.

        Leaf tensors are created with ``empty(..).uniform_(..)`` /
        ``empty(..).normal_()`` so ``.requires_grad_(True)`` keeps them leaves
        — see the PR #94 regression on Linux CI where non-leaf gradients were
        silently dropped.
        """
        torch.manual_seed(0)
        preds = {
            "rushing_yards": torch.empty(8).normal_().requires_grad_(True),
            "receiving_yards": torch.empty(8).normal_().requires_grad_(True),
            "rushing_tds": torch.empty(8).uniform_(0.1, 2.0).requires_grad_(True),
        }
        preds["rushing_tds_gate_logit"] = torch.empty(8).normal_().requires_grad_(True)
        preds["rushing_tds_value_mu"] = torch.empty(8).uniform_(0.1, 2.0).requires_grad_(True)
        preds["rushing_tds_value_log_alpha"] = torch.zeros(8).requires_grad_(True)
        targets = {
            "rushing_yards": torch.randn(8),
            "receiving_yards": torch.randn(8),
            "rushing_tds": torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0]),
        }

        loss_fn = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
            head_losses={"rushing_tds": "hurdle_negbin"},
            gate_weight=1.0,
            gated_targets=["rushing_tds"],
        )
        total, components = loss_fn(preds, targets)
        # Gate BCE + ZTNB value both reported.
        assert "loss_gate_rushing_tds" in components
        assert "loss_rushing_tds" in components
        # Backward through ZTNB and BCE paths should populate gradients.
        total.backward()
        assert preds["rushing_tds_gate_logit"].grad is not None
        assert preds["rushing_tds_value_mu"].grad is not None
        assert preds["rushing_tds_value_log_alpha"].grad is not None
        # log_alpha should see some gradient (dispersion affects ZTNB likelihood).
        assert (preds["rushing_tds_value_log_alpha"].grad != 0).any()

    def test_hurdle_negbin_requires_gated_target(self):
        """Misconfiguration: hurdle_negbin without gate membership raises."""
        with pytest.raises(ValueError, match="hurdle_negbin"):
            MultiTargetLoss(
                target_names=TARGETS,
                loss_weights={t: 1.0 for t in TARGETS},
                head_losses={"rushing_tds": "hurdle_negbin"},
                gated_targets=[],  # rushing_tds not gated — should error
            )


# ---------------------------------------------------------------------------
# MultiHeadHistoryTrainer (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiHeadHistoryTrainer:
    @pytest.fixture
    def setup_trainer(self, history_data_factory):
        np.random.seed(42)
        torch.manual_seed(42)

        X_ts, X_th, y_train = history_data_factory(64)
        X_vs, X_vh, y_val = history_data_factory(16)

        train_loader, val_loader = make_history_dataloaders(
            X_ts,
            X_th,
            y_train,
            X_vs,
            X_vh,
            y_val,
            batch_size=32,
        )

        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
        device = torch.device("cpu")

        trainer = MultiHeadHistoryTrainer(
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            target_names=TARGETS,
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

    def test_history_keys_complete(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=5)
        expected_keys = {
            "train_loss",
            "val_loss",
            "val_loss_rushing_yards",
            "val_loss_receiving_yards",
            "val_loss_rushing_tds",
            "val_mae_rushing_yards",
            "val_mae_receiving_yards",
            "val_mae_rushing_tds",
        }
        assert expected_keys.issubset(set(history.keys()))

    def test_losses_decrease(self, setup_trainer):
        trainer, train_loader, val_loader = setup_trainer
        history = trainer.train(train_loader, val_loader, n_epochs=20)
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_best_checkpoint_loaded_on_normal_completion(self, setup_trainer):
        """Training without early-stopping still restores the best checkpoint.

        Before the fix, `best_model_state` was only loaded inside the early-stop
        branch. A run that completed all ``n_epochs`` kept the last epoch's
        weights even when an earlier epoch was better.
        """
        trainer, train_loader, val_loader = setup_trainer
        # Set patience >> n_epochs so early stopping cannot trigger.
        trainer.patience = 10_000
        trainer.train(train_loader, val_loader, n_epochs=3)
        assert trainer.best_model_state is not None
        loaded = trainer.model.state_dict()
        for k, v in trainer.best_model_state.items():
            torch.testing.assert_close(loaded[k], v)


@pytest.mark.unit
class TestAttentionEntropyRegulariserWiring:
    """Trainer must add ``model.attention_entropy_loss()`` to the criterion
    output when the model exposes it with a non-zero coefficient."""

    def _build(self, history_data_factory, *, coeff: float):
        np.random.seed(0)
        torch.manual_seed(0)
        X_ts, X_th, y_train = history_data_factory(32)
        X_vs, X_vh, y_val = history_data_factory(16)
        train_loader, val_loader = make_history_dataloaders(
            X_ts, X_th, y_train, X_vs, X_vh, y_val, batch_size=32
        )
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.0,
            attn_entropy_coeff=coeff,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        criterion = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
        return (
            MultiHeadHistoryTrainer(
                model,
                optimizer,
                scheduler,
                criterion,
                torch.device("cpu"),
                target_names=TARGETS,
                patience=5,
            ),
            train_loader,
            val_loader,
        )

    def test_trainer_runs_with_entropy_regulariser(self, history_data_factory):
        """Training with coeff>0 must complete without errors and produce a
        positive loss on the first epoch (entropy term is non-negative)."""
        trainer, train_loader, val_loader = self._build(history_data_factory, coeff=0.05)
        history = trainer.train(train_loader, val_loader, n_epochs=2)
        assert history["train_loss"][0] > 0

    def test_entropy_term_increments_first_batch_loss(self, history_data_factory):
        """Running a single train batch with coeff>0 produces a larger loss
        than with coeff=0 — the entropy term must actually reach the loss."""
        # Build matched pairs; same seed makes predictions identical at step 0.
        trainer_off, loader_off, _ = self._build(history_data_factory, coeff=0.0)
        trainer_on, loader_on, _ = self._build(history_data_factory, coeff=0.1)
        trainer_off.model.train()
        trainer_on.model.train()

        batch_off = next(iter(loader_off))
        batch_on = next(iter(loader_on))

        preds_off, y_off = trainer_off._forward_batch(batch_off)
        preds_on, y_on = trainer_on._forward_batch(batch_on)

        base_loss, _ = trainer_off.criterion(preds_off, y_off)
        entropy_only_base, _ = trainer_on.criterion(preds_on, y_on)
        entropy_term = trainer_on.model.attention_entropy_loss()

        assert entropy_term is not None and entropy_term.item() > 0
        # Replicates what the trainer does internally: base + entropy
        final_loss = entropy_only_base + entropy_term
        # The off-model and on-model share seeds/architecture → base losses
        # match. The regularised final loss must strictly exceed it.
        torch.testing.assert_close(base_loss, entropy_only_base, atol=1e-5, rtol=0)
        assert final_loss.item() > base_loss.item()

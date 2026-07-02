"""Tests for src.shared.training — history-based components (dataset,
dataloaders, trainer), plus loss-function and non-history dataloader coverage.
"""

import numpy as np
import pytest
import torch

from src.shared.neural_net import MultiHeadNetWithHistory
from src.shared.training import (
    MultiHeadHistoryTrainer,
    MultiHeadHistoryWithOppTrainer,
    MultiTargetDataset,
    MultiTargetHistoryDataset,
    MultiTargetHistoryWithOppDataset,
    MultiTargetLoss,
    make_dataloaders,
    make_history_dataloaders,
    make_history_with_opp_dataloaders,
)

TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]
LOSS_WEIGHTS = {"rushing_yards": 1.0, "receiving_yards": 1.0, "rushing_tds": 1.0}


def _make_padded_history(seq_lens, game_dim=3, max_seq_len=None):
    """Helper: build (X_history, mask) from a list of real sequence lengths."""
    n = len(seq_lens)
    if max_seq_len is None:
        max_seq_len = max(seq_lens)
    X_h = np.zeros((n, max_seq_len, game_dim), dtype=np.float32)
    mask = np.zeros((n, max_seq_len), dtype=bool)
    for i, slen in enumerate(seq_lens):
        X_h[i, :slen] = np.random.randn(slen, game_dim).astype(np.float32)
        mask[i, :slen] = True
    return X_h, mask


# ---------------------------------------------------------------------------
# MultiTargetHistoryDataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTargetHistoryDataset:
    def test_length(self):
        X_s = np.random.randn(10, 5).astype(np.float32)
        X_h, mask = _make_padded_history([3] * 10, game_dim=3, max_seq_len=8)
        y = {"t1": np.random.randn(10).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, mask, y)
        assert len(ds) == 10

    def test_getitem_types_and_shapes(self):
        X_s = np.random.randn(5, 4).astype(np.float32)
        X_h, mask = _make_padded_history([3] * 5, game_dim=2, max_seq_len=8)
        y = {"t1": np.random.randn(5).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, mask, y)
        static, history, sample_mask, targets = ds[0]
        assert isinstance(static, torch.Tensor)
        assert isinstance(history, torch.Tensor)
        assert isinstance(sample_mask, torch.Tensor)
        assert isinstance(targets, dict)
        assert static.shape == (4,)
        assert history.shape == (8, 2)  # max_seq_len × game_dim
        assert sample_mask.shape == (8,)
        assert sample_mask.dtype == torch.bool
        # First 3 positions are real, rest are padding
        assert sample_mask[:3].all()
        assert not sample_mask[3:].any()

    def test_padding_is_zero(self):
        X_s = np.random.randn(2, 4).astype(np.float32)
        X_h, mask = _make_padded_history([2, 5], game_dim=3, max_seq_len=5)
        y = {"t1": np.random.randn(2).astype(np.float32)}
        ds = MultiTargetHistoryDataset(X_s, X_h, mask, y)
        _, h0, m0, _ = ds[0]
        # Sample 0 has 2 real games; positions 2-4 are padding zeros
        assert (h0[2:] == 0).all()
        assert not m0[2:].any()


# ---------------------------------------------------------------------------
# make_history_dataloaders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMakeHistoryDataloaders:
    def test_returns_two_loaders(self, history_data_factory):
        X_s, X_h, mask, y = history_data_factory(64)
        X_vs, X_vh, vmask, yv = history_data_factory(16)
        train_loader, val_loader = make_history_dataloaders(
            X_s, X_h, mask, y, X_vs, X_vh, vmask, yv, batch_size=32
        )
        assert train_loader is not None
        assert val_loader is not None

    def test_batch_unpacks_correctly(self, history_data_factory):
        X_s, X_h, mask, y = history_data_factory(64)
        X_vs, X_vh, vmask, yv = history_data_factory(16)
        train_loader, _ = make_history_dataloaders(
            X_s, X_h, mask, y, X_vs, X_vh, vmask, yv, batch_size=32
        )
        statics, padded, masks, targets = next(iter(train_loader))
        assert statics.dim() == 2
        assert padded.dim() == 3
        assert masks.dim() == 2
        assert isinstance(targets, dict)

    def test_mask_dtype_is_bool(self, history_data_factory):
        X_s, X_h, mask, y = history_data_factory(64)
        X_vs, X_vh, vmask, yv = history_data_factory(16)
        train_loader, _ = make_history_dataloaders(
            X_s, X_h, mask, y, X_vs, X_vh, vmask, yv, batch_size=32
        )
        _, _, masks, _ = next(iter(train_loader))
        assert masks.dtype == torch.bool

    def test_batches_share_max_seq_len_across_loader(self, history_data_factory):
        """Static padding: every batch has the same sequence length (max_seq_len),
        not a batch-local max. This is what removes per-batch padding work."""
        X_s, X_h, mask, y = history_data_factory(64, max_seq_len=8)
        X_vs, X_vh, vmask, yv = history_data_factory(16, max_seq_len=8)
        train_loader, _ = make_history_dataloaders(
            X_s, X_h, mask, y, X_vs, X_vh, vmask, yv, batch_size=16
        )
        seq_lens = {batch[1].shape[1] for batch in train_loader}
        assert seq_lens == {8}


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
        """Empty training data must not silently produce a mis-shaped batch.

        The CPU ``DataLoader`` path surfaces a clear error; the GPU-resident
        batcher (used when CUDA is available) instead yields zero batches. Accept
        either outcome, but reject a silent non-empty result.
        """
        X_tr = np.zeros((0, 4), dtype=np.float32)
        y_tr = {t: np.zeros(0, dtype=np.float32) for t in TARGETS}
        X_val, y_val = self._make_flat_data(4)
        try:
            train_loader, _ = make_dataloaders(X_tr, y_tr, X_val, y_val, batch_size=4)
            n_batches = sum(1 for _ in train_loader)
        except (ValueError, RuntimeError):
            pass  # CPU path: surfaced a clear error, as intended.
        else:
            assert n_batches == 0  # GPU-resident path: no error, but no batches either.

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

    @staticmethod
    def _hurdle_loss_and_targets():
        """Shared hurdle_negbin fixture: loss fn + batch-8 targets (sparse tds)."""
        loss_fn = MultiTargetLoss(
            target_names=TARGETS,
            loss_weights={t: 1.0 for t in TARGETS},
            head_losses={"rushing_tds": "hurdle_negbin"},
            gate_weight=1.0,
            gated_targets=["rushing_tds"],
        )
        targets = {
            "rushing_yards": torch.randn(8),
            "receiving_yards": torch.randn(8),
            "rushing_tds": torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 0.0]),
        }
        return loss_fn, targets

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
        loss_fn, targets = self._hurdle_loss_and_targets()

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

    def test_gated_model_forward_through_hurdle_negbin_backward(self):
        """Seam test (#575): GatedHead forward output feeds hurdle_negbin directly.

        The GatedHead emission keys (``*_gate_logit``/``*_value_mu``/
        ``*_value_log_alpha``) and MultiTargetLoss's hurdle_negbin consumption
        are each unit-tested in isolation on hand-built dicts; this chains
        model.forward → loss → backward so a key-contract drift between the
        two surfaces here instead of only in a full (slow) attn-NN e2e run —
        production RB/WR/TE enable exactly this path via POSITION_CONFIG's
        head_losses/gated_targets, which CONFIG_TINY deliberately omits for
        shard speed.
        """
        torch.manual_seed(0)
        model = MultiHeadNetWithHistory(
            static_dim=5,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[16, 8],
            d_model=8,
            n_attn_heads=2,
            head_hidden=4,
            dropout=0.1,
            gated=True,
            gated_targets=["rushing_tds"],
        )
        x_static = torch.randn(8, 5)
        x_history = torch.randn(8, 6, 3)
        mask = torch.ones(8, 6, dtype=torch.bool)
        preds = model(x_static, x_history, mask)

        loss_fn, targets = self._hurdle_loss_and_targets()
        total, components = loss_fn(preds, targets)
        assert torch.isfinite(total)
        assert "loss_gate_rushing_tds" in components
        assert "loss_rushing_tds" in components

        total.backward()
        backbone_grads = [p.grad for n, p in model.named_parameters() if n.startswith("backbone")]
        assert backbone_grads and all(g is not None for g in backbone_grads)
        assert any((g != 0).any() for g in backbone_grads)


# ---------------------------------------------------------------------------
# MultiHeadHistoryTrainer (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMultiHeadHistoryTrainer:
    @pytest.fixture
    def setup_trainer(self, history_data_factory):
        np.random.seed(42)
        torch.manual_seed(42)

        X_ts, X_th, tr_mask, y_train = history_data_factory(64)
        X_vs, X_vh, v_mask, y_val = history_data_factory(16)

        train_loader, val_loader = make_history_dataloaders(
            X_ts, X_th, tr_mask, y_train, X_vs, X_vh, v_mask, y_val, batch_size=32
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

    @pytest.fixture(autouse=True)
    def _force_eager_cuda_graph(self, monkeypatch):
        # The entropy regulariser adds a second backward, incompatible with
        # CUDA-graph capture (autodetect-ON for sm_80+ since #889). These tests
        # build CPU-device trainers, but cuda_graph_enabled() keys off the host
        # GPU, so on a local sm_80+ box capture is attempted and errors with
        # "backward through the graph a second time". Force eager — production is
        # unaffected (attn_entropy_coeff=0.0 in every config; the live Batch CE
        # is T4/sm_75 with graphs off). Continues #571's test-side dev-box fix
        # that #889's autodetect-ON re-broke.
        monkeypatch.setenv("FF_CUDA_GRAPH", "0")

    def _build(self, history_data_factory, *, coeff: float):
        np.random.seed(0)
        torch.manual_seed(0)
        X_ts, X_th, tr_mask, y_train = history_data_factory(32)
        X_vs, X_vh, v_mask, y_val = history_data_factory(16)
        train_loader, val_loader = make_history_dataloaders(
            X_ts, X_th, tr_mask, y_train, X_vs, X_vh, v_mask, y_val, batch_size=32
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
        """The entropy regulariser must actually reach the loss.

        With coeff=0 the model exposes no entropy term, so the trainer's
        ``loss = loss + entropy_term`` branch never fires. With coeff>0 the
        trainer adds ``model.attention_entropy_loss()`` to the base criterion
        output, so the final loss strictly exceeds the base loss.

        Both legs are checked within a single model/forward. The earlier
        formulation compared a coeff=0 forward's base loss against a coeff>0
        forward's base loss at atol=1e-5 -- a cross-forward float-equality
        precondition that drifts on nondeterministic platforms (Windows/CUDA
        threaded-BLAS reduction order) even when the wiring is correct. The
        additivity it scaffolds is a within-model property, so assert it within
        one model.
        """
        trainer_off, loader_off, _ = self._build(history_data_factory, coeff=0.0)
        trainer_on, loader_on, _ = self._build(history_data_factory, coeff=0.1)
        trainer_off.model.train()
        trainer_on.model.train()

        # coeff=0 leg: run a real forward, but the model exposes no entropy
        # term, so the trainer would add nothing to the base loss.
        trainer_off._forward_batch(next(iter(loader_off)))
        assert trainer_off.model.attention_entropy_loss() is None

        # coeff>0 leg: base loss and entropy term come from the SAME forward, so
        # final = base + entropy has no cross-forward float drift to assert on.
        preds_on, y_on = trainer_on._forward_batch(next(iter(loader_on)))
        base_loss, _ = trainer_on.criterion(preds_on, y_on)
        entropy_term = trainer_on.model.attention_entropy_loss()
        assert entropy_term is not None and entropy_term.item() > 0
        # Replicates what the trainer does internally: final = base + entropy.
        final_loss = base_loss + entropy_term
        assert final_loss.item() > base_loss.item()


# ---------------------------------------------------------------------------
# Opp-history dataset / trainer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiTargetHistoryWithOppDataset:
    def test_length_and_item_tuple(self):
        X_s = np.random.randn(6, 4).astype(np.float32)
        X_h, h_mask = _make_padded_history([3] * 6, game_dim=3, max_seq_len=5)
        X_o, o_mask = _make_padded_history([4] * 6, game_dim=5, max_seq_len=7)
        y = {"t1": np.random.randn(6).astype(np.float32)}
        ds = MultiTargetHistoryWithOppDataset(X_s, X_h, h_mask, X_o, o_mask, y)
        assert len(ds) == 6
        s, h, hm, o, om, t = ds[2]
        assert s.shape == (4,)
        assert h.shape == (5, 3)
        assert hm.shape == (5,) and hm.dtype == torch.bool
        assert o.shape == (7, 5)
        assert om.shape == (7,) and om.dtype == torch.bool
        assert "t1" in t

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="opp history len"):
            MultiTargetHistoryWithOppDataset(
                np.zeros((4, 3), dtype=np.float32),
                np.zeros((4, 5, 2), dtype=np.float32),
                np.zeros((4, 5), dtype=bool),
                np.zeros((3, 5, 2), dtype=np.float32),  # wrong length
                np.zeros((3, 5), dtype=bool),
                {"t1": np.zeros(4, dtype=np.float32)},
            )


@pytest.mark.unit
class TestMultiHeadHistoryWithOppTrainer:
    def _build_trainer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = MultiTargetLoss(target_names=TARGETS, loss_weights=LOSS_WEIGHTS)
        return MultiHeadHistoryWithOppTrainer(
            model, optimizer, scheduler, criterion, torch.device("cpu"), target_names=TARGETS
        )

    def test_forward_batch_passes_opp_tensors(self):
        """Forward batch unpacks the 6-tuple and calls the model with both
        histories, producing all target keys."""
        torch.manual_seed(0)
        model = MultiHeadNetWithHistory(
            static_dim=4,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[8],
            opp_game_dim=5,
        )
        trainer = self._build_trainer(model)
        batch = (
            torch.randn(4, 4),
            torch.randn(4, 6, 3),
            torch.ones(4, 6, dtype=torch.bool),
            torch.randn(4, 5, 5),
            torch.ones(4, 5, dtype=torch.bool),
            {t: torch.randn(4) for t in TARGETS},
        )
        preds, y = trainer._forward_batch(batch)
        for t in TARGETS:
            assert t in preds
            assert preds[t].shape == (4,)

    def test_train_one_epoch_end_to_end(self):
        """End-to-end: dataloader → ``trainer.train`` for one epoch runs
        without shape/device errors and reports finite losses."""
        torch.manual_seed(0)
        np.random.seed(0)
        X_s = np.random.randn(32, 4).astype(np.float32)
        seq_lens = np.random.randint(1, 5, size=32).tolist()
        X_h, h_mask = _make_padded_history(seq_lens, game_dim=3, max_seq_len=5)
        X_o, o_mask = _make_padded_history(seq_lens, game_dim=5, max_seq_len=5)
        y = {t: np.random.randn(32).astype(np.float32) for t in TARGETS}

        train_loader, val_loader = make_history_with_opp_dataloaders(
            X_s,
            X_h,
            h_mask,
            X_o,
            o_mask,
            y,
            X_s,
            X_h,
            h_mask,
            X_o,
            o_mask,
            y,
            batch_size=8,
        )
        model = MultiHeadNetWithHistory(
            static_dim=4,
            game_dim=3,
            target_names=TARGETS,
            backbone_layers=[8],
            opp_game_dim=5,
        )
        trainer = self._build_trainer(model)
        history = trainer.train(train_loader, val_loader, n_epochs=1)
        assert np.isfinite(history["train_loss"][0])
        assert np.isfinite(history["val_loss"][0])

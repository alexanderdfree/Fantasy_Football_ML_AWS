"""CPU smoke tests for ``src.shared.training`` internals exercised only on CUDA in production.

The ``_GPUResidentBatcher`` path (PR #309) replaces the DataLoader on CUDA hosts.
Because CI runs on CPU, the batcher's own logic — permutation, slicing,
``drop_last`` semantics, multi-feature shape preservation — had zero coverage
prior to this file. The class doesn't call ``.cuda()`` in ``__init__``, so a
plain CPU-tensor construction exercises the iteration path.

These tests guard against regressions where the production CUDA path silently
diverges from the contract documented in the class's docstring.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.shared.training import (
    MultiHeadNestedHistoryTrainer,
    MultiHeadTrainer,
    MultiTargetLoss,
    _gpu_resident_device,
    _GPUResidentBatcher,
    _GraphedTrainStep,
)


@pytest.mark.unit
def test_gpu_resident_device_opt_in_by_caller_device():
    """Residency is opt-in by the caller's *training* device, not global CUDA.

    A CPU-device trainer on a GPU box (the per-position ``@integration`` tests,
    and any ``FF_DEVICE=cpu`` run) must fall through to the plain ``DataLoader``;
    only an explicit CUDA device engages the resident batcher. Guards the
    GPU-box regression-test flake fix (see ``todo/fixed-archive.md``). Pure
    ``.type`` gating, so it needs no CUDA to run.
    """
    assert _gpu_resident_device(None) is None
    assert _gpu_resident_device(torch.device("cpu")) is None
    cuda_dev = torch.device("cuda")
    assert _gpu_resident_device(cuda_dev) is cuda_dev


@pytest.mark.unit
class TestGPUResidentBatcher:
    def test_single_feature_tensor_basic_iteration(self):
        """One feature tensor + dict targets — the ``make_dataloaders`` shape."""
        n, feat_dim, batch_size = 10, 5, 4
        X = torch.randn(n, feat_dim)
        y = {"target_a": torch.randn(n)}
        batcher = _GPUResidentBatcher(
            feature_tensors=(X,),
            y_dict=y,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        # drop_last=False, n=10, bs=4 → batches of 4, 4, 2 → 3 batches
        assert len(batcher) == 3
        batches = list(batcher)
        assert len(batches) == 3
        # Each batch is ``(*features, y_dict)`` — here ``(X_batch, y_dict)``.
        for batch in batches[:-1]:
            x_batch, y_batch = batch
            assert x_batch.shape == (batch_size, feat_dim)
            assert "target_a" in y_batch
            assert y_batch["target_a"].shape == (batch_size,)
        # Last batch is the remainder.
        x_last, y_last = batches[-1]
        assert x_last.shape == (2, feat_dim)
        assert y_last["target_a"].shape == (2,)

    def test_drop_last_true(self):
        """``drop_last=True`` mirrors ``DataLoader(drop_last=True)`` — drop ragged tail."""
        n, feat_dim, batch_size = 10, 5, 4
        X = torch.randn(n, feat_dim)
        y = {"target_a": torch.randn(n)}
        batcher = _GPUResidentBatcher(
            feature_tensors=(X,),
            y_dict=y,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        # drop_last=True, n=10, bs=4 → 2 batches of 4 (drop tail of 2)
        assert len(batcher) == 2
        batches = list(batcher)
        assert len(batches) == 2
        for batch in batches:
            x_batch, y_batch = batch
            assert x_batch.shape == (batch_size, feat_dim)
            assert y_batch["target_a"].shape == (batch_size,)

    def test_multi_feature_tuple(self):
        """Mirrors ``make_history_dataloaders``: (X_static, X_history, mask) + y_dict."""
        n, static_dim, max_seq, game_dim, batch_size = 8, 3, 4, 2, 4
        X_static = torch.randn(n, static_dim)
        X_history = torch.randn(n, max_seq, game_dim)
        mask = torch.ones(n, max_seq, dtype=torch.bool)
        y = {"t1": torch.randn(n), "t2": torch.randn(n)}
        batcher = _GPUResidentBatcher(
            feature_tensors=(X_static, X_history, mask),
            y_dict=y,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        assert len(batcher) == 2
        for batch in batcher:
            # 3 feature tensors + 1 dict = 4-tuple
            assert len(batch) == 4
            x_s, x_h, m, y_b = batch
            assert x_s.shape == (batch_size, static_dim)
            assert x_h.shape == (batch_size, max_seq, game_dim)
            assert m.shape == (batch_size, max_seq)
            assert m.dtype == torch.bool
            assert set(y_b.keys()) == {"t1", "t2"}
            assert y_b["t1"].shape == (batch_size,)

    def test_shuffle_permutes(self):
        """``shuffle=True`` should produce a non-identity permutation w.h.p."""
        n, feat_dim = 20, 3
        # Use unique identifiers so we can detect order.
        X = torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(-1, feat_dim).contiguous()
        y = {"t1": torch.arange(n, dtype=torch.float32)}
        torch.manual_seed(0)
        shuffled = _GPUResidentBatcher(
            feature_tensors=(X,),
            y_dict=y,
            batch_size=n,  # one batch covers all
            shuffle=True,
            drop_last=False,
        )
        ((x_batch, y_batch),) = list(shuffled)
        # Same set of values…
        assert torch.equal(torch.sort(y_batch["t1"]).values, torch.arange(n, dtype=torch.float32))
        # …in a different order than the input. Seed 0 should reliably permute.
        assert not torch.equal(y_batch["t1"], torch.arange(n, dtype=torch.float32))

    def test_no_shuffle_preserves_order(self):
        """``shuffle=False`` must preserve insertion order."""
        n, feat_dim = 8, 2
        X = torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(-1, feat_dim).contiguous()
        y = {"t1": torch.arange(n, dtype=torch.float32)}
        batcher = _GPUResidentBatcher(
            feature_tensors=(X,),
            y_dict=y,
            batch_size=n,
            shuffle=False,
            drop_last=False,
        )
        ((x_batch, y_batch),) = list(batcher)
        assert torch.equal(y_batch["t1"], torch.arange(n, dtype=torch.float32))

    def test_empty_feature_tensors_raises(self):
        """Empty feature tuple should fail fast, not silently produce no batches."""
        with pytest.raises(ValueError, match="at least one feature tensor"):
            _GPUResidentBatcher(
                feature_tensors=(),
                y_dict={"t": torch.randn(5)},
                batch_size=2,
                shuffle=False,
                drop_last=False,
            )

    def test_mismatched_feature_n_raises(self):
        """Feature tensors with mismatched N must fail at construction."""
        X1 = torch.randn(10, 3)
        X2 = torch.randn(8, 3)  # wrong N
        with pytest.raises(ValueError, match="must share N"):
            _GPUResidentBatcher(
                feature_tensors=(X1, X2),
                y_dict={"t": torch.randn(10)},
                batch_size=4,
                shuffle=False,
                drop_last=False,
            )

    def test_mismatched_target_n_raises(self):
        """y_dict tensor N must match feature N."""
        X = torch.randn(10, 3)
        y = {"t": torch.randn(8)}  # wrong N
        with pytest.raises(ValueError, match="shape 8 != N=10"):
            _GPUResidentBatcher(
                feature_tensors=(X,),
                y_dict=y,
                batch_size=4,
                shuffle=False,
                drop_last=False,
            )

    def test_drop_last_len_matches_iter_count(self):
        """``len(batcher)`` must match the number of yielded batches in both modes."""
        n, feat_dim, batch_size = 13, 4, 5
        X = torch.randn(n, feat_dim)
        y = {"t": torch.randn(n)}
        for drop_last in (True, False):
            batcher = _GPUResidentBatcher(
                feature_tensors=(X,),
                y_dict=y,
                batch_size=batch_size,
                shuffle=False,
                drop_last=drop_last,
            )
            batches = list(batcher)
            assert len(batches) == len(batcher), (
                f"drop_last={drop_last}: len()={len(batcher)} but yielded {len(batches)}"
            )


@pytest.mark.unit
class TestGraphedTrainStep:
    """CPU-eager checks for the full-step capture pieces. The capture itself
    is CUDA-only (GPU-dry-run-gated, like all graph paths); these pin the
    wrapper's math and the index iterator's RNG contract, which are
    device-agnostic."""

    def _tiny_setup(self, n=12, d=5, batch_size=4):
        torch.manual_seed(7)
        feats = (torch.randn(n, d),)
        y = {"a": torch.randn(n).abs(), "b": torch.randn(n).abs()}
        batcher = _GPUResidentBatcher(
            feature_tensors=feats, y_dict=y, batch_size=batch_size, shuffle=True, drop_last=True
        )

        class _TwoHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(d, 2)

            def forward(self, x):
                out = self.fc(x)
                # softplus keeps the poisson_nll head's rate positive — an
                # untrained Linear emits negatives, which NaN the NLL (the
                # production model clamps via non_negative_targets instead).
                return {"a": out[:, 0], "b": torch.nn.functional.softplus(out[:, 1])}

        model = _TwoHead()
        criterion = MultiTargetLoss(
            target_names=["a", "b"],
            loss_weights={"a": 1.0, "b": 2.0},
            head_losses={"a": "huber", "b": "poisson_nll"},
        )
        return feats, y, batcher, model, criterion

    def test_wrapper_matches_eager_gather_plus_loss(self):
        feats, y, batcher, model, criterion = self._tiny_setup()
        step = _GraphedTrainStep(model, criterion, batcher.features, batcher.y_dict)
        idx = torch.tensor([3, 0, 7, 5])
        wrapped = step(idx)
        preds = model(feats[0].index_select(0, idx))
        y_sel = {k: v.index_select(0, idx) for k, v in y.items()}
        eager, _ = criterion._compute_loss_components(preds, y_sel)
        assert wrapped.item() == pytest.approx(eager.item(), rel=1e-6)

    def test_wrapper_backward_reaches_model_params(self):
        _, _, batcher, model, criterion = self._tiny_setup()
        step = _GraphedTrainStep(model, criterion, batcher.features, batcher.y_dict)
        loss = step(torch.tensor([0, 1, 2, 3]))
        loss.backward()
        assert model.fc.weight.grad is not None

    def test_index_batches_matches_iter_rng_and_order(self):
        feats, y, batcher, _, _ = self._tiny_setup()
        torch.manual_seed(123)
        sliced = [b[0] for b in batcher]  # first feature tensor per batch
        torch.manual_seed(123)
        idxs = list(batcher.index_batches())
        assert len(idxs) == len(sliced) == len(batcher)
        for idx, ref in zip(idxs, sliced, strict=True):
            assert torch.equal(feats[0].index_select(0, idx), ref)

    def test_full_step_gating_noops_on_cpu_and_off_knob(self, monkeypatch):
        _, _, batcher, model, criterion = self._tiny_setup()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = MultiHeadTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            criterion=criterion,
            device=torch.device("cpu"),
            target_names=["a", "b"],
            patience=3,
        )
        # CPU device -> never engages, even with the env forced on.
        monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "1")
        assert trainer._maybe_graph_full_step(batcher) is False
        assert trainer._graphed_step is None
        # And a CPU train() runs the plain path end-to-end with the env on.
        val_loader = _GPUResidentBatcher(
            feature_tensors=batcher.features,
            y_dict=batcher.y_dict,
            batch_size=4,
            shuffle=False,
            drop_last=False,
        )
        history = trainer.train(batcher, val_loader, n_epochs=2)
        assert len(history["train_loss"]) == 2

    def test_full_step_gating_noops_for_nested_k_trainer(self, monkeypatch):
        _, _, batcher, model, criterion = self._tiny_setup()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = MultiHeadNestedHistoryTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            criterion=criterion,
            device=torch.device("cpu"),
            target_names=["a", "b"],
            patience=3,
        )
        monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "1")
        assert trainer._maybe_graph_full_step(batcher) is False


@pytest.mark.unit
class TestGraphedValPass:
    """CPU-side checks for the graphed val pass: prefix/tail arithmetic and
    body math are device-agnostic; the CUDA capture itself is covered by the
    skipif test below + the Batch smoke."""

    def _setup(self, n=11, d=5, batch_size=4):
        torch.manual_seed(3)
        feats = (torch.randn(n, d),)
        y = {"a": torch.randn(n).abs(), "b": torch.randn(n).abs()}
        loader = _GPUResidentBatcher(
            feature_tensors=feats, y_dict=y, batch_size=batch_size, shuffle=False, drop_last=False
        )

        class _TwoHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(d, 2)

            def forward(self, x):
                out = self.fc(x)
                return {"a": out[:, 0], "b": torch.nn.functional.softplus(out[:, 1])}

        criterion = MultiTargetLoss(
            target_names=["a", "b"],
            loss_weights={"a": 1.0, "b": 2.0},
            head_losses={"a": "huber", "b": "poisson_nll"},
        )
        from src.shared.training import _GraphedValPass

        gv = _GraphedValPass(_TwoHead(), criterion, loader, ["a", "b"], torch.device("cpu"))
        return feats, y, loader, gv

    def test_prefix_tail_arithmetic(self):
        _, _, _, gv = self._setup(n=11, batch_size=4)
        assert (gv.k, gv._n_fixed, gv._rem) == (2, 8, 3)
        _, _, _, gv0 = self._setup(n=8, batch_size=4)
        assert (gv0.k, gv0._rem) == (2, 0)
        assert list(gv0.tail_batches()) == []

    def test_tail_batch_matches_loader_last_batch(self):
        feats, y, loader, gv = self._setup(n=11, batch_size=4)
        (tail,) = list(gv.tail_batches())
        *tail_feats, tail_y = tail
        last = list(loader)[-1]
        *last_feats, last_y = last
        assert torch.equal(tail_feats[0], last_feats[0])
        for k in y:
            assert torch.equal(tail_y[k], last_y[k])

    def test_body_matches_eager_prefix_math(self):
        """_run_body's accumulators must equal an eager loop over the K
        full-size batches (CPU: same code path the graph captures)."""
        feats, y, loader, gv = self._setup(n=11, batch_size=4)
        # comp_sums normally preallocated by build(); do it manually on CPU.
        f0 = tuple(t.narrow(0, 0, 4) for t in feats)
        y0 = {k: v.narrow(0, 0, 4) for k, v in y.items()}
        _, comps = gv._criterion._compute_loss_components_capturable(gv._model(*f0), y0)
        gv.comp_sums = {k: torch.zeros((), dtype=torch.float32) for k in comps}
        with torch.no_grad():
            gv._run_body()

        loss_sum = torch.zeros((), dtype=torch.float32)
        with torch.no_grad():
            for i in range(2):
                fb = tuple(t.narrow(0, i * 4, 4) for t in feats)
                yb = {k: v.narrow(0, i * 4, 4) for k, v in y.items()}
                combined, _ = gv._criterion._compute_loss_components_capturable(gv._model(*fb), yb)
                loss_sum += combined.float()
        assert gv.loss_sum.item() == pytest.approx(loss_sum.item(), rel=1e-6)
        for k in ("a", "b"):
            assert gv.pred_bufs[k].shape == (8,)

    def test_gating_noop_without_full_step(self, monkeypatch):
        """_maybe_graph_val never builds when the train-side capture didn't
        engage (CPU trainers, K's nested trainer, knob off — all imply
        _graphed_step is None)."""
        _, _, loader, _ = self._setup()
        model = nn.Linear(5, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = MultiHeadTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            criterion=MultiTargetLoss(target_names=["a"], loss_weights={"a": 1.0}),
            device=torch.device("cpu"),
            target_names=["a"],
            patience=2,
        )
        monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "1")
        trainer._maybe_graph_val(loader)
        assert trainer._graphed_val is None
        assert trainer._graphed_val_failed is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only capture path")
def test_graphed_val_engages_and_matches_eager_on_cuda(monkeypatch):
    """GPU boxes only (5080/Batch dry-run): the captured val pass must engage
    behind an engaged train capture and produce val metrics that move across
    epochs (weight visibility) — full parity is covered by the Batch smoke."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    n, d = 64, 6
    feats = (torch.randn(n, d, device=device),)
    y = {"a": torch.randn(n, device=device).abs()}
    train_loader = _GPUResidentBatcher(
        feature_tensors=feats, y_dict=y, batch_size=16, shuffle=True, drop_last=True
    )
    val_loader = _GPUResidentBatcher(
        feature_tensors=feats, y_dict=y, batch_size=16, shuffle=False, drop_last=False
    )

    class _OneHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(d, 1)

        def forward(self, x):
            return {"a": self.fc(x).squeeze(-1)}

    model = _OneHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)
    trainer = MultiHeadTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        criterion=MultiTargetLoss(target_names=["a"], loss_weights={"a": 1.0}),
        device=device,
        target_names=["a"],
        patience=10,
    )
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "1")
    history = trainer.train(train_loader, val_loader, n_epochs=3)
    assert trainer._graphed_step is not None, "train-side capture must engage"
    assert trainer._graphed_val is not None, "val capture must engage"
    assert len(history["val_loss"]) == 3
    assert len(set(history["val_loss"])) > 1, "val loss must move across epochs"

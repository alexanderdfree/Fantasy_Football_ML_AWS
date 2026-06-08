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

from src.shared.training import _gpu_resident_device, _GPUResidentBatcher


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

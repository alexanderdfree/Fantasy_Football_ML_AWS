"""Unit tests for the opt-in CUDA-graph training path (``FF_CUDA_GRAPH``) and the
investigation knobs kept alongside it (``FF_NN_NORM``, ``FF_FORCE_DROPOUT_ZERO``,
``FF_NN_FIXED_EPOCHS``).

All four default to off / BatchNorm so production stays byte-identical. CI runs
on CPU, where ``cuda_graph_enabled()`` is False and ``make_graphed_callables`` is
never reached — so these tests pin the *gating contract* and the CPU/flag-off
no-ops, plus the CPU-observable effects of the norm / dropout / fixed-epoch
knobs. The GPU A/B that established the speed/inertness tradeoff (per-step
bit-exact, but FP16+GradScaler amplifies the multi-step trajectory ~0.5%) lives
in todo/gpu_launch_bound_levers.md (Lever A).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.shared.neural_net import MultiHeadNetWithHistory, _build_backbone
from src.shared.pipeline import _maybe_force_dropout_zero
from src.shared.training import (
    MultiHeadHistoryTrainer,
    MultiHeadHistoryWithOppTrainer,
    MultiHeadNestedHistoryTrainer,
    MultiHeadTrainer,
    MultiTargetLoss,
    _restore_batchnorm_state,
    _snapshot_batchnorm_state,
    make_history_dataloaders,
)
from src.shared.utils import cuda_graph_enabled

_TARGETS = ["rushing_yards", "receiving_yards", "rushing_tds"]
_WEIGHTS = {t: 1.0 for t in _TARGETS}


# ---------------------------------------------------------------------------
# cuda_graph_enabled() — autodetect-ON for CUDA sm_80+; FF_CUDA_GRAPH force-off override
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_cuda_graph_off_without_cuda(monkeypatch):
    """No CUDA (CPU/CI) → always off, with FF_CUDA_GRAPH unset OR forced on:
    the hardware floor keeps CPU/CI byte-identical to the eager path."""
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    assert cuda_graph_enabled() is False
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    assert cuda_graph_enabled() is False


@pytest.mark.unit
@pytest.mark.parametrize("env", ["unset", ""])
def test_cuda_graph_autodetect_on_sm80plus_when_unset(monkeypatch, env):
    """New default: sm_80+ with FF_CUDA_GRAPH unset/empty → graphs ON via
    device autodetect (no manual opt-in)."""
    if env == "unset":
        monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    else:
        monkeypatch.setenv("FF_CUDA_GRAPH", env)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    assert cuda_graph_enabled() is True


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["0", "false", "no", "off"])
def test_cuda_graph_force_off_override_on_sm80plus(monkeypatch, flag):
    """On capable hardware graphs autodetect ON; a falsy FF_CUDA_GRAPH is the
    force-off override (e.g. for a bit-comparable eager A/B)."""
    monkeypatch.setenv("FF_CUDA_GRAPH", flag)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    assert cuda_graph_enabled() is False


@pytest.mark.unit
def test_cuda_graph_noop_when_cuda_disabled(monkeypatch):
    """FF_CUDA_GRAPH=1 on a CPU/forced-CPU box is still off."""
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    monkeypatch.setenv("FF_DEVICE", "cpu")
    assert cuda_graph_enabled() is False


@pytest.mark.unit
def test_cuda_graph_noop_on_t4_sm75(monkeypatch):
    """T4 (sm_75) stays off even opted-in — parity with _maybe_compile."""
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert cuda_graph_enabled() is False


@pytest.mark.unit
@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (12, 0)])
def test_cuda_graph_enabled_on_sm80plus(monkeypatch, cap):
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)
    assert cuda_graph_enabled() is True


# ---------------------------------------------------------------------------
# Trainer hooks: _maybe_graph_model no-op (CPU) + _graph_inputs unpacking
# ---------------------------------------------------------------------------
def _bare_trainer(cls, model, targets=_TARGETS):
    return cls(
        model=model,
        optimizer=None,
        scheduler=None,
        criterion=None,
        device=torch.device("cpu"),
        target_names=targets,
    )


@pytest.mark.unit
def test_maybe_graph_model_noop_when_disabled(monkeypatch):
    """Flag off → _maybe_graph_model leaves self.model untouched (same object)."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    model = nn.Linear(3, 2)
    tr = _bare_trainer(MultiHeadTrainer, model, targets=["y"])
    assert tr._graphed is False
    tr._maybe_graph_model(train_loader=None)  # returns before touching the loader
    assert tr.model is model
    assert tr._graphed is False


@pytest.mark.unit
def test_graph_inputs_base():
    tr = _bare_trainer(MultiHeadTrainer, nn.Linear(3, 2), targets=["y"])
    X = torch.randn(4, 3)
    args = tr._graph_inputs((X, {"y": torch.randn(4)}))
    assert len(args) == 1 and torch.equal(args[0], X)


@pytest.mark.unit
def test_graph_inputs_history():
    tr = _bare_trainer(MultiHeadHistoryTrainer, nn.Linear(3, 2))
    xs, xh, m = torch.randn(4, 3), torch.randn(4, 5, 2), torch.ones(4, 5, dtype=torch.bool)
    args = tr._graph_inputs((xs, xh, m, {"y": torch.randn(4)}))
    assert len(args) == 3
    assert torch.equal(args[0], xs) and torch.equal(args[1], xh) and torch.equal(args[2], m)


@pytest.mark.unit
def test_graph_inputs_with_opp():
    tr = _bare_trainer(MultiHeadHistoryWithOppTrainer, nn.Linear(3, 2))
    xs, xh, m = torch.randn(4, 3), torch.randn(4, 5, 2), torch.ones(4, 5, dtype=torch.bool)
    xo, om = torch.randn(4, 5, 2), torch.ones(4, 5, dtype=torch.bool)
    args = tr._graph_inputs((xs, xh, m, xo, om, {"y": torch.randn(4)}))
    assert len(args) == 5
    assert torch.equal(args[3], xo) and torch.equal(args[4], om)


@pytest.mark.unit
def test_nested_history_trainer_graph_noop_even_when_enabled(monkeypatch):
    """K's nested attention path is explicitly eager when FF_CUDA_GRAPH=1."""
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    monkeypatch.setattr("src.shared.training.cuda_graph_enabled", lambda: True)
    model = nn.Linear(3, 2)
    tr = _bare_trainer(MultiHeadNestedHistoryTrainer, model)
    tr._maybe_graph_model(train_loader=object())
    assert tr.model is model
    assert tr._graphed is False


@pytest.mark.unit
def test_batchnorm_snapshot_restore_covers_running_buffers():
    bn = nn.BatchNorm1d(3)
    bn.train()
    _ = bn(torch.randn(8, 3) + 3.0)
    snapshot = _snapshot_batchnorm_state(bn)

    original_mean = bn.running_mean.clone()
    original_var = bn.running_var.clone()
    original_batches = bn.num_batches_tracked.clone()

    bn.running_mean.add_(10)
    bn.running_var.mul_(2)
    bn.num_batches_tracked.add_(5)
    _restore_batchnorm_state(snapshot)

    assert torch.equal(bn.running_mean, original_mean)
    assert torch.equal(bn.running_var, original_var)
    assert torch.equal(bn.num_batches_tracked, original_batches)


@pytest.mark.unit
def test_fixed_amp_scale_env_is_trainer_local(monkeypatch):
    monkeypatch.setenv("FF_AMP_FIXED_SCALE", "1")
    tr = _bare_trainer(MultiHeadTrainer, nn.Linear(3, 2), targets=["y"])
    assert tr._fixed_amp_scale is True


# ---------------------------------------------------------------------------
# FF_NN_NORM — backbone BatchNorm (default) vs LayerNorm
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_backbone_batchnorm_by_default(monkeypatch):
    monkeypatch.delenv("FF_NN_NORM", raising=False)
    bb = _build_backbone(8, [16, 8], 0.0)
    assert any(isinstance(m, nn.BatchNorm1d) for m in bb.modules())
    assert not any(isinstance(m, nn.LayerNorm) for m in bb.modules())


@pytest.mark.unit
@pytest.mark.parametrize("val", ["layer", "ln", "layernorm", "LAYER"])
def test_backbone_layernorm_when_ff_nn_norm(monkeypatch, val):
    monkeypatch.setenv("FF_NN_NORM", val)
    bb = _build_backbone(8, [16, 8], 0.0)
    assert any(isinstance(m, nn.LayerNorm) for m in bb.modules())
    assert not any(isinstance(m, nn.BatchNorm1d) for m in bb.modules())


# ---------------------------------------------------------------------------
# FF_FORCE_DROPOUT_ZERO — pure config transform
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_force_dropout_zero_noop_by_default(monkeypatch):
    monkeypatch.delenv("FF_FORCE_DROPOUT_ZERO", raising=False)
    cfg = {"nn_dropout": 0.3, "attn_dropout": 0.05}
    assert _maybe_force_dropout_zero(cfg) is cfg  # unchanged object


@pytest.mark.unit
def test_force_dropout_zero_zeros_all_when_set(monkeypatch):
    monkeypatch.setenv("FF_FORCE_DROPOUT_ZERO", "1")
    cfg = {
        "nn_dropout": 0.3,
        "attn_dropout": 0.05,
        "attn_history_dropout": 0.1,
        "attn_self_dropout": 0.2,
        "nn_lr": 1e-3,
    }
    out = _maybe_force_dropout_zero(cfg)
    assert out is not cfg  # returns a copy
    for k in ("nn_dropout", "attn_dropout", "attn_history_dropout", "attn_self_dropout"):
        assert out[k] == 0.0
    assert out["nn_lr"] == 1e-3  # non-dropout keys untouched
    assert cfg["nn_dropout"] == 0.3  # original not mutated


# ---------------------------------------------------------------------------
# FF_NN_FIXED_EPOCHS — overrides n_epochs, disables early-stop (CPU train run)
# ---------------------------------------------------------------------------
def _tiny_history_trainer():
    np.random.seed(0)
    torch.manual_seed(0)
    n, sd, gd, seq = 24, 5, 3, 6
    xs = np.random.randn(n, sd).astype(np.float32)
    xh = np.random.randn(n, seq, gd).astype(np.float32)
    mask = np.ones((n, seq), dtype=bool)
    y = {t: np.random.randn(n).astype(np.float32) for t in _TARGETS}
    train_loader, val_loader = make_history_dataloaders(
        xs, xh, mask, y, xs, xh, mask, y, batch_size=8
    )
    model = MultiHeadNetWithHistory(
        static_dim=sd,
        game_dim=gd,
        target_names=_TARGETS,
        backbone_layers=[8, 4],
        d_model=8,
        n_attn_heads=2,
        head_hidden=4,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    trainer = MultiHeadHistoryTrainer(
        model,
        optimizer,
        scheduler,
        MultiTargetLoss(target_names=_TARGETS, loss_weights=_WEIGHTS),
        torch.device("cpu"),
        target_names=_TARGETS,
        patience=1,  # would early-stop almost immediately if not overridden
    )
    return trainer, train_loader, val_loader


@pytest.mark.unit
def test_fixed_epochs_overrides_count_and_disables_early_stop(monkeypatch):
    """FF_NN_FIXED_EPOCHS=N: train exactly N epochs even with patience=1 and a
    larger n_epochs argument (proves both the override and no-early-stop)."""
    monkeypatch.setenv("FF_NN_FIXED_EPOCHS", "4")
    trainer, train_loader, val_loader = _tiny_history_trainer()
    history = trainer.train(train_loader, val_loader, n_epochs=10)
    assert len(history["train_loss"]) == 4


@pytest.mark.unit
def test_unset_does_not_override_epoch_count(monkeypatch):
    """Default (FF_NN_FIXED_EPOCHS unset): no epoch override — with early-stop
    disabled (high patience) the loop runs the full ``n_epochs``, i.e. it does
    NOT force the count the override test pins. (Asserting early-stop *fires* is
    deliberately avoided: a tiny synthetic val curve can improve monotonically,
    which is environment-sensitive and flaked CI once.)"""
    monkeypatch.delenv("FF_NN_FIXED_EPOCHS", raising=False)
    trainer, train_loader, val_loader = _tiny_history_trainer()
    trainer.patience = 1000  # never early-stop → the epoch count is deterministic
    history = trainer.train(train_loader, val_loader, n_epochs=5)
    assert len(history["train_loss"]) == 5

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
from src.shared.pipeline import _a3_capturable, _maybe_force_dropout_zero
from src.shared.training import (
    MultiHeadHistoryTrainer,
    MultiHeadHistoryWithOppTrainer,
    MultiHeadNestedHistoryTrainer,
    MultiHeadTrainer,
    MultiTargetLoss,
    _GPUResidentBatcher,
    _GraphedFullStep,
    _optimizer_is_fused_capturable,
    _restore_batchnorm_state,
    _snapshot_batchnorm_state,
    make_history_dataloaders,
)
from src.shared.utils import (
    cuda_graph_enabled,
    cuda_graph_full_enabled,
    cuda_graph_opt_enabled,
)

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
# cuda_graph_full_enabled() — promoted 2026-06-15 to autodetect-ON sm_80+
# (FF_CUDA_GRAPH_FULL = force-off override), mirroring cuda_graph_enabled()
# ---------------------------------------------------------------------------
def _sm89(monkeypatch):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))


@pytest.mark.unit
@pytest.mark.parametrize("env", ["unset", "", "1", "true"])
def test_cuda_graph_full_autodetect_on_sm80plus(monkeypatch, env):
    """New default: sm_80+ with FF_CUDA_GRAPH_FULL unset/empty/truthy → full-step ON."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    if env == "unset":
        monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    else:
        monkeypatch.setenv("FF_CUDA_GRAPH_FULL", env)
    _sm89(monkeypatch)
    assert cuda_graph_full_enabled() is True


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["0", "false", "no", "off"])
def test_cuda_graph_full_force_off_override(monkeypatch, flag):
    """A falsy FF_CUDA_GRAPH_FULL force-disables full-step (model-only still on)."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", flag)
    _sm89(monkeypatch)
    assert cuda_graph_full_enabled() is False
    assert cuda_graph_enabled() is True  # base capture unaffected by the full force-off


@pytest.mark.unit
def test_cuda_graph_full_requires_base_gate(monkeypatch):
    """Full-step is a superset of the base gate: FF_CUDA_GRAPH=0 disables BOTH,
    and CPU/CI is always off regardless of FF_CUDA_GRAPH_FULL."""
    monkeypatch.setenv("FF_CUDA_GRAPH", "0")
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    _sm89(monkeypatch)
    assert cuda_graph_full_enabled() is False  # base force-off cascades
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "1")
    assert cuda_graph_full_enabled() is False  # no CUDA → off


@pytest.mark.unit
def test_cuda_graph_full_noop_on_t4_sm75(monkeypatch):
    """T4 (sm_75) can't capture, so full-step is off even with the default on."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert cuda_graph_full_enabled() is False


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
    """Force-off override → _maybe_graph_model leaves self.model untouched (same object).

    Post-#889, ``FF_CUDA_GRAPH`` is a force-off override over the sm_80+ autodetect,
    not the trigger — so "disabled" is ``=0``, not unset (unset autodetects ON on a
    local sm_80+ box and would attempt capture on this CPU trainer).
    """
    monkeypatch.setenv("FF_CUDA_GRAPH", "0")
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
def test_base_trainer_graph_noop_when_device_cpu_even_if_enabled(monkeypatch):
    """A CPU-device trainer must NOT capture even when the host autodetects graphs ON.

    ``cuda_graph_enabled()`` reports *host* capability (sm_80+); the trainer's own
    ``self.device`` gates capture. Pre-fix the base trainer entered the guard on a
    GPU host regardless of device and called ``make_graphed_callables`` on a CPU
    model → NaN MAE / non-decreasing loss. Regression guard for the FF_CUDA_GRAPH
    "K NN test flake" (#690). With the bug present this raises (it would reach
    ``next(iter(object()))``); the device gate must short-circuit first.
    """
    monkeypatch.setenv("FF_CUDA_GRAPH", "1")
    monkeypatch.setattr("src.shared.training.cuda_graph_enabled", lambda: True)
    model = nn.Linear(3, 2)
    tr = _bare_trainer(MultiHeadTrainer, model, targets=["y"])  # device=cpu
    tr._maybe_graph_model(train_loader=object())  # must return before touching the loader
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


# ---------------------------------------------------------------------------
# cuda_graph_opt_enabled() — Lever A3 optimizer-tail capture gate
# (autodetect-ON sm_80+; FF_CUDA_GRAPH_OPT force-off; A3 ⊆ A2 ⊆ base gate)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("env", ["unset", "", "1", "true"])
def test_cuda_graph_opt_autodetect_on_sm80plus(monkeypatch, env):
    """sm_80+ with FF_CUDA_GRAPH_OPT unset/empty/truthy → optimizer-tail ON."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    if env == "unset":
        monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    else:
        monkeypatch.setenv("FF_CUDA_GRAPH_OPT", env)
    _sm89(monkeypatch)
    assert cuda_graph_opt_enabled() is True


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["0", "false", "no", "off"])
def test_cuda_graph_opt_force_off_override(monkeypatch, flag):
    """A falsy FF_CUDA_GRAPH_OPT force-disables A3; A2 + base stay on."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.setenv("FF_CUDA_GRAPH_OPT", flag)
    _sm89(monkeypatch)
    assert cuda_graph_opt_enabled() is False
    assert cuda_graph_full_enabled() is True  # A2 unaffected by the A3 force-off
    assert cuda_graph_enabled() is True


@pytest.mark.unit
def test_cuda_graph_opt_cascades_off_with_base(monkeypatch):
    """FF_CUDA_GRAPH=0 or FF_CUDA_GRAPH_FULL=0 cascades A3 off (A3 ⊆ A2 ⊆ base)."""
    monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    _sm89(monkeypatch)
    monkeypatch.setenv("FF_CUDA_GRAPH", "0")
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    assert cuda_graph_opt_enabled() is False  # base force-off cascades through A2
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", "0")
    assert cuda_graph_opt_enabled() is False  # full force-off cascades to A3


@pytest.mark.unit
def test_cuda_graph_opt_off_without_cuda(monkeypatch):
    """CPU/CI → A3 always off, even with FF_CUDA_GRAPH_OPT forced on."""
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.setenv("FF_CUDA_GRAPH_OPT", "1")
    assert cuda_graph_opt_enabled() is False


@pytest.mark.unit
def test_cuda_graph_opt_noop_on_t4_sm75(monkeypatch):
    """T4 (sm_75) can't capture → A3 off even with the default on."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    monkeypatch.delenv("FF_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    assert cuda_graph_opt_enabled() is False


# ---------------------------------------------------------------------------
# _a3_capturable() — AdamW capturable-flag wiring in _run_nn_training
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a3_capturable_true_when_fused_cosine_and_gate_on(monkeypatch):
    """fused + per-epoch scheduler + A3 gate on → capturable=True."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    _sm89(monkeypatch)
    assert _a3_capturable(fused=True, per_batch=False) is True


@pytest.mark.unit
def test_a3_capturable_false_for_onecycle(monkeypatch):
    """A per-batch scheduler (OneCycleLR) → capturable=False even when gated on."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    _sm89(monkeypatch)
    assert _a3_capturable(fused=True, per_batch=True) is False


@pytest.mark.unit
def test_a3_capturable_false_when_not_fused(monkeypatch):
    """Non-fused (CPU/MPS) → capturable=False regardless of the gate."""
    _sm89(monkeypatch)
    assert _a3_capturable(fused=False, per_batch=False) is False


@pytest.mark.unit
def test_a3_capturable_false_when_gate_off(monkeypatch):
    """FF_CUDA_GRAPH_OPT=0 → capturable=False even fused + cosine."""
    monkeypatch.setenv("FF_CUDA_GRAPH_OPT", "0")
    _sm89(monkeypatch)
    assert _a3_capturable(fused=True, per_batch=False) is False


@pytest.mark.unit
def test_a3_capturable_false_on_cpu(monkeypatch):
    """CPU/CI box: even fused-cosine resolves False (no CUDA)."""
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.setenv("FF_CUDA_GRAPH_OPT", "1")
    assert _a3_capturable(fused=True, per_batch=False) is False


# ---------------------------------------------------------------------------
# _optimizer_is_fused_capturable() — trainer-side defensive precondition
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_optimizer_is_fused_capturable():
    class _Opt:
        def __init__(self, groups):
            self.param_groups = groups

    assert _optimizer_is_fused_capturable(_Opt([{"fused": True, "capturable": True}])) is True
    assert _optimizer_is_fused_capturable(_Opt([{"fused": True, "capturable": False}])) is False
    assert _optimizer_is_fused_capturable(_Opt([{"fused": False, "capturable": True}])) is False
    assert _optimizer_is_fused_capturable(_Opt([])) is False
    assert _optimizer_is_fused_capturable(_Opt(None)) is False


# ---------------------------------------------------------------------------
# _maybe_graph_full_opt() — gate (onecycle exclusion + capture-fail fallback)
# ---------------------------------------------------------------------------
def _fake_cuda_resident_trainer(monkeypatch, *, scheduler_per_batch):
    """A bare trainer with a CUDA-TYPED device, A2 engaged, a CPU-resident train
    loader, and a fused+capturable mock optimizer — enough to exercise the A3
    gate logic without a real GPU. ``_maybe_graph_full_opt`` reads only
    ``self.device.type`` (string) before the build, so a ``torch.device("cuda")``
    constructed on a CPU box is safe up to (not including) capture."""
    monkeypatch.delenv("FF_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_FULL", raising=False)
    monkeypatch.delenv("FF_CUDA_GRAPH_OPT", raising=False)
    monkeypatch.setattr("src.shared.training.cuda_graph_opt_enabled", lambda: True)

    model = nn.Linear(3, 2)

    class _Opt:
        param_groups = [{"fused": True, "capturable": True, "lr": 1e-3}]
        state: dict = {}

        def step(self):  # pragma: no cover - never called in gate tests
            pass

    tr = MultiHeadTrainer(
        model=model,
        optimizer=_Opt(),
        scheduler=None,
        criterion=MultiTargetLoss(target_names=["y"], loss_weights={"y": 1.0}),
        device=torch.device("cuda"),
        target_names=["y"],
        scheduler_per_batch=scheduler_per_batch,
        use_amp=False,
    )
    tr._graphed_step = object()  # A2 "engaged" sentinel
    feats = (torch.randn(16, 3),)
    y = {"y": torch.randn(16)}
    train_loader = _GPUResidentBatcher(feats, y, batch_size=8, shuffle=True, drop_last=True)
    return tr, train_loader


@pytest.mark.unit
def test_maybe_graph_full_opt_excludes_onecycle(monkeypatch):
    """A per-batch (OneCycle) schedule → A3 returns False WITHOUT capturing.

    The build is monkeypatched to raise so any attempt to capture would fail the
    test loudly; the gate must short-circuit before reaching it.
    """
    monkeypatch.setattr(
        "src.shared.training._GraphedFullStep.build",
        lambda self: (_ for _ in ()).throw(AssertionError("must not build for onecycle")),
    )
    tr, train_loader = _fake_cuda_resident_trainer(monkeypatch, scheduler_per_batch=True)
    assert tr._maybe_graph_full_opt(train_loader) is False
    assert tr._graphed_opt is None
    assert tr._graphed_opt_failed is False


@pytest.mark.unit
def test_maybe_graph_full_opt_fallback_on_capture_failure(monkeypatch):
    """A capture failure latches _graphed_opt_failed and returns False (never raises)."""
    monkeypatch.setattr(
        "src.shared.training._GraphedFullStep.build",
        lambda self: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    tr, train_loader = _fake_cuda_resident_trainer(monkeypatch, scheduler_per_batch=False)
    assert tr._maybe_graph_full_opt(train_loader) is False
    assert tr._graphed_opt is None
    assert tr._graphed_opt_failed is True


@pytest.mark.unit
def test_maybe_graph_full_opt_off_when_a2_not_engaged(monkeypatch):
    """No A2 graph (``_graphed_step`` None) → A3 can't engage."""
    tr, train_loader = _fake_cuda_resident_trainer(monkeypatch, scheduler_per_batch=False)
    tr._graphed_step = None
    assert tr._maybe_graph_full_opt(train_loader) is False


@pytest.mark.unit
def test_maybe_graph_full_opt_off_when_gate_disabled(monkeypatch):
    """FF_CUDA_GRAPH_OPT force-off → A3 gate returns False even with A2 engaged."""
    tr, train_loader = _fake_cuda_resident_trainer(monkeypatch, scheduler_per_batch=False)
    monkeypatch.setattr("src.shared.training.cuda_graph_opt_enabled", lambda: False)
    assert tr._maybe_graph_full_opt(train_loader) is False


@pytest.mark.unit
def test_nested_k_trainer_full_opt_noop(monkeypatch):
    """K's nested trainer overrides _maybe_graph_full_opt to a hard no-op."""
    monkeypatch.setattr("src.shared.training.cuda_graph_opt_enabled", lambda: True)
    tr = _bare_trainer(MultiHeadNestedHistoryTrainer, nn.Linear(3, 2), targets=["y"])
    tr._graphed_step = object()
    assert tr._maybe_graph_full_opt(train_loader=object()) is False


# ---------------------------------------------------------------------------
# A3 LR refresh — baked LR tensor identity stable across scheduler.step()
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a3_refresh_lr_keeps_baked_tensor_identity_and_updates_value():
    """refresh_lr_from_scheduler writes the scheduler's new LR INTO the baked
    device tensor in place (same object) and restores the param_group binding,
    so the captured graph keeps reading the live LR each epoch.

    CPU-only: exercises the pure book-keeping (no GPU/graph needed). The baked
    tensor stands in for capturable AdamW's device LR tensor.
    """
    baked = torch.tensor(1e-3)
    pg = {"lr": baked}

    class _Opt:
        param_groups = [pg]

    gfs = _GraphedFullStep.__new__(_GraphedFullStep)  # bypass build/CUDA
    gfs.optimizer = _Opt()
    gfs._baked_lr_tensors = [baked]

    # A scheduler.step() rebound lr to a NEW float object (the stale-LR hazard).
    pg["lr"] = 5e-4
    gfs.refresh_lr_from_scheduler()

    assert pg["lr"] is baked  # binding restored to the baked tensor
    assert float(baked) == pytest.approx(5e-4)  # value updated in place

    # A second epoch's step updates the SAME tensor object again.
    pg["lr"] = 2.5e-4
    gfs.refresh_lr_from_scheduler()
    assert pg["lr"] is baked
    assert float(baked) == pytest.approx(2.5e-4)


# ---------------------------------------------------------------------------
# A3 in-place grad zero ≡ set_to_none — fixed-grad CPU micro-test
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_a3_inplace_grad_zero_equals_set_to_none():
    """``p.grad.zero_()`` (A3's _run_body, stable addresses) vs
    ``zero_grad(set_to_none=True)`` (eager tail) yield bit-identical params over
    N steps when backward OVERWRITES the grads each step (the autograd default).

    Two identical models trained with the two zeroing strategies on the same
    fixed batch must end byte-for-byte equal — proving the A3 zeroing choice is
    numerically inert.
    """
    torch.manual_seed(0)

    def _make():
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, opt

    x = torch.randn(8, 4)
    t = torch.randn(8, 1)

    m_a, opt_a = _make()  # set_to_none=True path
    m_b, opt_b = _make()  # in-place zero_() path
    params_b = list(m_b.parameters())

    for _ in range(15):
        opt_a.zero_grad(set_to_none=True)
        loss_a = ((m_a(x) - t) ** 2).mean()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(m_a.parameters(), 1.0)
        opt_a.step()

        for p in params_b:
            if p.grad is not None:
                p.grad.zero_()
        loss_b = ((m_b(x) - t) ** 2).mean()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(params_b, 1.0)
        opt_b.step()

    for pa, pb in zip(m_a.parameters(), m_b.parameters(), strict=True):
        assert torch.equal(pa, pb)

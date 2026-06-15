"""Unit tests for the stacked scheduler-type A/B spec (src/tuning/ab_scheduler_type.py).

Coverage: spec resolution (``plateau`` DROPPED — ``train_stacked`` rejects ``ReduceLROnPlateau``),
the ``attn_``-prefixed scheduler mutators (set the active type's params + clear the other type's),
and the load-bearing "scheduler actually changed" guard: because Ridge is scheduler-free the Ridge
sentinel is BLIND to a no-op scheduler mutator, so we prove each arm resolves to a *different*
scheduler via the same ``attn_``-prefixed ``_build_scheduler`` the stacked trainer uses.
"""

from __future__ import annotations

import pytest
import torch

from src.tuning import ab_harness as H
from src.tuning import ab_scheduler_type as S

pytestmark = pytest.mark.unit


def test_scheduler_type_spec_resolves_without_plateau():
    spec = H.resolve_spec("src.tuning.ab_scheduler_type")
    assert spec.dotted == "src.tuning.ab_scheduler_type"
    assert spec.positions == ["RB"]
    assert "plateau" not in spec.variants  # train_stacked rejects ReduceLROnPlateau — dropped
    assert set(spec.variants) == {"cosine", "onecycle"}
    assert spec.baseline == "cosine"
    for name in ("cosine", "onecycle"):
        v = spec.variants[name]
        assert v.cfg_mutator is not None
        assert v.frame_injector is None
        assert v.expect_ridge_identical is True  # scheduler is NN-only → Ridge byte-identical


def test_mutators_set_attn_prefixed_type_and_clear_other():
    """Each mutator pins the active type's ``attn_``-prefixed params and removes the OTHER type's
    stale ``attn_`` keys, so ``_build_scheduler(scheduler_prefix="attn_")`` reads an unambiguous
    config. onecycle peaks at 4*attn_lr (TE's production ratio), NOT 4*nn_lr."""
    base = {"nn_lr": 5e-4, "attn_lr": 1e-3, "attn_cosine_eta_min": 9.9, "attn_onecycle_max_lr": 9.9}

    oc = S._mut_onecycle(dict(base))
    assert oc["attn_scheduler_type"] == "onecycle"
    assert oc["attn_onecycle_max_lr"] == pytest.approx(4e-3)  # 4 * attn_lr, NOT 4 * nn_lr (=2e-3)
    assert oc["attn_onecycle_pct_start"] == pytest.approx(0.3)
    assert "attn_cosine_eta_min" not in oc  # stale cosine key cleared

    cos = S._mut_cosine(dict(base))
    assert cos["attn_scheduler_type"] == "cosine_warm_restarts"
    assert (cos["attn_cosine_t0"], cos["attn_cosine_t_mult"]) == (40, 2)
    assert cos["attn_cosine_eta_min"] == pytest.approx(1e-5)
    assert "attn_onecycle_max_lr" not in cos  # stale onecycle key cleared


def test_each_arm_resolves_to_a_distinct_scheduler_class():
    """The Ridge sentinel can't see a no-op scheduler mutator (Ridge is scheduler-free) — so prove
    the arms differ where it counts: the ``attn_``-prefixed ``_build_scheduler`` the stacked
    trainer calls. A production-pinned ``attn_scheduler_type`` would otherwise make an unprefixed
    swap a silent no-op, which this guards against."""
    from src.shared.pipeline import _build_scheduler

    base = {"nn_lr": 1e-3, "attn_lr": 1e-3, "nn_epochs": 10}
    loader = [0, 0, 0]  # only len() is read (onecycle steps_per_epoch)

    def _sched(cfg):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        sched, _per_batch = _build_scheduler(opt, cfg, loader, scheduler_prefix="attn_")
        return sched

    assert isinstance(_sched(S._mut_onecycle(dict(base))), torch.optim.lr_scheduler.OneCycleLR)
    assert isinstance(
        _sched(S._mut_cosine(dict(base))),
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    )

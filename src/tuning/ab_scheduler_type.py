"""A/B spec: which LR-scheduler TYPE is best for the attention NN — stacked-seed (N=24) form.

The GPU-default N=24 stacked-seed port of [ablate_scheduler_type.py](ablate_scheduler_type.py).
It compares the two stack-compatible scheduler types head-to-head for the attention NN under the
stacked ensemble regime.

Scope vs the eager script (which stays for the cross-position rollup and the dropped arm):

* ``plateau`` (``ReduceLROnPlateau``) is **DROPPED**. It steps on a per-epoch val metric, which
  the stacked trainer's per-batch loop cannot reproduce — ``train_stacked`` raises on it. Run
  plateau eager via ``ablate_scheduler_type.py`` if needed.
* **Symmetric CANONICAL params for both arms.** The legacy script gave the position's production
  type its tuned settings; here both arms use canonical settings, so an onecycle-vs-cosine win is
  read straight off the Δ with no tuned home-field advantage.
* The stacked attention trainer builds its scheduler via
  ``_build_scheduler(cfg, scheduler_prefix="attn_")``, which **prefers the ``attn_``-prefixed
  keys**. Production pins ``attn_cosine_eta_min`` / ``attn_onecycle_max_lr``, so unprefixed keys
  would NOT override it — each mutator therefore sets ``attn_scheduler_type`` + the type's
  ``attn_``-prefixed canonical params and clears the other type's stale ``attn_`` keys.
* **Scheduler-only ⇒ ``expect_ridge_identical=True``.** NOTE: Ridge is scheduler-free, so the
  Ridge sentinel is BLIND to a no-op scheduler mutator — the "scheduler actually changed" guard
  lives in [test_ab_scheduler_type.py](../../tests/tuning/test_ab_scheduler_type.py), not the
  sentinel.

Run::

    python -m src.tuning.ab_scheduler_type --list
    python -m src.tuning.launch_ab --spec src.tuning.ab_scheduler_type   # GPU fleet, N=24 stacked
    python -m src.tuning.ab_scheduler_type --positions QB RB WR TE        # skill-position sweep
"""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["RB"]  # cosine in production; --positions to sweep others (K/DST fall back to eager).
# No SEEDS — inherit the GPU-default N=24 stacked width (lean 3-seed on CPU/CI). See ab_attn_arch.

# attn_-prefixed scheduler param keys (mirrors ablate_scheduler_type._TYPE_KEYS, prefixed). The
# full set is cleared before the active type's canonical params are set, so a type swap leaves no
# stale cross-type key for _build_scheduler(scheduler_prefix="attn_") to read.
_ATTN_TYPE_PARAM_KEYS = (
    "attn_cosine_t0",
    "attn_cosine_t_mult",
    "attn_cosine_eta_min",
    "attn_onecycle_max_lr",
    "attn_onecycle_pct_start",
)


def _canonical_attn_params(cfg: dict, sched_type: str) -> dict:
    """Canonical ``attn_``-prefixed params for ``sched_type`` (deliberately NOT per-position-tuned).

    ``onecycle`` peaks at 4x the ATTENTION lr — TE's production ratio is
    ``attn_onecycle_max_lr=4e-3 = 4*attn_lr`` (``attn_lr=1e-3``); ``cosine_warm_restarts`` uses
    the QB/RB/WR family standard (``T_0=40``, ``T_mult=2``, ``eta_min=1e-5``).
    """
    attn_lr = float(cfg.get("attn_lr", cfg.get("nn_lr", 1e-3)))
    if sched_type == "cosine_warm_restarts":
        return {"attn_cosine_t0": 40, "attn_cosine_t_mult": 2, "attn_cosine_eta_min": 1e-5}
    if sched_type == "onecycle":
        return {"attn_onecycle_max_lr": 4.0 * attn_lr, "attn_onecycle_pct_start": 0.3}
    raise ValueError(f"unsupported stacked scheduler type: {sched_type!r}")


def _set_attn_scheduler(cfg: dict, sched_type: str) -> dict:
    """Force the ATTENTION scheduler to ``sched_type`` with canonical ``attn_``-prefixed params.

    Clears every ``attn_``-prefixed type param first (no stale cross-type key survives the swap),
    then sets ``attn_scheduler_type`` — which ``_build_scheduler`` reads with priority over the
    production-pinned unprefixed ``scheduler_type``, so the arm is never a silent no-op.
    """
    for k in _ATTN_TYPE_PARAM_KEYS:
        cfg.pop(k, None)
    cfg["attn_scheduler_type"] = sched_type
    cfg.update(_canonical_attn_params(cfg, sched_type))
    return cfg


def _mut_cosine(cfg: dict) -> dict:
    return _set_attn_scheduler(cfg, "cosine_warm_restarts")


def _mut_onecycle(cfg: dict) -> dict:
    return _set_attn_scheduler(cfg, "onecycle")


VARIANTS = [
    Variant(
        "cosine",
        cfg_mutator=_mut_cosine,
        expect_ridge_identical=True,  # scheduler is NN-only; Ridge must stay byte-identical
        label="cosine_warm_restarts (canonical, attn)",
    ),
    Variant(
        "onecycle",
        cfg_mutator=_mut_onecycle,
        expect_ridge_identical=True,
        label="onecycle (canonical, attn)",
    ),
]
BASELINE = "cosine"  # both arms carry a mutator → declare the reference arm explicitly


if __name__ == "__main__":
    ab_main(__spec__.name)

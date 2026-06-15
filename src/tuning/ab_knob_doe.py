"""A/B: stacked Plackett-Burman main-effects screen over the attention knobs.

This is the issue #720 ``doe`` knob screen re-homed onto the shared ``ab_harness``
so it inherits the GPU-gated **vmap seed-ensemble** stacked path (#1150/#1165 —
the measured ~14x/seed win, already validated) instead of running eager,
CPU-forced, per-seed like ``attn_knob_experiments.py``. Same eight knobs, same
12-run Plackett-Burman design, same high-minus-low main-effects estimator —
only the execution backend changes (eager → stacked), which is exactly the
"wire the speedup into the knob experiments" goal. We reuse ``ab_harness``'s
validated stacked path rather than hand-rolling a second vmap loop.

Each variant is one PB row: ``+1`` sets the knob to its high bound, ``-1`` to its
low bound (``attn_knob_experiments.doe_overrides``); ``baseline`` is the
identity (production knobs). The knobs are attention-only, so Ridge MUST stay
byte-identical across every arm: ``expect_ridge_identical=True`` turns the
harness sentinel into a hard guard that a knob change didn't leak into the
shared data path.

This runs the **full pipeline** (it does NOT disable the non-attention models).
The stacked harness's Phase-A run (``run_group_stacked``, attention OFF) must
reach the pipeline's normal return that carries ``test_df``; disabling the
non-attn models drops Phase-A into an early-return path that omits ``test_df``
and ``KeyError``s every cell — the #1172 stacked-validation failure. (The
``attn_knob_experiments`` eager path can disable them because it reads
``attn_nn_metrics`` directly and never goes through the Phase-A ``test_df``.)

Judge on the attention NN's ``test_df`` fantasy-point MAE (the harness default
metric). For per-knob main effects across the 12 runs, post-process the
per-variant table with ``knob_main_effects`` (reuses #720's
``estimate_doe_effects`` verbatim).

Caveat inherited from #720: a 2-level corner screen confounds main effects with
interactions and treats monotone knobs (e.g. ``attn_patience`` under a min-loss
objective) as suspiciously "important" — read it as a screen, then confirm
survivors with a focused run, never as a config-selection objective on its own.

Run (local authoring smoke only — the real screen runs on the GPU Batch fleet)::

    python -m src.tuning.ab_knob_doe --list                       # show the grid, run nothing
    python -m src.tuning.launch_ab --spec src.tuning.ab_knob_doe  # GPU fleet (RB default, stacked)
    python -m src.tuning.ab_knob_doe --positions RB --no-stacked-seeds   # force eager A/B
"""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.attn_knob_experiments import (
    ATTN_KNOBS,
    KNOB_NAMES,
    doe_overrides,
    estimate_doe_effects,
    plackett_burman_design,
)

POSITIONS = ["RB"]  # lead; run QB/WR/TE via --positions (flat-history, stackable)

# This screen runs the FULL pipeline — it does NOT disable the non-attention
# models. The stacked harness's Phase-A run (run_group_stacked, attention OFF)
# must reach the pipeline's normal return that carries ``test_df``; disabling
# LightGBM/base-NN/ElasticNet drops Phase-A into an early-return path that omits
# ``test_df`` and KeyErrors every cell (the #1172 stacked-validation failure).
# The knobs are attention-only, so Ridge/LightGBM are constant across arms anyway.


def _make_mutator(overrides: dict):
    """Bind one PB row's knob overrides into a cfg mutator (no late-binding)."""

    def _mut(cfg, _ov=overrides):
        cfg.update(_ov)
        return cfg

    return _mut


def _build_variants() -> list[Variant]:
    """baseline (production knobs) + one variant per 12-run PB row."""
    design = plackett_burman_design(len(ATTN_KNOBS))
    # Identity baseline = production knobs + full pipeline, so the stacked
    # Phase-A reaches the test_df return. Auto-picked as the sentinel baseline.
    variants = [Variant("baseline", label="production attention knobs")]
    for run_idx, signs in enumerate(design, start=1):
        overrides = doe_overrides(signs)
        variants.append(
            Variant(
                f"pb{run_idx:02d}",
                cfg_mutator=_make_mutator(overrides),
                expect_ridge_identical=True,  # attention-only → Ridge must not move
                label=", ".join(f"{k}={overrides[k]}" for k in KNOB_NAMES),
            )
        )
    return variants


VARIANTS = _build_variants()
BASELINE = "baseline"


def knob_main_effects(
    variant_seed_mae: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per knob, via #720's ``estimate_doe_effects``.

    ``variant_seed_mae`` maps each PB variant (``pb01``..``pb12``) to its
    ``{seed: attn_test_mae}`` (extract the attention NN row from the harness
    run; the baseline is ignored — the design carries the contrast). We rebuild
    the row shape ``estimate_doe_effects`` expects and delegate, so the estimator
    stays single-sourced in ``attn_knob_experiments``.
    """
    design = plackett_burman_design(len(ATTN_KNOBS))
    rows: list[dict] = []
    for idx, signs in enumerate(design, start=1):
        name = f"pb{idx:02d}"
        sign_map = dict(zip(KNOB_NAMES, signs, strict=True))
        for seed, mae in variant_seed_mae.get(name, {}).items():
            rows.append({"seed": seed, "signs": sign_map, "attn_test_mae": mae})
    return estimate_doe_effects(rows)


if __name__ == "__main__":
    ab_main(__spec__.name)

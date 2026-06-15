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
low bound (``attn_knob_experiments.doe_overrides``). All variants — including
``baseline`` — also disable the non-attention models (LightGBM / base NN /
ElasticNet), keeping only Ridge (for the data-identity sentinel) and the
attention NN, mirroring ``attn_knob_experiments._make_cfg``. The knobs are
attention-only, so Ridge MUST stay byte-identical across every arm:
``expect_ridge_identical=True`` turns the harness sentinel into a hard guard
that a knob change didn't leak into the shared data path.

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

# Train only what the screen reads: the attention NN, plus Ridge as the
# data-identity sentinel. Mirrors attn_knob_experiments._make_cfg.
_DISABLE_NON_ATTN = {
    "train_lightgbm": False,
    "train_base_nn": False,
    "train_elasticnet": False,
    "train_ridge": True,
}


def _apply(cfg: dict, overrides: dict) -> dict:
    cfg.update(_DISABLE_NON_ATTN)
    cfg.update(overrides)
    return cfg


def _make_mutator(overrides: dict):
    """Bind one PB row's knob overrides into a cfg mutator (no late-binding)."""

    def _mut(cfg, _ov=overrides):
        return _apply(cfg, _ov)

    return _mut


def _build_variants() -> list[Variant]:
    """baseline (production knobs) + one variant per 12-run PB row."""
    design = plackett_burman_design(len(ATTN_KNOBS))
    # baseline keeps production knob values but matches the model-disabling so
    # its attention-NN MAE is comparable; declared BASELINE since the mutator
    # makes it non-identity-shaped.
    variants = [
        Variant(
            "baseline",
            cfg_mutator=_make_mutator({}),
            expect_ridge_identical=True,
            label="production attention knobs",
        )
    ]
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

"""A/B: bounded-flag scaling for the injury-report ordinal codes.

``game_status`` / ``practice_status`` are fixed-domain ordinal codes, but the
NN path z-scores them like continuous features. Out/Doubtful players
self-eliminate (preprocessing drops no-play rows), so ``game_status`` is ~96%
constant at 1.0 and its train std collapses to ~0.099 — every Questionable row
lands at z=-4.83 and gets truncated to the -4.0 ``FEATURE_CLIP`` floor, where it
is indistinguishable from Doubtful and Out. Measured on the 2026-06-11 splits
the clip fires on 4.0% (RB) / 5.3% (WR) / 4.1% (TE) / 2.2% (QB) of rows.
LightGBM is scale-invariant and never sees this, which is why the two model
families disagree so sharply on questionable players.

``nn_bounded_flag_range`` replaces the fitted scaler stats for those two columns
so their semantic domain maps onto [-range, +range]
(``src/shared/feature_build.py::BOUNDED_FLAG_DOMAINS``). NN path only — Ridge
carries its own scaler inside ``RidgeMultiTarget`` — so the deterministic Ridge
fit must stay bit-identical (``expect_ridge_identical=True``).

**Why two treatment arms, not one.** Unit variance and a clip-free range are
mutually exclusive for a rare binary: forcing unit variance on a 4%-prevalence
column necessarily puts the minority value ~4.8 sigma out. So a single arm
confounds two changes — un-clipping the tiers, and altering the feature's
magnitude. The arms separate them:

  range=1.0  bounded and conservative; post-scale std ~0.15-0.23, i.e. ~5x
             quieter than a genuinely standardized feature.
  range=4.0  domain mapped onto exactly +/-FEATURE_CLIP — the largest provably
             clip-free magnitude; post-scale std ~0.58-0.90 (near unit).

If only ``range=4.0`` wins, the clip was the problem. If ``range=1.0`` also (or
only) wins, the magnitude was. If both are flat, the encoding is not the lever.

**Read the questionable-cohort rows, not GLOBAL.** Questionable is ~3-5% of
test rows, so the global MAE delta is near-unmeasurable by construction; the
same trap the ``ablate_injury_features`` docstring calls out. The decision
metric is the questionable cohort, and per AGENTS.md ("subgroup error = bias,
not MAE") the sign that matters is the *bias* — the printed table shows MAE, so
pull ``bias`` out of the per-cell JSON for the call.

Local (CPU/MPS). Pass ``--no-stacked-seeds`` on a CUDA box: stacked mode runs
Phase A once at ``seeds[0]`` and repeats it, so every non-attention row —
including ``NN @questionable`` and the Ridge sentinel — would be a single seed
reported as an N-seed mean with std=0.000, and this knob moves the base NN too::

    python -m src.tuning.ab_flag_scaling --list
    python -m src.tuning.ab_flag_scaling --positions WR --seeds 42 --no-stacked-seeds

Batch GPU fleet — the production metric path (L4/sm_89, FP32+TF32, graphs on),
one Spot job per position, eager (ADR-0020)::

    python -m src.tuning.launch_ab --spec src.tuning.ab_flag_scaling --positions WR --seeds 42
    python -m src.tuning.launch_ab --spec src.tuning.ab_flag_scaling
"""

from __future__ import annotations

import pandas as pd

from src.features.engineer import get_attn_static_columns
from src.shared.feature_build import BOUNDED_FLAG_DOMAINS
from src.tuning.ab_harness import Variant, ab_main

# K/DST carry neither flag in their explicit attention whitelists, so they have
# no arm to measure — the four skill positions get game_status/practice_status
# through the shared ``contextual`` INCLUDE_FEATURES category.
POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42, 123, 7]

# The two treatment magnitudes (see the module docstring).
BOUNDED_RANGES = {"bounded_r1": 1.0, "bounded_r4": 4.0}


def _make_mutator(target_range: float):
    """Set the knob, and fail loudly if it would be a silent no-op here.

    An NN-only arm that mutates a config nothing reads still passes the
    Ridge-identical sentinel — Ridge never sees NN config either way — so a
    no-op would be reported as a genuine "no effect" result. Assert the
    precondition per *scaled path*, not on the union of the whitelists: the
    attention trainer scales only ``get_attn_static_columns(feature_cols,
    attn_static_features)``, so a flag present in the base whitelist but absent
    from the attention one would leave the attention arm untouched while a
    union check still passed.
    """

    def mutator(cfg):
        flags = set(BOUNDED_FLAG_DOMAINS)
        base_cols = list(cfg["get_feature_columns_fn"]())
        missing = []
        if cfg.get("train_base_nn", True) and not flags & set(base_cols):
            missing.append("base NN (get_feature_columns_fn)")
        if cfg.get("train_attention_nn", False):
            attn_cols = get_attn_static_columns(base_cols, cfg["attn_static_features"])
            if not flags & set(attn_cols):
                missing.append("attention static branch (attn_static_features)")
        if missing:
            raise AssertionError(
                f"nn_bounded_flag_range would be a silent no-op on: {', '.join(missing)} "
                f"— none of {sorted(flags)} reaches those scaled paths."
            )
        cfg["nn_bounded_flag_range"] = target_range
        return cfg

    return mutator


def metric_fn(result: dict, position: str) -> dict[str, dict[str, float]]:
    """Per-model metrics on GLOBAL plus the healthy / questionable cohorts.

    Cohort rows are emitted as ``"<model> @questionable"`` pseudo-models so the
    harness aggregates and Δ's them exactly like a normal row. The plain
    ``"Ridge"`` key is preserved verbatim — the Ridge-invariance sentinel reads
    ``metrics["Ridge"]["mae"]``.

    Cohort masks come from ``cohort_analysis.SUBGROUP_SPECS`` (looked up by key,
    so this does not depend on that list's order) rather than being restated
    here — otherwise a change to the canonical questionable threshold would
    silently desynchronize this A/B from the repo's injury-cohort reporting.
    """
    from src.analysis.cohort_analysis import SUBGROUP_SPECS
    from src.evaluation.metrics import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    out: dict[str, dict[str, float]] = dict(per_model_metrics(df, models))

    specs = {key: (needed, mask_fn) for key, _label, needed, mask_fn in SUBGROUP_SPECS}
    for key in ("questionable", "healthy"):
        needed, mask_fn = specs[key]
        if needed and needed not in df.columns:
            raise ValueError(f"{position} test_df has no {needed}; cannot measure {key}")
        cohort_frame = df
        # A non-numeric / NaN status satisfies neither `< 1.0` nor `>= 1.0`, so
        # such rows would vanish from BOTH cohorts without a word. Count them.
        if needed:
            parsed = pd.to_numeric(df[needed], errors="coerce")
            cohort_frame = df.assign(**{needed: parsed})
            out[f"{needed} coverage"] = {"n": len(df), "unparseable": int(parsed.isna().sum())}
        mask = mask_fn(cohort_frame)
        out[f"{key} coverage"] = {"n": int(mask.sum())}
        for name, metrics in per_model_metrics(cohort_frame[mask], models).items():
            out[f"{name} @{key}"] = metrics
    return out


VARIANTS = [
    Variant("baseline", label="baseline (z-scored flags)"),
    *[
        Variant(
            name,
            cfg_mutator=_make_mutator(rng),
            expect_ridge_identical=True,  # NN scaler only — Ridge has its own
            label=f"bounded flags, range={rng}",
        )
        for name, rng in BOUNDED_RANGES.items()
    ],
]


if __name__ == "__main__":
    ab_main(__spec__.name)

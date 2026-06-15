"""A/B: Plackett-Burman feature-FAMILY selection screen (skill positions).

The project has no automated feature selection — features are a hand-maintained
``_INCLUDE_FEATURES`` category whitelist per position, toggled one-at-a-time by
bespoke ablations (e.g. ``ablate_injury_features.py``). This screen makes the
**which feature families carry signal** question systematic: a 12-run
Plackett-Burman main-effects design over the feature *categories*, so every
family's main effect is estimated from all 12 runs (tighter than OFAT) and
interactions don't alias onto the wrong family.

Design vs implementation split (the deliberate reuse):
  * The **design + main-effects engine** is reused verbatim from the issue #720
    attention-knob experiments (``plackett_burman_design`` / the high-minus-low
    estimator) — this screen is the same statistics applied to feature families
    instead of architecture knobs.
  * The **execution** rides the shared ``ab_harness`` so it inherits, for free,
    (a) the GPU-gated **vmap seed-ensemble** stacked path (#1150/#1165 — the
    measured ~14x/seed speedup, already validated) and (b) the **Ridge
    data-identity sentinel**. We deliberately do NOT hand-roll a vmap loop here:
    that would duplicate ``ab_harness.run_group_stacked``'s validated path.

Each variant is one PB row that DROPS the families flagged ``-1`` (``+1`` keeps
them); ``baseline`` keeps every family (production). Dropping a family filters
its columns out of BOTH model paths — the flat linear/tree list
(``get_feature_columns_fn``) and the attention static branch
(``attn_static_features``) — mirroring ``ablate_injury_features``'s mutator.

Every PB row declares ``expect_ridge_identical=False`` — dropping real feature
columns MUST move Ridge, so a Δ=0 means the drop silently didn't take and the
harness fails the run loudly. (The first cut used report-only ``None``, which
masked exactly that: the mutator was reading the absent ``cfg["include_features"]``
and no-opping, producing a baseline-identical null result that the report-only
sentinel happily reported as "data-identical" — the #1172 stacked-validation
finding. Source family→columns from :data:`_FAMILY_COLS`, never the cfg.) Each
row drops several of the eight screened families, all populated on every skill
position, so the assertion never false-trips.

Scope: skill positions only (QB/RB/WR/TE — they carry the ``include_features``
category dict; K/DST use a flat ``all_features`` list and are skipped). The
attention NN reads only the ``DEFAULT_ATTN_STATIC_CATEGORIES`` subset
(prior_season / matchup / contextual / weather_vegas) on its static branch, so
dropping a non-static family (rolling / trend / share / defense) moves Ridge/
LightGBM only; dropping a static family moves the attention NN too. ``ewma``
(usually empty) and ``specific`` (position-defining; dropping it can starve the
model) are intentionally outside the screened set — screen those separately.

Judge per model on ``test_df`` (the harness default metric): a family whose
removal raises a model's MAE carries signal *for that model*. For per-family
main effects across the 12 runs, post-process the per-variant table with
``feature_main_effects`` (see its docstring).

Run (local authoring smoke only — the real screen runs on the GPU Batch fleet;
local training SIGSEGVs on the macOS torch+lightgbm+sklearn libomp triple-load)::

    python -m src.tuning.ab_feature_screen --list                  # show the grid, run nothing
    python -m src.tuning.launch_ab --spec src.tuning.ab_feature_screen   # GPU fleet (RB default)
    python -m src.tuning.ab_feature_screen --positions WR --no-stacked-seeds   # force eager
"""

from __future__ import annotations

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.attn_knob_experiments import plackett_burman_design

POSITIONS = ["RB"]  # lead; run QB/WR/TE via --positions (all carry include_features)

# The feature families screened, in a fixed order so the PB design is stable
# across positions and runs. ``ewma`` (usually empty) and ``specific``
# (position-defining; dropping it can starve the model) are deliberately
# excluded — screen those with a bespoke ablation if needed.
SCREENED_FAMILIES: tuple[str, ...] = (
    "rolling",
    "prior_season",
    "trend",
    "share",
    "matchup",
    "defense",
    "contextual",
    "weather_vegas",
)


# The runtime pipeline cfg does NOT carry ``include_features`` (build_pipeline_config
# flattens it into ``get_feature_columns_fn``), so we cannot read family→columns
# off the cfg — doing so silently no-ops the screen (the #1172 null-result bug).
# Precompute it from each skill position's ``PositionConfig.include_features`` (the
# source of truth) as the UNION across positions: family membership is by column
# NAME, which is position-consistent, so the union intersected with this position's
# live ``get_feature_columns_fn()`` yields exactly this position's family columns.
_FAMILY_SOURCE_POSITIONS = ("QB", "RB", "WR", "TE")


def _family_columns() -> dict[str, frozenset[str]]:
    import importlib

    cols: dict[str, set[str]] = {fam: set() for fam in SCREENED_FAMILIES}
    for pos in _FAMILY_SOURCE_POSITIONS:
        inc = importlib.import_module(f"src.{pos.lower()}.config").POSITION_CONFIG.include_features
        for fam in SCREENED_FAMILIES:
            cols[fam].update(inc.get(fam, ()))
    return {fam: frozenset(c) for fam, c in cols.items()}


_FAMILY_COLS = _family_columns()


def _drop_families(cfg: dict, families: frozenset[str]) -> dict:
    """Filter every column belonging to ``families`` out of both model paths.

    Surgical filter (the ``ablate_injury_features`` pattern): removes the dropped
    families' columns from ``get_feature_columns_fn`` (linear/tree) and
    ``attn_static_features`` (attention static branch). Family→columns comes from
    :data:`_FAMILY_COLS` (NOT ``cfg["include_features"]``, which is absent at
    runtime — the #1172 null-result bug); intersecting the union with the live
    feature list removes only this position's columns.
    """
    dropped_cols = {c for fam in families for c in _FAMILY_COLS.get(fam, ())}
    if not dropped_cols:
        return cfg
    base_get = cfg.get("get_feature_columns_fn")
    if base_get is not None:
        cfg["get_feature_columns_fn"] = lambda _b=base_get: [
            c for c in _b() if c not in dropped_cols
        ]
    static = cfg.get("attn_static_features")
    if static is not None:
        cfg["attn_static_features"] = [c for c in static if c not in dropped_cols]
    return cfg


def _make_mutator(drop: frozenset[str]):
    """Bind one PB row's dropped-family set into a cfg mutator (no late-binding)."""

    def _mut(cfg, _drop=drop):
        return _drop_families(cfg, _drop)

    return _mut


def _build_variants() -> list[Variant]:
    """baseline (keep all) + one variant per 12-run PB row (drop the -1 families)."""
    design = plackett_burman_design(len(SCREENED_FAMILIES))
    variants = [Variant("baseline", label="all feature families (production)")]
    for run_idx, signs in enumerate(design, start=1):
        drop = frozenset(
            fam for fam, sign in zip(SCREENED_FAMILIES, signs, strict=True) if sign < 0
        )
        kept = [f for f in SCREENED_FAMILIES if f not in drop]
        variants.append(
            Variant(
                f"pb{run_idx:02d}",
                cfg_mutator=_make_mutator(drop),
                # MUST move Ridge: every PB row drops >=1 populated family, so a
                # Delta=0 means the drop didn't take (#1172 null-result bug) —
                # fail loud rather than silently emit a baseline-identical row.
                expect_ridge_identical=False,
                label=f"drop={sorted(drop)} keep={kept}",
            )
        )
    return variants


VARIANTS = _build_variants()
BASELINE = "baseline"


def feature_main_effects(
    variant_seed_mae: dict[str, dict[int, float]],
) -> dict[str, dict[str, float]]:
    """High-minus-low main effect per screened family, averaged across seeds.

    ``variant_seed_mae`` maps each PB variant name (``pb01``..``pb12``) to its
    ``{seed: mae}`` for one model — extract it from the harness run for the model
    you care about (``"Ridge"`` / ``"LightGBM"`` / ``"Attn NN"``; the baseline
    row is ignored, the design carries the contrast). The estimator mirrors
    ``attn_knob_experiments.estimate_doe_effects``: per seed, mean(MAE | family
    DROPPED) - mean(MAE | family KEPT), then averaged across seeds. A POSITIVE
    effect = dropping the family RAISES MAE = the family carries signal.

    Returns ``{family: {mean_effect, std_effect, n_seeds}}``. Off the hot path
    (a handful of arithmetic ops on ~12 values), so plain ``statistics`` is fine.
    """
    import statistics

    design = plackett_burman_design(len(SCREENED_FAMILIES))
    # Reconstruct each PB variant's dropped-family set from the design.
    row_drop = {
        f"pb{idx:02d}": {
            fam for fam, sign in zip(SCREENED_FAMILIES, signs, strict=True) if sign < 0
        }
        for idx, signs in enumerate(design, start=1)
    }
    seeds = sorted({s for m in variant_seed_mae.values() for s in m})
    out: dict[str, dict[str, float]] = {}
    for fam in SCREENED_FAMILIES:
        per_seed: list[float] = []
        for seed in seeds:
            dropped = [
                variant_seed_mae[v][seed]
                for v, drop in row_drop.items()
                if fam in drop and v in variant_seed_mae and seed in variant_seed_mae[v]
            ]
            kept = [
                variant_seed_mae[v][seed]
                for v, drop in row_drop.items()
                if fam not in drop and v in variant_seed_mae and seed in variant_seed_mae[v]
            ]
            if dropped and kept:
                per_seed.append(statistics.mean(dropped) - statistics.mean(kept))
        if per_seed:
            out[fam] = {
                "mean_effect": statistics.mean(per_seed),
                "std_effect": statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                "n_seeds": len(per_seed),
            }
    return out


if __name__ == "__main__":
    ab_main(__spec__.name)

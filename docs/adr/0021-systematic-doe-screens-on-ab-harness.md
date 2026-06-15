# ADR-0021: Systematic Knob + Feature-Family Screens As Stacked A/B-Harness Specs

**Status:** Proposed
**Date:** 2026-06-15
**Supersedes:** none
**Related:** ADR-0015 (Optuna tuning), ADR-0020 (Batch A/B path), ADR-0017 (platform/AMP policy)

## Context

Two systematic-search needs were raised after the vmap seed-ensemble ("stacked")
speedup landed (#1150/#1165 — measured ~14×/seed on CUDA, N=24 the per-seed
optimum):

1. **Knob attribution.** Issue #720 already ships
   [src/tuning/attn_knob_experiments.py](../../src/tuning/attn_knob_experiments.py):
   a `fanova` mode (repeated Optuna studies over eight attention knobs, scored on
   **test FP MAE**, summarized with Optuna fANOVA importances) and a `doe` mode
   (12-run Plackett–Burman main-effects screen). It is correct but runs **eager,
   CPU-forced, per-seed** (`_force_cpu_device`, `ProcessPoolExecutor`) — it does
   **not** get the stacked speedup (#1165 explicitly deferred experiment/ablation
   stacking to a follow-up).
2. **Feature selection.** There is **no automated feature selection** — features
   are a hand-maintained `_INCLUDE_FEATURES` category whitelist, toggled
   one-at-a-time by bespoke ablations (`ablate_injury_features.py`). No
   family-level main-effects screen exists.

The shared `ab_harness` ([src/tuning/ab_harness.py](../../src/tuning/ab_harness.py))
already has a **GPU-gated stacked path** (`run_group_stacked`, default-ON on CUDA)
and the **Ridge data-identity sentinel**, both validated by #1150/#1165.

## Decision

Implement both screens as **`ab_harness` specs**, reusing #720's PB design +
main-effects estimator for the statistics and `ab_harness` for execution — so
they inherit the validated stacked path and sentinel rather than re-implementing
either.

- **[src/tuning/ab_knob_doe.py](../../src/tuning/ab_knob_doe.py)** — the #720
  `doe` knob screen re-homed onto `ab_harness`. Same `ATTN_KNOBS` / `doe_overrides`
  / `plackett_burman_design` / `estimate_doe_effects`; each PB row is a variant
  that applies the knob overrides and disables the non-attention models
  (attention-only ⇒ `expect_ridge_identical=True`). Only the execution backend
  changes (eager → stacked) — this is the "wire the speedup into the knob
  experiments" deliverable.
- **[src/tuning/ab_feature_screen.py](../../src/tuning/ab_feature_screen.py)** —
  net-new feature-FAMILY PB screen. Each PB row drops the `-1` families,
  filtering their columns out of both `get_feature_columns_fn` (linear/tree) and
  `attn_static_features` (attention static). `expect_ridge_identical` is left
  unset (report-only): a row dropping only families empty for a given position
  removes nothing, so a hard "must differ" assertion would false-trip.

Both stay under `src/tuning/` (no 6-position retrain trigger). fANOVA-with-
stacking is **not** re-implemented: it is already available via the stacked
`tune_nn` study (ADR-0015) feeding `attn_knob_experiments`'s fANOVA importances.

## Chosen vs Rejected

- **Chosen:** specs on `ab_harness` → reuse the validated stacked path + sentinel;
  reuse #720's PB engine. Lowest new-code surface; no GPU code to re-validate.
- **Rejected — hand-roll a vmap loop inside `attn_knob_experiments.py`:**
  duplicates `ab_harness.run_group_stacked`'s validated path, doubling the
  GPU-numerics surface that must be re-validated off-box.
- **Rejected — a fresh `doe.py` design + ANOVA engine:** #720 already has
  `plackett_burman_design` / `estimate_doe_effects` / `ridge_sentinel_ok`.

## Consequences

- **Within-mode only.** Stacked runs are not production-comparable (FP32/LN/
  fixed-epochs; ADR-0017); compare stacked-vs-stacked, then confirm survivors
  eager (`--no-stacked-seeds`) on the shipped metric before any config change.
- **Validation gate.** The stacked execution path is GPU-only and cannot be run
  on the dev Mac (no CUDA; local training SIGSEGVs on the libomp triple-load).
  CPU-safe logic (variant construction, mutators, main-effects estimators) is
  unit-tested ([tests/tuning/test_ab_doe_specs.py](../../tests/tuning/test_ab_doe_specs.py));
  the stacked GPU run needs off-box validation on the Batch fleet before this ADR
  moves to Accepted.
- **PB screen caveat (inherited from #720):** a 2-level corner design confounds
  main effects with interactions and flatters monotone knobs — a screen, not a
  config-selection objective.

## References

- [src/tuning/ab_knob_doe.py](../../src/tuning/ab_knob_doe.py),
  [src/tuning/ab_feature_screen.py](../../src/tuning/ab_feature_screen.py)
- [src/tuning/attn_knob_experiments.py](../../src/tuning/attn_knob_experiments.py) (#720 engine reused)
- [src/tuning/ab_harness.py](../../src/tuning/ab_harness.py) (stacked path + sentinel)

## Changelog

- 2026-06-15 · Proposed: knob DoE + feature-family screens as stacked `ab_harness` specs reusing the #720 PB engine; awaiting off-box GPU validation.

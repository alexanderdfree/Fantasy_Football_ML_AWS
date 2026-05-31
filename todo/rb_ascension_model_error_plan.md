# RB ascension — measure the literal per-model undershoot (continuation note)

**Status:** deferred (heavy local run not yet executed). This is a ready-to-run
plan for a future session. Tracked in [TODO.md](../TODO.md) under
`[NEXT] Measure literal RB-model undershoot on ascension games`.

## The question

Do the trained RB models actually **undershoot** the week a backup ascends into a
workhorse role (starter injured), now that 2025 `depth_chart_rank` is confirmed
present? The earlier "76% under-prediction" figure
([rb_ascension_findings.md](../src/analysis/rb_ascension_findings.md)) is an **input-information
bound** (realized FP vs an L3-mean volume baseline), *not* the literal error of
the trained ensemble. This task gets the real per-model number.

## Already done (context)

- #588 — shipped the ascension diagnostic `src/analysis/rb_ascension.py` (cohort,
  lag gap, convergence) + `--with-model-error` mode (not yet run on real data).
- #592 — fixed a bug where `_print_depth_coverage` pulled the raw
  `nfl_source.depth_charts` shim and dropped 2025's ESPN-schema rows; it now loads
  depth the loader's way (`_normalize_espn_depth`, #370). 2025 is ~99% RB-covered.
- #593 — corrected the stale `src/shared/feature_build.py` "2025 entirely missing"
  comment (`[docs-only]`).
- Confirmed: the cohort/lag/convergence path is **depth-independent** (weekly
  volume + fantasy_points only), so the 76% bound was never contaminated.

## What's left — run it

```
/Users/alex/miniforge3/bin/python -m src.analysis.rb_ascension --with-model-error
```

This calls `src.rb.run_pipeline.run()` — self-builds the 2025 splits from scratch
(incl. correct ESPN `depth_chart_rank`) and trains Ridge / NN / Attn-NN / LightGBM
at seed 42 — then `_print_model_error` → `label_ascension_rows` + `cohort_model_table`
print per-model **MAE + signed bias** on the `ascension` vs `established` buckets.
No new code expected; the path is unit-tested but never run on real data, so fix
localized breakage if it surfaces (likely `label_ascension_rows` column deps:
`rolling_mean_carries_L3`, `rolling_mean_targets_L3`, `carries`, `targets`; or
`cohort_model_table`: `pred_*_total`, `fantasy_points`, `role_change`). The
0-ascension-row guard already exists.

## Interpretation — honest caveats (critical)

- **Headline Ridge + LightGBM** (deterministic; LightGBM is RB production-best).
  Their ascension-bucket bias is stable and citable.
- **NN + Attn-NN are single-seed (42) + N≈14 → directional only.** Do not headline
  a precise NN/Attn-NN number or any "Attn-NN beats X" claim from one seed (see
  auto-memory `feedback_nn_seed_sensitive_overall_mae`; #596). For a robust NN
  claim, re-run ≥2 seeds and judge the *direction*.
- **Undershoot = signed bias** `mean(pred − actual)` on the ascension bucket
  (large negative = undershooting). Contrast with the `established` bucket to
  isolate the ascension-specific gap; compare to the ≈ −10.8 FP (std) input bound —
  does the present-but-weak depth signal (median rank 2 on ascension weeks) claw
  any of it back?
- **Do not** substitute a reduced-config proxy to save time — it can flip the sign
  (auto-memory `feedback_validation_proxy_must_match_production_model`).

## Risks

- **Heavy:** no local splits → full build (network pulls incl. PBP redzone, the
  bulk) + CPU training of NN/Attn-NN (torch 2.11 CPU-only on this box). ~30–60+ min
  and potentially fragile; surface progress/failures, don't silently retry.
- **N≈14** ascension events in the 2025 holdout → noisy; the 205-event input bound
  is the structural backstop. If 14 is too thin, escalate to rolling-origin
  (2023–25, ~40 events) — needs a small code addition to label + aggregate each
  fold (`ROLLING_ORIGIN_TEST_SEASONS` in `src/config.py`).

## Verification & reporting

- Run completes, prints the cohort table; sanity: ascension-bucket bias **negative**
  and N≈14; `established`-bucket bias near the global RB bias (~−0.8, per
  [rb_lgbm_disagreement_findings.md](../src/analysis/rb_lgbm_disagreement_findings.md)) as an
  anchor; Ridge/LGBM reproduce on a re-run.
- If material, add a "Literal model undershoot (2025)" subsection to
  [rb_ascension_findings.md](../src/analysis/rb_ascension_findings.md) with the measured per-model
  bias/MAE, framed against the input bound. `src/analysis/` only → no retrain; ship
  as a small PR.

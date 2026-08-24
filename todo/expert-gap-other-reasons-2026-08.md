<!-- Research answer to "consider other reasons why the experts are still beating my models"
     (2026-08-24). Read-only codebase investigation; no model/loader/serving change. Each
     hypothesis below was verified against the actual code/data reality (the "verify the
     activation precondition, not just the code smell" lesson) — file:line citations inline.
     Companion to todo/expert-gap-investigation-2026-06.md (what IS known) and
     todo/expert-edge-action-plan.md (the roadmap). [docs-only] -->

# Expert-gap — other candidate reasons (2026-08)

**Scope.** As of the 2026-07-13 refresh the experts' MAE edge is gone; what survives is
RotoWire's **RMSE (boom/bust-tail) edge at RB/WR** (Δ +0.23/+0.25, DM p ≤ 8.4e-4) and the
**rank-ordering edge at RB + WR + TE** (iso_edge +0.17/+0.16/+0.12, CIs exclude 0), with the
QB residual attributed to current-week starter news ([#1134](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/issues/1134))
and the WR/TE deep gap to player-level coverage data ([#1210](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/issues/1210),
data-blocked ≥2013). This doc catalogs candidate reasons **not already in** the
[investigation](expert-gap-investigation-2026-06.md), the [action plan](expert-edge-action-plan.md),
or the [new-sources sweep](new-sources-research-2026-06.md) — split into measurement artifacts
(the measured edge may be partly inflated), unbuilt in-house modeling levers, and framing.
Ordered by plausibility × testability within each section.

---

## A. Measurement — part of the measured edge may not be expert skill at all

### A1. Every expert head-to-head grades the season's final rest-noise week — #615 was never wired into the expert comparisons — **HIGH, trivially testable**
- **Code reality:** [src/analysis/analysis_expert_comparison.py](../src/analysis/analysis_expert_comparison.py)
  joins on all `(player_id, season, week)` overlap with **no max-week filter**; same for the
  rolling-origin study ([src/tuning/ab_rolling_origin_rotowire.py](../src/tuning/ab_rolling_origin_rotowire.py))
  and the Comparison-tab metrics path (`src/shared/evaluation.py` iterates every
  `test_df['week'].unique()` — the known [#615](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/issues/615)
  deferral, tracked in TODO.md "Late-week eval exclusion").
- **Mechanism:** [late_week_effect_findings](../src/analysis/late_week_effect_findings.md)
  already measured the final week (wk17 ≤2020 / wk18 2021+) as depressed **−10% to −24%**
  and *fat-tailed*. Rest decisions are **announced news**: experts project rested starters
  near zero; the model sees `game_status=1.0` (healthy — rest is not an injury designation)
  plus full role features and projects a normal week. Starters who play a series or two
  before sitting *do* have a stat row, so they are scored — and one such row can carry
  ~10–15 FP of error, squared, exactly where ΔRMSE / NDCG / lineup-regret live. Fully-inactive
  rested starters drop out of the joined sample (no stat row), so the surviving rows are the
  worst-case partial-rest ones.
- **Why the prior "late info" test missed it:** the investigation's "experts have late injury
  info is false" verdict sliced *injury-designation* rows; healthy rest is not on any injury
  slice.
- **Test (an afternoon):** re-run `analysis_expert_comparison` and the iso_edge /
  rolling-origin decomposition with each season's `max(week)` excluded (season-aware, per the
  findings doc — never a flat `week==18`). If RB/WR ΔRMSE or iso_edge shrinks materially,
  wire the #615 eval exclusion (owner-approved rebaseline) and re-read the surviving gap.
  Decision-relevance argues for excluding it regardless — fantasy championships end wk17.

### A2. The evaluation protocol hands experts a 1–2-season parameter-vintage advantage — **MEDIUM, structural**
- **Code reality:** `TRAIN_SEASONS = 2013–2023`, `VAL_SEASONS = [2024]` (early-stopping/tuning
  only), `TEST_SEASONS = [2025]` ([src/config.py:38-40](../src/config.py)); rolling-origin
  mirrors it (train ≤T−2, val T−1). Every measured comparison scores parameters fit through
  T−2 against humans operating with full T−1 *and in-season-T* knowledge. Features carry
  in-season form (rolling/EWMA), but coefficients, interactions, and league-era calibration
  are frozen ≥2 seasons back.
- **Diagnostic (cheap, uses the existing substrate):** slice the expert edge by week-of-season.
  An edge that *grows* over the season = in-season-adaptation gap (see B4); flat = mostly not
  this. (Week 1 is separately known — the tracked empty-history cohort, QB/K-specific per
  PR #1475.)

### A3. Ranking-metric plumbing was silently broken for a month — housekeeping
- `*_ranking` was lost from split-mode benchmark runs 2026-06-11 → 2026-07 (every split-mode
  top12 read 0; fixed in [#1486](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/1486)).
  Any ordering impression formed from `benchmark_history/` / the History tab in that window
  undercounts the model — re-verify ordering trends only on post-#1486 runs. (The expert
  studies computed their own rank metrics and are unaffected.)

### A4. The Comparison tab scores on the experts' home turf — **LOW**
- The committed top-30 slices ([src/serving/comparison.py](../src/serving/comparison.py),
  `comparison_experts.json`) derive from expert-scored pools; per-row scoring is fair (same
  players both sides), but the slice *chooser* names players it understands. One-off
  sensitivity check: recompute on a symmetric slice (top-30 by actual, or by model) and
  compare verdicts. (The reliability look-ahead in this file was already fixed — PR #1476.)

## B. Modeling structure — in-house levers nobody has tried

### B1. Nothing in the stack is trained to rank — the "non-monotonic re-ranker" the investigation itself named was never built — **HIGH**
- **Code reality:** no `lambdarank` / `LGBMRanker` / listwise objective anywhere in `src/`
  (grep). All heads are per-stat magnitude regressors (MSE / Poisson / hurdle); the only
  ordering lever shipped is head *selection* (LightGBM ranks RB/WR on the NextWeek sort,
  PR #1477). Yet the investigation's own conclusion (§1a) is that the durable edge is an
  **ordering** edge closable only by new signal (#1210) *or a non-monotonic re-ranker* —
  and iso_edge is, by construction, exactly the slice a rank-trained model could attack
  **with zero new data**.
- **Two shapes:**
  (a) **Clean under ADR-0003** — a new *independent comparison column* (TabPFN precedent):
  LightGBM with `lambdarank` objective, groups = `(season, week, position)`, same features,
  judged on NDCG / recall@k / lineup regret on the shared RotoWire-covered slate.
  (b) **Stacked re-ranker** over the four heads' predictions + a small static feature set —
  this is the blend [ADR-0003](../docs/adr/0003-three-way-model-comparison-no-ensemble.md)
  explicitly anticipates as a follow-up ("a future production system would of course blend
  these"); needs an owner call before building.
- Head errors are co-linear (ρ 0.95–0.97) so a *linear* blend recovers ≈0 (already tested);
  a rank objective and feature-conditioned non-monotonicity are the two axes that result
  does not rule out.

### B2. The per-head loss allocation is fantasy-point-blind — and the tool built to quantify it has no recorded result — **HIGH, cheapest first step**
- FP error is dominated by ×6 TD leverage (a 0.3-TD miss = 1.8 FP vs a 30-yd miss = 3 FP).
  The yards heads got tuned `1/δ` weights (PR #870), but the count heads (Poisson /
  hurdle-NegBin) sit at weight 1.0 with **nothing tying their loss share to their FP-error
  share**. [src/analysis/rmse_gap_decomposition.py](../src/analysis/rmse_gap_decomposition.py)
  (PR #1365) computes the exact per-head share of squared FP error on the served artifacts —
  **run it and record the answer** (nothing in-repo carries its conclusions). If TD/count
  heads dominate the RB/WR squared-FP gap, the lever is a hand-tuned count-head loss-weight
  ablation via the `ablate_rb_gate` decision-table pattern (per the stop-rule: never into
  `tune_nn`'s search space), or an FP-space auxiliary loss term. Complements — does not
  duplicate — #1354's per-sample weighting (that is *which rows*; this is *which heads*).

### B3. Training weights 2013 rows equal to 2023 rows — no recency weighting anywhere — **MEDIUM**
- **Code reality:** no `sample_weight` in any fit path (grep; all hits are optimizer
  `weight_decay`). The league drifted across the 11-season window (pass-rate environment,
  committee backfields, two-high-shell era); experts run a fully-current prior.
  `season_recency` exists only as the harness *demo feature* in
  [ab_example.py](../src/tuning/ab_example.py) ("intentionally low-stakes", result never
  recorded) — a feature lets the model shift by era; **loss-level down-weighting of old
  seasons is a different, untested lever**.
- A/B: exponential season-decay sample weights (LightGBM `sample_weight`; per-sample loss
  scaling for the NNs). Small pipeline change, screen via `ab_harness` cfg-mutator, judge on
  rank metrics + overall MAE non-regression.

### B4. No in-season adaptation: parameters are frozen all season while experts learn weekly — **MEDIUM, gate on A2's diagnostic**
- Distinct from the **rejected** recalibration: that was *monotone* (rank-preserving) pooled
  isotonic, which provably cannot move ordering. An in-season, feature-conditioned residual /
  re-rank layer updated on test-season weeks 1..W−1 **can** change ordering and adapts to
  the current era (also the only structural answer to A2 short of retraining in-season).
  Only worth building if A2's week-slice shows the edge growing over the season.

### B5. Cross-position teammate context is absent — a WR model cannot see that its QB is out — **MEDIUM-HIGH**
- **Code reality:** the inheritance features are same-position-only by construction —
  `_build_inheritance_features` ranks "same-team, **same-position** teammates"
  ([src/features/engineer.py:436-447](../src/features/engineer.py)), and the WR contextual
  whitelist carries no QB-context column ([src/wr/config.py:103-124](../src/wr/config.py)).
  So a backup-QB start (or an elite-QB return) is invisible to every WR/TE/RB receiving
  head, while experts propagate QB changes onto the whole receiving corps instantly. Same
  blind spot class as the E2 O-line-for-RB candidate — but *this* one is not in the
  new-sources list (B1 CPOE is QB-as-a-QB-feature; E2 is OL-for-RB).
- **2013-safe, $0, already-ingested feeds:** the QB inheritance machinery already computes
  per-QB role values from ff_opportunity exp-FP + injuries + weekly rosters — broadcast the
  team's expected-starter status/quality to its pass-catchers (`team_qb_out`,
  expected-starter prior exp-FP, `qb_change_flag`). Judge on a QB-change-week receiving
  cohort (bias + rank), never overall MAE (n is small). Extends #1102/#1104/#1106
  within-position work to cross-position.
- **Status: screen BUILT (this PR), not yet run** —
  [src/tuning/ab_qb_context_receivers.py](../src/tuning/ab_qb_context_receivers.py)
  (`baseline` / `+qb_out` / `+qb_quality` / `+both` × WR/TE/RB, qbout-cohort metric;
  unit-tested). Run `python -m src.tuning.ab_qb_context_receivers` on a GPU box (smoke one
  `--positions WR --only +qb_out --seeds 42` cell first), or the Batch fleet via
  `launch_ab --spec src.tuning.ab_qb_context_receivers` (dispatch `batch-image.yml` on the
  branch first, ADR-0020).

## C. Framing — why "still beating" may persist regardless

- **C1. The four heads are one model.** Elite-tier error correlation between our heads is
  0.95–0.97 — anything trained on the same features converges to the same mistakes. Only new
  information (news feeds, coverage data) or a new *objective* (B1) decorrelates; this is why
  architecture knobs keep landing in the tested-rejected pile.
- **C2. The live-season gap will read worse than the 2025 backtest parity.** Backtest
  features carry **closing** Vegas lines (nflverse `spread_line`/`total_line` are closing —
  [new-sources §C2](new-sources-research-2026-06.md)), a small backtest-only tailwind. Live,
  the upcoming-week artifact refreshes every 3h
  ([refresh-upcoming-week.yml](../.github/workflows/refresh-upcoming-week.yml)) — lines and
  injuries stay ~fresh, but **same-day actives/healthy scratches never reach it** (action-plan
  Phase 4). Expect the Timeline tab's live weekly head-to-head to trail the backtest read;
  prioritize the actives feed before 2026 wk1 and set expectations accordingly.
- **C3. The remaining measured edge is small.** ΔRMSE ~0.24 at RB/WR and iso_edge 0.12–0.17
  pooled — each lever above is realistically worth hundredths. The plan should be *stacking
  small levers* (and A1 may show the honest floor is lower still), not hunting one big fix.

## Checked and ruled out en route (so nobody re-proposes)

- **Short-week / TNF rest blindness** — already carried: `days_rest_improved` /
  `rest_advantage` come from real schedule rest days
  ([src/shared/weather_features.py:267-273](../src/shared/weather_features.py)); the crude
  `week.diff()*7` `days_rest` in engineer.py is the legacy column, not the only rest signal.
- **Same-position vacancy blindness** — built and validated (#1102/#1104/#1106,
  `is_top_available` + `inherited_opportunity`, all four skill positions).
- **Late injury info** — tested false (investigation §1, no error spike on late-info slices);
  A1 above is the one *healthy-rest* case that test could not see.
- **Trade/team-change window resets** — rolling windows deliberately carry cross-trade
  (wr/features.py comment); and the D1 finding (RB edge concentrates in *established-alpha*
  RBs) argues role-discontinuity weeks are not where the RB edge lives.

## Recommended order

1. **A1** final-week exclusion sensitivity on the existing comparison scripts — an afternoon;
   could shrink the headline surviving gap before anything else is built.
2. **B2** run `rmse_gap_decomposition`, record the per-head answer; count-head loss-weight
   ablation if TD-dominated.
3. **B1(a)** lambdarank comparison column (clean ADR-0003 shape), judged on the shared-slate
   rank harness.
4. **B5** cross-position QB-context features (pairs naturally with the Phase-2 B2 PROE
   screen infra).
5. **B3** recency-weight A/B; **C2** actives feed before 2026 wk1; **A2/B4** week-slice
   diagnostic → in-season layer only if it shows growth.

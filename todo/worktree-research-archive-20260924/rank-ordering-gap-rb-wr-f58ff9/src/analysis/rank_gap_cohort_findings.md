# Rank-ordering gap × player-subtype decomposition (2026-07-13) — findings

**Question.** The surviving expert edge at RB/WR is a rank-ordering edge (RMSE + Spearman, MAE now
parity — [docs/expert_comparison.md](../../docs/expert_comparison.md) as refreshed by PR #1485;
derivation in [todo/expert-gap-investigation-2026-06.md](../../todo/expert-gap-investigation-2026-06.md)).
Is that ordering gap uniform, or concentrated in specific player subtypes / specific models?

**Tool.** [`src/analysis/rank_gap_cohorts.py`](rank_gap_cohorts.py) — pandas-only, runs locally from
the production predictions cache (no artifact loads, no torch inference). Raw tables land in
gitignored `analysis_output/rank_gap_cohorts/`; reproduce with
`python -m src.analysis.rank_gap_cohorts --positions RB WR TE QB`.

## 0. Method + substrate

- **Substrate:** `predictions_cache/predictions.parquet` (S3, built 2026-07-05, schema v7) — all 4
  models + both experts pre-joined on the served 2025 rows — cohort columns joined from
  `data/splits/test.parquet` (join rate **100.0%** at all four positions, 0 unmatched keys).
- **Calibration gate: PASSED.** The attn_nn×RotoWire shared-slate ΔMAE reproduces the published
  (PR #1485, same snapshot) table exactly: RB +0.075 (n=1447), WR +0.011 (n=2295), TE −0.063
  (n=1165), QB +0.142 vs anchor +0.139 (n=558 vs 545 — the 13 extra rows are exactly Taysom Hill,
  whom Sleeper labels TE and serving's player-id join keeps). Weekly Spearman also reproduces the
  published ρ to 4 decimals (RB 0.7151 vs 0.7492; WR 0.6203 vs 0.6542).
- **Metrics** (all paired on the per-(model, expert) shared slate; sign convention: positive =
  expert advantage): percentile-rank displacement (signed + abs, week-block bootstrap CIs, DM +
  BH q-values); **Kendall pair-bucket attribution** (the ordering gap decomposes *exactly and
  additively* into cohort-pair buckets — "share_of_gap" is an accounting identity); top-N
  miss/false-positive composition; leave-cohort-out vs a week-stratified random-removal
  permutation baseline; per-cohort expert coverage.
- **Cohorts:** pre-kickoff (prior-season expectation tier, rookie/veteran, inheritor / returning /
  questionable, RB carry-share role, WR target-share role, implied-total quartiles, season thirds)
  + clearly-separated hindsight views (actual weekly finish tier, RB ascension). Leakage guard:
  no pre-kickoff mask reads same-week realized stats (unit-tested).
- Primary contrast pre-registered: **LightGBM vs RotoWire at RB/WR** (the served NextWeek ranker
  vs the strongest expert). Everything else exploratory. Schedule cohorts carry a `small_n` flag
  only because they span < 6 weeks by construction (rows are in the hundreds).

## 1. Is it all models? Yes at RB/WR vs RotoWire — every head trails; TE/QB clean

Mean weekly shared-slate gaps (positive = expert better):

| Pos vs RotoWire | metric | ridge | nn | attn_nn | **lgbm (served)** |
|---|---|---:|---:|---:|---:|
| RB | Spearman gap | +0.035 | +0.049 | +0.034 | **+0.029** |
| RB | regret@24 (pts/wk) | +10.9 | +11.5 | +8.8 | **+6.6** |
| WR | Spearman gap | +0.029 | +0.030 | +0.034 | **+0.022** |
| WR | regret@24 (pts/wk) | +10.2 | +11.2 | +14.0 | **+10.5** |

- **All four models trail RotoWire's ordering at RB and WR** — LightGBM (the served ranker since
  PR #1477) is the closest at both, but still clearly behind. The gap is a *signal* deficit, not a
  head-selection artifact.
- **Vs NFL.com the served head is at ordering parity** (RB lgbm Spearman gap +0.004, regret@24
  −0.02; WR −0.001, +2.7): the ordering edge is RotoWire-specific, consistent with all prior work.
- **TE: no gap — we lead.** Every model's Spearman gap is *negative* vs both experts (lgbm −0.016
  vs RotoWire), regret ~0. **QB (negative control): clean** — the best QB heads beat RotoWire's
  ordering (lgbm −0.034); no systematic expert edge for the machinery to invent. Both controls
  behave, which is evidence the method is sound.

## 2. Is it all subtypes? ~Broad-based, with reproducible hot spots — and clean zones

Total pairwise-ordering gap (lgbm vs RotoWire): RB **+0.0138** [0.0055, 0.0224] concordance
points; WR **+0.0091** [0.0035, 0.0148]. Attribution by cohort (share_of_gap ÷ pair_share > 1 ⇒
the cohort carries more gap than its pair volume):

**RB hot spots** (lgbm×RotoWire; attn_nn ratios in parens — same pattern, so it's model-agnostic):
| cohort | n rows | share of gap | pair share | ratio |
|---|---:|---:|---:|---:|
| inheritor (`inherited_opportunity>0`) | 99 | 27.0% | 13.3% | **2.0×** (1.8×) |
| weeks 1–4 | 337 | 49.3% | 24.2% | **2.0×** |
| questionable | 51 | 12.9% | 6.9% | 1.9× |
| rookie | 296 | 54.0% | 37.0% | **1.5×** (1.3×) |
| no prior season | 337 | 55.2% | 41.4% | **1.3×** (1.2×) |
| **clean:** bellcow | 258 | 11.7% | 32.4% | **0.36×** (0.03×) |
| **clean:** elite_top12 | 192 | 14.3% | 25.2% | **0.57×** (0.24×) |

**WR hot spots:**
| cohort | n rows | share of gap | pair share | ratio |
|---|---:|---:|---:|---:|
| returning_2plus | 127 | 23.0% | 10.4% | **2.2×** |
| weeks 14–18 | 661 | 57.0% | 29.7% | **1.9×** |
| finish_25_36 (hindsight) | 207 | 31.1% | 17.9% | 1.7× |
| rookie | 391 | 39.2% | 31.0% | **1.3×** (1.3×) |
| **clean:** alpha (target_share_L5 ≥ .22) | 426 | 30.5% | 34.6% | 0.88× (0.63×) |
| **clean:** elite_top12 | 159 | 8.2% | 14.0% | 0.59× (0.51×) |

- The bulk of both gaps still rides the big veteran/deep slate at ~proportional volume (RB veteran
  93% of gap on 96% of pairs) — **the gap is not *confined* to any subtype**. But the hot spots are
  real, reproduce on the attention head, and are directionally confirmed by three independent
  metrics (below).
- **Seasonal asymmetry:** the RB gap is an *early-season* phenomenon (|displacement| delta weeks
  1–4 = 0.018 [0.004, 0.035], decaying to 0.004 n.s. by weeks 14–18); the WR gap is *late-season*
  (weeks 14–18 = 0.010 [0.002, 0.020], early weeks n.s.). RotoWire out-orders us on RB before
  rolling histories exist, and on WR when injuries/rest/matchup context dominates.

## 3. The single clearest signal: we under-rank rookies, everywhere

Signed percentile-rank displacement delta on rookies (positive = we place rookies further down the
board than the expert; the expert is right):

- **RB rookies: +0.029** [0.008, 0.061], BH q = 0.02 (lgbm×RotoWire); positive in **7/8**
  model×expert pairs (max +0.034 on lgbm×NFL.com).
- **WR rookies: +0.033** [0.014, 0.053]; positive in **8/8** pairs (ridge worst at +0.045).
- Same story for `no_prior` (rookie-dominated): RB +0.031, WR +0.031.
- It is a *directional placement* error, not noise: RotoWire is systematically bolder on rookies
  and correct. The served LightGBM is the **most** rookie-pessimistic of our four heads.

**Miss composition @24 makes it concrete** (lgbm vs RotoWire, pooled 2025): of rookies who
*actually finished* weekly top-24 — RB: we caught **38.9%**, RotoWire **55.6%**; WR: we caught
**5.6%**, RotoWire **25.0%**. Inheritor RBs: 50.0% vs 67.6%. Meanwhile on elite_top12 we *beat*
RotoWire (RB 97.6% vs 95.3%). The studs are not the problem; the ascending names are.

**Coverage is not the confound:** RotoWire projects 88–92% of slate rows and ≥98% of actual
top-24 finishers in every cohort (rookies/inheritors: 100%). The expert's edge is ranking skill on
a shared slate, not slate breadth.

**Leave-cohort-out (permutation-calibrated):** removing RB `no_prior`/`rookie` rows shrinks the
lgbm×RotoWire Spearman gap by −0.009/−0.008 — more than ~97% of random same-size removals
(perm pctile 0.025/0.035). The WR picture is more diffuse (removing the deep `tertiary` slate
*raises* the remaining gap — the WR gap sits proportionally in the startable alpha/secondary
tiers, matching §2's clean-zone ratios).

## 4. The dangling `ab_air_yards` fleet A/B — collected (dispatched 2026-06-15, never read)

Run `ab_air_yards-20260615T035629Z-1d05ace` (24/24 cells ok, 6 seeds × 4 variants ×
QB/RB/WR/TE, stacked-FP32 screen regime, Ridge-identity across variants = 0.0 exactly). Judged on
the boom-tail metrics it targeted (Attention-NN Q4 corr/RMSE):

| Pos | variant | ΔMAE | Δq4_corr | Δq4_rmse | q4_corr beats baseline |
|---|---|---:|---:|---:|---|
| **WR** | +air | +0.014 | **−0.013** | **+0.173** | 1/6 seeds |
| **WR** | +air_yac | −0.010 | −0.005 | +0.082 | 2/6 |
| **WR** | +air_yac_fd | +0.020 | −0.007 | +0.056 | 1/6 |
| RB | +air | +0.013 | **+0.016** | −0.074 | **5/6** |
| RB | +air_yac(_fd) | +0.023 | +0.003 | +0.051 | 3/6 |
| TE | best (+air_yac) | +0.009 | +0.010 | −0.071 | 3/6 |
| QB | all | ~flat/worse | ≤0 | mixed (one diverged seed) | ≤2/6 |

**Verdict — WR: TESTED, NEGATIVE.** Raw air-yards-decomposition history tokens *hurt* the WR boom
tail on every variant — the very lever issue #1353 lists as its primary WR play. This closes the
2026-06-14 correlation-lever hypothesis the way the coverage screen closed the team-proxy one:
WR boom/bust is not forecastable from the player's own opportunity decomposition
(consistent with the CB-coverage data-gap read, #1210). **RB `+air`: weak directional positive**
(q4_corr +0.016, 5/6 seeds; ~1σ) but MAE pays +0.013, it was screened in the stacked-FP32 regime
(≠ eager FP32 production), and it does not target where §2 says the RB gap lives — below the ship
bar; re-screen only if the rookie/inheritor levers stall. TE/QB: flat/noise.

## 5. Single-season caveat — what needs multi-season confirmation before spending

All cohort numbers above are 2025-only; 2025-only findings have flipped on multi-season substrates
before (expert-gap investigation §1–2). The metric functions here are pure frame-in/rows-out
specifically so they can ride the rolling-origin harness
([`src/tuning/ab_rolling_origin_rotowire.py`](../tuning/ab_rolling_origin_rotowire.py) pattern:
per-origin frame injection, 2022–2025, RotoWire-covered slate). **Confirm before any model
change:** the rookie/no-prior signed-displacement direction and the RB inheritor share. The
seasonal-asymmetry and WR-returner reads are weaker (schedule cohorts span <6 weeks; WR
returning_2plus n=127) — treat as hypotheses only.

## 6. Recommendation — the best way to close the RB/WR ordering gap, ranked

1. **Rookie ordering lever (RB+WR): re-open draft capital, judged on the rookie-cohort rank
   metrics this analysis defines.** The `[TESTED, REJECTED]` draft-capital archive entry
   explicitly left this door open: *"benchmark-flat … the gain concentrates in LightGBM … don't
   re-propose without a tracked rookie-subgroup metric."* That metric now exists and points
   here: rookies carry 1.3–1.5× their volume in ordering gap at both positions, across all four
   models and both experts; the served LightGBM — exactly where draft capital showed its gain —
   is the most rookie-pessimistic head; and the top-24 rookie hit-rate gap (17–19 points vs
   RotoWire) is the concrete payoff surface. $0 data, wiring already prototyped once. Screen via
   `ab_harness` with acceptance = rookie signed-displacement delta → 0 and rookie hit@24 up,
   overall MAE flat, on ≥3 seeds; multi-season confirm (§5) before the retrain PR.
2. **RB inheritor magnitude (#1134 Phase-4 work, already planned — now quantified).** Inheritors
   are 2.0× disproportionate and cap at ~27% of the RB gap; the top-24 hit gap on them is 18
   points. This bounds the lever's value: real, worth its already-scheduled slot, not a silver
   bullet.
3. **Early-season RB ≈ the same cold-start root cause.** Weeks 1–4 carry ~49% of the RB annual
   gap at 2× density, and rookies/inheritors are precisely the players with empty/ambiguous early
   histories. Expect lever 1 (+2) to eat most of this; don't build a separate week-1 ordering
   calibration first (the #1475 games-gap A/B already rejected the generic empty-history feature).
4. **WR: no in-house feature lever survives — localize, don't build.** The WR gap is diffuse
   (alphas/secondaries carry it in proportion; no pre-kickoff cohort concentrates it except
   small-n returners/late-season), the air-yards lever is now tested-negative (§4), and CB-level
   coverage data is blocked ≥2013 (#1210). The honest WR plays are: (a) ride lever 1's WR rookie
   component; (b) the FFToday 2013+ rolling-origin iso_edge extension (action-plan Phase 1b.4) to
   localize whether the WR edge is stable expert skill or era-dependent; (c) keep condq as the
   forward bet. Accept the residual as external-information-shaped until #1210's data gate moves.
5. **Bookkeeping:** record §4's WR verdict on issue #1353 (its primary lever is dead; WOPR/aDOT
   variants are the same opportunity-decomposition family and should inherit the negative prior),
   and cross-link this doc from the expert-edge action plan — both after PR #1479 (which owns
   that file) merges.

**What this rules out:** head-swap fixes (all four heads trail — §1), coverage artifacts (§3),
boom-tail opportunity tokens at WR (§4), and treating the elite tier as the problem (we win it).

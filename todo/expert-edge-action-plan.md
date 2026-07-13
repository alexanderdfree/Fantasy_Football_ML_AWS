<!-- Action plan distilled from the expert-comparison body of work (docs/expert_comparison.md,
     todo/expert-gap-investigation-2026-06.md, todo/new-sources-research-2026-06.md,
     src/analysis/rookie_cohort_findings.md, todo/attn_accuracy_findings.md). One tracked
     roadmap for "get an edge over the experts"; each phase is its own future PR/session. -->

# Expert-edge action plan (2026-07)

**Where we stand (2025 test + 2022–2025 rolling-origin, PPR; MAE re-checked 2026-07-13 on the
2026-07-05 served artifacts):** the experts' *MAE* edge is gone — the served model **ties all three
experts at QB/RB/WR/TE** (QB vs NFL.com Δ +0.133, CI [−0.09, +0.36]; the earlier "lose to NFL.com"
Δ +0.84 was the pre-regen stale-QB-artifact era, and the RB/WR RotoWire MAE losses +0.217/+0.132 are
now +0.075/+0.011, p ≥ 0.20 — [docs/expert_comparison.md](../docs/expert_comparison.md) carries the
refreshed tables), ties DST, **wins K**, beats FFToday on TE, and ties/wins the top-12 elite tier.
What survives is the tail + ordering: RotoWire keeps a significant **RMSE edge at RB/WR**
(Δ +0.23/+0.25, DM p ≤ 8.4e-4) and the durable **rank-ordering edge at RB + WR + TE** (iso_edge
+0.168/+0.161/+0.123, CIs exclude 0) that **no monotone calibration can close** (tested, rejected);
the QB residual is **current-week starter news** (#1134); the WR/TE deep gap is **player-level
coverage data, blocked ≥2013** (#1210). Full derivation:
[todo/expert-gap-investigation-2026-06.md](expert-gap-investigation-2026-06.md).

Ruled out — do not re-propose without new evidence: monotone/isotonic recalibration, team-level
coverage proxies, history-derived QB spot-start features (#1042), draft capital, expert projections as
*features* (leakage / relearns the consensus — benchmarks only).

## Phase 1 — quick wins, no retrain (this PR's session)

### 1a. Per-position best-ranker selection on the upcoming-week board — SHIPPED HERE
The homepage/NextWeek default sort ranked every position by `attn_nn → lgbm → nn`, but **LightGBM beats
the attention head on RB/WR lineup regret in 4/4 seasons** (shared RotoWire-covered slate;
[expert-gap-investigation §3](expert-gap-investigation-2026-06.md)). `upcomingProjection()`
([src/serving/frontend/src/views/NextWeek.jsx](../src/serving/frontend/src/views/NextWeek.jsx)) is now
position-aware: **RB/WR rank by `lgbm_pred` first**; other positions keep the attention-first chain.
Display unchanged (all heads stay visible); ADR-0003-compatible head *selection*, not ensembling.
Real but **partial** — LightGBM still trails RotoWire on RB/WR ordering.

### 1b. FFToday as the third expert benchmark (analysis-only)
The ≥2013 archive that extends the ordering diagnostic back a decade (current panel is genuinely
pre-kickoff only 2024+ NFL.com / 2018+ RotoWire). Loader shipped + wired
([src/analysis/fftoday_loader.py](../src/analysis/fftoday_loader.py) →
`analysis_expert_comparison._build_experts`, PR #1376). Steps, in order:
1. ~~Per-source backfill vet (mandatory before trust)~~ — **DONE 2026-07-06, verdict CLEAN on the
   FULL archive.** The hvpkod near-exact-match test (`expert_intervals.lookahead_seasons`) over all
   13 seasons 2013–2025 × QB/RB/WR/TE (n=36,788 joined rows; actuals = the S3-mirrored
   `weekly_2012_2025.parquet` scored through `_position_actuals`): per-season near-exact fractions
   **2.5–9.4%** (flag threshold 30%), residual σ **5.6–9.3** (a backfilled feed reads σ≈0),
   **zero flagged seasons**. Genuine ex-ante forecasts. Side finding: FFToday **over-projects WR**
   (pooled bias +2.66; other positions +0.4–0.7).
2. ~~Full multi-season pull~~ — **DONE 2026-07-06:** 2013–2025, 38,275 rows, 99.0% gsis match.
   **Gotcha fixed en route:** the loader cache keyed on min/max season only, so the sampled
   2013/2019/2024 pull silently satisfied a full 2013–2024 request — cache key now disambiguates
   non-contiguous lists (`_seasons_sig`, `_CACHE_VERSION` v2, regression-tested).
3. ~~Same-sample head-to-head + publish~~ — **DONE 2026-07-06**, published in
   [docs/expert_comparison.md](../docs/expert_comparison.md) § "Third expert: FFToday". Model =
   served attention NN via `artifact_eval.build_test_df_from_artifacts` (Ridge-identity validation
   Δ=0.0000 all four positions). Result (PPR 2025, FFToday's deeper covered slate): **model beats
   FFToday on TE** (Δ −0.181, CI [−0.32, −0.05], DM p=0.03 — first significant same-sample win vs
   an expert at a skill position), leans better on WR (−0.144, NS), ties QB (+0.163) and RB (+0.097).
   Expert-strength ordering: NFL.com > RotoWire > FFToday.
4. **Remaining:** extend the iso_edge / rank-skill ordering diagnostic to the 2013+ FFToday substrate
   (needs rolling-origin per-season model retrains — GPU-fleet work, pairs naturally with the Phase 2
   screen below).
- **ToS: internal benchmark only** — restricted redistribution; never a serving tab, never a feature.
- (FantasyPros stays out: current-season-only pool, survivorship-biased history — see the
  [new-sources correction](new-sources-research-2026-06.md).)

## Phase 2 — RB ordering research (highest-EV model lever): B2 PROE/pace screen
The RB ordering edge + bust-avoidance concentrates in **established-alpha RBs** (D1, 4/4 seasons) and —
unlike WR's external CB gap — may be forecastable from workload/role/game-script. Top candidate:
**B2 team PROE + neutral pace** ([new-sources-research §B2](new-sources-research-2026-06.md) — $0,
2013-verified from already-ingested PBP, leakage-clean with a season-to-date-through-W-1 group-shift lag).
- Screen via `ab_harness` frame-injection (QB/RB/WR/TE), ≥3 seeds (5–8 if the delta sits in the band),
  judged on **rank metrics** — NDCG / recall@k / mean-rank / lineup regret on the shared
  RotoWire-covered slate + the D1 established-alpha and D4 culprit slices — **not MAE** (the edge being
  chased is ordering). Batch-fleet path per ADR-0020 if no local GPU.
- Ship gate: rank-metric win on RB (direction holds ≥2/3 seeds) with overall MAE flat → wire loader
  (`src/data/`), `engineer.py` merge, `INCLUDE_FEATURES` + `ATTN_STATIC_FEATURES` (team-level
  season-to-date rates are static-eligible; **not** a windowed player stat), fixtures, retrain PR with
  benchmark evidence. Flat/negative → `[TESTED, REJECTED]` archive entry (draft-capital precedent).
- Runner-up if B2 is flat: **E2 O-line continuity** (orthogonal, from already-ingested PFR snaps).

## Phase 3 — cohort-bias calibrations (bias, not MAE; judged on the tracked `cohorts` block)
- ~~Games-gap A/B (`career_weeks_since_last_game`)~~ — **executed + tested-rejected on the fleet by
  PR #1475** (46/46 cells): `career_gap`/`itt_empty` destabilize QB attention (+5.6–6.6% MAE) and fail
  the #1137 passing-yards guard; week-1 under-projection is **QB- and K-specific, not RB/WR/TE**. Do not
  redo; see the `[TESTED, REJECTED]` archive entry it adds.
- **Week-1 conditional calibration, narrowed per #1475:** **K first** (consistent across seeds,
  un-destabilized, Ridge-served), then QB only with an attention-MAE non-regression guard.
- **Rookie-early bias calibration** ([rookie_cohort_findings](../src/analysis/rookie_cohort_findings.md)):
  QB/WR/TE rookies over-predicted +3–4.4 FP/g in their first ~3 games, RB under-predicted throughout;
  ~1 FP/g calibration-recoverable at QB. Judge on rookie-cohort bias + ranking, never headline MAE.
- **Questionable-streak A/B** (`consec_weeks_questionable`, TODO.md deferred-(c)): prior-weeks-only
  run-length; current-week exclusion mandatory for serving parity; judge on the `questionable` cohort.

## Phase 4 — QB current-week infra (#1134)
1. `inherited_opportunity` **magnitude** angles via `ab_harness` (scale/normalize by team positional
   volume; whitelist coverage audit) — judged on the beneficiary-cohort bias (−1.57 post-activation).
2. **Same-day actives/inactives feed** for the serving upcoming-week path (healthy/coach scratches never
   reach the artifact today). Source-scoping first; train/serve skew check per AGENTS.md. Serving-only —
   cannot backtest; this is the one place experts hold a genuine pre-kickoff information edge.

## Gated / parked
- **#1354 loss-shaping** (per-sample weighting, quantile tilt): pursue only if the standing
  `elite_top24` bias metric stays material post-#870-MSE — data-gated as written in the issue.
- **#1210 player-level WR/TE coverage:** data-blocked under the ≥2013 gate. Use Phase 1b's FFToday
  panel to *localize* the residual across seasons; a post-2015 secondary experiment is the only opening.
- **FantasyPros:** valid only as a current-season consensus benchmark; optional, after FFToday.

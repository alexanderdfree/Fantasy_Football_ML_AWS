# Expert-gap investigation (2026-06-30) — where/how RotoWire + NFL.com beat the model, and an artifact-drift bug

Read-only diagnostic investigation into exactly where and how the two expert projection sources beat
our model on the elite tier, plus the stale-artifact bug it surfaced. **No production change**: every
file shipped is under `src/analysis/`, `src/scripts/`, `src/tuning/`, `tests/`, or `todo/` — none in
`src/shared|data|features`, `src/config.py`, `src/{pos}/`, or `src/serving/`, so `scope_positions` is
empty (no retrain), nothing redeploys, and no model artifacts are committed (gitignored).

> **Status (2026-06-30):** the 17 angles were first run **single-season (2025) on the served artifacts**,
> which carried the stale QB artifact (§2), then **re-run on a corrected multi-season substrate**
> (rolling-origin 2022–2025, fresh calibrated model per origin, agg-of-target-stats truth, RotoWire
> 4 seasons + NFL.com 2024–25 only). **The multi-season re-run (§1, §1a) is authoritative** and flipped
> five single-season headlines (the QB stale-artifact casualties + two 2025 flukes). The full v2 numbers
> live in the gitignored `scratchpad/multiseason/expert_gap_report_v2.md`; the durable conclusions are
> below.

## 1. Where/how the experts win (17 metric angles, corrected multi-season 2022–2025)
A fan-out of 17 metric angles (rank-aware, decision-relevant, projection-rank, forensics, mechanism),
re-run on the corrected 2022–2025 substrate, replaced the coarse "RB/WR Q4 boom, ~78% irreducible"
story. Out-of-sample the experts win in exactly **two separable pockets, both skill-position only**:

- **Ordering / rank skill (irreducible by rescaling) — RB + WR + TE** (not WR-only). Experts rank the
  week's actual top-12 ~1.4 ranks shallower (WR), out-NDCG every season, and field more startable points
  (+11.3 WR / +5.8 RB pts/wk). The WR residual is a genuine **missing-data** problem — the signed
  error-gap is unpredictable from any pre-kickoff feature we carry (held-out R² −0.0004; tracks only
  *realized* receptions) → #1210 (CB/coverage signal absent).
- **Calibration / magnitude (closable by rescaling, but small + rejected) — mostly NFL.com's ceiling** +
  a residual ~0.7–1.2 pt boom-undercall on skill positions. The one *large* magnitude gap was QB's
  −3.46 "under-leveling" — a **stale-artifact bug** (§2), now gone.

**k-shape reversed vs the single-season story:** the model now **ties/wins the studs** (k=12; QB hit@12
**+0.020 sig**) and the **experts win the deep tier** (k=24–30, the gap *grows* with k).

**Two single-season headlines were 2025 flukes** (do NOT replicate over 4 seasons): the "apex-WR collapse"
(actual top-3 WRs in a top-12 slot ~39% vs RotoWire ~57% — pooled Δ **−2.8pp, NS**; the model wins the
apex in 2023 & 2024, and the 2025 gap was the stale artifact) and "the model wins the RB1 studs /
boom-capture" (**dead tie pooled**; the win lived entirely in 2025).

- Cross-cutting: "experts have late injury info" is **false** (no error spike on any late-info slice). The
  one genuine narrow pre-kickoff edge is **QB depth≥2 spot-start fill-ins** (D4: 61% of QB culprits are
  depth≥2 vs 9% of the pool; RB/WR/TE culprits are mostly *starter* booms) — and #1042 proved
  history-derived features can't close it; it needs a current-week starter-news feed (#1134), not a
  derivable feature.

### 1a. The definitive split — ordering vs calibration (E4 isotonic decomposition)
`iso_edge` = the expert's elite-tier RMSE advantage that **remains after both sides are optimally
isotonic-recalibrated** (the best rank-preserving monotone map pred→actual). It cleanly separates the two
axes: `iso_edge > 0` (CI excludes 0) ⟺ an **ordering** edge (a better *ranker*), **irreducible by any
monotonic/isotonic calibration by construction** — a monotone map preserves a model's rank order, so it
cannot move NDCG / mean_rank / recall@k / regret; `raw_edge − iso_edge` is the **calibration/magnitude**
slice (closable by rescaling). Pooled, 4 seasons, RotoWire (90% block-bootstrap CI on (season,week)):

| Pos | raw_edge | **iso_edge** (survives recalibration) | 90% CI | verdict |
|---|---|---|---|---|
| **RB** | +0.218 | **+0.168** | [+0.055, +0.281] | **ordering edge** |
| **WR** | +0.268 | **+0.161** | [+0.075, +0.246] | **ordering edge** |
| **TE** | +0.128 | **+0.123** | [+0.044, +0.198] | **ordering edge** |
| QB  | +0.099 | −0.005 | [−0.083, +0.073] | none (was the stale artifact) |
| *NFL.com — any pos* | — | — | all straddle 0 | **calibration-only** |

**Consequence (answers "can the rest be fixed with monotonic/isotonic calibration?"): no.** The durable
expert edge is an **ordering** edge at the three skill positions, which **calibration cannot close** — that
is *why* recalibration was rejected (§2/§3). Calibration only moves the magnitude axis, which is already
resolved for QB and net-negative on decisions for the rest. Closing the ordering gap needs **new ranking
signal** (#1210) or a **non-monotonic re-ranker**; our own two heads are co-linear at the elite tier (blend
recovers ≈0, error ρ 0.95–0.97), so the missing information is external, not latent in our heads. The one
in-house *partial* lever is **selecting** the LightGBM head for RB/WR ranking (a non-monotone re-ranker
that carries some orthogonal ordering).

## 2. P0/P1 (recalibration + head selection) — RETRACTED, and the bug behind it
A single-season pass over the **served artifacts** claimed a LOWO monotone recalibration closes
95.9% (QB) / 100% (TE) of the elite RMSE gap and that LightGBM out-decides the served attn head. The
**multi-season replication refuted the strong claims and exposed an `artifact_eval` bug**:

**The bug.** `artifact_eval.build_test_df_from_artifacts` silently reconstructed a **stale QB model** —
the deterministic Ridge head reconstructed at MAE **6.74 vs the QB model's recorded training MAE 5.88
(Δ+0.86)**, while WR/RB/TE were within 0.03. Root cause: the local `data/splits` (2026-06-08 vintage)
are *older* than the dbc8bf9 Batch artifacts (2026-06-22 data); the QB-only feeds (ff-opportunity
inheritance #1104, ESPN QBR #445, `season_starts_to_date` #1042) shifted value between vintages, so the
fixed QB artifact mis-predicts on the local splits. The spurious −3.46 "elite under-leveling" and the
recalibration "win" were this drift, not real model behavior.

## 3. Multi-season confirmation (authoritative — fresh rolling-origin retrain, 2022–2025 vs RotoWire)
Trained fresh per origin (train `[2013..T-2]`, val `T-1`, test `T`, no leakage; RotoWire is clean
2018+), bug-free:
- **QB is well-calibrated every season** (elite bias ≈ 0, like RotoWire); recalibration *hurts* it.
- **RB/WR/TE elite boom under-leveling is REAL and replicates every season** (attn_nn ≈ −1.6/−2.3/−1.5;
  RotoWire ≈ −1.0). RotoWire beats LightGBM on RB/WR RMSE + lineup regret every season. (This boom-magnitude
  deficit is the *calibration* slice; the larger, irreducible piece is the **ordering** edge in §1a — the
  part recalibration cannot touch, which is why the next bullet rejects recalibration.)
- **Recalibration REJECTED:** it reduces bias toward the expert level but leaves elite RMSE neutral and
  **hurts RB/WR lineup regret every season** — a bias-centering trick, not a decision win.
- **LightGBM-for-RB/WR head selection holds** (beats our attn head on RB/WR regret every season) but
  **still trails RotoWire** — a real but partial gain. (Caveat: the original run measured model regret
  on the full slate vs RotoWire on its covered subset — an apples-to-oranges ceiling; the spec now
  computes regret on a shared RotoWire-covered slate. The **bias/RMSE** conclusions — recalibration
  rejected, QB calibrated, RB/WR/TE under-leveling real — are regret-independent and unaffected.)

## 4. P3 — WR opponent-coverage feature (net-new, tested, NEGATIVE)
Verified WR carries only team-level pass-defense *volume* (no coverage-*quality* signal; NGS/PFR
ingested nowhere). Built a leakage-safe opponent coverage-openness feature from NGS `avg_separation` /
`avg_cushion` (`ab_opp_coverage_wr.py`) and A/B'd it on the WR boom subgroup — whitelist AND attention
static branch + `condq` (#1210's landing spot). **Noise everywhere** (boom-tier correlation moves
≤±0.003). Empirically confirms #1210 is a **player-level CB-matchup data** problem (PFF/FTN charting,
paid), not a feature-engineering one. (PFR advanced-defense uses non-nflverse team codes — needs
normalization if pursued.)

## 5. P4 — QB spot-start signal — NOT built (already present / tested-rejected)
The injury-driven spot-start case is already built (`is_top_available` + `inherited_opportunity` for QB,
#1102), and the derivable depth-chart-shift alternative was already tested-rejected
(`src/tuning/ab_qb_inheritance.py`). The genuine residual (non-injury benchings/trades/news) needs a new
current-week starter-news feed (#1134), not a derivable feature.

## 6. The fix shipped here (so this can't recur)
- **`artifact_eval.validate_reconstruction()`** — the Ridge-tell stale-artifact check, **warn-by-default
  inside `build_test_df_from_artifacts`** (fires for every diagnostic consumer) + CLI `--validate` /
  `--strict`. Warn 0.10 / fail 0.30 FP MAE (healthy ≈0.02–0.03; the QB bug was 0.86). PPR + skill
  positions only; skips gracefully without a reference. CI-safe unit tests in
  `tests/analysis/test_artifact_eval.py`. (Not the feature-manifest guard #1348 — that is config/feature
  drift, deliberately data-free; this is artifact-vs-data drift. Not a CI gate — CI has no splits/artifacts.)
- **`src/scripts/regen_served_artifacts.py`** — realign a drifted position: run the pipeline on the
  current splits → promote producer→served → write a fresh `benchmark_metrics.json` (bare `run_pipeline`
  doesn't). Ran for QB locally; post-regen QB validates OK (Δ0.018) and dcr1 bias is +0.32 (calibrated).

## Code shipped
| file | what |
|---|---|
| `src/analysis/artifact_eval.py` | `validate_reconstruction()` Ridge-tell check + `--validate`/`--strict` CLI |
| `tests/analysis/test_artifact_eval.py` | CI-safe verdict-logic + skip tests |
| `src/scripts/regen_served_artifacts.py` | realign a drifted position's served artifacts |
| `src/analysis/recalibration_eval.py` | LOWO-isotonic recalibration diagnostic (lever rejected; see §3) |
| `src/tuning/ab_opp_coverage_wr.py` | WR opponent-coverage A/B (negative; §4) |
| `src/tuning/ab_rolling_origin_rotowire.py` | multi-season rolling-origin × RotoWire confirmation (§3) |

## Reproduce
- Stale-artifact check: `python -m src.analysis.artifact_eval --positions QB RB WR TE --validate`
- Regen a drifted position: `python -m src.scripts.regen_served_artifacts --positions QB`
- Multi-season confirmation: `python -m src.tuning.ab_rolling_origin_rotowire` (eager, 4 origins × QB/RB/WR/TE)
- WR coverage A/B: `python -m src.tuning.ab_opp_coverage_wr`
- (Transient harness glue + raw outputs — substrate parquets, per-angle results — live under the
  gitignored `scratchpad/`; the reusable pieces are the tracked files above.)

## Open items (unchanged by this PR)
- **The ordering edge is RB + WR + TE** (not WR-only, §1a) and is **NOT calibration-fixable**. WR/TE need
  **player-level CB-matchup data** (#1210); RB may be partly closable from role/workload signal we already
  carry (the tractable research lever — see below). Recalibration and team-coverage proxies are both
  ruled out (§3/§4).
- **Product lever (owner call, serving-only, no retrain):** the homepage ranking key
  (`upcomingProjection()`, `src/serving/static/js/app.js:596-599`) globally prefers `attn_nn→lgbm→nn`;
  for **RB/WR the LightGBM head ranks better**. Per-position best-ranker selection is
  **ADR-0003-compatible** (head *selection*, not ensembling/stacking — there is no single "served head"
  today; all four are shown). Real but partial.
- **RB research lever (no new data):** the RB ordering edge (iso +0.168) + bust-avoidance (+1.3pp top-24,
  4/4 seasons) concentrates in established-alpha RBs (D1, 4/4 positive) — unlike WR's external CB gap, part
  of RB bust risk may be forecastable from workload/role/game-script we already have. Investigate via
  `ab_harness` against the substrate's D4 culprits / D1 archetype slices.

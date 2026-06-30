# Expert-gap investigation (2026-06-30) — where/how RotoWire + NFL.com beat the model, and an artifact-drift bug

Read-only diagnostic investigation into exactly where and how the two expert projection sources beat
our model on the elite tier, plus the stale-artifact bug it surfaced. **No production change**: every
file shipped is under `src/analysis/`, `src/scripts/`, `src/tuning/`, `tests/`, or `todo/` — none in
`src/shared|data|features`, `src/config.py`, `src/{pos}/`, or `src/serving/`, so `scope_positions` is
empty (no retrain), nothing redeploys, and no model artifacts are committed (gitignored).

## 1. Where/how the experts win (17 new metric angles, 2025; adversarially verified)
A fan-out of 17 new metric angles (rank-aware, decision-relevant, projection-rank, forensics,
mechanism) replaced the coarse "RB/WR Q4 boom, ~78% irreducible" story with four position-specific
mechanisms:
- **WR** — an **apex-studs ranking** gap: the model lands the week's actual top-3 WRs in a top-12 slot
  only ~39% of weeks vs RotoWire ~57% / NFL.com ~50%; slots 4–12 are even. The residual is a genuine
  **missing-data** problem — the signed error-gap is unpredictable from any pre-kickoff feature we have
  (held-out R²≤0; it tracks only *realized* receptions) → confirms #1210 (CB/coverage signal absent).
- **RB** — a deep-tier (RB2/flex) **bust-avoidance** edge; the model wins the RB1 studs.
- **QB / TE** — pricing/calibration, not ranking (we pick the right players); the model is competitive
  on lineups.
- Cross-cutting: "experts have late injury info" is **false** (no error spike on any late-info slice);
  the one genuine pre-kickoff info edge is narrow **QB spot-start backups** (~10% of rows, a week-5+
  role-change effect) — and #1042 already proved history-derived features can't close it.

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
  RotoWire ≈ −1.0). RotoWire beats LightGBM on RB/WR RMSE + lineup regret every season.
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
- RB/WR boom gap remains open — needs **player-level CB-matchup data** (#1210), not recalibration or a
  team-coverage proxy.
- Optional product lever: **serve LightGBM (not attn) for RB/WR ranking** — real but partial; owner call.

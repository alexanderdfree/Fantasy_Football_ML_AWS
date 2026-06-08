# Closing the RB/WR accuracy gap vs RotoWire

Plan/handoff doc. **Read [the diagnosis](#diagnosis) first** — it constrains the plan hard.
Measurement instrument: [`src/analysis/rmse_gap_decomposition.py`](../src/analysis/rmse_gap_decomposition.py)
(PR #1037) — re-run before/after any change.

## Diagnosis (already done — what the gap *is*)

On the Comparison tab's top-30 slice (2025), experts beat our best model (LightGBM) on **RMSE/R²**
for RB/WR but **not MAE**. The decomposition diagnostic pinned the cause:

- **The gap is entirely the Q4 boom tier.** Our model *beats* both experts on Q1–Q3 RMSE; it loses
  only on the top scoring quartile, and because Q4 errors are ~3× larger that one tier flips the
  aggregate. (RB: model Q4 RMSE 13.86 vs nflcom 12.93 / rotowire 12.54.)
- **It is not the loss family.** L2 LightGBM and L2 Ridge show the same TD-dominated error as the
  Poisson NN (Ridge has the *highest* TD share). Swapping the NN's TD loss won't move it.
- **It is not calibration.** Our model is already the best-calibrated of the three (overall slope
  0.87 vs experts' 0.75–0.81); the best RMSE any rescale could reach (`recal_rmse`) saves only ~0.11.
- **~78% of Q4 error is the systematic boom under-call, shared with the experts** (they under-call
  booms too; de-biased residual floors are nearly identical, ~6.3). That part is near-irreducible.
- **The one closable edge is RotoWire's correlation** (overall corr 0.412 vs our 0.365; we're *tied*
  with NFL.com). Correlation = signal, so only **more signal** closes it — and notably, in Q4 our
  correlation already **exceeds** both experts', so we are *not* worse at identifying booms.

**Net:** small gap (~0.2 RMSE, RotoWire only), boom-concentrated, mostly irreducible. The lever is
a modest correlation lift, not a model/loss/calibration change.

## What we already have (do NOT rebuild)

Background feature-landscape audit (2026-06-07) — all built, RB/WR-whitelisted, serving-parity-safe
(`build_position_features` is shared between training and serving):

| Signal | Status |
|---|---|
| Vegas: `spread_line`, `total_line`, `implied_team_total`/`implied_opp_total` | ✅ whitelist **+ attention history** ([weather_features.py](../src/shared/weather_features.py)) |
| Snap share: `snap_pct`, `snap_pct_raw` | ✅ whitelist + attention history |
| Injury: `game_status`, `practice_status` (ordinal) | ✅ whitelist — **static-only; never alignment-audited (OPEN)** ([loader.py](../src/data/loader.py):399) |
| Depth chart: `depth_chart_rank` | ✅ whitelist — static-only; audited/realigned (#595) |

So RotoWire's edge is **not** "they have injury/Vegas/snap data we lack." It is **role/opportunity
intelligence** — who inherits volume when a starter sits — which we hold only as crude raw inputs and
never turn into the derived signal that drives booms. **No new data source is required.**

## Hypothesis

The model's worst Q4 boom under-calls are **role-vacancy events** (a backup RB who gets a workhorse
load because the starter is inactive; a WR promoted up the depth chart). We have injury + depth +
teammate snaps as raw columns but never compute *vacated opportunity*; RotoWire's analysts encode it
by hand.

## Plan (validation-gated)

### Phase 0 — Validate the lever (read-only, no model change) ← GO/NO-GO
Per the project rule "don't build a feature without a tracked subgroup metric showing the gap" and
"verify the activation precondition, not just the code smell."
- **0a. Audit injury alignment.** Reuse the [`audit_depth_alignment.py`](../src/analysis/audit_depth_alignment.py)
  pattern: does `game_status` for week W reflect the *pre-kickoff* designation, or is it stale/leaky?
  Restrict to transition rows (new OUT designations). Injury alignment is an OPEN, never-audited gap
  and a prerequisite for trusting any injury-derived feature.
- **0b. Vacancy attribution on the worst Q4 misses.** The diagnostic already lists them; for each,
  check whether a higher-depth same-team/position teammate was OUT/inactive that week. **Quantify the
  fraction of the model's Q4 *excess* error (vs RotoWire) that coincides with vacated roles.**
- **Gate:** meaningful vacancy share → Phase 1. Mostly established-star monster games (no vacancy) →
  gap is irreducible; report and stop (don't build a feature that can't move the metric).

### Phase 1 — Build opportunity-vacancy features (derived from existing data; no new source)
- `teammate_snap_vacated` — Σ prior-week `snap_pct` of higher-depth same-team/position players who are
  OUT/Doubtful this week.
- `effective_depth_rank` — depth rank after removing injured-out players above.
- `is_promoted` / `lead_back_out` — is this player now the top *available* at their position/team.
- Wire per conventions: [`src/{rb,wr}/features.py`](../src/rb/features.py) + `include_features`
  whitelist (RB + WR) + `ATTN_STATIC_FEATURES` (pre-kickoff, week-specific, **non-temporal** → static
  branch; NOT rolling, per the stop-rule) + test fixtures. Build in the **shared** path so serving
  gets parity. Align on pre-kickoff injury status (Phase 0a guards this).

### Phase 2 — Validate with the real pipeline (the metric that matters)
- `python -m src.{rb,wr}.run_pipeline`, **≥3 seeds** (5–8 if borderline), then re-run
  `rmse_gap_decomposition` before/after.
- **Success = Q4 correlation lift + RMSE/R² toward RotoWire, without regressing the Q1–Q3 wins or
  overall MAE.** Judge by Q4 correlation across seeds (subgroup direction), not single-seed overall
  MAE (noise). Slice `result["test_df"]`; don't reimplement models. Ceiling ~0.2 RMSE; if Q4 corr
  doesn't move across seeds, revert.

### Phase 3 — only if Phase 2 pays off
Add per-game availability/role to `ATTN_HISTORY_STATS` (temporal branch — the NN sees no role
trajectory today); Vegas up-weighting for the week-1 empty-history regime (separate OPEN issue).

## Stop-rules / risks
- Editing `src/{rb,wr}/features.py`+config fires an RB/WR retrain (expected — we're changing features).
- Injury alignment is the main landmine (Phase 0a is the guard). Attention static must stay
  non-temporal (stop-rule). Maintain train/serve parity (build in the shared path).
- Expected payoff is modest and boom-concentrated. Phase 0 protects against spending feature work on
  an irreducible gap.

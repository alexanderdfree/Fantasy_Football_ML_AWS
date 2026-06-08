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

## Phase 0 result — COMPLETE (2026-06-08)

**0a — injury alignment: PASS.** Across all 2013+ seasons, 0.0% (RB) / 0.2% (WR) of player-weeks
labelled OUT/Doubtful appear in the played frame — OUT reliably means "did not play," so the injury
report is a clean pre-kickoff signal (no leakage).

**0b — the lever, correctly specified:**
- **Crude "any teammate OUT" is a NULL across 2013+** (it was a false positive on 2025 alone). Pooled
  WR lift −0.12 FP (positive in only 6/13 seasons), RB +0.26; boom rate flat for both. A usage-weighted
  "a ≥0.4-snap starter is OUT" measure is *also* null (WR corr −0.013, RB +0.001; boom rate flat). The
  dilution: averaging over *all* remaining same-position players drowns the one who actually inherits.
- **The inheritor (next-man-up), measured within-player, is a strong multi-season signal.** Restricting
  to the player who becomes the snap-leader when a starter sits, vs that player's own other weeks:
  **RB +6.36 FP (±0.38 se), 84% of players; WR +2.08 FP (±0.26 se), 59%**; boom rate (≥20 FP)
  **RB 21% vs 8%, WR 19% vs 8%.** The model is blind to it — `snap_pct` is lagged (still the backup's
  prior-week value) and `depth_chart_rank` is inertial — which is exactly the RotoWire edge.

**Independent confirmation + the success metric:** the existing **ascension cohort**
([`src/analysis/cohort_analysis.py`](../src/analysis/cohort_analysis.py): `find_ascension_events`,
`label_ascension_rows`, `add_injury_attribution`) already measured this for RB — **every model
under-calls the ascension week by ~12 FP** (LightGBM −12.5, Ridge −11.8, NN −12.1, Attn −12.2; 205
events 2012–2025, ~60% injury-linked) — and [`rb_ascension_findings.md`](../src/analysis/rb_ascension_findings.md)
explicitly recommends a forward-looking vacated-volume feature. **That cohort is the tracked subgroup
metric** (the cohort is ~1% of rows, so judge the fix on cohort bias, not overall MAE).

**Verdict: GO**, correctly specified as *role inheritance* (next-man-up) — RB primary (bigger,
documented), WR secondary.

## Placement decision (where the feature goes)

The signal is **this week's vacancy** — a pre-kickoff, point-in-time, non-windowed fact. So:
- **Ridge + LightGBM + NN static branch — YES, in one edit.** Add to `include_features["contextual"]`
  for RB/WR. That feeds Ridge + LGBM automatically *and* the NN static branch, because
  `attn_static_features = derive_attn_static_features(_INCLUDE_FEATURES, ["prior_season","matchup",
  "contextual","weather_vegas"])` and `contextual` is in that set — the same path `depth_chart_rank`
  already takes. No separate `ATTN_STATIC_FEATURES` edit needed.
- **Attention HISTORY branch — NO.** (1) The history sequence is the player's *past* games (lagged);
  the vacancy is about *this* week — wrong temporal frame. (2) A per-game "was-inheritor" stat is
  redundant — a past inheritance game already shows up as a high `snap_pct_raw`/`game_carry_share` game
  (RB) the attention pool can attend to. (3) The per-target head fuses static×history by concatenation,
  so the NN learns "big vacancy this week × this player absorbs volume" with the vacancy in static
  alone. Adding it to history violates the "don't double-feed the temporal branch" stop-rule for no
  marginal signal.
- **Separate WR-history note (not this feature):** WR's `ATTN_HISTORY_STATS` carries only `snap_pct_raw`
  for usage and **lacks `game_target_share`** (RB has it) — a real per-game-usage gap worth its own A/B
  if WR underperforms, but distinct from inheritance.

## Phase 1 — Build the role-inheritance feature (RB + WR)

Current-week, pre-kickoff, derived from existing injury + depth + prior usage (no new data source):
- `vacated_opportunity` — Σ prior usage (season-mean snap share / prior-3 opp) of same-team/position
  players **OUT/Doubtful on week-W's report** and ranked above this player. Sizes the freed volume.
- `effective_depth_rank` — depth rank after removing OUT players above (captures "I'm now the lead").

Reuse `add_injury_attribution`, `compute_team_{rb,wr}_totals`, `safe_divide`, `label_ascension_rows`
(`cohort_analysis.py` / `src/{rb,wr}/data.py`). **Data-plumbing note:** OUT players are dropped from the
splits (no snap), so the teammate-vacancy aggregate must be computed where the full injury table + all
players are present (loader / engineer, *pre-filter*), then joined onto surviving rows — NOT in the
per-position `features.py` (which only sees survivors). **Leakage guard:** only the injured teammate's
*prior* usage + the pre-kickoff injury report (0a confirmed OUT→0 snaps).

**Wiring:** add to `include_features["contextual"]` (RB + WR) → Ridge + LGBM + NN static automatically.
Build in the **shared** feature path for serving parity, and ensure serving fetches the current-week
injury report (today it leans on pre-computed splits — the main integration risk). Update fixtures
(`tests/conftest.py` / `tests/{rb,wr}/conftest.py`).

## Phase 2 — Validate (the metric that matters)
- `python -m src.{rb,wr}.run_pipeline`, **≥3 seeds**, **all four models**; re-run `cohort_analysis`
  (ascension bias, RB; build a targets-based WR analog) + the committed `rmse_gap_decomposition`
  Q4/inheritor cut, before/after.
- **Success = ascension-cohort / Q4 bias reduction, with overall MAE held flat.** Cohort is 1–4% of
  rows, so judge the subgroup (the draft-capital trap), across seeds, all models. If the cohort bias
  doesn't move across seeds, revert.

## Stop-rules / risks
- Subgroup-not-overall-MAE (cohort is ~1–4% of rows). Pre-kickoff-only (leakage). Serving parity for
  live injuries. Static stays non-temporal. Editing `src/{rb,wr}/features.py`+config (and possibly the
  loader/engineer for the teammate aggregate) fires the RB/WR (or 6-position, if loader) retrain.
- The attention **history branch is decided out** for this feature (see Placement) — do not re-propose.
- The **crude "any teammate OUT" measure is a tested NULL across 2013+** — the signal is the *inheritor*
  (next-man-up), not generic vacancy. Don't rebuild the crude version.

## Outcome — SHIPPED (2026-06-08, PR #1053)

Done via the shared parallel A/B harness ([src/tuning/ab_history_token.py](../src/tuning/ab_history_token.py)), not bespoke loops.

- **Shipped feature** (final names): `is_top_available` + `inherited_opportunity` (= the plan's
  "effective_depth_rank" / "vacated_opportunity" concept). Computed in
  [src/features/engineer.py](../src/features/engineer.py)`._build_inheritance_features` — the shared,
  pre-split path, with the raw injuries frame threaded through `build_features(df, injuries_df)` from
  `refresh-splits.yml` (OUT teammates are dropped from splits, so the out-set must come from the report).
  Whitelisted into RB+WR `include_features["contextual"]` → Ridge+LGBM+NN-static + the NN static branch.
  Role proxy: RB `snap_pct_raw`, **WR `targets`** (snap-share is the wrong opportunity proxy for WR).
- **Parity:** baked into `data/splits/*.parquet`; serving reads the same columns → **no serving change**.
  `refresh-splits` regenerates splits (fail-loud on injury-fetch error); `train-batch` race-gate waits.
- **Validated** (3 seeds, **inheritor subgroup** `inherited_opportunity>0`, n=33 RB / 41 WR — NOT overall
  MAE, which dilutes a ~1%-of-rows feature to noise): **RB strong, all models** (inh-MAE Ridge −1.61 /
  LGBM −0.71 / Attn −0.93; inh-bias under-call ~halved, +2.2…+3.3). **WR modest-but-real on bias**
  (inh-bias +0.2…+1.2 all models; inh-MAE flat — diffuse target redistribution). Production
  `_build_inheritance_features` is **byte-identical** to the A/B injector on the real test split.
- **History branch REJECTED** (the user's hypothesis, tested): routing the signal through
  `attn_history_stats` is −0.32 FP / ~3σ *worse* on the RB cohort — a past spot-start is already encoded
  by the history branch's `snap_pct_raw`/usage tokens (redundant; averaging an average). AGENTS.md
  stop-rule added. The *static* current-week value is the win (the vacancy is in no past sequence).
- **Merge fires** refresh-splits + a 6-position retrain (`engineer.py` is global; RB/WR use the feature,
  the other four are inert / byte-identical).

## Round 2 — WR red-zone + opportunity (boom tier), SHIPPED 2026-06-08

Lever #2 after inheritance. A broader signal scan (two-agent feature-landscape audit) found **WR is
blind to red-zone receiving in every branch** (RB carries it everywhere) and thin on per-game usage /
priors vs RB. The closable edge is correlation/signal on the boom tier, so the bet was red-zone +
opportunity parity. A/B: [src/tuning/ab_boom_signals_wr.py](../src/tuning/ab_boom_signals_wr.py).

- **Shipped feature (`+all` arm)**: red-zone rolling (`redzone_targets_L3`, `redzone_target_share_L3`)
  + prior (`prior_season_total_redzone_touches`/`_per_game`, already in splits) + parity priors
  (`prior_season_games_played`, `prior_season_mean_catch_rate`) + `opportunity_index_L3` → WR whitelist
  (Ridge/LGBM/NN-static); raw per-game `redzone_targets`/`_share` + `game_target_share`/`_hhi`/
  `game_opportunity_index` → `attn_history_stats` (NN history, AGENTS.md reach #2 — raw per-game,
  genuinely absent from WR's sequence; NOT the rejected windowed inheritance token). Built in
  [src/wr/features.py](../src/wr/features.py) (pipeline-time, mirrors RB) + the already-baked priors
  whitelisted in [src/wr/config.py](../src/wr/config.py).
- **Parity / scope**: serving auto-parity via shared `build_position_features` (no serving change); no
  `engineer.py`/splits change → **WR-only retrain** (not 6-position). Production build is **byte-identical**
  to the validated A/B injector on the test split (max|Δ|=0).
- **Validated** (6 seeds, **boom subgroup** Q4 / receiving-TD games — NOT overall MAE): **real but
  modest, direction-robust** — 12/12 boom deltas positive across Ridge/LGBM/Attn for `+all`. Attention NN
  (best WR model): q4_bias +0.218, q4_corr +0.004, rztd_bias +0.177 (~1.5σ, all positive). LightGBM
  q4_bias +0.058 (~1σ), corr flat. `+rz` and `+parity` alone are each weaker/mixed — the win is the
  interaction (`+all` dominates every arm on every model's boom bias). Effect is ~3% of the boom
  under-call — exactly the "incremental, mostly-irreducible" gap the diagnosis predicted.
- **Deferred** (next levers if pursued): air-yards opportunity (WOPR/aDOT/RACR — buildable, no new
  source) and route participation / TPRR (needs a new FTN/nflverse source, 2022+ only).

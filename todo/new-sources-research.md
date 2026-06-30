# New data-source research: candidates for the per-position fantasy ML system

Research synthesis for new **feature-track** signals and new **expert-comparison** sources, ranked by ROI × feasibility. Scope: per-position weekly fantasy-points models (QB/RB/WR/TE/K/DST), raw-stat targets, 2012–2025 training with the 2025 holdout. Every candidate below is **free + buildable** (paywalled/rejected items are in the appendix). All survivors passed an adversarial refutation pass; the ROI/Feas numbers are the post-refutation values.

**The hard truth up front:** every verified survivor scored **feasibility 6–9 but ROI 3–5**. There is no free, deep, leakage-safe source that is *also* a confident benchmark mover — the high-ROI signals (player-level CB coverage for #1210) are paywalled or shallow, and the deep/clean signals (PBP-derived team context, NGS efficiency) carry real **benchmark-flat risk** because the served models already triangulate most of what they add. This doc is therefore a *screen-first* backlog: build the cheapest survivor, gate it on a **subgroup metric**, and ship only if it moves the **served best model**.

---

## Orchestrator reconciliation — independent verification (read this first)

This report fuses two parallel passes: the **8-lens, 70-agent discover → deep-dive → adversarial-verify → synthesize workflow** (the body below), and **5 independent Opus-max verifiers** run concurrently against the highest-confidence headline leads. Where they agree, confidence is high. The independent pass also **corrects two places** the workflow synthesis was off and **reframes the top-pick decision** around the stated goal: *new information*, not re-expressing signal we already have.

**Independent verifier cross-check (free + buildable + leakage-safe):**

| Lead | Independent verdict | Net |
|---|---|---|
| **NGS `load_nextgen_stats`** — receiving sep/cushion (#1210) + RB RYOE + QB CPOE/pressure | ✅ CONFIRMED · live-safe (updates **nightly in-season**) · 2016/2017+ · repo-unused (only the POC) | **Headline feature bet.** Ship whitelist-only to Ridge/LGBM (keep `_L5` out of `ATTN_STATIC_FEATURES`); gate on WR boom subgroup. |
| **FantasyPros ECR `load_ff_rankings(type="all")`** | ✅ CONFIRMED · free 2019–2025 consensus of ~100 experts · leakage-safe (scrape-date pre-kickoff) · **analysis-only / no retrain** · rank-only | **Headline expert bet** (with the "market"/implied-total expert) — a true consensus vs Sleeper's single RotoWire provider. |
| **PFR `load_pfr_advstats`** | ✅ CONFIRMED · pressure/blitz + YBC/broken-tackles/drops are the net-new part (YAC/air-yards are dups of weekly) · 2018+ | Solid Tier-2 feature; subgroup-screen on the 2018+ cohort. |
| Betting **player props** (dual-use) | ❌ REFUTED (free) · no free multi-season historical prop archive (Odds-API free tier excludes NFL props; paid prop history only ~2023+) · public archives are *game* lines we already have | Drop as feature/benchmark; live-prop **serving overlay only**. |
| **DFS salary** (RotoGuru/`nfldfs`) | ⚠️ free fallback · 2014/15+ · feature-only (market-rank proxy), **not** a benchmark · benchmark-flat risk | Tier-3 feature. |
| **`load_participation`** (man/zone coverage) | ❌ REFUTED for production · play-level (routes-run weak) · **updates only post-season → no in-season feed**, so the leakage-safe opponent-tendency can't be built live | Historical-screen only (matches the appendix). |

**Two corrections to the body below:**

1. **The "NGS coverage POC already *regressed* WR" claim is an overstatement.** What regressed on WR is the **`condq` architecture** (conditioning queries on static features) — per AGENTS.md/the WR config, an owner *forward-bet on a data gap*, **not** a completed test of the NGS separation/cushion feature. The NGS-coverage feature itself was only ever a measurement-harness POC (`src/tuning/ab_opp_coverage_wr.py`) that was **never wired or conclusively A/B'd**. So #1210's named missing input (defense-allowed separation/cushion) is **untested, not failed** — which is exactly why it's the strongest *new-information* bet.
2. **Elevate FantasyPros ECR.** The body files it in Tier 3; the independent pass confirms it as the strongest *new* expert-track add — free, deep (2019–2025), leakage-safe, **analysis-only (no retrain)**, a ~100-expert consensus (vs. Sleeper's single provider). Ship it **alongside** the Tier-1 "market"/closing-implied-total expert: two no-retrain benchmark wins.

**Top-pick reframing.** The body picks `pbp_pace` (game-script/pace) on pure ROI×feasibility — correct as the *cheapest, lowest-risk warm-up PR* (zero new fetch, full depth, validates the ingestion harness). But by the report's own repeated admission it is a **second-order signal the served models already triangulate** (`implied_team_total` + team-volume history tokens) → likely benchmark-flat. Since the ask was explicitly **new information / new angles**, the genuinely-new-information bets are:

- **Feature track → NGS receiving defense-allowed separation/cushion (#1210)** — the one source that targets the project's marquee unsolved problem with information it does **not** already have, and is **live-safe** (unlike participation). Spec immediately below.
- **Expert track → FantasyPros ECR + closing-implied-total**, both analysis-only/no-retrain — immediate benchmark breadth at near-zero risk.

**Recommended sequencing:** (1) ship the two no-retrain experts first (free wins, no benchmark risk); (2) run the **NGS-coverage A/B** as the headline feature experiment; (3) keep `pbp_pace` (full spec in the body) as the low-risk harness warm-up if a safe first feature PR is wanted.

### Orchestrator top pick — NGS defense-allowed separation/cushion for WR (#1210)

The explicitly-named missing input for the project's #1 open gap; live-safe (NGS updates nightly in-season); a POC seam already exists. **Design only.** Fires a WR retrain.

- **Ingestion (mirror `ff_opportunity`):** (1) `src/data/nfl_source.py` — add a `next_gen_stats(seasons, stat_type)` wrapper through the Polars→pandas boundary (the POC bypasses it; move it here), with `_native_int_seasons` coercion. (2) `src/data/external_sources.py` — `load_ngs_coverage(seasons)`: pull `stat_type="receiving"`, take `avg_separation`/`avg_cushion` per `(player, team_abbr, season, week)`, target-weight to a per-offense game value, map offense→**defense via `load_schedules`** to get separation/cushion **allowed**, cache like `ff_opportunity`; return an `(opponent_team, season, week)`-keyed frame. (3) `src/data/loader.py::load_raw_data` — `_fetch_ngs_cov` in the ThreadPool; left-merge on `(opponent_team, season, week)`; NaN→train-mean (not 0) — 2012–2015 (pre-NGS) NaN-filled, and **empirically confirm `avg_separation`/`avg_cushion` are populated in 2016 vs only 2017+** before claiming that season.
- **Feature build:** `src/features/engineer.py` — `opp_sep_allowed_L5`/`opp_cushion_allowed_L5` = per-defense `shift(1).rolling(OPP_ROLLING_WINDOW)` **within season**, alongside `_build_defense_matchup_features`.
- **Whitelist:** `src/wr/config.py` `_INCLUDE_FEATURES['defense']` — **Ridge + LightGBM only** (WR's served best models). **Do NOT** add the `_L5` columns to `ATTN_STATIC_FEATURES` (windowed-in-static stop-rule; mirrors how `opp_def_*_L5` is excluded from `DEFAULT_ATTN_STATIC_CATEGORIES`). Extend RB (RYOE-allowed) / QB (pressure-allowed) later. Update `tests/wr/conftest.py` + the manifest snapshot.
- **Alignment audit (mirror `src/analysis/audit_depth_alignment.py`):** schedule offense→defense join hits ~100% of WR rows; `team_abbr` is modern codes but **schedules carry legacy OAK/SD/STL** — verify join keys match; target-weighted `_wmean` is the intended "allowed" semantic; week-W value uses only the defense's games < W.
- **A/B + gate (`ab_harness`, frame-injection, WR):** mirror the existing `ab_opp_coverage_wr.py` arms (baseline vs +coverage→Ridge/LGBM), `expect_ridge_identical=False`. **Gate on the WR boom subgroup** (corr/bias/RMSE on `result["test_df"]`, ≥3 seeds), **not** overall MAE (the dilution trap). Ship only if it reaches WR's served best model. Caveat: only ~2017+ rows carry signal — judge era-aware.

---

## TL;DR

The 3–5 highest-value NEW sources, one line each:

1. **Team game-script / pace context from already-cached PBP** `[feature]` — neutral-script sec/play, PROE (`pass_oe`/`xpass`), plays/game, no-huddle rate, aggregated per `(posteam, season, week)`. **Zero new fetch**, full **2012–2025** depth, leakage-safe via the proven `shift(1).rolling` team pattern. Fills the named *game-script/pace* gap. Highest feasibility (9); ROI capped because volume is partly proxied by `implied_team_total` + realized team-volume history tokens.
2. **Sleeper market `started%` / `owned%`** `[feature]` — free no-auth research endpoint (`api.sleeper.com/players/nfl/research/...`); a crowd **probability-of-start / role-clarity** scalar (Dobbins 89%→11% across his injury week). Genuinely net-new (repo uses zero market signal), leakage-proven. Capped by **2020–2025-only** coverage (8/14 train seasons NaN).
3. **NGS rushing RYOE / stacked-box rate for RB** `[feature]` — `load_nextgen_stats(stat_type="rushing")`, gsis-keyed (no bridge), into the **RB attention history branch** (RB's served best model is the NN). Tracking-talent prior beyond box score. Capped by **2016+** floor and a 10-carry minimum that covers only ~43% of RB-weeks.
4. **NGS passing CPOE / time-to-throw / aggressiveness for QB** `[feature]` — same loader, `stat_type="passing"`; drops into the exact **ADR-0016 QBR seam**. Accuracy-net-of-difficulty signal orthogonal to volume. Capped by 2016+ and a high benchmark-flat risk (QB is the most feature-saturated position; the rich per-game channel bypasses the *served plain MLP*).
5. **Closing implied team total as a "market" expert** `[expert]` — `build_implied_team_total_lookup` already exists; adds the missing **"are we beating Vegas?"** baseline, **analysis-only / no retrain**, and gives DST a clean 2nd expert (NFL.com has none). Capped because the linear DST component (sacks/INT/TD) is not market-derivable, so it is a floor projection.

**Honest gap note:** none of these closes the marquee **#1210 WR boom/bust** gap — that is player-level **CB coverage quality** (separation/cushion/shadow), which is either paywalled (PFF) or shallow (NGS receiving 2016+/FTN 2022+). **[Correction — see reconciliation above]** the NGS-coverage *feature* was only ever a measurement POC, never conclusively A/B'd; what regressed on WR is the separate `condq` *architecture* (a forward-bet on this very data gap), not this input. The team-level proxies here (PROE, opponent EPA-DvP, PFR/participation scheme) touch its *environment*, not its core.

---

## Why now

The model has **plateaued on architecture**: six attention-arch knobs (#109–121, except `condq`), dual-branch opponent-defense attention, `hurdle_poisson`, and stacked-seed regimes have all been screened, and the documented gaps (`#1210` WR coverage, week-1 cold-start, richer opponent/role/pace) are **information-starved, not architecture-starved** — the attention NN consumes the *same signal* LightGBM does, just shaped differently. The remaining lever is **new pre-kickoff information**, but the cheap/deep sources (PBP-derived team context, NGS efficiency) mostly re-express what the served models already infer, so the discipline is: build the cheapest survivor, judge it on a **subgroup metric against the served best model**, and treat "benchmark-flat ⇒ don't ship" as the default outcome rather than the exception.

---

## Ranked candidates

ROI and Feas are the post-refutation 1–9 scores. **DUAL-USE** = the same source feeds both a feature and an expert column. All non-NGS PBP candidates are **zero-new-fetch** (reuse `nfl_source.pbp_data(seasons, cols)`).

### Tier 1 — High (feasibility 8–9, build-ready, leakage-clean)

| Source | Track | What it adds | Free access path | Gap / position | Leakage audit | ROI×Feas |
|---|---|---|---|---|---|---|
| **Team neutral-script pace** (sec/play, neutral plays/game) | feature | Snap-volume multiplier orthogonal to per-play efficiency; team tempo on neutral-script plays only | `pbp_data` + new `PBP_PACE_COLS` (zero fetch) | game-script/pace; all skill pos | Team-agg then `shift(1).rolling(5)`; PHI/MIA face-validity ranking | 5×9 |
| **PBP game-script / PROE** (PROE, neutral sec/play, no-huddle, plays/game, early-down pass rate) | feature | Coordinator pass-lean net of game script + opponent-allowed mirror | `pbp_data` + `PBP_PACE_COLS`; `pass_oe`/`xpass` valid 2006+ | game-script/pace + richer opp adj; all pos | `shift(1).rolling(OPP_ROLLING_WINDOW)` per team & per opp; non-null-only aggregation | 5×9 |
| **Sleeper `started%` / `owned%`** | feature | Crowd probability-of-start / role-clarity + value prior | `GET api.sleeper.com/players/nfl/research/{type}/{season}/{week}` (no auth) | role/usage clarity, week-1; all pos | In-game-injury discriminator (Dobbins 89%→11% at W2) — **PASSED** | 5×8 |
| **Closing implied team total = "market" expert** | expert | Missing Vegas baseline; clean 2nd DST expert | `build_implied_team_total_lookup` (already in-source) | benchmark breadth (DST/team) | Game-keyed (no W-vs-W-1); implied vs realized separation regression | 3×8 |

### Tier 2 — Medium (real signal, but coverage gap and/or benchmark-flat risk)

| Source | Track | What it adds | Free access path | Gap / position | Leakage audit | ROI×Feas |
|---|---|---|---|---|---|---|
| **Team PROE** (standalone offense pass-tendency prior) | feature | `mean(pass)−mean(xpass)` team prior; volume-scaling axis | `pbp_data` + `PBP_PROE_COLS` (zero fetch) | game-script; QB/WR/TE/RB | `shift(1).rolling(5)` per `recent_team`; non-null-only | 4×9 |
| **Opponent pass-rush pressure** (qb_hit/sack rate per dropback) | both | `qb_hit` genuinely new; dropback-normalized opp pressure | `pbp_data` + `PBP_PRESSURE_COLS` (zero fetch) | richer opp adj; QB/RB matchup + DST | `shift(1).rolling(5)` per defense; positive control on elite front | 4×9 |
| **Opponent EPA-allowed-by-position** (per-play DvP) | feature | Format-agnostic per-play DvP replacing TD-noisy pts-allowed rank | `pbp_data` + `PBP_EPA_COLS` + reuse `roster_pos` (loader.py:409) | richer opp adj; WR/RB/TE | `shift(1).rolling` per `(defteam, faced-pos)`; **discrimination audit** (WR vs RB split must separate) | 3×9 |
| **NGS rushing RYOE / stacked-box** (RB) | feature | Tracking-talent prior net of blocking; box-rate game-script proxy | `load_nextgen_stats(stat_type="rushing")` (gsis-keyed) | role/usage; **RB (served NN)** | Drop `week==0` summary; `shift(1)` history + season+1 prior | 5×7 |
| **NGS passing CPOE / time-to-throw** (QB) | feature | Accuracy-net-of-difficulty + aggressiveness | `load_nextgen_stats(stat_type="passing")` (gsis-keyed) | QB quality/efficiency | Drop `week==0`; history-branch + season+1 only | 4×8 |
| **Injury-TYPE (body-part) severity prior** | feature | Soft-tissue/concussion vs finger/illness bucket the pipeline discards | Already-cached `load_injuries` (extend the inj_agg groupby, **no new fetch**) | returning-player over-prediction bias | Inherits the shipped `game_status` merge semantics; transition-row check | 4×8 |
| **PFR advanced stats** (QB pressure / WR-TE drops, broken-tackles) | feature | Manual charting axes the project lacks | `load_pfr_advstats(stat_type, summary_level="week")` + reuse pfr_id↔gsis crosswalk | QB pressure / WR-TE hands | Game-keyed; history + season+1; `2018+` floor | 4×8 |

### Tier 3 — Speculative (kept for context; build only behind a tracked subgroup gate)

| Source | Track | What it adds | Free access path | Gap / position | Leakage audit | ROI×Feas |
|---|---|---|---|---|---|---|
| Intended air-yards / aDOT-by-zone (team pass-shape) | feature | Team aDOT, deep-rate, air-yards HHI + corrected air-share denom | `pbp_data` (`air_yards`/`pass_length`), 2006+ | #1210 environment; WR/TE | `shift(1).rolling(5)` team & player | 4×9 |
| Schedule-distance strength-of-schedule | feature | Cumulative SOS-faced normalizer + upcoming-opp prior | No fetch (schedules + box scores already pulled) | richer opp adj; all pos | `shift(1)` per defense; cumulative excludes week-W | 4×9 |
| FantasyPros ECR + sd/best/worst (features) | feature | News-aware value prior + boom/bust **width** | `load_ff_rankings(type="all")` (DynastyProcess) | week-1 + #1210 width; all pos | **TNF leakage edge** (Fri scrape post-TNF); strict scrape-instant < first kickoff | 6×6→4 |
| NGS receiving xYAC + air-yards share (WR/TE) | feature | Tracking YAC skill + true alpha-role air-share | `load_nextgen_stats(stat_type="receiving")` | player-side of #1210 | Drop `week==0`; history + season+1; `2017+` for alpha cols | 5×7 |
| Participation man/zone/blitz scheme (defense) | feature | Per-defense coverage **scheme** (cause, vs NGS outcome) | `load_participation` (2016–2025; man/zone usable 2018+) | #1210 scheme; WR | `shift(1).rolling` per defense; **season-batch FTN feed — can't self-update live** | 5×7 |
| PFR Advanced Defense coverage (opp matchup) | feature | Per-defender passer-rating/comp%/yds-per-tgt **allowed** | `load_pfr_advstats(stat_type="def", week)` (native `team` col) | #1210 (team proxy); WR | target-weighted mean → `shift(1).rolling(5)`; `2018+` floor | 5×7 |
| FantasyPros ECR (expert baseline) | expert | Consensus-of-~100 ranking baseline | `load_ff_rankings(type="all")` | benchmark breadth | Fri-scrape pre-kickoff; **rank-only** (no MAE) → needs ranking-only harness branch | 5×7 |
| Open-Meteo archive weather (precip/humidity/gusts/snow) | feature | Genuinely-new weather; fills 33% schedule-weather hole | `archive-api.open-meteo.com/v1/archive` (no key); serving = `/v1/forecast` | game context; DST/K/bad-weather | Dome-neutralize + kickoff-hour + hole-fill agreement (r>0.9) | 3×7 |
| DFS salary (DK/FD) market composite | both | `salary_implied_rank` + `salary_delta_wow` ascension flag | RotoGuru `fyday.pl` (no auth) / `nfldfs` | week-1 + ascension; all pos | name-key `merge_name` bridge; co-located post-game points **backfill trap**; ~2014 floor | 4×6 |
| ESPN historical weekly projections | expert | 3rd decomposable expert house (raw stats) | `lm-api-reads.fantasy.espn.com/.../players?view=kona_player_info` | benchmark breadth | **Provenance uncertifiable** (no freeze flag); retention contradicted | 5×4 |

---

## Per-source scorecards

### Tier 1

#### 1. Team neutral-script pace (sec/play + neutral plays/game) — `[feature]` · ROI 5 · Feas 9

- **Data:** per-`(posteam, season, week)` (a) sec/play = mean negative diff of `game_seconds_remaining` over a team's consecutive offensive snaps; (b) neutral plays/game = count of run/pass plays — both computed **only on neutral-script plays** (`wp`/`vegas_wp` ∈ [0.2, 0.8] **and** `|score_differential| ≤ 8`) to strip hurry-up / clock-kill.
- **Granularity:** team-level `(recent_team, season, week)`, per-game; broadcast to every skill player via a team-keyed left-merge.
- **Historical coverage:** **full 2012–2025.** `game_seconds_remaining`/`score_differential` core since 1999; `wp`/`vegas_wp` modeled 2000+. A real depth edge over NGS (2016+)/participation (2016+)/FTN (2022+). (Note: the single 2024-game `vegas_wp` NA bug was nflfastR **PR #500**, not #503; the `wp`+`score_differential` co-filter is Vegas-line-independent anyway.)
- **Integration seam (fires a retrain):**
  1. `src/data/nfl_source.py` — add a `PBP_PACE_COLS` tuple (`game_seconds_remaining`, `half_seconds_remaining`, `wp`, `vegas_wp`, `score_differential`, `posteam`, `play_type`, `season`, `season_type`, `week`).
  2. **NEW** `src/data/pace_pbp.py` — mirror `src/data/redzone_pbp.py` (schema-gated cache via `_seasons_cache_signature`, per-year try/except, `play_type ∈ {pass, run}` filter, `_aggregate_one_season`).
  3. `src/data/loader.py::load_raw_data` — add `_fetch_pace` to the `ThreadPoolExecutor(max_workers=10)` (loader.py:378) + a team-keyed left-merge on `(recent_team, season, week)`.
  4. `src/features/engineer.py` — `shift(1).rolling(5)` of the team pace (mirror `target_share_L{w}` at engineer.py:309, but team-grouped).
  5. `src/{pos}/config.py` — add to `_INCLUDE_FEATURES['contextual']` (an `ATTN_STATIC` category) for QB/RB/WR/TE; **enumerate** in DST/K `_CONTEXTUAL_FEATURES`.
  6. Update `tests/{pos}/conftest.py` fixtures + whitelist tests.
- **Named alignment audit:** (a) **face-validity ranking** — fast offenses (PHI/MIA/BUF/recent BAL) top the sec/play (lowest) ranking, clock-control units (recent ATL/TEN) at the bottom; a top-vs-bottom mismatch ⇒ sign error in the `game_seconds_remaining` diff. (b) **lag control** — assert the production feature == `shift(1).rolling(5)` of team-game pace (a known week-1 row must be the prior-season carry, not realized week-1 pace). (c) cross-check neutral-play count vs a published tempo source (RBSDM situation-neutral pace) for 3–4 teams.
- **Stop-rule check:** PASS. Zero pace features exist (grep hits were namespace/comment noise). Distinct from draft-capital (player cold-start), the disabled dual-branch opp-def attention (defensive matchup), and the rejected role-inheritance token. **Sharp guardrail:** the season-rolled value is **static-branch-only** — routing it into `attn_history_stats` would violate the "no windowed aggregate in the raw-per-game history" rule (the same class as the rejected `snap_pct` expanding-mean). A *raw* per-game team sec/play could legitimately go to `attn_history_stats`, but that is a different feature — keep the spec to the lagged static contextual slot.

#### 2. PBP game-script / PROE bundle (PROE, neutral sec/play, pass rate, plays/game, early-down pass rate, no-huddle) — `[feature]` · ROI 5 · Feas 9

- **Data:** per-`(team, season, week)` tempo/tendency from the `load_pbp` frame already pulled: PROE (`mean(pass_oe)` or `mean(pass) − mean(xpass)`), neutral sec/play, plays/game, early-down (1st/2nd) pass rate, `no_huddle` rate — each with an **opponent-allowed mirror** and an own-team prior-season prior. `build_nflfastR_pbp` already runs `add_xpass`, so the public release ships `xpass`/`pass_oe` populated (no user-side recompute).
- **Granularity:** team-level `(team, season, week)` + opponent mirror; joins via `recent_team` and `opponent_team` like the existing `opp_def_*_L5` block. Position-agnostic (all six).
- **Historical coverage:** **full 2012–2025, no depth wall.** `xpass`/`pass_oe` return NA only pre-2006 (scramble marking began 2006). No backfill leak (deterministic PBP aggregates, not box-score backfills).
- **Integration seam (fires a retrain):** `PBP_PACE_COLS` + `pbp_data()` in `nfl_source.py`; **NEW** `src/data/pbp_pace.py` cloning `redzone_pbp.py`; `_fetch_pace` in the loader ThreadPool + a left-merge on `(recent_team, season, week)` *and* a second merge on `(opponent_team, season, week)` for the allowed-mirror; the opponent-allowed rolling + prior-season aggregation in `engineer.py` alongside `_build_defense_matchup_features` (engineer.py:373), reusing `OPP_ROLLING_WINDOW` + `TEAM_CODE_NORMALIZATION`; whitelist into a `matchup`/`contextual` subkey (eligible for `attn_static` via `DEFAULT_ATTN_STATIC_CATEGORIES`).
- **Named alignment audit:** assert every rolling column == `groupby([team_key, 'season'])[raw].transform(lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean())` for **both** own-team and opponent keys (the accepted `_build_defense_matchup_features` template); positive control on one `(team, week-W)` row (PROE excludes week-W plays); week-1 fills from the prior-season prior (not silent-0); apply `TEAM_CODE_NORMALIZATION` (OAK/SD/STL) on both `posteam` and the opponent join key before groupby.
- **Stop-rule check:** PASS. The "no rolling/EWMA in `ATTN_STATIC_FEATURES`" rule targets the *player's own* windowed history; these are **team/opponent prior-games** rolling values, the same class as the already-shipped `opp_def_sacks_L5` living in the static `matchup` category. Not the dual-branch opp-def **attention** (this is static/Ridge/LGBM columns). **Caveat — benchmark-flat:** PROE/pace mostly re-expresses game-script the served models get from `implied_team_total`, `opp_def_pts_allowed_L5`, snap share, targets, and `ff_opportunity`; A/B with a game-script subgroup lens, not overall MAE.

> **Tier-1 overlap note:** Candidates 1, 2, and the Tier-2 standalone PROE are the **same PBP-derived family**. Build them as **one `pbp_pace.py` module** emitting `{sec/play, neutral plays, PROE, no-huddle, plays/game}` in a single pass — do not ship three separate fetch wrappers.

#### 3. Sleeper market `started%` / `owned%` — `[feature]` · ROI 5 · Feas 8

- **Data:** per-player weekly market-consensus scalars from Sleeper's **research** endpoint: `owned` (% leagues rostering) and `started` (% leagues starting). `started%` is a crowd pre-kickoff probability-of-start / role-clarity signal; `owned%` is a continuous value prior richer than `depth_chart_rank`. Verified live: `regular/2023/5` (823 players), `regular/2020/1` (459), all HTTP 200, no auth.
- **Granularity:** per-`(player, season, week)`; offense bridges `sleeper_id → gsis_id` (6146 players carry both), DST is team-keyed (LAR→LA fixup), K skipped for `started%` (rostered starters span 60.6–94.7, but conservatively skip).
- **Historical coverage:** **2020–2025 only** (verified: 2012/2015/2017/2018/2019 return literal `null`; 2020 W1 is first populated). 8 of 14 train seasons NaN → train-mean fill (same shape as the accepted NGS POC). 2025 holdout fully available. The claimed preseason `pre` prior is **illusory** — `pre/{season}/1` is byte-identical to `regular/{season}/1` across all six covered seasons.
- **Integration seam (fires a retrain) — exact `load_qbr_weekly` clone:** `sleeper_research(season_type, season, week)` wrapper in `nfl_source.py` (urllib GET like `sleeper_loader._default_reader`); `load_sleeper_research(seasons)` in `external_sources.py` mirroring `load_qbr_weekly` (cache `sleeper_research_{sig}.parquet`, bridge mirroring `bridge_qbr_to_gsis`, dedupe + deterministic pre-sort, return `[player_id, season, week, sleeper_owned_pct, sleeper_started_pct]`); `_fetch_sleeper` in the loader ThreadPool + a left-merge on `(player_id, season, week)` — **NaN→missing-indicator + median (not 0-fill)**, since 0% own is meaningful-low and clashes with pre-2020 missing rows; add to `_INCLUDE_FEATURES` + `attn_static_features` (current-week market state = non-temporal).
- **Named alignment audit (DONE, PASSED):** in-game-injury pre/post discriminator — Dobbins `started=88.9%` at `regular/2023/1` then `11.5%` at W2; Rodgers `40.3%`→`3.6%`. Both publicly done for the season by Sunday/Monday night yet W1 stays HIGH and the crash lands at W2, proving `regular/{season}/{week}` is the **pre-W1-kickoff** consensus. Re-run on 2–3 cases per covered season; for serving, pull at a **fixed pre-lock timestamp** (the live endpoint serves a moving snapshot for the in-progress week).
- **Stop-rule check:** PASS. A raw **external current-week scalar** in the static branch — not the rejected `ATTN_HISTORY_STATS` expanding-mean inheritance token, not draft-capital. **WATCH:** `owned%` is a value prior whose gain may concentrate in LightGBM/RB (draft-capital flat risk); judge `started%` (the net-new role axis) by an RB/spot-starter **subgroup metric** across ≥2 seeds, not overall MAE.

#### 4. Closing implied team total = "market" expert — `[expert]` · ROI 3 · Feas 8

- **Data:** the closing Vegas line as a pre-kickoff "market" projection — a 3rd `ExpertSource`. `implied_opp_total` feeds the DST points-allowed tier bonus (`_PTS_ALLOWED_TIERS`/`_pts_allowed_to_bonus` in `src/dst/targets.py`). A per-player baseline (team-total × usage share) is a **v2**, not what ships first.
- **Granularity:** game/team-level per-week; cleanest as a **team/DST** expert (the DST model's `player_id` IS the team abbrev).
- **Historical coverage:** **strong** — `spread_line`/`total_line` are PFR **closing** lines (nflfastR 2.0.0), dense well before 2012; `implied_team_total` is a live production feature in 5/6 configs. The sparse schedule columns are the auxiliary `*_odds`/moneyline fields, NOT these two.
- **Integration seam (analysis-only — NO retrain):** **NEW** `src/analysis/market_loader.py` mirroring `sleeper_loader.py` — read the cached schedules parquet, call `build_implied_team_total_lookup` (`src/shared/weather_features.py`), emit `[player_id(=team for DST), season, week, expert_pred_total]` where `expert_pred_total = _pts_allowed_to_bonus(implied_opp_total) + fitted_constant` for the non-market linear component; add `ExpertSource(name="market", ...)` to `analysis_expert_comparison.py::_build_experts` (skipped except DST for v1); regenerate `src/serving/comparison_experts.json` via `build_comparison_summary.py` (reuses `paired_bootstrap_metric_ci` + `diebold_mariano_test` unchanged). **Integration is slightly more than "add a frozenset":** `build_summary` hardcodes a two-expert output schema (`{nflcom, rotowire}` + `experts_meta` + the serving render layer), so adding `market` touches the output schema and frontend (still no retrain).
- **Named alignment audit:** (1) game-key sanity — exactly 2 team-rows/game, `implied_team_total + implied_opp_total == total_line` (the home/away sign split is the only failure point; already unit-tested + audited #414/#598/#849). (2) realized-vs-implied separation — regress `implied_opp_total` against actual opponent points on a holdout; a near-perfect fit ⇒ the line is secretly realized; a loose-but-calibrated fit (slope≈1, wide residual) confirms a forecast. (3) closing-line provenance spot-check vs PFR's published number.
- **Stop-rule check:** PASS (analysis-only, no feature/loss/training change, no `fantasy_points` target). **Honest ROI cap:** the DST expert is a **floor** — `_pts_allowed_to_bonus` emits only ~7 discrete tier values and folds the entire high-variance linear DST component (def TDs/turnovers/sacks) into one fitted constant, so DST units cluster in the same tier with near-zero ranking resolution. It is a measurement/reporting win, **not** a served-model improvement.

### Tier 2

#### 5. Team PROE (standalone) — `[feature]` · ROI 4 · Feas 9
- **Data/coverage:** `mean(pass_oe)/100` per `(posteam, season, week)`; full 2012–2025 (xpass valid 2006+; 2012 = 76.0% xpass / 73.9% `pass_oe` non-null, reproduced at source). **Seam:** `PBP_PROE_COLS` + new `team_proe_pbp.py` (REG-only, `groupby(['posteam','season','week'])` → `team_proe = mean(pass_oe over non-null)`); `_fetch_team_proe` + left-merge on `(recent_team, season, week)`; `shift(1).rolling(5)` → `team_proe_trailing` in `engineer.py`; add to QB/RB/WR/TE `_INCLUDE_FEATURES` + `attn_static_features`.
- **Alignment audit:** trivial merge key (PBP `posteam`/`week` authoritative); (A) validity — trailing PROE for a known aerial unit ranks top-quartile vs a run-lean one, week≥7; (B) leakage — `corr(trailing_proe[W], realized_pass_oe[W])` moderate-not-1.0; season-opener value NaN/0.
- **Stop-rule:** PASS — season-to-date team rate = sanctioned static reach #1; do **not** route into `attn_history_stats`. **ROI haircut (→4):** all four skill positions already feed raw `team_pass_attempts`/`team_rush_attempts` into the attention **history** sequence (QB's config comment says these "carry the run/pass game-script"), so raw pass-tendency is *not* new — only PROE's `xpass` expectation-adjustment is, a narrow second-order increment squarely in the benchmark-flat zone.

#### 6. Opponent pass-rush pressure (qb_hit/sack rate per dropback) — `[both]` · ROI 4 · Feas 9
- **Data/coverage:** per-`(defteam, season, week)` `sum(sack)/sum(qb_dropback)` and `sum(qb_hit)/sum(qb_dropback)`; full 2012–2025 (qb_hit/qb_dropback reliable from 2011, base PBP). `qb_dropback` **includes** scrambles (correct standard denominator). **Seam:** `PBP_PRESSURE_COLS` + new `defense_pressure.py`; `_fetch` + team-keyed left-merge on `opponent_team` (QB/RB matchup) and `recent_team` (DST input); `shift(1).rolling(5)` per defense near `_build_defense_matchup_features`; QB `_INCLUDE_FEATURES['defense']` + `attn_static_features`, DST `_CONTEXTUAL_FEATURES`.
- **Alignment audit:** trailing `qb_hit_rate_L5`/`sack_rate_generated_L5` for a known elite front (2023 BAL) ranks top-5; rolling value at W uses only weeks < W; `sum(qb_dropback)` denominator sanity; confirm `qb_hit` rate is **not** ~1:1 collinear with the existing `opp_def_sacks_L5`.
- **Stop-rule:** PASS — a **new flat input** to Ridge+LGBM+NN-static, not the rejected opp-def *attention* branch (which re-routed the same six existing stats through a 2nd `AttentionPool`). **ROI haircut (→4):** `qb_hit` is genuinely new (zero grep hits), but the sack-rate half overlaps three shipped features (QB `sack_rate_L3`, DST `opp_qb_sack_rate_L5`, DST `opp_sacks_allowed_L5`), and QB/RB already carry a full opponent-defense matchup block — the project's own dual-branch-disabled prior signals low headroom.

#### 7. Opponent EPA-allowed-by-position (per-play DvP) — `[feature]` · ROI 3 · Feas 9
- **Data/coverage:** per-`(defteam, faced-pos, season, week)` mean `epa` allowed on plays targeting/carried-by that position, joined via `roster_pos` (loader.py:409, already built). Full 2012–2025 (`epa` since 1999). Replaces the TD-noise-laden `opp_fantasy_pts_allowed_to_pos`/`opp_def_rank_vs_pos`. **Seam:** `PBP_EPA_COLS` + new `reconstruct_epa_dvp_from_pbp` (clone `redzone_pbp`, group by `defteam`+faced-pos); `_build_epa_dvp_features(df)` in `engineer.py` paralleling `_build_defense_matchup_features`; add to the `matchup` list in WR/RB/TE/QB `_INCLUDE_FEATURES` (flows to `attn_static` via `DEFAULT_ATTN_STATIC_CATEGORIES`).
- **Alignment audit:** (1) leakage/semantics — every value at `(D, week W)` from D's games < W; `defteam` per play == opponent of `posteam`; REG-only. (2) **discrimination (binding)** — known shadow-corner defenses must show EPA-allowed-to-WR diverging from EPA-allowed-to-RB/TE, AND EPA-allowed-to-WR must beat `opp_def_rank_vs_pos` on next-game Spearman. **If the buckets don't separate, keep=false.**
- **Stop-rule:** PASS — a static scalar into the matchup whitelist (not the disabled opp-def attention; not the `opp_*_pts_allowed` reject — that was format-bound fantasy-points). **ROI capped at 3:** external DvP literature is damning (WR/TE DvP is *more* volatile and *less* predictive; EPA/play is "extremely noisy, needs heavy regularization"), the raw L5 mean is the un-regularized form the literature warns against, RB already **dropped** the points-allowed DvP (VIF 193), and LightGBM (WR/RB's best model) is collinearity-robust — textbook benchmark-flat. It does **not** close #1210 (still a team-level OUTCOME aggregate, not CB coverage).

#### 8. NGS rushing RYOE / stacked-box (RB) — `[feature]` · ROI 5 · Feas 7
- **Data/coverage:** `rush_yards_over_expected(_per_att)`, `efficiency`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los` per `(player_gsis_id, season, week)`, RB-only. **2016+** floor (verified `ValueError` for 2015); the 10-carry minimum covers only **42.4%** of `≥1`-carry / 63.3% of `≥5`-carry RB-weeks (572/904 in 2023, reproduced) — the missing ~57% are the committee/spot-start backs. NGS week-W rush_yards match the weekly box at **99.8%** (not stale-by-1).
- **Seam (exact ff_opportunity mirror):** `nextgen_stats(seasons, stat_type)` wrapper; `load_ngs_rushing` + `NGS_RUSH_FEATURE_COLUMNS` (filter `week>0` & `season_type=='REG'`, rename `player_gsis_id→player_id`, dedup, cache); append per-game cols to `EXTERNAL_PRIOR_STATS` (engineer.py:246 auto-builds `prior_season_mean_*`); `_fetch_ngs_rush` + left-merge (NaN, not 0); RB config — raw per-game tokens (`rush_yards_over_expected`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los`, `efficiency`) → `attn_history_stats`, `prior_season_mean_*` → `_INCLUDE_FEATURES['prior_season']`; box-rate **not** in `attn_static`.
- **Alignment audit:** drop `week==0` summary rows; per-game gsis/team_abbr join check (NGS week-W `rush_yards_over_expected` == single game, not season total); Ridge-identity tell (`expect_ridge_identical=False`) to confirm the merge moved inputs.
- **Stop-rule:** PASS — raw per-game NGS (legit history token, not an averaged-average), prior-season MEAN (static-legal). **WATCH:** shares draft-capital's flat *failure mode* (RB FP-MAE is volume-dominated; `expected_rush_yards` is 0.84-correlated with the ingested `ff_opp rush_yards_gained_exp`), but it reaches the **RB attn NN (the served best model)** so it's not pre-doomed. Judge on a high-volume-back subgroup (RMSE/bias, ≥3 seeds), never overall MAE.

#### 9. NGS passing CPOE / time-to-throw / aggressiveness (QB) — `[feature]` · ROI 4 · Feas 8
- **Data/coverage:** `completion_percentage_above_expectation` (CPOE), `avg_time_to_throw`, `aggressiveness`, `avg_air_yards_differential`, `avg_intended_air_yards` per `(player_gsis_id, season, week)`, QB-only, gsis-native (no ESPN bridge). **2016+** floor (most fields 2017+). NGS-tracking CPOE is r²=0.88 to the PBP-derivable version — orthogonal increment is modest.
- **Seam (ADR-0016 QBR seam):** `nextgen_stats` wrapper; `load_ngs_passing` + `NGS_PASS_FEATURE_COLUMNS` (filter `week>0` & REG); append to `EXTERNAL_PRIOR_STATS`; `_fetch_ngs` + left-merge (copy loader.py:568-576); QB config — per-game cols → `attn_history_stats` (next to `qbr_total`), `prior_season_mean_*` → `_INCLUDE_FEATURES['prior_season']`.
- **Alignment audit:** assert NO NGS column in `QB_ATTN_STATIC_FEATURES` (`tests/test_attn_static_columns.py`); leakage A/B — `n_prior_games==0` row has all-zero/masked NGS tokens, `prior_season_mean_cpoe` at season S == mean over S-1 only; positive control — drop `week==0`, assert no "game" has attempts == full-season attempts; gsis join ~100%.
- **Stop-rule:** PASS — per-game → history, aggregates → `prior_season_mean` (static-legal). **ROI haircut (→4):** the served best QB model is the **plain MultiHeadNet** (6.514), and the rich per-game CPOE-trajectory tokens feed **only** the attention variants — they reach the served plain MLP **only** through the diluted `prior_season_mean_*` static channel, which prior-season efficiency + realized `passing_yards` already partly encode. High benchmark-flat risk; gate on a served-model A/B, not NN-MAE.

#### 10. Injury-TYPE (body-part) severity prior — `[feature]` · ROI 4 · Feas 8
- **Data/coverage:** the free-text body-part strings the pipeline **discards** at the inj_agg groupby (`report_primary_injury`, `practice_primary_injury`, ...) — mapped to a curated ordinal severity bucket (soft-tissue/concussion = high; finger/illness = low) + a lower-vs-upper-body axis. Full 2012–2025 (`load_injuries` since 2009; `practice_primary_injury` ~99–100% populated every season). **No new fetch** — the frame is already cached. (Caveat: a 2014 injury-reporting-regime change weakens body-part *coding consistency* for 2012–2013, even where fill is present.)
- **Seam (extend the existing merge):** `loader.py:480-487` inj_agg groupby — add per-`(gsis_id,season,week)` aggregates of the type strings (severity-bucket max / one-hot) **before** they are dropped; neutral fill in the `engineer.py` "Injury status" defaults block; curated free-text→ordinal dict (small new module); add to `_INCLUDE_FEATURES['contextual']` (auto-flows to `attn_static` via `DEFAULT_ATTN_STATIC_CATEGORIES`); K/DST explicit allowlists.
- **Alignment audit:** `#595`-style — for players flipping `Out↔active` between W-1 and W, confirm `report_primary_injury` at `(player, season, W)` describes the injury carried *into* W (cross-check vs whether they actually played); positive control on a multi-week soft-tissue absence. `date_modified` has **0 matches** in `src/`, so the merge keys only on `(gsis_id, season, week)` — leakage-safe **by inheritance** from the already-shipped `game_status`.
- **Stop-rule:** PASS — a distinct field never ablated (the ablated set was `{game_status, practice_status, days_rest, is_returning_from_absence}`), and a non-temporal static descriptor (not an `ATTN_HISTORY_STATS` expanding-mean). **ROI capped:** `game_status` is 96.8% constant in the 2025 test split (Out/Doubtful self-eliminate), and the returning-player issue (#623) is **bias-not-MAE**; even the STATUS features were never shown to move the returning subgroup. Clone `ablate_injury_features.py` (its `INJURY_FEATURES` frozenset + returning/questionable subgroups), judge on **subgroup bias** across ≥2–3 seeds.

#### 11. PFR advanced stats (QB pressure / WR-TE drops, broken-tackles) — `[feature]` · ROI 4 · Feas 8
- **Data/coverage:** `load_pfr_advstats(stat_type, summary_level="week")` — `pass` (QB: `times_pressured(_pct)`, `times_blitzed/hurried/hit`, `passing_bad_throws(_pct)`, `def_times_*`), `rec` (`receiving_drop(_pct)`, `receiving_broken_tackles`, `receiving_rat`). Live-verified: 0 dup keys on `(pfr_player_id, season, week)`; pfr_id↔gsis bridge resolves **100.0%/99.9%** of REG rows. **2018+** hard floor (`ValueError` for 2017; 6/14 train seasons NaN). **Correction:** `receiving_yac` does **not** exist in the weekly rec table (NGS carries YAC; PFR weekly does not).
- **Seam:** `pfr_advstats(seasons, stat_type, summary_level)` in `nfl_source.py`; `load_pfr_pass`/`load_pfr_rec` mirroring `load_qbr_weekly` (reuse the pfr_id↔gsis crosswalk the snap-count merge already runs at loader.py, `right_on='pfr_id'` + sort/keep=first); extend `EXTERNAL_PRIOR_STATS`; `_fetch_pfr_*` + left-merge; QB/WR/TE config — raw per-game → `attn_history_stats`, prior-season means → static (**never** raw current-week to static). Triggers a QB/WR/TE retrain.
- **Alignment audit:** game-keyed week (1–22 with `game_type` REG/WC/DIV/CON/SB) — a post-game table, so safe via history-branch + season+1 (the established ff_opp/QBR routing); assert no current-game PFR column in `_INCLUDE_FEATURES` static; bridge positive control (≥99% REG resolution); REG filter + dedup before merge.
- **Stop-rule:** PASS — in-game charting (not static draft-capital), player-level history tokens (not the disabled opp-def attention). **WATCH:** the draft-capital benchmark-flat precedent is directly analogous (real signal, washed out, helped only non-best LightGBM); PFR has better odds (per-game signal into the WR/TE attention history branch, ~100% coverage, net-new QB usage-quality) but the wash-out risk is real — judge on QB pressure-cohort bias / WR-TE drop-boom RMSE+corr, ≥3 seeds.

---

## Top pick — full spec

**Pick: Team game-script / pace context from already-cached PBP (the unified `pbp_pace.py` family).** Final ROI 5 × Feas 9 — the highest product among verified survivors, and the only Tier-1 feature that is **zero-new-fetch**, **full 2012–2025 depth**, and fills an **explicitly-named** documented gap (game-script/pace). Ship candidates 1 + 2 + the standalone PROE as one module.

> **Why this over Sleeper `started%` (also Tier 1):** Sleeper has marginally higher novelty but is gated by 2020–2025-only coverage (8/14 seasons NaN) **and** a network-boundary fetch with a moving in-progress-week snapshot. PBP-pace is offline, deterministic, and full-depth — strictly lower-risk to build and validate, and the right first PR.

This is a **feature-track** change → it **fires a retrain** via `refresh-splits` (touches `src/data/` + `src/features/engineer.py`). DESIGN ONLY below.

### Ingestion (zero new network fetch)
1. **`src/data/nfl_source.py`** — add one tuple:
   `PBP_PACE_COLS = ('season', 'season_type', 'week', 'posteam', 'play_type', 'game_seconds_remaining', 'half_seconds_remaining', 'wp', 'vegas_wp', 'score_differential', 'pass', 'xpass', 'pass_oe', 'no_huddle', 'down')`.
   No new fetch function — reuse `pbp_data(seasons, PBP_PACE_COLS)` (the loader `redzone_pbp.py`/`k/data.py` already call).
2. **NEW `src/data/pbp_pace.py`** — mirror `src/data/redzone_pbp.py` end-to-end:
   - `reconstruct_pace_from_pbp(seasons, cache_dir)` with a `PACE_FEATURE_COLUMNS` tuple as the single source of truth, schema-gated cache (`_seasons_cache_signature`, `_cached_*_is_current`), per-year `try/except`, `atomic_write_parquet`.
   - `_aggregate_one_season`: REG-only; **filter `play_type ∈ {pass, run}` before any seconds diff** (else timeouts/penalties/kneels/spikes distort sec/play — mirrors `redzone_pbp`'s play_type guard); compute per `(posteam, season, week)`:
     - **neutral mask** = `wp.between(0.2, 0.8) | vegas_wp.between(0.2, 0.8)` **AND** `score_differential.abs() <= 8`,
     - `team_sec_per_play` = mean negative diff of `game_seconds_remaining` over the team's consecutive *neutral* offensive snaps (sort by play within game),
     - `team_neutral_plays` = count of neutral run/pass plays,
     - `team_proe` = `mean(pass_oe over non-null rows)` (NOT `fillna(0)` — `xpass` is NA on non-valid plays),
     - `team_no_huddle_rate`, `team_plays_per_game`, `team_early_down_pass_rate` (down ∈ {1,2}).
   - rename `posteam → recent_team`.

### `external_sources.py` merge-key plan
- This is a **PBP-aggregate**, not a weekly-player frame, so it follows the **`redzone_pbp` path, not the `external_sources.load_ff_opportunity` path** (the denominator is play-grain). No `external_sources.py` loader needed.
- Merge key: **`(recent_team, season, week)`** (team-level), distinct from the player-keyed `(player_id, season, week)` ff_opp merge. For the opponent-allowed mirror, a **second** left-merge on `(opponent_team, season, week)`.
- NaN/zero-backfill: single-source the column list in a `PACE_FEATURE_COLUMNS` constant (like `RZ_PBP_FEATURE_COLUMNS`); leave pre-window rows NaN at loader time (the engineer lag handles week-1).

### `loader.py` wiring
- `src/data/loader.py::load_raw_data` — add `_fetch_pace()` to the `ThreadPoolExecutor(max_workers=10)` block (loader.py:378, alongside `_fetch_redzone`), then a team-keyed left-merge on `(recent_team, season, week)` (parallel to the redzone merge) + the opponent-mirror merge on `(opponent_team, season, week)`.

### `engineer.py` aggregation
- Add the lag next to the `team_totals` share block (engineer.py:309) and the opponent-allowed rolling next to `_build_defense_matchup_features` (engineer.py:373):
  - **own-team:** `team_pace_trailing` = `groupby(['recent_team','season'])[raw].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())` for each of `{sec_per_play, neutral_plays, proe, no_huddle_rate, plays_per_game}`,
  - **opponent-allowed mirror:** same transform grouped by `(opponent_team, season)`, reusing `OPP_ROLLING_WINDOW` + `TEAM_CODE_NORMALIZATION`,
  - optional `prior_season_mean_team_pace_*` (season+1 shift) for the week-1 lever.

### `{pos}/config.py` whitelist additions
- **`attn_static` [non-temporal]:** add `team_pace_trailing_*` (the lagged season-to-date rates) to `_INCLUDE_FEATURES['contextual']` for **QB/RB/WR/TE** — `contextual` is an `ATTN_STATIC_CATEGORY`, so it auto-feeds the NN static branch (Ridge + LightGBM + NN-static). **Enumerate** the same names in DST/K `_CONTEXTUAL_FEATURES`.
- **`attn_history` [raw per-game]:** **do NOT add the rolled value here.** The lagged aggregate is windowed → static-branch-only (the rejected-snap_pct-expanding-mean class). A *raw* per-game team `sec_per_play` token would be eligible for `attn_history_stats`, but that is a **separate feature** — keep this PR to the static contextual slot.
- Update `tests/{pos}/conftest.py` + `tests/conftest.py` fixtures + the feature-whitelist/manifest snapshot tests.

### Named alignment audit (mirror `src/analysis/audit_depth_alignment.py`)
1. **Face-validity ranking (positive control):** build 2024 neutral-script pace; assert fast offenses (PHI/MIA/BUF/recent BAL) top the `sec_per_play` (lowest) and `neutral_plays` (highest) rankings and clock-control units (recent ATL/TEN) sit at the bottom. A top-vs-bottom mismatch ⇒ a sign error in the `game_seconds_remaining` diff or leaked-script contamination.
2. **Lag/leakage control:** assert the production `team_pace_trailing` == `shift(1).rolling(5)` of the team-game pace (no current-game value in any row); a known week-1 row must equal the prior-season carry (or NaN→prior-season prior), never the realized week-1 pace. Confirm `corr(team_pace_trailing[W], realized_pace[W])` is moderate-not-1.0.
3. **Source-semantics:** `game_seconds_remaining`/`score_differential` carry no week-mislabel (game-keyed, not snapshot-keyed like #595) — light check; assert `play_type` filter applied before any seconds diff.
4. **Independent cross-check:** neutral-play count vs RBSDM/Football-Outsiders situation-neutral pace for 3–4 teams reproduces a published tempo ordering.

### A/B spec (`src/tuning/ab_harness.py`)
- **Injector shape:** a **cfg mutator** that toggles the pace columns into the whitelist (works for all six positions), plus a **frame injector** for QB/RB/WR/TE (frame injection is QB/RB/WR/TE-only; K/DST `run(seed, config)` build their own splits → cfg-injection only). Mirror the `ab_opp_coverage_wr.py` matchup-key merge pattern with `expect_ridge_identical=False` (the merge must move inputs).
- **Positions:** QB, RB, WR, TE first (the volume positions). Add DST/K only if the skill-position screen is positive.
- **Seeds:** **3** for the first FP-MAE read (mean±std); bump to **5–8** (`--seeds`) if the delta lands inside the seed band. On CUDA the harness fans out the position×variant×seed grid; the GPU-default N=24 stacked path applies for any NN tuning, but a plain feature A/B uses the lean seed set.
- **Leakage-safe injector:** the frame injector must be pre-kickoff (the `expect_ridge_identical` sentinel won't catch a feature-side leak) — inject the *lagged* `team_pace_trailing`, never the realized current-game value.

### GATE metric (ship/no-ship)
- **Subgroup cohort:** **fast-offense / high-pace games** (top-quartile `team_pace_trailing`) and **volume-floor games** for WR/RB/TE — judge **direction of bias** on the subgroup (per the "subgroup error = bias, not MAE" lesson), per model, across ≥3 seeds, on `result["test_df"]` (`pred_{model}_total`).
- **Non-regression:** overall per-position MAE on the **served best model** must not regress beyond the seed band. **Ship only if** the feature both (a) moves the fast-offense subgroup in the right direction AND (b) reaches the **served best model** (QB plain-NN, RB/WR attn-NN/LGBM, TE plain-NN, K Ridge) — not just a non-best model. If it is benchmark-flat on the served model (the likely outcome — a second-order volume multiplier partly proxied by `implied_team_total` + realized team-volume history tokens), **do not ship**, per the project's "benchmark-flat ⇒ don't ship" rule.

---

## Appendix: paywalled / rejected

None of the survivors above are paywalled. The items below were investigated and **excluded** from the ranked cut — listed so a follow-up sweep dedupes them.

| Source | Track | Paywalled? | Why excluded |
|---|---|---|---|
| **PFF / ETR / 4for4 / FantasyData premium / paid odds feeds** | both | **Yes** | Out of scope by constraint. PFF is the only source that *directly* answers #1210 (CB-level coverage grades, separation/cushion, shadow/man-zone) — note it separately as the high-value paid option, but it is disqualified from the free cut. |
| No-vig de-juiced spread/total + vig-asymmetry (sharp-side) | feature | No | **Measured payload ≈ 0.** no-vig P(over) is a near-constant 0.500 (std 0.0104; corr −0.011 w/ total residual); the one live axis (spread-odds skew, +0.136) predicts team **margin**, two abstraction layers above a raw-stat target — draft-capital-style real-but-flat. |
| ESPN anytime-TD odds → implied TD prob | feature | No | Fails depth (no free 2012–2025 player-prop archive, conceded) **and** is architecturally inert: a serving-only column is zero-filled by `feature_build.py:96` since it was absent from training. Honest residual = a display-only UI rail. |
| ESPN FPI game predictor / win-prob | both | No | Team/game-level only (no per-player stats); ~collinear with the ingested Vegas spread (disguised dual-branch-opp-defense axis); depth ceilings at 2015; leakage-unproven. |
| FantasyPros ECR as a points-paired expert | expert | No | No multi-season weekly archive of *points* (only "latest" ordinal snapshots); can't enter the points-error/Diebold-Mariano panel. (The **rank-only** ECR baseline survives in Tier 3 — a different, ranking-metrics-only use.) |
| Referee/crew penalty-tendency prior | feature | No | Real-but-tiny (~1 penalty/game crew spread; PFF pegs penalty↔points corr +0.08 = noise); helps only flat-column models — benchmark-flat. Analysis-only A/B at most. |
| `load_draft_picks` / `load_combine` | feature | No | **The reverted draft-capital/combine features** (fixed-archive line 330): benchmark-flat at every position, benefit reaches only LightGBM, rookie effect MAE-invisible. Re-proposing as a "new loader" is the disguised-rejected-item failure mode. |
| `load_players` / `load_team_stats` / `load_stats` / `load_ffverse` | feature | No | Redundant with current ingestion (`ff_playerids` + `team_week_stats_release` + `player_stats` already pulled). |
| `load_trades` | feature | No | Fails pre-kickoff alignment; no new per-game signal. |
| Opponent EPA-allowed per dropback/carry (static defense) | feature | No | Collides with the repo's own 24-seed feature-selection screen, which flags the opponent-`defense`/`matchup` block (incl. the shipped EPA variant) as a **drop candidate** for the best models; fails the served-best-model bar. |
| Participation routes-run & box counts (production feature) | feature | No | Routes-run only exists 2023+ (FTN era) → can't train across 2012–2025; participation releases **once after the postseason** → all-NaN for the live slate (the #1069→#1076 "heavy data not serviceable in serving" class). |
| Stadium altitude / air-density (K) | feature | No | Lands on K, whose served best model (Ridge) has R²≈0 on the FG head it touches; ~3% of rows → overall-MAE-flat by construction. |
| Full officiating-crew tendencies (`load_officials`) | feature | No | Canonical PFF analysis: total-penalty↔points corr +0.08 (noise), explicit "don't use crew data" → near-certain benchmark-flat. |
| Stadium turf-subtype & retractable-roof open/closed | feature | No | `is_retractable_open` is leakage-unsafe (reversible game-time decision, degenerate=0 in 2024/2025); surface-subtype is a flat relative of already-pruned `is_grass`. |
| Depth-chart CHANGE / promotion-event signal | feature | No | Disguised version of BOTH the **shipped** `is_top_available`/`inherited_opportunity` (#1053/#1353) AND the **tested-rejected** `depth_shift` arm of `ab_qb_inheritance.py`; the repo's gap doc says the residual needs a starter-news feed (#1134), not this derivable delta. |
| Sleeper trending add/drop velocity | feature | No | Endpoint is current-only (no as-of/week param) → zero historical training rows → benchmark-flat by construction; leakage-fragile serving-only display. |
| Travel direction / timezone-shift / circadian | feature | No | Documented effect is **team-level ATS/margin**, not player raw-stats; concentrated in a tiny Thu/Mon/West-coast cohort → high benchmark-flat risk; neutral-site `stadium_id` corrupts travel distance for international games. (Buildable + leakage-safe, but a cheap subgroup-screened experiment at most.) |
| FTN play-action/motion/RPO/screen scheme rates | feature | No | The only team-tempo/scheme axis the project lacks, **but** `load_ftn_charting` is **2022+** only (3 train seasons, ~25% row coverage); persistent low-variance team priors → high-risk-of-flat. CC-BY-SA attribution required in served output. (Tier-3-adjacent; subgroup-gate only.) |

**Net recommendation:** open the first build PR on the **unified `pbp_pace.py` game-script/pace family** (Top pick), screened via `ab_harness` on a fast-offense subgroup against the served best model. If it ships, the next cheapest deep candidates are **Sleeper `started%`** (role-clarity, accept the 2020+ floor) and **NGS rushing RYOE for RB** (into the served RB attn NN). The **closing-implied-total "market" expert** is a parallel, no-retrain, analysis-only add that strengthens the benchmark narrative regardless of the feature outcomes.

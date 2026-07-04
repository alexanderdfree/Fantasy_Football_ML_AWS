<!-- Generated 2026-06-30 by the multi-agent new-sources research workflow (Opus-4.8 discovery -> adversarial >=2013-history + backfill verification -> rubric scoring). Read-only research; no model/loader code changed. [docs-only] -->

# Fantasy-Football External Data-Source Research — Synthesis Report

## ⚠️ Correction (2026-06-30, post-publication direct verification)

Before building the rank-1 recommendation, a direct scrape of the actual live pages overturned two of this report's conclusions. **Read this first — it changes which source to build.**

- **F1 FantasyPros is NOT a usable ≥2013 benchmark** (contra its 0.81 rank-1 score below). Its historical pages (`/nfl/projections/{pos}.php?year=YYYY&week=W`) return real historical projected *values* but a player pool **filtered to players still rostered in the current (2025–26) season, labeled with their current teams**. The 2013 wk1 QB page returns only **~9 QBs** (Rodgers→"PIT", Stafford→"LAR", Cousins, Keenum, Tyrod Taylor…) — the entire since-retired 2013 cohort (Brees, Manning, Rivers, Roethlisberger) is **absent**. The original skeptic spot-checked a single still-active player (Rodgers), saw a genuine "Sep 5, 2013" value, and missed the pool truncation. Scoring the model against this decimated, survivorship-biased pool would be meaningless. **FantasyPros is valid only as a *current-season* consensus benchmark — it does not serve the ≥2013 goal.**
- **F3 FFToday is the genuine ≥2013 archive and is promoted to THE primary new benchmark.** Its 2013 wk1 pages carry the **correct historical teams** (Calvin Johnson→DET, Dez Bryant→DAL, Demaryius Thomas→DEN, Wes Welker→DEN, Reggie Wayne→IND) and **include since-retired players** — a real, complete historical pool (a current-only page could not list retired players). Coverage: QB/RB/WR/TE/K (no 2013 DST); the table carries separate Team/Opp columns + per-position stat columns; archive floor 2010.

**Net:** the "new per-game expert benchmark" lever survives, but **FFToday, not FantasyPros, is the ≥2013 source to build.** This is a textbook "smoke one real cell before building" catch — the adversarial skeptic verified *a* 2013 value existed but not pool completeness. The build-not-buy feature findings (B1 CPOE / B2 PROE-pace / B3 opp-adj-EPA / E2 O-line continuity / H1 injury body-part text) and the headline coverage-data negative result are **unaffected**.

## 1. Executive summary

- **Eleven candidates clear the project's hard ≥2013-history + leakage/backfill gates and are accessible.** They split into two actionable families: (a) **two genuinely new, independent per-game EXPERT BENCHMARKS** that both reach 2013 and are leakage-clean — **F1 FantasyPros Expert Consensus** (`score 0.81`) and **F3 FFToday weekly archive** (`score 0.69`); and (b) a cluster of **zero-cost, build-not-buy 2013-safe FEATURES** derived from data we already ingest — **B1 CPOE**, **B2 PROE/pace**, **B3 opponent-adjusted EPA**, **E2 O-line continuity**, and **H1 injury body-part TEXT** (all `score 0.72` except H1, all leakage/backfill-clean).
- **Best pick per lever:** *Expert benchmark* → **F1 FantasyPros** (true multi-source consensus, the toughest ordering bar; F3 FFToday is the corroborating second source). *RB workload-derivative* → **B2 PROE/pace** (team pass/play-volume environment, the highest-confidence ordering re-sorter among the build-not-buy features; E2 O-line continuity is the orthogonal runner-up). *WR/TE coverage* → **no viable source** (see headline below).
- **HEADLINE NEGATIVE RESULT: no viable ≥2013 player-level WR/TE coverage-charting source exists.** Every product that carries the player-vs-defender CB/coverage granularity that issue **#1210** actually needs provably starts *after* 2013 and/or is license-barred from ML use: **PFF Advanced Coverage Grade = 2019+** (and PFF terms ban ML inputs outright), **SIS DataHub = 2015+**, **FTN charting = 2022+ free / 2019+ paid**, **PFR advanced tables = 2018+**, **NGS player tracking = 2016+**. The exact axis #1210 demands is a 2015–2020-era data product that cannot backtest to 2013.
- **Be honest: none of the 11 viable candidates directly closes the #1210 WR/TE coverage gap.** The benchmarks *measure* the gap across more seasons but add no coverage feature; the build-not-buy features move team-volume / efficiency / injury-severity ordering on adjacent axes, not the player-vs-CB axis the residual names. Team-level coverage proxies were already A/B-ruled-out negative, and every 2013-reaching candidate here is offense-side or team-level.
- **The actionable yield is real and should not be buried by the negative result.** Two independent expert benchmarks materially extend the rank-skill diagnostic *back to 2013* (the current panel — NFL.com/hvpkod + RotoWire/Sleeper — is genuinely pre-kickoff only 2024+), and the build-not-buy features are all $0, license-clean, and ride existing loaders/cron with no new infra.
- **The RB lever may need no external data.** B2/B3/E2 are the strongest external RB options, but the gap diagnostic flags RB as forecastable from workload/role/game-script already carried — so these are A/B-screen candidates judged on rank metrics (NDCG/recall@k/mean-rank), not assumed wins.
- **The QB residual (#1134, current-week starter news) is structurally serving-only** and cannot backtest to 2013; the one accuracy-adjacent build (B1 CPOE) does not address it.
- **Two candidates reached partial history but were zeroed by the leakage/backfill HARD gate:** **F4 SportsDataIO** (skill-position 2013 start unsourced + suspected box-score backfill) and **G1 deep weather** (observed station data is post-game = leakage; the advertised peak-gust column is empty in 2013 anyway). Nine more failed the ≥2013 gate, were redundant, license-barred, or unverifiable (appendix §6).

## 2. The gap this targets

The project trains 2013–2023, validates 2024, tests 2025, with a **hard constraint that every candidate's usable history reach ≥2013**. The durable, authoritative gap is an **ordering / rank-skill edge** that human experts hold at **RB + WR + TE** — irreducible by rescaling (it is a *ranking* edge, not a magnitude one). Diagnostically this splits three ways: the **WR/TE residual (#1210) is a MISSING-DATA problem** needing **player-vs-defender CB / coverage granularity** (team-level coverage proxies are already A/B-ruled-out negative); the **RB lever may need no new external data** (forecastable from workload/role/game-script already carried); and the **QB residual (#1134) is current-week starter news** — a live feed, serving-only, un-backtestable to 2013. This research asked whether any external source can supply the missing player-level coverage axis, a new expert benchmark to measure the ordering edge, or a build-not-buy feature that re-sorts RB/WR/TE rank within the 2013 constraint.

## 3. Method

Discovery used Opus-4.8 with maximum parallel agents fanning out across source categories (expert benchmarks, PBP-derivable features, charting providers, weather, DFS/betting markets, college, ADP). Every surviving candidate then passed an **adversarial verification stage run by a SEPARATE skeptic agent**: (1) a **≥2013 history check requiring a citable 2013 artifact** (a live URL, a downloaded parquet/CSV with 2013 rows, or a dated contemporaneous page — *not* a marketing claim); (2) a **backfill/leakage check** including the **hvpkod near-exact-match test** — a projection feed whose "historical" values match realized box scores to within ~0.5 pt is retro-backfilled and leakage-unsafe, and is hard-gated to zero. Survivors were integration-scored against the **rubric 0.40 signal / 0.25 history / 0.20 access / 0.15 leakage** (history a near-gate, leakage a hard gate). Scores below are presented **as computed; not recomputed here.**

## 4. Ranked viable candidates

| # | Candidate | Cat | Type | ≥2013 | Access | Expected signal | Score | One-line |
|---|-----------|-----|------|-------|--------|-----------------|-------|----------|
| 1 | **F1 FantasyPros consensus** | F | expert benchmark | yes | free scrape | medium | **0.81** | 3rd independent multi-source expert consensus, the toughest ordering bar |
| 2 | **B1 CPOE derivatives** | B | feature (PBP) | yes | free API | low | **0.72** | QB-accuracy + receiver catch-over-expected, build-not-buy from ingested PBP |
| 3 | **B2 PROE / pace** | B | feature (PBP) | yes | free API | low | **0.72** | Team pass-rate-over-expected + neutral pace = forward volume environment |
| 4 | **B3 opp-adjusted EPA** | B | feature (PBP) | yes | free scrape | low | **0.72** | Home-rolled SoS-adjusted EPA (free DVOA substitute) for matchup ordering |
| 5 | **E2 O-line continuity** | E | feature (snaps) | yes | free scrape | low | **0.72** | Starting-five stability from already-ingested PFR snaps (RB/QB) |
| 6 | **H1 injury body-part TEXT** | H | feature (injuries) | yes | free API | low | **0.72** | Injury-type severity already in the loaded feed, collapsed to 2 ordinals |
| 7 | **F3 FFToday archive** | F | expert benchmark | yes | free scrape | low | **0.69** | 2nd independent deep weekly projection archive (5/6 positions) |
| 8 | **E1 scheme/coaching** | E | feature (PBP) | partial | free API | low | **0.595** | Free PBP scheme tendencies + coaching-change flags (team-level) |
| 9 | **I1 college production** | I | feature (CFBD) | partial | free API | low | **0.595** | College market-share / pedigree for rookie cold-start (~14% of rows) |
| 10 | **D1 DFS ownership/salary** | D | feature (DFS) | partial | free scrape | low | **0.565** | RotoGuru FD salary reaches 2013; realized ownership history-blocked |
| 11 | **C2 line movement** | C | feature (odds) | partial | free scrape | low | **0.52** | Opening line + open→close delta; player props fail 2013 |

### F1 — FantasyPros Expert Consensus weekly projections (score 0.81)

**What it is.** FantasyPros' multi-source Expert Consensus weekly projections expose per-player per-game fantasy POINTS plus raw stat lines (Standard/PPR), for all six positions. Intended use is a **third independent expert benchmark** to score the model's ordering edge against — **not a feature** (training on expert points would relearn a consensus we are trying to beat and re-introduce the forbidden fantasy-point-target/leakage problems).

**≥2013 verdict + artifact.** PASS. The live page accepts `year`/`week` for old seasons: `https://www.fantasypros.com/nfl/projections/qb.php?year=2013&week=1&scoring=STD` renders "2013 Week 1, Consensus of 7 Sources, Sep 5 2013" with per-player FPTS (Rodgers 21.0, Stafford 19.3, Dalton 15.5); `rb.php?year=2013&week=10` shows "2013 Week 10, Nov 9 2013". The skeptic falsified the prior pessimism that only Wayback wk-1 captures existed — the live `.php` grid serves a full in-season weekly history.

**Leakage/backfill.** Leakage-safe. The 2013 Rodgers projection (291.4 yd / 2.2 TD) vs his realized 333/3 is a large miss — no ≤0.5-pt hvpkod backfill match — and the Sep-5 / Nov-9 2013 consensus dates are pre-kickoff. `backfill_risk = low`.

**Access/cost/license.** $0 monetary. Real cost is ToS friction: the public API is free but requires submitting an access request (personal/non-commercial, attribution required), and **redistribution of projections is restricted**. The real working mechanism is the HTML `.php` scrape (what the `ffpros` wrapper uses), **not** the 403-ing `api.fantasypros.com/v2` URL. Effort **M**.

**Integration.** `id_bridge`: name+team+pos fuzzy-match to `gsis_id` (no native id; bridge via `pfr_id`/`espn_id` crosswalk). `merge_key` `(gsis_id, season, week)`. Lands in `src/data/external_sources.py` as an expert-benchmark feeding the eval / Diebold-Mariano harness — **never** into `_INCLUDE_FEATURES`/`ATTN_*`.

**Ordering hypothesis.** A *true multi-expert consensus* (the page shows "Consensus of N Sources" = ESPN/CBS/FFToday/NFL.com/FanDuel) rank-dominates any single expert, so it is the toughest ordering bar and the most informative DM comparison for the RB/WR/TE rank-skill edge under the existing player-clustered bootstrap.

**Biggest unresolved risk.** **Redundancy as a benchmark:** FantasyPros aggregates the same upstream sources (NFL.com, ESPN, CBS, FFToday) the project partly already benchmarks against, so its consensus is partially correlated with the existing NFL.com benchmark — value is the consensus aggregation + DM head-to-head, not a fully orthogonal opinion. Secondary risk: the 3rd-party FFA archive only claims weekly ≥2015, so a bulk historical pull means self-scraping the live grid.

### B1 — CPOE & air-yards-model derivatives (score 0.72)

**What it is.** Build-not-buy QB-accuracy and receiver-catch-quality features aggregating nflfastR's per-play completion-probability outputs (`cp`, `cpoe`, `complete_pass`, `air_yards`) — already in the PBP we ingest — to player-week CPOE, catch-rate-over-expected, and catchable-target rate.

**≥2013 verdict + artifact.** PASS. `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2013.csv.gz` is a real asset; the OSF nflfastR post states verbatim that CP eras are "2006–2013, 2014–2017, 2018+" — `cp`/`cpoe` are populated since 2006, null only pre-2006/non-pass. Primary: `https://opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/`.

**Leakage/backfill.** Clean. Not a 3rd-party projection, so the hvpkod backfill trap is N/A; the model is fixed-fit with no per-row lookahead. Leakage-safe **only** as a lagged `cpoe_L3`/prior-season aggregate like the existing `_L3` builders. `backfill_risk = none`.

**Access/cost/license.** $0, already-ingested PBP, license `ok`. Effort **S**.

**Integration.** `gsis_id` native (PBP `passer_player_id`/`receiver_player_id` match project `player_id`) — no cross-id bridge. Aggregate per-play → one row per `(gsis_id, season, week)`, lag before merge; land in QB `_INCLUDE_FEATURES`/`ATTN_STATIC` and WR/TE receiver-CROE.

**Ordering hypothesis.** Lagged passer CPOE isolates throwing accuracy from scheme/target-depth (a deep-ball QB has low completion% but positive CPOE), potentially reordering QBs whose efficiency is masked by aDOT; receiver catch-over-expected is a hands/target-quality signal absent from the current whitelist.

**Biggest unresolved risk.** **Axis mismatch + partial redundancy:** every CPOE derivative is offense-side and adds no opponent-coverage information — the specific missing axis for #1210 — and receiver catch-over-expected overlaps the model's existing `completion_pct_L3`, `passing_epa_per_dropback_L3`, `deep_ball_rate_L3`, `yac_rate_L3`, racr/wopr. Expected benchmark-flat-to-marginal; worth a low-cost ab_harness screen on NDCG/recall@k, not a coverage fix.

### B2 — Team PROE / pace / neutral-script tendencies (score 0.72)

**What it is.** Per-team season-to-date Pass-Rate-Over-Expected (from nflfastR `xpass`/`pass_oe`), seconds-per-play / plays-per-game pace, and neutral-game-script pass tendency (win-probability-filtered), all derivable from the PBP we already load.

**≥2013 verdict + artifact.** PASS, **empirically verified**. The skeptic downloaded `play_by_play_2013.parquet` (19.5 MB): 2013 wks 1–21, 48,158 plays, `xpass` 75.8% non-null (0.01–0.997), `pass_oe` 73.8% (−98..+97), pace inputs `wp`/`vegas_wp`/`qb_dropback`/`no_huddle` 97–100% non-null. Primary: `https://nflfastr.com/reference/add_xpass.html`.

**Leakage/backfill.** Clean — build-not-buy from raw PBP, hvpkod trap N/A. Only leak vector is in-week contamination, killed by a season-to-date-through-W-1 group-shift lag. `backfill_risk = none`.

**Access/cost/license.** $0, free nflverse data (MIT code / CC-BY data) already loaded via `nflreadpy.load_pbp()`, license `ok`. Effort **M**.

**Integration.** `id_bridge`: `team_code`; `merge_key` `(team, season, week)` left-join on player frame `recent_team` (team feature broadcast, no player-id bridge). New PBP-derived team-week loader → `engineer.py` → RB/WR/TE/QB `ATTN_STATIC_FEATURES`.

**Ordering hypothesis.** PROE and neutral pace forecast a team's *future* pass/play volume better than realized pass rate (which is contaminated by extreme game scripts). This re-ranks players with stable usage *share* but shifting team *absolute* volume — a different axis from the already-carried `implied_team_total` (Vegas points, not pass-tendency) and from player-level snap/target share. Strongest target is WR/TE team-volume ordering.

**Biggest unresolved risk.** **Unproven lift + overlap:** it is orthogonal to — and does not claim to close — the #1210 CB-coverage gap, and overlaps existing implied-totals/EPA/air-yards/ff_opportunity; RB value is lower (RB volume may already be forecastable from carried features). Must be A/B-screened on rank metrics.

### B3 — Home-rolled opponent-adjusted EPA (free DVOA substitute) (score 0.72)

**What it is.** A build-not-buy strength-of-schedule-adjusted EPA (per-team offense/defense EPA-per-play with ridge or lagged-moving-average opponent adjustment, à la Open Source Football / `henrygise nfl-adjusted-epa`) plus a PBP-derived aDOT / team-pass-volume "route-participation" proxy. Primary: `https://opensourcefootball.com/posts/2020-08-20-adjusting-epa-for-strenght-of-opponent/`.

**≥2013 verdict + artifact.** PASS. `play_by_play_2013.parquet` (18.6 MB) + `.csv` (91 MB) exist; nflfastR EP model has an explicit "2006–2013" era bucket; `epa` populated 1999+. 

**Leakage/backfill.** Clean — not a projection (hvpkod trap N/A); temporal-only risk solved by `shift(1).rolling` per-(def, season) like the existing `_build_defense_matchup_features`. `backfill_risk = none`. License `ok`, $0, effort **M**. `merge_key` `(opponent_team/team, season, week)`.

**Ordering hypothesis.** Raw defensive EPA isn't opponent-adjusted, so a defense that faced weak offenses looks better than it is; a lagged opponent-adjusted pass/rush-EPA-allowed rating could re-rank favorable/unfavorable matchups mid-season for streamer/mid-tier RB/WR/TE.

**Biggest unresolved risk.** **It is a refinement of the already-A/B-rejected team-aggregate coverage class.** The published SoS edge is marginal (OSF: 64.04% vs 63.48% on *game outcome*, ~0.5pp, zero fantasy-ordering evidence; henrygise adj-vs-raw r²=0.85 — barely re-ranks), and an opponent-adjusted *team-defense* rating still lacks the player-vs-defender granularity #1210 names. Lowest-confidence of the three PBP features.

### E2 — Offensive-line continuity (score 0.72)

**What it is.** Build-not-buy per-team-week starting-five stability (top-5 OL by `offense_pct`, week-to-week roster overlap, games-played-together) derived from the nflverse PFR snap_counts table **already ingested** in `loader.py::_fetch_snap_counts` (which currently keeps skill-player `offense_pct` and discards OL rows). PFF OL grades are assessed only as a paid/redistribution-forbidden add-on.

**≥2013 verdict + artifact.** PASS. `nflreadr load_snap_counts` docs + `dictionary_snap_counts`: PFR snap counts "starting with the 2012 season"; nflverse issue #128 and the PFR coverage page corroborate 2012/2013. Primary: `https://github.com/nflverse/nflverse-data/releases/tag/snap_counts`.

**Leakage/backfill.** Clean — continuity is *realized* snap participation, not a projection (hvpkod trap N/A); needs only a week≤W-1 lag. `backfill_risk = none`. **Free continuity = $0** (reuses already-fetched table); the PFF OL-grades alternative is $34.99/mo personal-use-only viewing, with bulk/served use requiring an unpriced enterprise license — **license-barred**. Effort **M**.

**Integration.** `pfr_player_id → gsis_id` via the `pfr_to_gsis` map already built in `loader.py`; `merge_key` `(team, season, week)` for continuity, attached to player rows on `(gsis_id, season, week)`. Reuse the fetched snap_counts (keep OL rows) → `engineer.py` team-week continuity → RB/QB (and WR/TE) `ATTN_STATIC_FEATURES`.

**Ordering hypothesis.** O-line continuity is a known driver of run-game efficiency and pass protection, largely orthogonal to the workload/role/game-script signal the RB lever already carries — two RBs with identical projected volume can diverge in realized efficiency behind a stable vs shuffled line. A grep confirmed the project carries skill-player `snap_pct` but **zero** OL-cohort continuity/pressure/pass-block feature, so this is new.

**Biggest unresolved risk.** **Free continuity is a team-environment proxy, not the player-level CB/coverage axis** #1210 names; the sharper signal (PFF pass-block grades / pressures-allowed) is license-barred. Best fit is RB — the lower-EV external lever — so its marginal rank-lift is the open A/B question.

### H1 — Injury-report body-part TEXT severity (score 0.72)

**What it is.** The backtest-able piece of "deeper injury depth": the injury-report body-part TEXT (`report_primary_injury`, `practice_primary_injury`, secondary-injury counts, and the pre-2016 "Probable" tier) that the **same already-loaded nflverse injuries feed** carries back to 2009 but the project currently collapses into a single participation ordinal. The genuinely-new starter-news pieces are serving-only (QB #1134) or already derivable from integrated snaps.

**≥2013 verdict + artifact.** PASS, **downloaded in-session**: `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2013.parquet` — 5,070 rows, season==2013, `report_primary_injury` 97.3% filled (Knee/Ankle/Hamstring), `report_status` Probable 2,772 / Q / D / Out, `date_modified` 2013-09; 2009 the same.

**Leakage/backfill.** Clean — a contemporaneous official report, no projection-to-realized match (hvpkod N/A). Leakage-safe: 85% of rows are Friday-stamped (NFL final-designation day), wk1 max Sat 09-07 before Sun 09-08 games; only ~1 Sun + 4 Mon edits/season form a droppable tail. `backfill_risk = none`. $0, license `ok`, effort **S**.

**Integration.** `gsis_id` native; extend the existing `loader.py` injury block (lines 469–503) to derive body-part text features → RB/WR/TE `ATTN_STATIC_FEATURES`.

**Ordering hypothesis.** The current single ordinal treats all "Questionable" alike, but injury TYPE/LOCATION encodes performance-while-playing severity: a soft-tissue lower-body injury (Hamstring/Groin/Ankle) on a player who suits up throttles explosiveness vs an upper-body/non-load-bearing injury (Hand/Shoulder/Illness) at the same tier — a body-part severity tier could push hobbled-but-active players down and healthy peers up. The pre-2016 "Probable" tier (currently dropped to a neutral 1.0) adds resolution in 2013–2015.

**Biggest unresolved risk.** **Low-EV, off-axis, already partly ablated.** It is the same already-integrated feed (not new external data), orthogonal to the #1210 coverage axis, and injury features were already ablated near-flat overall (`game_status` is ~96.8% constant in the 2025 test split). Realistic upside is a small returning/hobbled SUBGROUP rank effect, measurable only with a tracked subgroup metric — not overall MAE.

### F3 — FFToday weekly per-game projection archive (score 0.69)

**What it is.** FFToday's free, public weekly per-game fantasy projection pages (QB/RB/WR/TE/K) expose a scrapeable archive of pre-kickoff projected stat lines keyed by Season+GameWeek+PosID — a **second independent expert-comparison benchmark** alongside NFL.com and RotoWire. (Footballguys, the sibling source, is subscriber-paywalled and not a free deep archive.)

**≥2013 verdict + artifact.** PASS, **re-fetched**. `https://www.fftoday.com/rankings/playerwkproj.php?Season=2013&GameWeek=1&PosID=30` shows real 2013 Wk1 WR projections (Dez Bryant 7-110-1TD=17.0, Calvin Johnson, Julio Jones); `Season=2009` returns "No Player Found!" → archive floor = 2010, 2013 verified.

**Leakage/backfill.** Clean — the **inverse** of the hvpkod trap: all pass yds round to /10 (360, 300, 310), TDs are small ints 0–3, and many players share identical lines = a projection *grid*, not realized box scores (which are messy/unique). FantasyPros independently tracks FFToday's site-projection accuracy, confirming genuine ex-ante forecasts. `backfill_risk = low`.

**Access/cost/license.** $0 free public HTML scrape, no login. Cost is scrape-politeness: ~16 seasons × ~17 weeks × 5 positions ≈ 1,400–4,000 light GETs for a one-time backfill + a rate-limit/sleep. License `restricted` (redistribution). Effort **M**. DST (PosID=99) is empty in 2013 → **5 of 6 positions**.

**Integration.** Name+team+season+week fuzzy-match to `gsis_id` (bridge via roster name/team like the existing NFL.com/RotoWire join); `merge_key` `(gsis_id, season, week)`. Lands in `src/data/` expert-benchmark loader → evaluation/backtest comparison panel, **not** a feature.

**Ordering hypothesis.** A methodologically distinct 3rd projector whose per-week ordering can be measured against the model with the existing NDCG / mean-rank / recall@k + DM harness. Concrete uses: test whether the human-expert ordering edge at RB/WR/TE *replicates across a 3rd independent source* (robustness of the #1210 claim); build a stronger consensus-of-3 rank ceiling; localize weeks/players where the model's rank diverges most.

**Biggest unresolved risk.** **It adds no player-level coverage signal** (benchmark-only, doesn't close #1210), has **no DST arm**, and its highest marginal value sits at WR/TE while the RB lever may be forecastable without external data.

### E1 — Coaching changes + scheme/personnel tendencies (score 0.595, partial history)

Bundle split by derivability: (1) **FREE PBP-derivable team-scheme tendencies** (PROE/xpass, no-huddle, shotgun, QB-dropback rate) reaching 1999/2006 and genuinely new (the project loads `load_pbp()` but features none of these); (2) **charting-only fields** (play_action, personnel 11/12/21, pre-snap motion, defenders-in-box) that via free nflverse start at **2016 (NGS participation) / 2022 (FTN)** and **FAIL the 2013 gate**; (3) **coaching-change flags** (HC/OC/DC) from PFR, free and reaching 2013. Primary: `https://nflreadr.nflverse.com/reference/load_participation.html`. Leakage-safe as lagged team rates; `merge_key` `(team, season, week)`, effort **S**. **Biggest risk:** the 2013-reachable tier is a *team aggregate* — the same class as the already-rejected NGS team-coverage proxies — while the player-level pieces that might add more are exactly the ones that fail 2013 or are PFF-paywalled. Expected ordering lift LOW, concentrated at RB-/WR-volume tiers.

### I1 — College production / market-share & recruiting pedigree (score 0.595, partial history)

College-career production, target/carry market-share (derivable from CFB PBP), and 247Sports pedigree/breakout-age for **rookie cold-start**, via the CollegeFootballData.com key-gated free tier and the no-key cfbfastR-data parquet mirror. Verified: `play_by_play_2013.parquet` (and 2002–2012) return HTTP 200 in the no-key mirror, so market-share is PBP-derivable to 2013 — **but the pre-built `player_stats` mirror starts 2014**, so 2013 production needs PBP-derivation. Recruiting (247Sports) = 2000+. Primary: `https://collegefootballdata.com/`. Leakage-safe (college stats predate NFL debut), effort **L**, `merge_key` static per-player draft-year attribute joined onto `(gsis_id, season, week)` rows from rookie season onward. **Biggest risk:** touches only ~14% rookie rows (invisible in overall MAE/NDCG), does not address the #1210 weekly coverage axis, and is **adjacent to the already-REJECTED rookie draft-capital/combine work** — draft capital already encodes much of what scouts infer from college production, so marginal lift over the (benchmark-flat) draft-pick signal is the binding uncertainty. Pursue only scoped to a rookie WR/RB subgroup rank metric.

### D1 — DFS ownership % / salary archives (score 0.565, partial history)

Two distinct products: (a) **RotoGuru free salary+fantasy-point archive** (FanDuel 2011+, DraftKings 2014+, **NO ownership column**) and (b) **realized contest OWNERSHIP %** (RotoGrinders ResultsDB / FantasyLabs), the candidate's actual target signal but **history-blocked to ~2014+ and effectively ~2016-2017+** for accessible archives. Verified via curl(UA): `http://rotoguru1.com/cgi-bin/fyday.pl?week=1&year=2013&game=fd&scsv=1` returns real 2013 wk1 FD rows; DK 2013 is empty (starts 2014). FantasyLabs dashboard launched Feb-2017; no documented pre-2013 realized-ownership feed. **Biggest risk:** the part that clears 2013 (salary) overlaps heavily with already-carried implied team totals / ff_opportunity / snap-role features (likely benchmark-flat), while the differentiated piece (realized ownership) is history-blocked. Treat as a low-priority **post-2016 secondary study**, not a 2013-back feature.

### C2 — Opening-vs-closing line movement + player props (score 0.52, partial history)

Pre-kickoff betting-market movement: the opening line and open→close delta on game spread/total (a sharp-money signal beyond the closing line we already carry), plus player-prop lines — but **the prop component does not exist before ~2019-2023**. Game-level open/close reaches 2013 (SBR verified: `https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm`, free SBR/AusSportsBetting xlsx). **Biggest risk:** nflverse `spread_line`/`total_line` **already carry the CLOSING line**, so the only new piece is the opening value + movement — a thin *team-level* increment overlapping already-carried/already-rejected team signal; the one player-level piece (props) fails the hard 2013 gate. Low priority as a 2013-back feature; props would be a strong candidate only if the horizon is ever relaxed to 2019+.

## 5. Prototype next: three picks (one per lever)

### (a) New per-game expert benchmark → **F1 FantasyPros** (corroborated by F3 FFToday)
- **First step:** scrape the FantasyPros `.php` consensus grid (`/nfl/projections/{pos}.php?year=&week=&scoring=STD`) for a single backfill season (e.g. 2013), build the name+team → `gsis_id` bridge, and run the existing player-clustered-bootstrap + Diebold-Mariano harness to measure the model's per-week NDCG/recall@k/mean-rank vs the consensus at RB/WR/TE. Add **F3 FFToday** (`playerwkproj.php?Season=&GameWeek=&PosID=`) as the corroborating second arm to test whether the human-expert ordering edge *replicates across an independent 3rd projector* and to build a consensus-of-3 rank ceiling.
- **Biggest risk:** redistribution ToS on both sources (benchmark-only/internal use mitigates), and FantasyPros' partial correlation with the already-integrated NFL.com benchmark (it aggregates the same upstream sources) — F3's independent methodology is the orthogonality check.

### (b) RB workload-derivative → **B2 PROE / pace** (best of B2/B3/E2)
- **Why B2 over B3/E2:** all three are $0, 2013-verified, leakage-clean build-not-buy features, but **B2 carries the cleanest, highest-confidence ordering mechanism** — PROE + neutral pace forecast a team's *future* pass/play volume (an established result) on an axis genuinely distinct from the carried `implied_team_total` and player-level share, and it was *empirically verified* (real 2013 parquet, `xpass`/`pass_oe` densely populated). **B3** is explicitly a refinement of the already-A/B-rejected team-aggregate coverage class with a marginal (~0.5pp, game-outcome-only) published edge; **E2** is orthogonal and worth a screen but its best fit (RB) is the lower-EV lever and the sharp variant (PFF grades) is license-barred. B2 is the strongest single A/B.
- **First step:** add a PBP-derived team-week loader computing season-to-date-through-W-1 PROE + neutral-script pass rate + pace, broadcast onto `recent_team`, and run a 3-seed ab_harness cfg-injection screen judged on **rank metrics (NDCG/recall@k/mean-rank)**, not overall MAE.
- **Biggest risk:** unproven ordering lift and overlap with implied-totals/EPA/ff_opportunity; RB volume may already be forecastable from carried features, so the screen must clear the seed band before shipping.

### (c) WR/TE coverage → **honest verdict: NO 2013-back source exists**
- **The realistic move is the benchmarks + build-not-buy proxies, NOT a coverage feature.** Every player-vs-defender CB/coverage product provably starts after 2013 and/or is license-barred (PFF 2019+ & ML-banned, SIS 2015+, FTN 2022+/2019+, PFR 2018+, NGS 2016+ — appendix §6). **#1210 is data-blocked under the ≥2013 hard constraint.** Use F1/F3 to *measure and localize* the WR/TE residual across more seasons, and screen B1 (receiver catch-over-expected) / B2 (team volume) as adjacent — but acknowledged off-axis — proxies.
- **First step:** flag #1210 as **data-blocked pending a post-2015 secondary experiment** — a truncated-window (2016+ NGS, or 2019+ PFF-advanced if a license is ever obtained) study isolating whether player-level coverage granularity moves the WR/TE rank tail at all, before any 2013-back integration is attempted.
- **Biggest risk:** treating any of the 2013-reaching team-level proxies as the coverage fix — they are the exact class already A/B-ruled-out negative; the gap is genuinely missing data, not an unexploited feature.

## 6. Excluded / blocked appendix ("we considered it")

| Candidate | Category | Reaches 2013 | Why excluded |
|-----------|----------|--------------|--------------|
| **F4 SportsDataIO / FantasyData** | F (benchmark) | partial | **HARD-GATED (leakage/backfill):** DST projections "from 2013/2014" in dict, but skill-position start-year is **unsourced (marketing-only)**, no citable 2013 player row (paid-key-gated), and the BAKER engine debuted 2022 → 2013-era "projections" suspected **box-score-backfilled** (hvpkod failure mode). Must buy key + run near-exact-match vet before any trust. |
| **G1 Deep stadium weather (NOAA/Meteostat/VisualCrossing)** | G (feature) | partial | **HARD-GATED (leakage/backfill):** observed station weather is **post-game = leakage-unsafe** as a backtest feature; and the advertised **peak-GUST column is empty in 2013** (`wpgt` 0/8758 via Meteostat ISD-Lite). A forecast archive would be a different, thinner product. |
| **A1 PFF grades + WR/CB coverage charting** | A (charting) | partial | **License-barred + history:** 2013 slot charting + coverage grades exist (to 2006), but per-snap separation / **Advanced Coverage Grade = 2019+**; PFF terms **explicitly ban PFF Data as ML inputs / AI training**, personal non-commercial only. No legal feature path. |
| **A2 SIS DataHub charting** | A (charting) | **no (2015)** | **History gate:** earliest selectable NFL season = **2015** (verbatim product quote, 2018); zero coverage for 2013/2014 + the 2012 context season. Some subsets shallower (Tendency Reports "back to 2018"). |
| **A3 FTN Data charting + DVOA** | A (charting) | partial | **History gate:** granular charting (routes/coverage/pressure) — the part fitting #1210 — is **2022+ free / 2019+ paid**; legacy DVOA reaches 1981 but is a sub-gated team/player rate (≈ruled-out team coverage proxy). |
| **A4 PFR advanced passing/rushing/receiving** | A (charting) | **no (2018)** | **History gate:** advanced column family uniformly **2018-back** (pressure/pocket-time even 2019); 2013–2017 fully NA across 5 of 11 train seasons → silent zero-fill failure mode. Basic pre-2018 columns already covered by nflverse. |
| **C1 Historical player-prop closing lines** | C (market) | **no (2020)** | **History gate:** per-player prop archives start **2020 (SportsDataIO) / 2023 (The Odds API)**; free archives (BigDataBall 2016, SBR 2007) are game-level only. Structural: weekly props became a core product only post-May-2018 PASPA repeal. |
| **C3 DFS salaries (DK/FD)** | C (market) | partial | **History gate (both ends):** FD reaches 2013 (RotoGuru) but **RotoGuru stops after 2021** — the 2024 val / 2025 test seasons are empty; DK starts 2014; DK draftables API is live-only. No single leakage-safe source spans 2013→2025. |
| **F2 ESPN / Yahoo / CBS projection archives** | F (benchmark) | partial | **Redundant + format mismatch:** ESPN/CBS 2012–2013 CSVs exist but are **full-season totals (no week col)** → fatal mismatch with the weekly `(id, season, week)` contract; Yahoo starts 2014; and as weekly benchmarks they'd overlap existing NFL.com + RotoWire. |
| **X1 NGS player tracking (separation/cushion/air-yards)** | A (charting) | **no (2016)** | **History gate:** RFID chips not league-wide until **2016** (18/32 stadiums in 2014, "best effort"); 2013–2015 fully NaN (3 of 11 train seasons). The canonical 2016-cutoff example this gate was written to catch. |
| **X2 Historical ADP archives (FFC / MFL)** | I (market) | yes | **Redundant / wrong granularity:** FFC ADP verified to 2010/2013 live, but ADP is **season-level draft consensus**, not weekly — redundant with carried role/expected-value features and off the weekly-ordering axis; MFL per-2013-season isolation also claimed-but-unverified. |

## 7. Caveats

- **Per-source backfill vet is still required before trusting any new benchmark.** F1 and F3 passed the skeptic's leakage/backfill checks on sampled seasons (F1: a large 2013 Rodgers projection-vs-realized miss; F3: a rounded projection *grid* unlike messy realized box scores), but a **full per-season hvpkod-style near-exact-match scan (`_LOOKAHEAD_EPS=0.5`)** across the whole 2013–2023 backfill is mandatory before either feeds the DM harness as ground truth — exactly the trap that hard-gated F4.
- **Scraping ToS:** both new benchmarks are `license_redistribution = restricted`. FantasyPros' API access is personal/non-commercial with attribution and restricted redistribution; FFToday is a free public scrape but redistribution-restricted. Use them as **internal benchmarks only**, add rate-limit/sleep + caching, and do not redistribute the projections. Use the working `.php`/`playerwkproj.php` HTML endpoints — not the 403-ing `api.fantasypros.com/v2` path.
- **The hard-gated and blocked items are the post-2015 secondary-experiment backlog.** If the backtest horizon is ever relaxed past 2013, these become live: **NGS player-tracking (2016+)** is the closest thing to the #1210 coverage axis and the realistic vehicle for a truncated-window WR/TE coverage experiment; **deep weather gusts** would need a *forecast* archive (not observed) to be leakage-safe; **DFS realized ownership (2016-2017+)** and **player props (2019/2020+)** are player-level market signals worth a relaxed-horizon study. None of these can serve the current 2013-back primary backtest — but they are the documented "next horizon" set, and #1210 should be tracked as **data-blocked**, not closed.

# Comparison Against Expert Projection Sources

## Methodology

This document compares our model's weekly fantasy point predictions against published accuracy benchmarks from industry expert sources and academic ML research.

**Our evaluation setup:**
- **Training data:** 2012-2023 NFL seasons (~12 seasons of weekly player data)
- **Validation set:** 2024 NFL season (for hyperparameter tuning and early stopping)
- **Test set:** 2025 NFL season (held out entirely from training and validation)
- **Scoring format:** Full PPR (1 point per reception)
- **Primary metric:** MAE (Mean Absolute Error) on total weekly fantasy points
- **Player pool:** All rostered players at each position with recorded game stats
- **Prediction style:** Pure pre-game projections using only data available before kickoff
- **Architectures:** Ridge regression (per-target), Multi-Head NN (shared backbone + per-target heads), Attention NN (per-target heads with per-target attention over player history + opponent context), LightGBM (per-target boosted trees). Trained per position (QB, RB, WR, TE, K, DST).

**Important caveats on comparability:**
- Expert sites and our model were not evaluated on identical player pools or seasons. Expert accuracy rankings (e.g., from Fantasy Football Analytics) cover 2019-2023 and use curated pools (top 20 QBs, top 50 RBs/WRs, top 20 TEs), while our test set covers all 2025 rostered players.
- Most expert accuracy rankings report *relative* rankings (1st, 2nd, 3rd by position) rather than raw MAE values, making direct numerical comparison difficult.
- Scoring format differences (standard vs. half-PPR vs. full PPR) shift absolute point totals and therefore MAE.
- One source — **NFL.com** — publishes per-game projections we can score directly; see "NFL.com Head-to-Head Baseline" below for the apples-to-apples MAE.
- **Training loss ≠ evaluation metric.** Our models train with Huber loss (robust to the boom/bust tail of fantasy scoring), but every comparison here is on MAE / RMSE / R² over held-out predictions — metrics that don't depend on what any model trained on. Squared error is the consistent scoring rule for the conditional *mean* (which mean-oriented expert projections implicitly target) and absolute error for the *median* (Gneiting 2011); Huber is consistent only for an intermediate robust "Huber mean" functional (Taggart 2022), so we never score a head-to-head *with* Huber. We report MAE (the better-matched summary for heavy-tailed errors) and RMSE side by side — RMSE because it favors the experts' implicit squared-error objective, which makes beating them on it the stronger claim.
- Despite these limitations, published accuracy thresholds and academic benchmarks provide meaningful context for interpreting our results.

---

## Our Model Performance

From the 2026-05-29 benchmark run (`benchmark_history/2026-05-29T11-00-49_9de4d84.json`), evaluated on the 2025 test season:

| Model | QB MAE | QB R² | RB MAE | RB R² | WR MAE | WR R² | TE MAE | TE R² | K MAE | K R² | DST MAE | DST R² |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge Regression | 6.539 | 0.273 | 4.502 | 0.360 | 4.779 | 0.312 | 3.726 | 0.316 | **4.008** | **0.018** | 5.203 | 0.044 |
| Multi-Head NN | **6.514** | **0.275** | 4.302 | 0.419 | 4.256 | **0.368** | 3.515 | **0.336** | 4.167 | -0.096 | 5.115 | **0.084** |
| Attention NN | 6.585 | 0.249 | **4.179** | 0.418 | 4.292 | 0.352 | **3.508** | 0.325 | 4.221 | -0.099 | **5.107** | 0.061 |
| LightGBM | 6.693 | 0.242 | 4.195 | **0.425** | **4.238** | 0.361 | 3.595 | 0.317 | 4.133 | -0.051 | 5.271 | 0.016 |
| **Best (MAE)** | NN 6.514 | NN 0.275 | Attn NN 4.179 | LGBM 0.425 | LGBM 4.238 | NN 0.368 | Attn NN 3.508 | NN 0.336 | Ridge 4.008 | Ridge 0.018 | Attn NN 5.107 | NN 0.084 |

**Per-target MAE breakdown** (Attention NN, in native stat units):

| Position | Per-target MAE |
|----------|----------------|
| **QB** | passing_yards: 67.8 yds, rushing_yards: 11.0 yds, passing_tds: 0.89, rushing_tds: 0.25, interceptions: 0.64, fumbles_lost: 0.29 |
| **RB** | rushing_yards: 18.1 yds, receiving_yards: 9.5 yds, receptions: 0.98, rushing_tds: 0.32, receiving_tds: 0.10, fumbles_lost: 0.08 |
| **WR** | receiving_yards: 20.3 yds, receptions: 1.37, receiving_tds: 0.30, fumbles_lost: 0.01 |
| **TE** | receiving_yards: 14.9 yds, receptions: 1.20, receiving_tds: 0.29, fumbles_lost: 0.01 |

K and DST use bespoke aggregators (signed FG/PAT for K; tier-mapped points/yards-allowed for DST); see [`src/k/config.py`](../src/k/config.py) and [`src/dst/config.py`](../src/dst/config.py) for the per-target target lists.

---

## Industry Expert Source Rankings

The following rankings come from Fantasy Football Analytics (FFA), which tracks projection accuracy across major sources using MAE on weekly fantasy points (2019-2023 seasons).

### Expert Accuracy Rankings by Position

| Position | #1 Source (3-year) | #2 Source | #3 Source | Notes |
|----------|-------------------|-----------|-----------|-------|
| **QB** | FantasySharks | FFToday | FFA Average | CBS historically strong over 5-year window |
| **RB** | FFToday | FFA Average | CBS | CBS led the 5-year period; FFToday surged recently |
| **WR** | FFToday | FFA Average | NumberFire | FFToday dominates both 3-year and 5-year windows |
| **TE** | FFToday | NumberFire | FFA Average | NumberFire historically strong |

**FFA Average** (a consensus aggregate of multiple sources) consistently ranks in the top 3 across all positions, demonstrating the power of ensemble approaches --- similar to how our Ridge / NN / Attention NN / LightGBM comparison explores model diversity.

### FantasyPros Consensus

FantasyPros aggregates projections from 40+ expert sources. A 2012 analysis found the consensus explained approximately 67% of variance in actual fantasy points (R² ~ 0.67), outperforming any individual source. This remains the gold standard for expert projections.

### ESPN, Yahoo, CBS Individual Performance

- **ESPN:** Limited public accuracy data (only available for 2019 and 2023 seasons in FFA tracking). Known to systematically underpredict quarterbacks and kickers.
- **Yahoo:** Not prominently featured in multi-year accuracy comparisons. Justin Boone (Yahoo Fantasy) won FantasyPros' 2025 in-season accuracy award.
- **CBS:** Consistently in the top 5 across all positions over 5-year windows; historically the most balanced single source.

---

## NFL.com Head-to-Head Baseline

Unlike the FFA-tracked sources above (which publish only relative accuracy rankings), NFL.com publishes per-game projected raw stats we can score directly through our PPR aggregator and compare to actual results — giving a raw-MAE number that's directly comparable to our model's. Numbers below are produced by [`src/analysis/analysis_nflcom_baseline.py`](../src/analysis/analysis_nflcom_baseline.py), which writes a `nflcom_baseline.json` report (per-position raw MAE for NFL.com projections scored through our PPR aggregator, alongside our model's MAE). That report is regenerated on demand and is not committed to the repo.

**Methodology:** NFL.com weekly projections sourced from the [hvpkod/NFL-Data](https://github.com/hvpkod/NFL-Data) archive, joined to actual stats from the [nflverse stats_player_week](https://github.com/nflverse/nflverse-data) parquet on `(player_id, season, week)`. Both projections and actuals are scored through our internal PPR aggregator (`src.shared.aggregate_targets.predictions_to_fantasy_points`) so the comparison is apples-to-apples. K uses raw FG/PAT distance stats directly (the parquet's `fantasy_points` column is offensive-only). DST is skipped — NFL.com has no DST projections in this archive.

### NFL.com Accuracy on the 2025 Test Season

| Position | n | NFL.com MAE | NFL.com RMSE | NFL.com R² |
|----------|---:|---:|---:|---:|
| **QB** | 664 | 5.713 | 7.607 | 0.321 |
| **RB** | 1,577 | 4.110 | 5.965 | 0.471 |
| **WR** | 2,498 | 3.983 | 5.720 | 0.378 |
| **TE** | 1,281 | 3.390 | 4.991 | 0.331 |
| **K** | 542 | 4.358 | 5.510 | -0.125 |
| **DST** | — | — | — | (no NFL.com data) |

### Side-by-side With Our Model (PPR, 2025)

Pairing the NFL.com MAEs above with the per-position best from "Our Model Performance":

| Position | Our MAE (best model) | NFL.com MAE | Delta |
|----------|---:|---:|---:|
| **QB** | 6.514 (Multi-Head NN) | 5.713 | NFL.com leads by 0.80 |
| **RB** | 4.179 (Attention NN) | 4.110 | NFL.com leads by 0.07 (≈ tied) |
| **WR** | 4.238 (LightGBM) | 3.983 | NFL.com leads by 0.26 |
| **TE** | 3.508 (Attention NN) | 3.390 | NFL.com leads by 0.12 |
| **K**  | 4.008 (Ridge) | 4.358 | We lead by 0.35 |
| **DST** | 5.107 (Attention NN) | — | (no NFL.com DST data) |

QB is the position where NFL.com beats us by the largest margin (~0.8 MAE pts); RB and TE are within ~0.1 pts (below the run-to-run variance we typically see between benchmark snapshots), and WR is ~0.26 pts behind. K is the only position where we beat NFL.com cleanly — partly because their kicker projection schema is per-distance-bucket FG attempts that doesn't aggregate well, and partly because our K model now keys directly off cached PBP after the `fg_yards_made` schema fix in [#227](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/227).

The QB gap is consistent with the structural QB limitation noted under "Position-by-Position Analysis": touchdown variance dominates QB error, and a curated expert projection that incorporates context (matchup, narrative, etc.) can suppress that volatility better than a stats-driven model trained on broader player pools.

**Caveats:**
- Single test season (2025). 2024 was used for hyperparameter tuning, so it's not a clean held-out comparison; the analysis script is multi-season-capable and will pool across years automatically as `TEST_SEASONS` grows.
- NFL.com's player pool is implicitly curated (they only project players likely to play); our model evaluates every rostered player at each position. NFL.com's MAE is on `n_matched` rows where both sides have a projection, so this is partially controlled, but coverage differences in marginal players still affect the per-position N.
- The 92.2% NFL.com name-match rate (see the loader logs) drops ~7.8% of NFL.com rows that couldn't be joined to a `gsis_id` — mostly free-agent placeholders ("Alex Hale (K, FA)", etc.).

### Significance-tested head-to-head (`analysis_expert_comparison.py`)

The baseline table above scores NFL.com against actuals across every NFL.com-matched row. A second script — [`src/analysis/analysis_expert_comparison.py`](../src/analysis/analysis_expert_comparison.py) — runs the stricter, decision-relevant comparison against **each expert** (NFL.com and Sleeper/RotoWire — see below). For a given expert it joins our model's held-out per-player-week predictions to that expert on the **intersection** where both project, scores both against the same ground-truth `fantasy_points` column (genuinely *paired* errors), and adds:

- **MAE and RMSE** for each side (see the loss-vs-metric caveat above for why both).
- **Ranking** quality — top-K hit-rate and Spearman ρ — because fantasy is a selection problem and rank metrics are loss-agnostic.
- A **paired significance test** — a player-clustered bootstrap CI on ΔMAE / ΔRMSE (primary) plus a Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction (Diebold & Mariano 1995). This is what separates a real edge from run-to-run noise.

Because it restricts to the model ∩ NFL.com sample, its per-position N and NFL.com MAE differ slightly from the baseline table above (which is on the full NFL.com-matched set).

**Illustrative QB 2025 output** (n = 663 common player-weeks; model = production Attention NN):

| Metric | Model | NFL.com | Δ (model − expert) | 95% CI | DM p |
|--------|---:|---:|---:|:--:|---:|
| MAE | 6.596 | 5.756 | +0.840 | [0.54, 1.15] | 8.6e-7 |
| RMSE | 8.235 | 7.702 | +0.534 | [0.13, 0.88] | 0.011 |
| Top-12 hit rate | 0.468 | 0.491 | — | — | — |
| Spearman ρ | 0.480 | 0.536 | — | — | — |

The ΔMAE CI excludes zero and DM p ≪ 0.05, so NFL.com's QB edge is **statistically real**, not benchmark-snapshot noise — and it holds on RMSE too (the metric tilted *toward* the expert's squared-error objective) and on both ranking metrics. This is the rigorous version of the QB gap discussed above. Run it with `python -m src.analysis.analysis_expert_comparison` (defaults to PPR / `TEST_SEASONS` / all positions; writes both experts to the nested `experts` block of `analysis_output/expert_comparison.json`, regenerated on demand and not committed).

### Second expert: Sleeper (RotoWire)

A second expert is wired into the same tool: **RotoWire projections via Sleeper's API** ([`src/analysis/sleeper_loader.py`](../src/analysis/sleeper_loader.py)), offense only (QB/RB/WR/TE), joined to actuals through the nflverse `sleeper_id → gsis_id` crosswalk. It's a *single provider* (not a consensus), and Sleeper's unofficial API is use-at-your-own-risk.

**Provenance check (the gate before trusting it).** Sleeper doesn't document whether its historical projections are the as-of-kickoff snapshot or a later backfill — and a backfill would inflate RotoWire's apparent accuracy via look-ahead. So we score RotoWire against 2025 actuals — on the players it *actually* projects (Sleeper returns a row for every rostered player; the unprojected placeholders are dropped, else they'd be scored as confident 0.0s and deflate the MAE) — and confirm the error sits in the *plausible expert range*: MAE **QB 6.21, RB 4.32, WR 4.20, TE 3.57** — close to NFL.com's (5.71 / 4.11 / 3.98 / 3.39), a touch higher (RotoWire is marginally less accurate and projects a more selective pool). A backfill would show near-zero error; instead these are genuine pre-game projections (e.g. Lamar Jackson, 2025 Wk1: a fractional **1.78** projected passing TDs, not an integer actual).

**Model vs RotoWire (PPR, 2025)** (model = production Attention NN; offense MAEs are on RotoWire's own — more selective — player pool, so the N and model MAE differ from the NFL.com illustration above). **DST is included because Sleeper provides it and NFL.com does not** — so this is the model's *only* DST head-to-head (team-keyed; all 10 DST targets map, with `special_teams_tds` ← Sleeper's `st_td`):

| Pos | n | Model MAE | RW MAE | ΔMAE | 95% CI | DM p |
|-----|---:|---:|---:|---:|:--:|---:|
| QB | 545 | 6.362 | 6.243 | +0.119 | [-0.17, 0.43] | 0.38 |
| RB | 1447 | 4.545 | 4.328 | +0.217 | [0.06, 0.37] | 0.003 |
| WR | 2295 | 4.360 | 4.228 | +0.132 | [0.03, 0.23] | 0.002 |
| TE | 1165 | 3.629 | 3.611 | +0.018 | [-0.07, 0.12] | 0.71 |
| DST | 544 | 5.144 | 5.117 | +0.026 | [-0.19, 0.23] | 0.76 |

RotoWire is marginally *less* accurate than NFL.com (QB 6.24 vs 5.76). Both experts agree the model **trails on RB and WR** (CIs exclude zero) and **ties on TE** (CI spans zero); the model also **ties RotoWire on DST** (Δ +0.03, p = 0.76) — a head-to-head NFL.com can't provide. They differ on **QB**: the model trails NFL.com (Δ +0.90, significant) but **ties RotoWire** (Δ +0.12, p = 0.38) — because RotoWire is itself weaker on QB, the model matches it there. Two experts converging on the RB/WR/TE verdict — and bracketing the QB result — is a more robust read than either alone.

---

## Published Accuracy Thresholds

Fantasy Projection Lab and Fantasy Football Analytics provide general benchmarks for weekly projection quality:

| MAE Range | Interpretation |
|-----------|---------------|
| **< 5.0 pts** | Competitive with professional projection sources |
| **5.0 - 7.0 pts** | Within the range of established projection systems |
| **> 7.0 pts** | Likely adding noise relative to simple baselines |

---

## Academic ML Benchmarks

| Study | Method | Key Result | Reference |
|-------|--------|------------|-----------|
| **Kapania (Stanford CS 229, 2012)** | Ridge, Bayesian Ridge, ElasticNet, Random Forest, Gradient Boosting | R² = 0.53-0.61 across seasons; performance comparable to FantasyData.com | [cs229.stanford.edu](https://cs229.stanford.edu/proj2012/Kapania-FantasyFootballAndMachineLearning.pdf) |
| **Dross (INST 414, 2023)** | k-NN, Linear Regression | RB RMSE: 4.19 (k-NN); WR RMSE: 3.64 (Linear Regression); R² up to 0.59 | [Medium](https://medium.com/inst414-data-science-tech/predicting-nfl-players-fantasy-performance-362f5f1828d3) |
| **Hua et al. (arXiv:2111.02874, 2021)** | Deep Learning + NLP (ESPN text features) | RMSE: 6.78 across positions | [arXiv](https://arxiv.org/pdf/2111.02874) |
| **SMU Data Science Review (2024)** | Ensemble Neural Networks | Improved prediction accuracy over single models for top-12 classification | [SMU Scholar](https://scholar.smu.edu/datasciencereview/vol8/iss2/7/) |
| **Chant (2023)** | Multi-season regression models | R² = 0.53 (1 year), 0.61 (2 years), 0.81 (in-season with 5+ weeks) | [jadenchant.github.io](https://jadenchant.github.io/pages/projects/nfl_fantasy_pt/) |
| **Beal, Norman & Ramchurn (IJCSS, 2020)** | LR / LSTM / RBF / Random Forest (best per position) | Per-game RMSE (FanDuel scoring): QB 6.89, RB 6.07, WR 4.39, TE 4.10, DEF 3.85, K 3.28 | [Southampton ePrints](https://eprints.soton.ac.uk/445995/1/DFS_IJCSS.pdf) |
| **Lutz (arXiv:1505.06918, 2015)** | SVR, Neural Net (QB only) | QB RMSE ≈ 7.8 (standard scoring, top-24 QBs) | [arXiv](https://arxiv.org/abs/1505.06918) |

> **Scoring-format caveat:** these RMSE figures are *context, not a like-for-like target*. Beal et al. and Lutz use FanDuel / standard scoring (not full PPR), and Beal et al.'s per-position models use player-history-only features (narrower inputs than ours), which deflates their WR/TE/K RMSE. Also note many "fantasy RMSE" numbers online are actually MAE, R², normalized error, or win-rate — verify what each measures before citing.

---

## Position-by-Position Analysis

### Quarterbacks

**Our best MAE: 6.514 (Multi-Head NN)** --- falls within the 5.0-7.0 "established systems" range.

QBs are generally considered the most predictable position due to consistent volume (30+ pass attempts per game), yet our R² is moderate (0.242–0.275 across models). The per-target breakdown reveals why: `passing_tds` and `rushing_tds` carry meaningful per-game volatility (a QB throwing 1 vs. 3 TDs swings the score by 8 fantasy points), and this volatility limits any model's predictive power on a weekly basis. The `passing_yards` head alone accounts for the bulk of fantasy-point variance (MAE 68 yards × 0.04 pts/yd ≈ 2.7 fantasy points).

The deep learning NLP study (arXiv:2111.02874) reported a comparable RMSE of 6.78 using ESPN text features, suggesting our QB MAE of 6.51 is in line with neural-network approaches in the literature. The Multi-Head NN edges the other architectures here by a narrow margin (and LightGBM is actually the weakest on QB this run), so no single family has a decisive edge on this high-variance position.

### Running Backs

**Our best MAE: 4.179 (Attention NN)** --- in the "competitive" range (< 5.0), narrowly.

RBs are among the most volatile fantasy positions due to game-script dependence and TD variance, making a sub-5 MAE a solid result. For context:
- The k-NN academic benchmark reported RB RMSE of 4.19, and our MAE of 4.18 is below that (MAE ≤ RMSE by definition).
- Our best R² of 0.425 (LightGBM) is just under the 0.53–0.61 range reported by Kapania's multi-season Stanford study — but our test set is the full 2025 RB pool, not just starters, which suppresses R² on a noisier denominator.
- Per-target MAE distributes the error: rushing_yards 18.1 yds × 0.1 ≈ 1.8 pts; receiving_yards 9.5 yds × 0.1 ≈ 1.0 pt; receptions 0.98 × 1.0 ≈ 1.0 pt; rushing_tds 0.32 × 6 ≈ 1.9 pts. TDs remain the dominant single source of error in fantasy-point space.

### Wide Receivers

**Our best MAE: 4.238 (LightGBM)** --- competitive (< 5.0), with LightGBM edging the attention NN by ~0.05 MAE points.

WR is a tight race between the trees and the neural models: receiver production has strong route/scheme/coverage dependencies that the attention NN captures via learned attention over player history + opponent defensive context, but LightGBM narrowly takes the MAE this run while the Multi-Head and attention NNs stay within ~0.05.
- The academic linear regression benchmark for WRs reported RMSE of 3.64, which is lower but used a curated player pool (top 50 WRs only) and a different season.
- Our best R² of 0.368 (Multi-Head NN) sits within the published WR consistency range (R² ~ 0.30–0.65 depending on methodology).
- Per-target MAE (Attention NN): receiving_yards 20.3 yds × 0.1 ≈ 2.0 pts is the largest contributor; receptions 1.37 × 1.0 (PPR) ≈ 1.4 pts; receiving_tds 0.30 × 6 ≈ 1.8 pts.

### Tight Ends

**Our best MAE: 3.508 (Attention NN)** --- comfortably in the competitive range.

TEs are the lowest-volume offensive skill position and produce the lowest-variance fantasy outputs of QB/RB/WR/TE, which gives the model a structurally easier target — most starting TEs land in a 4–10 point band most weeks. The attention NN narrowly edges LightGBM (3.508 vs 3.595 MAE); per-target the receptions head (1.20 MAE) carries proportionally more of the fantasy-point error than for WR because TE target volume is lower (3–5 targets/game typical).

### Kickers

**Our best MAE: 4.008 (Ridge)** --- in the competitive band, on a position no public benchmark covers carefully.

K is interesting because kicker fantasy production is *highly* opponent/weather/coach dependent — and our model now beats NFL.com here (4.008 vs 4.358 MAE) after the `fg_yards_made` schema fix in [#227](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/227) restored the FG-distance signal that had been silently dropped from the PBP cache. K's R² hovers near zero across all four models (−0.10 to 0.02); on this position, "low MAE + near-zero R²" is the right operating point — kicker variance week-to-week is too high to extract a meaningful rank-order signal, but absolute prediction error is tight.

### Defense / Special Teams

**Our best MAE: 5.107 (Attention NN)** --- in the 5.0–7.0 "established systems" band.

DST landed via [commit `cc0c627`](../) with a 10-target attention model (sacks, INTs, fumble recoveries, fumbles forced, safeties, def TDs, blocked kicks, ST TDs, points allowed, yards allowed) plus a parallel opp-OFFENSE attention branch (PR [#223](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/223)) that conditions defensive predictions on the offense being faced. The Attention NN and the Multi-Head NN (no attention) are effectively tied on DST (5.107 vs 5.115 MAE) — DST signal is structurally simpler than skill-position production, so the attention branch is roughly on par with the plain NN rather than pulling decisively ahead on the current feature set.

---

## Key Findings

1. **All six positions are now in the "competitive" or "established" MAE bands.** QB / DST sit in the 5.0–7.0 "established systems" band; RB / WR / TE / K all clear the < 5.0 "competitive" threshold. The previous gap where K/DST results were pending is closed (this iteration of the doc).

2. **No single architecture wins everywhere.** Multi-Head NN wins QB; Attention NN wins RB / TE / DST; LightGBM wins WR; Ridge wins K. This is the strongest signal yet that maintaining a diverse model portfolio is worth the training cost — the per-position routing in `src.benchmarking.benchmark` picks the actual best architecture per position rather than locking in one family.

3. **The experts beat the model on RB/WR; QB depends on which expert; TE is a tie.** NFL.com and RotoWire (via Sleeper) both publish scoring-grade raw projections we can join to actuals. On 2025 the paired significance test (player-clustered bootstrap CI + Diebold-Mariano) shows the model **trails both experts on RB and WR** (CIs exclude zero) and **ties both on TE** (CIs span zero). On **QB the experts split**: the model trails the stronger NFL.com (Δ +0.90, significant) but **ties the weaker RotoWire** (Δ +0.12, p = 0.38) — RotoWire is less accurate at QB, so the model matches it there. On **DST** — a head-to-head only Sleeper enables (NFL.com has none) — the model **ties RotoWire** (Δ +0.03, p = 0.76). We beat NFL.com on K (Sleeper K is out of scope). Three scripts back this: `src/analysis/analysis_nflcom_baseline.py` (NFL.com vs actuals), `src/analysis/sleeper_loader.py` (RotoWire ingest + a provenance/look-ahead gate), and `src/analysis/analysis_expert_comparison.py` (same-sample model-vs-each-expert with ranking + paired bootstrap / Diebold-Mariano significance). Reports regenerate on demand; none is committed.

4. **Other expert sites largely publish relative rankings, not per-game error.** FantasyPros ranks experts 1st/2nd/3rd by position — its "accuracy gap" is converted to z-scores before aggregation, so no raw MAE/RMSE is surfaced. FFA *does* compute the full error suite: its [2013 post](https://fantasyfootballanalytics.net/2013/05/the-best-fantasy-football-projections.html) published raw R²/MAE/RMSE (e.g., NFL.com RMSE 62.47, FantasyPros 45.36), but those are **single-league, full-season totals** (not per-game), and its current per-game MAE/RMSE lives behind the FFA Insider paywall. So a like-for-like *per-game* numeric comparison against those sources isn't possible from free, citable data — though our results fall within the ranges where professional projection systems operate. (DFS/industry sources — 4for4, Establish The Run, Subvertadown, PFF — publish "most accurate" marketing with no citable raw error metric.)

5. **TD variance remains the structural ceiling at QB.** Per-target QB MAE is dominated by `passing_yards` (68 yds = 2.7 fantasy pts) and `passing_tds` (0.89 TDs = 3.5 fantasy pts at 4 pts/TD). The TD head is the single largest contributor and has the lowest R² of any QB target — week-to-week QB TD output is closer to noise than to a learnable signal at this player-pool size.

---

## Sources

1. Fantasy Football Analytics. "Which Fantasy Football Projections Are Most Accurate?" (Dec 2024). https://fantasyfootballanalytics.net/2024/12/which-fantasy-football-projections-are-most-accurate.html
2. FantasyPros. "2025 Fantasy Football Accuracy Scores." https://www.fantasypros.com/nfl/accuracy/
3. Fantasy Projection Lab. "What Makes a Fantasy Projection Accurate: Metrics and Benchmarks." https://fantasyprojectionlab.com/what-makes-a-projection-accurate
4. Kapania, N. "Fantasy Football and Machine Learning." Stanford CS 229 (2012). https://cs229.stanford.edu/proj2012/Kapania-FantasyFootballAndMachineLearning.pdf
5. Hua, R. et al. "Deep Artificial Intelligence for Fantasy Football Language Understanding." arXiv:2111.02874 (2021). https://arxiv.org/pdf/2111.02874
6. Dross, E. "Predicting NFL Players Fantasy Performance." INST 414 (2023). https://medium.com/inst414-data-science-tech/predicting-nfl-players-fantasy-performance-362f5f1828d3
7. "Data Analysis on Predicting the Top 12 Fantasy Football Players by Position." SMU Data Science Review, Vol. 8, No. 2 (2024). https://scholar.smu.edu/datasciencereview/vol8/iss2/7/
8. Chant, J. "NFL Fantasy Point Prediction." (2023). https://jadenchant.github.io/pages/projects/nfl_fantasy_pt/
9. Gneiting, T. "Making and Evaluating Point Forecasts." *Journal of the American Statistical Association* 106(494), 746–762 (2011). https://arxiv.org/abs/0912.0902
10. Taggart, R. J. "Point forecasting and forecast evaluation with generalized Huber loss." *Electronic Journal of Statistics* 16(1), 201–231 (2022). https://arxiv.org/abs/2108.12426
11. Diebold, F. X. & Mariano, R. S. "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3), 253–263 (1995). https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599
12. Beal, R., Norman, T. J. & Ramchurn, S. D. "Optimising Daily Fantasy Sports Teams with Artificial Intelligence." *International Journal of Computer Science in Sport* 19(2), 21–35 (2020). https://eprints.soton.ac.uk/445995/1/DFS_IJCSS.pdf
13. Lutz, R. "Fantasy Football Prediction." arXiv:1505.06918 (2015). https://arxiv.org/abs/1505.06918

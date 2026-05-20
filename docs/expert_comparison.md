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
- Despite these limitations, published accuracy thresholds and academic benchmarks provide meaningful context for interpreting our results.

---

## Our Model Performance

From the 2026-05-20 benchmark run (`benchmark_history/2026-05-20T06-11-13_de8b961.json`, plus K from the post-`#227` re-run `2026-05-20T06-39-23_4af6a9d.json`), evaluated on the 2025 test season:

| Model | QB MAE | QB R² | RB MAE | RB R² | WR MAE | WR R² | TE MAE | TE R² | K MAE | K R² | DST MAE | DST R² |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge Regression | 6.546 | 0.217 | 4.399 | 0.339 | 4.351 | 0.342 | 3.546 | 0.315 | 4.079 | 0.012 | 5.212 | 0.044 |
| Multi-Head NN | 6.435 | 0.229 | 4.278 | 0.398 | 4.182 | 0.348 | 3.494 | 0.305 | 4.128 | -0.042 | **5.162** | **0.067** |
| Attention NN | 6.544 | 0.222 | 4.247 | 0.376 | **4.106** | 0.350 | **3.445** | 0.301 | 4.132 | -0.021 | 5.288 | -0.009 |
| LightGBM | **6.256** | **0.269** | **4.159** | **0.417** | 4.231 | **0.356** | 3.517 | **0.317** | **4.061** | -0.013 | 5.274 | 0.015 |
| **Best (MAE)** | LGBM 6.256 | LGBM 0.269 | LGBM 4.159 | LGBM 0.417 | Attn NN 4.106 | LGBM 0.356 | Attn NN 3.445 | LGBM 0.317 | LGBM 4.061 | Ridge 0.012 | NN 5.162 | NN 0.067 |

**Per-target MAE breakdown** (Attention NN, in native stat units):

| Position | Per-target MAE |
|----------|----------------|
| **QB** | passing_yards: 68.4 yds, rushing_yards: 11.4 yds, passing_tds: 0.88, rushing_tds: 0.25, interceptions: 0.61, fumbles_lost: 0.25 |
| **RB** | rushing_yards: 18.3 yds, receiving_yards: 9.2 yds, receptions: 0.98, rushing_tds: 0.31, receiving_tds: 0.10, fumbles_lost: 0.04 |
| **WR** | receiving_yards: 19.9 yds, receptions: 1.32, receiving_tds: 0.29, fumbles_lost: 0.01 |
| **TE** | receiving_yards: 14.8 yds, receptions: 1.22, receiving_tds: 0.28, fumbles_lost: 0.01 |

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

Unlike the FFA-tracked sources above (which publish only relative accuracy rankings), NFL.com publishes per-game projected raw stats we can score directly through our PPR aggregator and compare to actual results — giving a raw-MAE number that's directly comparable to our model's. Numbers below come from [`analysis_output/nflcom_baseline.json`](../analysis_output/nflcom_baseline.json), produced by [`src/analysis/analysis_nflcom_baseline.py`](../src/analysis/analysis_nflcom_baseline.py).

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
| **QB** | 6.256 (LightGBM) | 5.713 | NFL.com leads by 0.54 |
| **RB** | 4.159 (LightGBM) | 4.110 | NFL.com leads by 0.05 (≈ tied) |
| **WR** | 4.106 (Attention NN) | 3.983 | NFL.com leads by 0.12 |
| **TE** | 3.445 (Attention NN) | 3.390 | NFL.com leads by 0.06 (≈ tied) |
| **K**  | 4.061 (LightGBM) | 4.358 | We lead by 0.30 |
| **DST** | 5.162 (Multi-Head NN) | — | (no NFL.com DST data) |

QB is the only position where NFL.com beats us by a meaningful margin (~0.5 MAE pts); RB / WR / TE are within 0.1–0.15 pts, which is below the run-to-run variance we typically see between benchmark snapshots. K is the only position where we beat NFL.com cleanly — partly because their kicker projection schema is per-distance-bucket FG attempts that doesn't aggregate well, and partly because our K model now keys directly off cached PBP after the `fg_yards_made` schema fix in [#227](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/227).

The QB gap is consistent with the structural QB limitation noted under "Position-by-Position Analysis": touchdown variance dominates QB error, and a curated expert projection that incorporates context (matchup, narrative, etc.) can suppress that volatility better than a stats-driven model trained on broader player pools.

**Caveats:**
- Single test season (2025). 2024 was used for hyperparameter tuning, so it's not a clean held-out comparison; the analysis script is multi-season-capable and will pool across years automatically as `TEST_SEASONS` grows.
- NFL.com's player pool is implicitly curated (they only project players likely to play); our model evaluates every rostered player at each position. NFL.com's MAE is on `n_matched` rows where both sides have a projection, so this is partially controlled, but coverage differences in marginal players still affect the per-position N.
- The 92.2% NFL.com name-match rate (see the loader logs) drops ~7.8% of NFL.com rows that couldn't be joined to a `gsis_id` — mostly free-agent placeholders ("Alex Hale (K, FA)", etc.).

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

---

## Position-by-Position Analysis

### Quarterbacks

**Our best MAE: 6.256 (LightGBM)** --- falls within the 5.0-7.0 "established systems" range.

QBs are generally considered the most predictable position due to consistent volume (30+ pass attempts per game), yet our R² is moderate (0.217–0.269 across models). The per-target breakdown reveals why: `passing_tds` and `rushing_tds` carry meaningful per-game volatility (a QB throwing 1 vs. 3 TDs swings the score by 8 fantasy points), and this volatility limits any model's predictive power on a weekly basis. The `passing_yards` head alone accounts for the bulk of fantasy-point variance (MAE 68 yards × 0.04 pts/yd ≈ 2.7 fantasy points).

The deep learning NLP study (arXiv:2111.02874) reported a comparable RMSE of 6.78 using ESPN text features, suggesting our QB MAE of 6.26 is in line with neural-network approaches in the literature. LightGBM edges the attention NN here by ~0.3 MAE points, which is consistent with tree models being well-suited to high-variance count targets (TDs) where smooth gradients don't help much.

### Running Backs

**Our best MAE: 4.159 (LightGBM)** --- in the "competitive" range (< 5.0), narrowly.

RBs are among the most volatile fantasy positions due to game-script dependence and TD variance, making a sub-5 MAE a solid result. For context:
- The k-NN academic benchmark reported RB RMSE of 4.19, and our MAE of 4.16 is below that (MAE ≤ RMSE by definition).
- Our R² of 0.417 (LightGBM) is just under the 0.53–0.61 range reported by Kapania's multi-season Stanford study — but our test set is the full 2025 RB pool, not just starters, which suppresses R² on a noisier denominator.
- Per-target MAE distributes the error: rushing_yards 18.3 yds × 0.1 ≈ 1.8 pts; receiving_yards 9.2 yds × 0.1 ≈ 0.9 pts; receptions 0.98 × 1.0 ≈ 1.0 pt; rushing_tds 0.31 × 6 ≈ 1.9 pts. TDs remain the dominant single source of error in fantasy-point space.

### Wide Receivers

**Our best MAE: 4.106 (Attention NN)** --- competitive (< 5.0), with the attention NN edging LightGBM by 0.12 MAE points.

WR is the position where the attention architecture pulls clearest of the trees: receiver production has strong route/scheme/coverage dependencies that benefit from learned attention over player history + opponent defensive context.
- The academic linear regression benchmark for WRs reported RMSE of 3.64, which is lower but used a curated player pool (top 50 WRs only) and a different season.
- Our R² of 0.350 (Attention NN) sits within the published WR consistency range (R² ~ 0.30–0.65 depending on methodology).
- Per-target MAE: receiving_yards 19.9 yds × 0.1 ≈ 2.0 pts is the largest contributor; receptions 1.32 × 1.0 (PPR) ≈ 1.3 pts; receiving_tds 0.29 × 6 ≈ 1.7 pts.

### Tight Ends

**Our best MAE: 3.445 (Attention NN)** --- comfortably in the competitive range.

TEs are the lowest-volume offensive skill position and produce the lowest-variance fantasy outputs of QB/RB/WR/TE, which gives the model a structurally easier target — most starting TEs land in a 4–10 point band most weeks. The attention NN narrowly edges LightGBM (3.445 vs 3.517 MAE); per-target the receptions head (1.22 MAE) carries proportionally more of the fantasy-point error than for WR because TE target volume is lower (3–5 targets/game typical).

### Kickers

**Our best MAE: 4.061 (LightGBM)** --- in the competitive band, on a position no public benchmark covers carefully.

K is interesting because kicker fantasy production is *highly* opponent/weather/coach dependent — and our model now beats NFL.com here (4.061 vs 4.358 MAE) after the `fg_yards_made` schema fix in [#227](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/227) restored the FG-distance signal that had been silently dropped from the PBP cache. K's R² hovers near zero across all four models (−0.04 to 0.01); on this position, "low MAE + near-zero R²" is the right operating point — kicker variance week-to-week is too high to extract a meaningful rank-order signal, but absolute prediction error is tight.

### Defense / Special Teams

**Our best MAE: 5.162 (Multi-Head NN)** --- in the 5.0–7.0 "established systems" band.

DST landed via [commit `cc0c627`](../) with a 10-target attention model (sacks, INTs, fumble recoveries, fumbles forced, safeties, def TDs, blocked kicks, ST TDs, points allowed, yards allowed) plus a parallel opp-OFFENSE attention branch (PR [#223](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/223)) that conditions defensive predictions on the offense being faced. The Multi-Head NN (no attention) slightly outperforms the attention variant on DST (5.162 vs 5.288 MAE) — DST signal is structurally simpler than skill-position production and the attention branch hasn't added clear value yet on the current feature set.

---

## Key Findings

1. **All six positions are now in the "competitive" or "established" MAE bands.** QB / DST sit in the 5.0–7.0 "established systems" band; RB / WR / TE / K all clear the < 5.0 "competitive" threshold. The previous gap where K/DST results were pending is closed (this iteration of the doc).

2. **No single architecture wins everywhere.** LightGBM wins QB / RB / K (and ties WR / TE on R²); Attention NN wins WR / TE; Multi-Head NN wins DST. This is the strongest signal yet that maintaining a diverse model portfolio is worth the training cost — the per-position routing in `src.benchmarking.benchmark` picks the actual best architecture per position rather than locking in one family.

3. **NFL.com gives us a direct head-to-head.** The only mainstream source publishing scoring-grade raw projections we can join to actuals. On 2025: QB is the only position where NFL.com beats us by a meaningful margin (~0.5 MAE pts); RB / WR / TE are within run-to-run variance; K we beat cleanly. Full per-target and weekly breakouts in [`analysis_output/nflcom_baseline.json`](../analysis_output/nflcom_baseline.json).

4. **Other expert sites publish only relative rankings.** Industry accuracy tracking (FantasyPros, FFA) ranks sources 1st/2nd/3rd by position rather than reporting raw MAE. This makes exact head-to-head comparison with those sources impossible, but our results fall within the ranges where professional projection systems operate.

5. **TD variance remains the structural ceiling at QB.** Per-target QB MAE is dominated by `passing_yards` (68 yds = 2.7 fantasy pts) and `passing_tds` (0.88 TDs = 3.5 fantasy pts at 4 pts/TD). The TD head is the single largest contributor and has the lowest R² of any QB target — week-to-week QB TD output is closer to noise than to a learnable signal at this player-pool size.

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

# Fantasy Football Weekly Points Predictor — Code Design Document

## For use with Claude Code / AI-assisted development on CS 372 Final Project

---

## 1. Project Overview

**Goal:** Build a machine learning system that predicts weekly fantasy football points for NFL players (QB, RB, WR, TE, K, DST), comparing a Ridge Regression baseline against a custom PyTorch multi-head neural network. Supports three scoring formats: Standard (0 PPR), Half-PPR, and Full PPR.

**Data source:** `nfl_data_py` (Python wrapper for nflverse), pulling weekly player stats, roster data, and schedule data from the 2012–2025 NFL seasons.

**Core question:** Can a multi-head neural network with engineered temporal features and target decomposition meaningfully outperform Ridge regression at predicting weekly fantasy output, and what features matter most?

---

## 2. Rubric Strategy (15 ML Items → Target ~73 pts)

Below are the 15 Machine Learning rubric items this project is designed to hit. Every architectural decision in this document maps back to one or more of these.

| # | Rubric Item | Pts | Where in Code |
|---|-------------|-----|---------------|
| 1 | **Solo project credit** | 10 | N/A |
| 2 | **Collected/constructed original dataset through API integration** (`nfl_data_py` ingestion, multi-season join, feature build) | 10 | `src/data/` |
| 3 | **Compared multiple model architectures quantitatively** (Ridge vs Multi-Head Neural Net, controlled setup) | 7 | `shared/models.py`, `shared/neural_net.py`, position configs |
| 4 | **Error analysis with visualization and discussion of failure cases** | 7 | `docs/expert_comparison.md`, pipeline outputs |
| 5 | **Applied ML to time-series forecasting** (weekly player projections using rolling temporal features) | 7 | `src/features/engineer.py`, position pipelines |
| 6 | **Simulation-based evaluation** (week-by-week prediction accuracy simulation) | 7 | `shared/backtest.py` |
| 7 | **Feature engineering** (rolling averages, snap %, target share, matchup strength, etc.) | 5 | `src/features/engineer.py` |
| 8 | **Defined and trained a custom neural network architecture** (PyTorch) | 5 | `shared/neural_net.py` |
| 9 | **Systematic hyperparameter tuning** (≥3 configs documented) | 5 | Position-specific configs (`QB/qb_config.py`, etc.) |
| 10 | **Regularization** (L2 + dropout + early stopping — at least 2 required) | 5 | `shared/neural_net.py`, `shared/training.py` |
| 11 | **Modular code design** with reusable functions/classes | 3 | `src/`, `shared/`, position folders |
| 12 | **Train/validation/test split** with documented ratios | 3 | `src/data/split.py` |
| 13 | **Training curves** (loss and metrics over time) | 3 | `shared/training.py` |
| 14 | **Baseline model** (constant prediction = player's season average) | 3 | `src/models/baseline.py` |
| 15 | **Used ≥3 distinct evaluation metrics** | 3 | `src/evaluation/metrics.py` |
| 16 | **Interpretable model design or explainability analysis** (Ridge coefficients + permutation importance for NN) | 7 | Pipeline outputs, `docs/expert_comparison.md` |

**Total if all hit: 90 pts (capped at 73).** Only 15 of 16 items need to land to cap out, giving a 17-pt buffer. The rubric grades only the first 15 selected items, so list the strongest 15 in the self-assessment and keep the 16th as insurance.

### Rubric Overflow: Items Implemented But Not Counted in the 16

These items are naturally implemented by the project but are **not** claimed as part of the 15/16. They further demonstrate depth:

| Rubric Item | Pts | Where in Code |
|-------------|-----|---------------|
| Properly normalized/standardized input features | 3 | `shared/models.py` (StandardScaler in RidgeMultiTarget) |
| Applied basic preprocessing appropriate to modality | 3 | `src/data/preprocessing.py` |
| Used appropriate data loading with batching and shuffling | 3 | `shared/training.py` (DataLoader) |
| Used learning rate scheduling | 3 | `shared/training.py` (ReduceLROnPlateau, OneCycleLR, CosineWarmRestarts) |
| Applied batch normalization | 3 | `shared/neural_net.py` (BatchNorm1d) |
| Documented design decision with technical tradeoffs | 3 | Section 6 of this doc + technical walkthrough |
| Substantive ATTRIBUTION.md on AI tool usage | 3 | `ATTRIBUTION.md` |

### One Piece of Work → One Item Rule

The rubric states each piece of work can claim at most one item (plus stackable add-ons). Here is the explicit mapping to avoid double-counting:

| Work | Claims Item # | Does NOT claim |
|------|--------------|----------------|
| Rolling window features + share stats + matchup features | #7 (Feature engineering, 5 pts) | — |
| Temporal split + rolling window modeling paradigm | #5 (Time-series forecasting, 7 pts) | — |
| Multi-head PyTorch architecture (shared backbone + per-target heads) | #8 (Custom neural network, 5 pts) | — |
| Dropout + L2 weight decay + early stopping + BatchNorm | #10 (Regularization, 5 pts) | — |
| Per-position config tuning (6 positions × architecture/loss/scheduler) | #9 (Systematic tuning, 5 pts) | — |
| RidgeMultiTarget vs MultiHeadNet controlled comparison | #3 (Architecture comparison, 7 pts) | — |
| Ridge coefficients + permutation importance | #16 (Explainability, 7 pts) | — |

---

## 3. Repository Structure

```
Final-Project/
├── app.py                         # Flask web application (main entry point + predictions dashboard)
├── requirements.txt               # Pinned dependencies
├── benchmark_nn.py                # Neural network benchmarking script
├── .gitignore
│
├── src/                           # General multi-position pipeline
│   ├── config.py                  # Central config: scoring, seasons, hyperparams
│   ├── data/
│   │   ├── loader.py              # nfl_data_py ingestion + caching
│   │   ├── preprocessing.py       # Cleaning, missing values, filtering
│   │   └── split.py               # Temporal train/val/test split
│   ├── features/
│   │   └── engineer.py            # All feature engineering logic
│   ├── models/
│   │   ├── baseline.py            # Constant-prediction baseline
│   │   └── linear.py              # Scikit-learn Ridge Regression
│   ├── training/
│   │   └── trainer.py             # Legacy trainer (see shared/training.py)
│   └── evaluation/
│       ├── metrics.py             # MAE, RMSE, R², per-position breakdown
│       └── backtest.py            # Season-long fantasy simulation
│
├── shared/                        # Generic multi-target infrastructure
│   ├── neural_net.py              # MultiHeadNet, MultiHeadNetWithHistory (attention), GatedTDHead
│   ├── models.py                  # RidgeMultiTarget, TwoStageRidge
│   ├── training.py                # MultiHeadTrainer, MultiTargetLoss, dataloaders
│   ├── pipeline.py                # Position pipeline template
│   ├── evaluation.py              # Evaluation utilities
│   ├── backtest.py                # Simulation/evaluation helpers
│   └── weather_features.py        # Vegas odds + venue/weather feature engineering
│
├── QB/                            # QB-specific model (config, data, targets, features, pipeline)
│   ├── qb_config.py
│   ├── qb_data.py
│   ├── qb_targets.py
│   ├── qb_features.py
│   ├── run_qb_pipeline.py
│   └── outputs/                   # Trained models + figures
│
├── RB/                            # RB-specific model
│   ├── rb_config.py               # (same structure as QB/)
│   ├── rb_data.py
│   ├── rb_targets.py
│   ├── rb_features.py
│   ├── run_rb_pipeline.py
│   ├── outputs/
│   └── tests/
│
├── WR/                            # WR-specific model (same structure)
├── TE/                            # TE-specific model (same structure)
├── K/                             # Kicker model (custom feature pipeline, bypasses general features)
├── DST/                           # D/ST model (custom feature pipeline, bypasses general features)
│
├── templates/
│   └── index.html                 # Main dashboard HTML
├── static/
│   └── css/style.css              # CSS styling
│
├── tests/
│   └── test_feature_leakage.py    # Verifies no future data leakage
│
├── data/                          # .gitignore — generated at runtime
│   ├── raw/                       # Cached nfl_data_py pulls
│   └── splits/                    # Train/val/test parquet files
│
├── instructions/                  # Project specification documents
│   ├── DESIGN_DOC.md              # This file
│   ├── METHOD_CONTRACTS.md        # Function signatures & data schemas
│   └── final_project_handout.html # Original assignment handout
│
└── docs/                          # Technical design documents
    ├── expert_comparison.md       # Comparison against published benchmarks
    ├── design_lstm_multihead.md   # LSTM + sequential modeling proposal
    └── design_weather_and_odds.md # Vegas odds & venue features proposal
```

---

## 4. Detailed Module Specifications

### 4.1 `src/config.py` — Central Configuration

All magic numbers live here. Nothing is hardcoded in other files.

```python
# Key constants to define:
SEASONS = list(range(2012, 2026))       # nfl_data_py seasons to pull (2012-2025)
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
MIN_GAMES_PER_SEASON = 6

# Base scoring (shared across all formats)
_BASE_SCORING = {"passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
                 "rushing_yards": 0.1, "rushing_tds": 6,
                 "receiving_yards": 0.1, "receiving_tds": 6,
                 "fumbles_lost": -2}

# Reception weights per format
PPR_FORMATS = {"standard": 0.0, "half_ppr": 0.5, "ppr": 1.0}

# Full scoring dicts per format
SCORING_STANDARD = {**_BASE_SCORING, "receptions": 0.0}
SCORING_HALF_PPR = {**_BASE_SCORING, "receptions": 0.5}
SCORING_PPR      = {**_BASE_SCORING, "receptions": 1.0}
SCORING = SCORING_PPR  # Default (full PPR)

# Temporal split boundaries (season-based, NOT random)
TRAIN_SEASONS = list(range(2012, 2024))  # 2012-2023
VAL_SEASONS   = [2024]                   # 2024
TEST_SEASONS  = [2025]                   # 2025

# Cross-validation (expanding window)
CV_VAL_SEASONS = [2021, 2022, 2023, 2024]

# Rolling feature windows
ROLLING_WINDOWS = [3, 5, 8]             # weeks lookback
ROLL_STATS = [                           # stats to compute rolling features for
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "passing_yards",
    "attempts", "snap_pct",
]
ROLL_AGGS = ["mean", "std", "max"]

# EWMA (exponentially weighted moving averages)
EWMA_STATS = ["fantasy_points", "targets", "carries", "receiving_yards",
              "rushing_yards", "passing_yards", "snap_pct"]
EWMA_SPANS = [3, 5]

# Trend/momentum
TREND_STATS = ["fantasy_points", "targets", "carries", "snap_pct"]

# Share features (multi-window)
SHARE_WINDOWS = [3, 5]

# Neural net defaults (global; overridden per-position in {POS}/{pos}_config.py)
NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4                  # L2 regularization
NN_EPOCHS = 200
NN_BATCH_SIZE = 256
NN_PATIENCE = 15                        # early stopping
```

**Position-specific overrides (from `{POS}/{pos}_config.py`):**

| Position | Backbone | Head Hidden | Dropout | LR | Epochs | Patience | Batch | Scheduler |
|----------|----------|-------------|---------|-----|--------|----------|-------|-----------|
| QB | [128] | 32 | 0.20 | 5e-4 | 300 | 25 | 128 | CosineWarmRestarts |
| RB | [128, 64] | 48 (td: 64) | 0.15 | 1e-3 | 300 | 30 | 256 | CosineWarmRestarts |
| WR | [128] | 32 | 0.20 | 1e-3 | 250 | 25 | 512 | CosineWarmRestarts |
| TE | [96, 48] | 24 (td: 32) | 0.30 | 5e-4 | 300 | 25 | 128 | OneCycleLR |
| K | [64, 32] | 16 | 0.25 | 3e-4 | 250 | 30 | 128 | OneCycleLR |
| DST | [128, 64] | 32/16/48 | 0.30 | 3e-4 | 300 | 35 | 128 | CosineWarmRestarts |

### 4.2 `src/data/loader.py` — Data Ingestion

**Rubric target: "Collected/constructed original dataset through API integration" (10 pts)**

```python
# Responsibilities:
# 1. Pull weekly player stats via nfl_data_py.import_weekly_data(SEASONS)  # 2012-2025
# 2. Pull roster data via nfl_data_py.import_rosters(SEASONS) for position info
# 3. Pull schedule/game data via nfl_data_py.import_schedules(SEASONS) for opponent info
# 4. Pull snap counts via nfl_data_py.import_snap_counts(SEASONS) for snap_pct
# 5. Merge into a single player-week DataFrame
# 6. Cache to data/raw/ as parquet to avoid re-pulling
#
# Key function signatures:
#   load_raw_data(seasons, cache_dir) -> pd.DataFrame
#   compute_fantasy_points(df, scoring_dict) -> pd.Series
```

**Important nfl_data_py notes:**
- `import_weekly_data()` returns one row per player per week with all stat columns
- Columns include: `player_id`, `player_name`, `position`, `recent_team`, `season`, `week`, plus stat columns like `passing_yards`, `rushing_yards`, `receptions`, `targets`, `carries`, `passing_tds`, `rushing_tds`, `receiving_tds`, `interceptions`, `sack_fumbles_lost`, etc.
- The `player_id` field (GSIS ID) is the stable key across seasons
- Use `import_rosters()` to get reliable position labels (weekly data position can be inconsistent)
- `import_snap_counts()` gives snap count and snap percentage — very useful feature

**Snap count integration:**
- `import_snap_counts()` uses `pfr_player_id` (Pro Football Reference ID), NOT the GSIS `player_id` used in weekly data. These are different ID systems.
- Merge strategy: use `pfr_player_id` if roster data provides it, otherwise fall back to name + team + season + week matching
- After merge, rename `offense_pct` → `snap_pct`
- If snap count data is unavailable for certain seasons/weeks, fill `snap_pct` with NaN (handled in preprocessing)

**Fantasy point computation — fumble column mapping:**
- `nfl_data_py` may split fumbles into `sack_fumbles_lost` and `rushing_fumbles_lost`
- Sum all `*_fumbles_lost` columns to compute total `fumbles_lost` for scoring
- Fill any missing stat columns with 0 before computing (player simply didn't record that stat)

**Parquet caching strategy:**
- Save separate files per data type: `data/raw/weekly_{min_season}_{max_season}.parquet`, `data/raw/rosters_{...}.parquet`, `data/raw/schedules_{...}.parquet`, `data/raw/snap_counts_{...}.parquet`
- On load: check if cached file exists → use it; otherwise pull from API and save
- Re-pull only if cache file is missing (manual delete to force refresh)

### 4.3 `src/data/preprocessing.py` — Cleaning

**Rubric target: "Applied basic preprocessing" (supports the 10-pt dataset item)**

```python
# Responsibilities:
# 1. Filter to positions (QB, RB, WR, TE, K, DST)
# 2. Remove bye weeks / players with 0 snaps (did not play)
# 3. Handle missing values:
#    - Statistical columns: fill with 0 (player didn't record that stat)
#    - Snap percentage: fill with position median for that week
# 4. Compute target variable: fantasy_points using config.SCORING
# 5. Compute fantasy_points_floor (yardage + receptions only, no TDs)
#
# NOTE: The min-games filter (< 6 games/season) is NOT applied here.
# It is deferred to build_features() AFTER team totals are computed,
# so that fringe players' stats contribute to correct team denominators
# for share features. Filtering before team totals would inflate
# target_share and carry_share for remaining players.
#
# Key function:
#   preprocess(raw_df) -> pd.DataFrame
```

### 4.4 `src/data/split.py` — Temporal Split

**Rubric target: "Train/val/test split with documented ratios" (3 pts)**

```python
# CRITICAL: This is time-series data. DO NOT use random splits.
# Split by SEASON to prevent data leakage from future weeks.
#
#   Train: 2012-2023 seasons (~12 seasons of data)
#   Val:   2024 season
#   Test:  2025 season
#
# Cross-validation uses expanding window with CV_VAL_SEASONS = [2021, 2022, 2023, 2024]
#
# Key function:
#   temporal_split(df, train_seasons, val_seasons, test_seasons)
#       -> (train_df, val_df, test_df)
#
# Also export split sizes to outputs/ for documentation.
```

### 4.5 `src/features/engineer.py` — Feature Engineering

**Rubric targets: "Feature engineering" (5 pts) + "Time-series forecasting" (7 pts)**

This is the most critical module. The model is only as good as its features.

```python
# === PLAYER ROLLING FEATURES (per player, per season, per window) ===
# For each stat and window w, compute using ONLY past data (shift(1) before rolling):
#   - rolling_mean_{stat}_L{w}       (e.g., rolling_mean_fantasy_points_L3)
#   - rolling_std_{stat}_L{w}        (volatility / consistency)
#   - rolling_max_{stat}_L{w}        (ceiling indicator)
#   - rolling_min_fantasy_points_L{w} (floor indicator — fantasy_points only)
#
# Key stats to roll: fantasy_points, fantasy_points_floor, targets, receptions,
#                     carries, rushing_yards, receiving_yards, passing_yards,
#                     attempts (passing_attempts), snap_pct
# NOTE: "attempts" added for QB volume signal; "fantasy_points_floor" separates
#       stable yardage production from volatile TD-dependent upside.
#
# === PRIOR-SEASON SUMMARY FEATURES ===
# Per-player, per-season aggregates (mean/std/max) shifted by 1 season.
# Provides stable signal for early-season weeks where within-season
# rolling features are NaN. Also captures offseason changes in role.
#
# === EWMA FEATURES (exponentially weighted moving averages) ===
# Recency-weighted averages with spans [3, 5] for key stats.
# Weight recent games more heavily — a player who scored 25 last week
# is more likely to score well than one who scored 25 three weeks ago.
#
# === TREND / MOMENTUM FEATURES ===
# trend_{stat} = rolling_mean_L3 - rolling_mean_L8
# Positive = player trending up (breakout signal); negative = declining.
#
# === USAGE / OPPORTUNITY FEATURES (multi-window rolling shares) ===
#   - target_share_L3, target_share_L5: multi-window to distinguish
#     recent usage breakouts from stable workload
#   - carry_share_L3, carry_share_L5: same pattern
#   - snap_pct:          from snap count data (direct feature, not derived)
#   - air_yards_share:   rolling_sum(player_air_yards, L5) / team, shifted
#   NOTE: Share features use stint-aware grouping to handle mid-season trades.
#
# === MATCHUP / OPPONENT FEATURES ===
#   - opp_fantasy_pts_allowed_to_pos: rolling average of total fantasy points the
#     upcoming opponent has allowed to this position over last 5 weeks
#   - opp_rush_pts_allowed_to_pos: rushing-only fantasy points allowed (rush yds + rush TDs)
#   - opp_recv_pts_allowed_to_pos: receiving-only fantasy points allowed (recv yds + recv TDs + receptions)
#   - opp_def_rank_vs_pos: rank version of total (1 = worst D = best matchup)
#
# === CONTEXTUAL FEATURES ===
#   - is_home:           binary, home vs away
#   - week:              integer 1-18 (captures early/late season patterns)
#   - is_returning_from_absence: binary, did player miss 1+ games before this week
#   - days_rest:         days since last game (bye week detection)
#
# === POSITION ENCODING ===
#   - One-hot encode position (QB, RB, WR, TE)
#
# === TEAM-LEVEL AGGREGATION (for share features) ===
# 1. Group by (recent_team, season, week) and sum targets, carries, air_yards
#    — computed BEFORE min-games filter so fringe players' stats are included
# 2. Merge team totals back to player rows
# 3. Apply min-games filter AFTER team totals are computed
# 4. Detect mid-season team changes, create stint_id per (player_id, season)
# 5. Compute multi-window rolling sums for BOTH player and team stats (shifted)
#    using (player_id, season, stint_id) grouping for trade safety
# 6. Share = player_rolling_sum / team_rolling_sum
# 7. IMPORTANT: Both numerator and denominator use shift(1) + rolling to
#    prevent leakage from the current week. Handle 0/0 by filling with 0.
#
# === OPPONENT FEATURE PIPELINE (leakage-safe) ===
# 1. From schedule data, get each team's opponent for each week
# 2. Merge opponent info onto player rows (via player's team + week)
# 3. Compute: for each (team_as_defense, position, week), sum fantasy points
#    that opposing players of that position scored against them
# 4. shift(1) the defense stats per (team, position) BEFORE rolling —
#    this ensures week W's matchup feature does NOT include week W's game
# 5. Rolling 5-week mean of the shifted defense stats
# 6. Rank within each week (1 = most points allowed = best matchup)
# 7. Join back to player rows by (opponent_team, position, week)
#
# === CRITICAL IMPLEMENTATION NOTES ===
# 0. SORT the DataFrame by (player_id, season, week) before any rolling
#    operations. Unsorted data causes shift(1) to return wrong rows.
# 1. ALL rolling features must use .shift(1) before .rolling() to prevent
#    data leakage — you cannot use the current week's stats to predict
#    the current week's points.
# 2. Rolling operations MUST stay within (player_id, season) groups. Use
#    df.groupby(["player_id", "season"])[stat].transform(lambda x: x.shift(1).rolling(...))
#    Grouping by player_id alone causes cross-season contamination:
#    shift(1) on Week 1 of 2020 would return Week 17 of 2019, bridging
#    the 4-8 month offseason gap with stale data from potentially
#    different teams, coaches, and schemes.
#    Also do NOT use: groupby().shift() followed by bare .rolling() —
#    the latter rolls across adjacent players' rows.
# 3. rolling std with min_periods=1 still returns NaN for single observations
#    (std of one value is undefined). First row per player per season will
#    have NaN for all std features. Handled by fill_nans_safe().
# 4. The first few weeks of each season will have NaN rolling features.
#    Explicit prior-season summary features (mean/std/max per player per
#    prior season) provide stable fallback signal for early-season weeks.
#    Remaining NaNs are filled by fill_nans_safe() AFTER temporal_split(),
#    using ONLY training set statistics to prevent data leakage.
# 5. Group all rolling computations by (player_id, season).
# 6. Rookies in test season: no prior season data → fill_nans_safe() uses
#    position-level training set averages for all rolling features.
#
# === EXPECTED FEATURE COUNT ===
# Rolling (mean/std/max): 10 stats × 3 windows × 3 aggs = 90
# Rolling min (fp only):  1 stat × 3 windows = 3
# Prior-season summary:   10 stats × 3 aggs = 30
# EWMA:                   7 stats × 2 spans = 14
# Trend:                  4 stats = 4
# Share:                  target_share × 2 windows + carry_share × 2 windows
#                         + snap_pct + air_yards_share = 6
# Matchup:                opp_fantasy_pts_allowed_to_pos + opp_rush_pts_allowed_to_pos
#                         + opp_recv_pts_allowed_to_pos + opp_def_rank_vs_pos = 4
# Contextual:             is_home, week, is_returning_from_absence, days_rest = 4
# Position:               one-hot (QB, RB, WR, TE) = 4
# TOTAL: ~155 general features (before position-specific adds/drops)
# Each position then adds specific features and drops irrelevant ones.
# Weather/Vegas features (from shared/weather_features.py) are added per-position.
#
# NOTE: Exact count depends on which nfl_data_py columns are available
#       (air_yards_share may be absent). The NN input_dim should be set
#       dynamically from len(get_feature_columns()).
#
# Key function:
#   build_features(df) -> pd.DataFrame   # adds feature columns, returns full df
#   get_feature_columns() -> list[str]   # returns list of all feature column names
```

### 4.6 `src/models/baseline.py` — Constant Baseline

**Rubric target: "Baseline model for comparison" (3 pts)**

```python
# Two simple baselines:
#
# 1. SeasonAverageBaseline: Predict each player's season-to-date average
#    fantasy points (expanding mean up to but not including current week).
#    This is what a "naive" fantasy manager would roughly estimate.
#
# 2. LastWeekBaseline: Predict the player scored the same as last week.
#
# Both implement:
#   .predict(df) -> np.array of predictions
#
# These set the floor — if our ML models can't beat these, something is wrong.
```

### 4.7 `src/models/linear.py` — Linear Regression

**Rubric target: "Compared multiple architectures" (7 pts, part 1)**

```python
# Use scikit-learn's Ridge regression (L2-regularized linear regression).
#
# Pipeline:
#   1. StandardScaler (fit on train only, transform val/test)
#   2. Ridge(alpha=...) — tune alpha via validation set
#
# Wrap in a class with .fit(X_train, y_train) and .predict(X) interface.
# Store the fitted scaler for use by the neural net too (shared preprocessing).
#
# Scaler sharing — the pipeline (run_pipeline.py) owns the scaling step:
#   1. RidgeModel.fit() internally fits a StandardScaler on training data
#   2. After fitting Ridge, the pipeline extracts the scaler: ridge_model.scaler
#   3. The pipeline uses this scaler to transform train/val/test data for the NN
#   4. The NN Trainer receives PRE-SCALED numpy arrays — it has no scaler awareness
#   5. Save fitted scaler to outputs/models/scaler.pkl via joblib (for inference)
#   - This ensures identical feature scaling across models for fair comparison
#   - The Trainer class does NOT handle scaling — the pipeline does
#
# Also extract and save .coef_ for feature importance analysis.
```

### 4.8 `shared/neural_net.py` — Multi-Head Neural Network

**Rubric targets: "Custom neural network" (5 pts) + "Regularization" (5 pts) + "Compared multiple architectures" (7 pts, part 2)**

```python
# Architecture: Multi-Head MLP for position-agnostic fantasy point decomposition
#
# class MultiHeadNet(nn.Module):
#     def __init__(self, input_dim, target_names, backbone_layers, head_hidden=32,
#                  dropout=0.3, head_hidden_overrides=None):
#         # input_dim set dynamically from feature count (varies per position)
#         #
#         # Shared backbone:
#         #   For each layer in backbone_layers:
#         #     Linear -> BatchNorm1d -> ReLU -> Dropout
#         #
#         # Per-target output heads (one per target_names entry):
#         #   Linear(backbone_out, head_hidden) -> ReLU -> Linear(head_hidden, 1)
#         #   -> clamp(min=0)  # ensures non-negative outputs (configurable per-target)
#         #
#         # Total prediction = sum of all heads
#
#     def forward(self, x) -> dict:
#         # Returns dict with per-target predictions + "total"
#
# Example target decompositions:
#   QB: ["passing_floor", "rushing_floor", "td_points"]
#   RB: ["rushing_floor", "receiving_floor", "td_points"]
#   WR: ["receiving_floor", "rushing_floor", "td_points"]
#   TE: ["receiving_floor", "rushing_floor", "td_points"]
#   K:  ["fg_points", "pat_points"]
#   DST: ["defensive_scoring", "td_points", "pts_allowed_bonus"]
#
# clamp(min=0) ensures non-negativity with exact zeros (replaced
# Softplus, which created a ~0.69/head floor — see TODO.md [FIXED]).
# The non_negative_targets parameter controls which heads are clamped,
# allowing targets like DST pts_allowed_bonus to go negative.
#
# Backward compatibility: load_state_dict() handles old naming convention
# (e.g., "rushing_head" -> "heads.rushing_floor").
#
# Regularization employed (need at least 2 for rubric):
#   1. Dropout (position-specific, 0.20–0.35)
#   2. L2 weight decay (1e-4 to 3e-4) via AdamW optimizer
#   3. Early stopping (patience 25–30 on validation loss)
#   4. BatchNorm acts as mild regularizer
#
# Loss function: Huber loss (per-target + total, with per-target weights and deltas)
# Optimizer: AdamW (Adam with decoupled weight decay)
#
# === Additional Architectures ===
#
# MultiHeadNetWithHistory: Extends MultiHeadNet with a parallel attention branch
# over raw game history sequences (up to 17 games). Uses learned-query
# AttentionPool to compress variable-length history into a fixed representation,
# concatenated with static features before the shared backbone.
# Trained for QB, RB, WR, TE (not K/DST).
#
# GatedTDHead: Two-stage hurdle head for zero-inflated TD prediction.
# Stage 1 (gate): P(TD > 0) via sigmoid
# Stage 2 (value): E[TD | TD > 0] via Softplus (always positive)
# Output: gate_prob × cond_value = E[TD]
# Used by attention NN variants across all skill positions.
#
# LightGBM: Gradient-boosted tree model (Optuna-tuned) trained for QB, RB, WR, TE.
# Provides a non-neural, non-linear baseline. Uses the same feature set as Ridge.
#
# TwoStageRidge / Ordinal / Gated-Ordinal: Alternative TD models for RB.
# Currently using gated_ordinal (binary gate + ordinal classes {0,6,12,18}).
```

### 4.9 `shared/training.py` — Training Loop

**Rubric targets: "Training curves" (3 pts) + "Regularization/early stopping" (5 pts)**

```python
# class MultiTargetLoss(nn.Module):
#     # Combined Huber loss for multi-head network
#     # Loss = sum(weight[t] * Huber(pred[t], target[t])) + w_total * Huber(total, actual_total)
#     # Per-target Huber deltas allow different MSE-to-MAE thresholds
#
# class MultiTargetDataset(Dataset):
#     # Returns features + dict of per-target values
#
# def make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size):
#     # Train: shuffle=True, drop_last=True; Val: shuffle=False
#     # num_workers=0, pin_memory=False (CPU-only)
#
# class MultiHeadTrainer:
#     def __init__(self, model, optimizer, scheduler, criterion, device,
#                  target_names, patience, scheduler_per_batch=False):
#
#     def train(self, train_loader, val_loader, n_epochs) -> dict:
#         # For each epoch:
#         #   1. Train pass: forward, loss, backward, gradient clipping (max_norm=1.0), step
#         #   2. Validation pass: per-target + total MAE/RMSE (no grad)
#         #   3. Log per-target losses, per-target MAE, total MAE/RMSE
#         #   4. Step LR scheduler (supports ReduceLROnPlateau, OneCycleLR, CosineWarmRestarts)
#         #   5. Early stopping: restore best weights if no improvement in `patience` epochs
#         # Return: history dict with all logged metrics per epoch
#
# def plot_training_curves(history, target_names, save_path):
#     # Four-panel figure:
#     #   Top-left: overall train/val loss
#     #   Top-right: per-target val losses
#     #   Bottom-left: per-target MAE
#     #   Bottom-right: total MAE and RMSE
```

### 4.10 `src/evaluation/metrics.py` — Evaluation Metrics

**Rubric target: "≥3 distinct evaluation metrics" (3 pts)**

```python
# Compute all of:
#   1. MAE  (Mean Absolute Error) — most interpretable for fantasy
#   2. RMSE (Root Mean Squared Error) — penalizes big misses
#   3. R²   (Coefficient of Determination) — variance explained
#
# Additional breakdowns:
#   - Per-position metrics (QB/RB/WR/TE separately)
#   - Per-scoring-tier: top-12 at position vs rest (star players vs bench)
#
# Key functions:
#   compute_metrics(y_true, y_pred) -> dict
#   compute_positional_metrics(df, y_pred_col, y_true_col) -> pd.DataFrame
#   print_comparison_table(results_dict) -> None  # pretty-print baseline vs LR vs NN
```

### 4.11 `src/evaluation/backtest.py` — Week-by-Week Prediction Simulation

**Rubric target: "Simulation-based evaluation" (7 pts)**

```python
# Simulate deploying the model week-by-week across the 2024 test season,
# evaluating individual player prediction quality over time.
# NO lineup construction — this is purely about projection accuracy.
#
# For each week W in the test season (2024):
#   1. Generate predictions for all active players using each model
#   2. Compute per-week metrics: MAE, RMSE, R² (per model)
#   3. Compute per-week ranking accuracy per position:
#      - Among top-12 actual scorers at each position, how many did the
#        model rank in its own top-12? (hit rate / precision@12)
#      - Spearman rank correlation between predicted and actual rankings
#   4. Track whether accuracy improves as the season progresses
#      (more rolling history available → potentially better features)
#
# Output:
#   - Per-week MAE for each model (line chart across weeks 1-18)
#   - Per-week top-12 hit rate per position (shows ranking quality)
#   - Aggregate season metrics for each model
#   - Visualization: accuracy trend over the season
#   - Discussion: does more data (later weeks) improve predictions?
#
# This answers: "how reliable are these projections for individual players,
# and does prediction quality change over the course of a season?"
```

### 4.12 Hyperparameter Tuning — Per-Position Config Files

**Rubric target: "Systematic hyperparameter tuning" (5 pts)**

Hyperparameter tuning is documented through the per-position config files (`{POS}/{pos}_config.py`).
Each position has been tuned independently with different architectures, dropout rates, learning rates,
loss weights, Huber deltas, and LR schedulers. The config files serve as the record of tuning decisions.

Key tuning dimensions per position:
- **Backbone architecture**: Single wide layer vs two-layer (e.g., RB [96] vs QB [96, 48])
- **Dropout**: 0.20 (WR) to 0.35 (QB) — higher for positions with less training data
- **Loss weights**: TD weight elevated (2.0–3.0) for zero-inflated target; floor weights vary by position
- **Huber deltas**: Per-target thresholds controlling MSE-to-MAE transition
- **LR scheduler**: ReduceLROnPlateau (RB), OneCycleLR (QB, TE, K), CosineWarmRestarts (WR, DST)
- **Ridge alpha grids**: Position-specific logspace grids per target

### 4.13 Error Analysis + Explainability

**Rubric targets: "Error analysis with visualization and discussion" (7 pts) + "Interpretable model design or explainability analysis" (7 pts)**

Error analysis and benchmark comparison are documented in `docs/expert_comparison.md`, which includes:
- Per-position MAE and R² for both Ridge and NN models
- Per-target MAE breakdown (floor, TD points)
- Comparison against academic benchmarks (Stanford CS 229, INST 414, arXiv)
- Comparison against industry expert sources (FantasyPros, FFA rankings)
- Position-by-position analysis of where/why the model succeeds or fails

Ridge coefficient analysis provides feature importance per target via `RidgeMultiTarget.get_feature_importance()`.

---

## 5. End-to-End Pipeline (per-position: `{POS}/run_{pos}_pipeline.py`)

Each position has its own pipeline script that orchestrates training. The general flow:

```python
# Per-position pipeline (e.g., RB/run_rb_pipeline.py):
#
# 1. LOAD:      loader.load_raw_data(SEASONS) + cache as parquet
# 2. PREPROCESS: preprocessing.preprocess(raw_df)  — NO min-games filter here
# 3. FEATURES:  engineer.build_features(clean_df)  — computes team totals THEN filters min-games
# 4. FILTER:    {pos}_data.filter_to_{pos}(featured_df)  — position-specific rows
# 5. TARGETS:   {pos}_targets.compute_{pos}_targets(df)  — target decomposition
# 6. POS FEATS: {pos}_features.add_{pos}_specific_features(df)  — position-specific features
# 7. DROP:      Remove features listed in {POS}_DROP_FEATURES (QB-only stats, etc.)
# 8. SPLIT:     split.temporal_split(df, ...)
# 9. NaN FILL:  Position-specific NaN filling using training set statistics
# 10. RIDGE:    Train RidgeMultiTarget (one Ridge per target), evaluate
# 11. NEURAL:   Train MultiHeadNet with MultiHeadTrainer, evaluate
# 12. SAVE:     Models to {POS}/outputs/models/, figures to {POS}/outputs/figures/
#
# The Flask web app (app.py) loads pre-trained models from all positions and
# serves predictions via a dashboard with API endpoints:
#   /api/predictions, /api/metrics, /api/weekly_accuracy,
#   /api/player/<player_id>, /api/top_players, /api/position_details
```

---

## 6. Key Design Decisions to Document

These are decisions you should explain in the technical walkthrough video and README. They directly map to the rubric item "Documented a design decision where you chose between ML approaches based on technical tradeoffs" (3 pts — claim as a 16th item if room, or use as backup).

1. **Temporal split vs random split:** Random split would leak future information (a player's week 15 stats informing a week 10 prediction). Season-based split is the correct approach for time-series, even though it gives less training data.

2. **Ridge Regression vs plain Linear Regression:** Ridge adds L2 regularization which prevents overfitting on correlated features (many rolling stats are correlated). Implemented as `RidgeMultiTarget` in `shared/models.py` — one Ridge per sub-target, with position-specific alpha grids.

3. **Multi-head MLP vs single-output MLP vs RNN/LSTM:** Target decomposition (e.g., rushing_floor + receiving_floor + td_points) with a shared backbone + per-target heads outperforms a single-output network. An LSTM could model sequences more naturally, but the MLP with hand-crafted rolling features is simpler and performs comparably on tabular data. LSTM is documented as future work in `docs/design_lstm_multihead.md`.

4. **Huber loss vs MSE:** Huber loss is robust to outlier games (e.g., a RB scoring 40+ in a blowout). Per-target deltas allow different thresholds — higher delta for TD points (more volatile) makes the loss more MAE-like for that target.

5. **Clamp for non-negativity (replaced Softplus):** Fantasy point components are non-negative. Softplus was initially chosen for differentiability, but `softplus(0) ≈ 0.693` per head created a ~2 point floor across 3 heads (no player could be predicted below ~2 pts). Replaced with `torch.clamp(min=0)` which allows exact zeros. A `non_negative_targets` parameter controls which heads are clamped, so DST's `pts_allowed_bonus` (range: -4 to +10) can go negative.

6. **Multi-format scoring:** The system computes fantasy points for Standard (0 PPR), Half-PPR (0.5), and Full PPR (1.0) formats. The only difference is the reception weight. All three columns are computed during preprocessing, enabling format-specific modeling and comparison. Full PPR remains the default.

7. **Weather and Vegas features:** Vegas-derived features (implied_team_total, implied_opp_total, total_line, spread_line) and venue features (is_dome, wind_adjusted, temp_adjusted) are computed in `shared/weather_features.py`. These provide game-environment context that pure player stats miss — a high implied total signals a projected shootout, while dome games boost passing. Implemented for all positions, with position-specific subsets.

---

## 7. Following Directions Checklist

| Item | Pts | File | Status |
|------|-----|------|--------|
| Self-assessment submitted | 3 | Gradescope | To do |
| SETUP.md with install instructions | 1 | SETUP.md | To do |
| ATTRIBUTION.md with AI usage | 1 | ATTRIBUTION.md | To do |
| requirements.txt | 1 | requirements.txt | To do |
| README: What it Does | 1 | README.md | To do |
| README: Quick Start | 1 | README.md | To do |
| README: Video Links | 1 | README.md | To do |
| README: Evaluation section | 1 | README.md | To do |
| Demo video (correct length, no code) | 2 | linked in README | To do |
| Technical walkthrough video | 2 | linked in README | To do |
| Workshop attendance (4 sessions) | 4 | N/A | Done |

**Target: 15/15 pts** (Individual Contributions section is N/A for solo projects). Workshop attendance adds up to 4 pts as buffer.

### 7.1 SETUP.md Content Outline

```
# Setup Instructions
1. Prerequisites: Python 3.10+, pip
2. Clone the repo: git clone <url> && cd fantasy-football-predictor
3. Create virtual environment: python -m venv venv && source venv/bin/activate
4. Install dependencies: pip install -r requirements.txt
5. Download/cache data: data is auto-downloaded on first pipeline run
6. Run position pipelines: python -m RB.run_rb_pipeline (etc.)
7. Start web dashboard: python app.py
8. Expected runtime: under a minute per position on CPU (no GPU required)
```

### 7.2 ATTRIBUTION.md Content Outline

```
# Attribution

## AI Development Tools
- Tool: Claude Code (Anthropic)
- What was AI-generated: [list modules/functions where AI wrote first draft]
- What was human-designed: [overall architecture, rubric strategy, feature selection]
- What required debugging/rework: [list any AI-generated code that needed fixes]
- How AI was used iteratively: [describe the collaboration workflow]

## Data Sources
- nfl_data_py (Python wrapper for nflverse): weekly player stats, roster data, schedules, snap counts
- nflverse project: https://github.com/nflverse
- Data license: [check nflverse license]

## External Code / References
- [List any code snippets, tutorials, or papers that influenced the implementation]
```

### 7.3 README Section Outlines

- **What it Does:** A machine learning system that predicts weekly fantasy football PPR points for all NFL positions (QB, RB, WR, TE, K, DST). Compares Ridge Regression against a custom PyTorch multi-head neural network using engineered temporal features from 14 seasons (2012-2025) of NFL data. Includes a Flask web dashboard for predictions and analysis.
- **Quick Start:** Clone, install, run position pipelines, start web dashboard. 3-5 lines max.
- **Video Links:** Direct links to demo video and technical walkthrough (YouTube or similar).
- **Evaluation:** Summary table of MAE, RMSE, R² for each model (baseline, Ridge, NN) overall and per position. Week-by-week accuracy trends and per-position ranking quality.

### 7.4 Video Content Plans

**Demo Video** (non-technical audience, no code shown):
- Open with the problem: "Every week, millions of fantasy football managers need to decide which players to start"
- Show what the system does: takes NFL stats → predicts weekly fantasy points per player
- Show prediction accuracy: weekly MAE trends, per-position ranking accuracy
- Show example predictions: "here's what the model predicted vs actual for top players in week X"
- Conclude with key finding: does the NN meaningfully outperform Ridge for individual projections?
- Check course website for exact duration requirements

**Technical Walkthrough Video** (code + architecture):
- Repository structure walkthrough
- Data pipeline: nfl_data_py → preprocessing → feature engineering (show shift(1) for leakage prevention)
- Feature engineering deep-dive: rolling windows, share features, matchup features
- Model architecture: MLP diagram, explain BatchNorm + Dropout + Early Stopping
- Training curves: show loss convergence, explain early stopping trigger
- Design decisions: why MLP over LSTM, why temporal split, why Ridge as fair comparison
- Error analysis highlights: where/why the model fails, fundamental unpredictability
- Explainability: feature importance comparison between Ridge and NN

---

## 8. Project Cohesion Narrative

Everything in this project serves one goal: **predict weekly fantasy football points better than simple baselines.** The cohesion argument flows naturally:

- **Problem** → Fantasy football managers need weekly projections; existing free tools are often inaccurate.
- **Data** → NFL weekly player stats from nflverse, the gold standard open NFL data source.
- **Approach** → Feature engineering captures temporal trends and matchup context; two model families (linear and neural) are compared fairly.
- **Evaluation** → Standard ML metrics *plus* a week-by-week simulation showing how prediction accuracy evolves over a season and how well models rank players within each position.

No component is superfluous — every feature, model, and evaluation ties back to that central goal.

### 8.1 Cohesion Rubric Mapping

| Rubric Checkbox | Pts | Where Demonstrated |
|----------------|-----|-------------------|
| README clearly articulates a single, unified project goal | 3 | README.md "What it Does" section |
| Demo video communicates why it matters to non-technical audience | 3 | Demo video (see 7.4) |
| Addresses real-world problem connected to research papers/competitions | 3 | README + references below |
| Technical walkthrough shows synergistic components | 2 | Technical walkthrough video |
| Clear progression: problem → approach → solution → evaluation | 2 | README + technical walkthrough |
| Design choices explicitly justified | 2 | Section 6 + technical walkthrough |
| Evaluation metrics directly measure stated objectives | 2 | MAE on fantasy points = direct measurement of prediction quality |
| No superfluous components | 2 | Every module maps to the core goal (see above) |
| Clean codebase, no extraneous files | 2 | `.gitignore` enforced, no unused code in final submission |

**Maximum possible: 21 pts (capped at 15).**

### 8.2 Research Context

This project connects to existing work in sports analytics and ML-based forecasting:

- [cite: sports analytics ML paper on player performance prediction]
- [cite: NFL/fantasy football forecasting study or Kaggle competition]
- [cite: temporal feature engineering for sports data or time-series prediction paper]

These references should be included in the README to satisfy the "meaningful research question connected to concrete research papers or competitions" requirement.

### 8.3 Clean Codebase Commitment

- No unused files in the repository
- `.gitignore` covers all generated/cached data (see Section 15)
- Clear file naming conventions matching the repository structure in Section 3
- No stale experimental code — all experiments live in notebooks, not in `src/`
- No hardcoded values outside `config.py`

---

## 9. Dependencies (`requirements.txt`)

```
numpy==2.4.1
pandas==2.3.3
scikit-learn==1.8.0
torch==2.11.0
scipy==1.17.1
matplotlib==3.10.8
pytest==9.0.3
flask==3.1.3
nfl_data_py==0.3.3
```

No GPU required. The dataset trains in under a minute per position on CPU. Flask serves the prediction dashboard.

---

## 10. Implementation Order

The actual implementation order followed:

1. **`src/config.py`** — Central constants and scoring
2. **`src/data/loader.py`** — nfl_data_py ingestion + caching
3. **`src/data/preprocessing.py`** — Cleaning, fantasy points computation
4. **`src/features/engineer.py`** — Rolling, EWMA, trend, share, matchup features
5. **`src/data/split.py`** — Temporal split
6. **`shared/neural_net.py`** — MultiHeadNet architecture
7. **`shared/models.py`** — RidgeMultiTarget wrapper
8. **`shared/training.py`** — MultiHeadTrainer, MultiTargetLoss, dataloaders
9. **Position folders** — Per-position configs, targets, features, data filters, pipelines
   (QB, RB, WR, TE first; K and DST added later with custom pipelines)
10. **`tests/`** — Feature leakage tests, RB target tests
11. **`app.py`** — Flask web dashboard + prediction API
12. **`docs/`** — Expert comparison, future work proposals

---

## 11. Potential Pitfalls & Edge Cases

- **nfl_data_py API changes:** Pin the version. If a function signature changes, the cache files in `data/raw/` will still work.
- **Players changing teams mid-season (trades):** Use `recent_team` from the weekly data, not roster data, for that week's team context. Share features (target_share, carry_share) must use stint-aware grouping — detect team changes via `stint_id` and reset rolling denominators on trade. Rolling player stats (non-share) do NOT reset because a player's individual production is meaningful across teams.
- **Bye weeks:** A player on bye has no row in weekly data — that's correct. But the *following* week's rolling features should NOT treat the bye as a zero-point game. Filter on `games_played > 0` or similar before rolling.
- **Injured players:** Players who are active but leave early will have low stats. This is inherently unpredictable and should appear in error analysis as a known failure mode. The `is_returning_from_absence` feature flags players coming back from missed games.
- **Rookies in test season:** Rookies in 2024 have no prior-season data. `fill_nans_safe()` uses position-level training set averages for all their features. Flag this in error analysis.
- **Week 1 of each season:** No in-season rolling features exist. Prior-season summary features provide stable fallback signal. Remaining NaNs filled by `fill_nans_safe()` using train-set statistics only.
- **Players changing positions mid-career:** Some players (e.g., Taysom Hill) appear as QB in some weeks and TE in others. Use the roster-based position (from `import_rosters()`) which is the season-level designation. If a player's position differs between seasons, use the most recent season's roster position. Do not use weekly position labels.
- **Snap count data coverage:** `nfl_data_py.import_snap_counts()` may not cover all seasons (earlier seasons may lack data). If snap counts are unavailable for certain seasons, `snap_pct` will be NaN for those rows — handled by the position-median fill in preprocessing.

---

## 12. Self-Assessment Plan

The Gradescope self-assessment (3 pts) requires mapping each claimed rubric item to evidence. Template to fill in post-implementation:

| # | Rubric Item | Evidence Location | Description |
|---|-------------|-------------------|-------------|
| 1 | Solo project credit | N/A | Completed individually |
| 2 | Original dataset via API | `src/data/loader.py` | nfl_data_py multi-table ingestion + merge (2012-2025) |
| 3 | Compared architectures | `shared/models.py`, `shared/neural_net.py`, position configs | RidgeMultiTarget vs MultiHeadNet per position |
| 4 | Error analysis | `docs/expert_comparison.md`, pipeline outputs | Per-target breakdown, benchmark comparison |
| 5 | Time-series forecasting | `src/features/engineer.py` | Rolling temporal features + temporal split |
| 6 | Simulation-based eval | `shared/backtest.py` | Week-by-week prediction accuracy simulation |
| 7 | Feature engineering | `src/features/engineer.py`, `{POS}/{pos}_features.py` | 130+ general + position-specific derived features |
| 8 | Custom neural network | `shared/neural_net.py` | Multi-head PyTorch architecture |
| 9 | Hyperparameter tuning | `{POS}/{pos}_config.py` files | Per-position configs with tuned architectures |
| 10 | Regularization | `shared/neural_net.py`, `shared/training.py` | Dropout + L2 + early stopping + BatchNorm |
| 11 | Modular code design | `src/`, `shared/`, position folders | Reusable modules and classes |
| 12 | Train/val/test split | `src/data/split.py` | Temporal split (2012-2023 / 2024 / 2025) |
| 13 | Training curves | `shared/training.py` | 4-panel: loss, per-target loss, per-target MAE, total MAE/RMSE |
| 14 | Baseline model | `src/models/baseline.py` | Season average + last week baselines |
| 15 | ≥3 evaluation metrics | `src/evaluation/metrics.py`, `shared/evaluation.py` | MAE, RMSE, R² |
| 16 | Explainability analysis | Pipeline outputs, `docs/expert_comparison.md` | Ridge coefficients + permutation importance |

---

## 13. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `nfl_data_py` API breaking changes | Low | High | Pin version in requirements.txt; cache raw data early as parquet |
| Neural net does NOT beat linear model | Medium | Low | This is a valid and interesting finding — explain in error analysis why tabular data often favors linear models |
| Insufficient data for rolling features | Low | Medium | 5 seasons of data (~25K+ rows); minimum 6-game filter ensures history |
| Time constraints before deadline | Medium | High | Implementation order (Section 10) prioritizes highest-value rubric items first; lower-value items can be simplified |
| Snap count data unavailable for some seasons | Low | Low | snap_pct is one of many features; fill with NaN → position median |

---

## 14. Testing Strategy

Validate correctness at each pipeline stage before moving on:

1. **Feature leakage tests:** `tests/test_feature_leakage.py` (328 lines) comprehensively verifies no future data leaks into features
2. **Unit tests:** `RB/tests/test_rb_targets.py` validates target decomposition correctness
3. **Shape checks:** At each pipeline step, print and verify DataFrame shape
4. **Target decomposition sanity checks:** Each `compute_{pos}_targets()` function includes a
   `fantasy_points_check` that verifies the sum of decomposed targets matches total fantasy points
5. **Model sanity check:** Verify trained models produce predictions in a reasonable range (0-50 PPR points typically)
6. **Cross-validation with nflverse:** `rb_targets.py` cross-validates computed fantasy points against
   nflverse pre-computed `fantasy_points_ppr` column when available

---

## 15. `.gitignore` Specification

```
# Data (generated at runtime)
data/

# Model outputs (large binary files)
outputs/models/

# Keep figures in git for documentation
# outputs/figures/ — NOT ignored

# Python
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store

# IDE
.vscode/
.idea/
```

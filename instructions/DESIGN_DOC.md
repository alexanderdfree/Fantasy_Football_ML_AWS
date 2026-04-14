# Fantasy Football Weekly Points Predictor — Code Design Document

## For use with Claude Code / AI-assisted development on CS 372 Final Project

---

## 1. Project Overview

**Goal:** Build a machine learning system that predicts weekly fantasy football points (PPR scoring) for NFL skill-position players (QB, RB, WR, TE), comparing a Linear Regression baseline against a custom PyTorch neural network.

**Data source:** `nfl_data_py` (Python wrapper for nflverse), pulling weekly player stats, roster data, and schedule data from the 2018–2024 NFL seasons.

**Core question:** Can a neural network with engineered temporal features meaningfully outperform linear regression at predicting weekly fantasy output, and what features matter most?

---

## 2. Rubric Strategy (15 ML Items → Target ~73 pts)

Below are the 15 Machine Learning rubric items this project is designed to hit. Every architectural decision in this document maps back to one or more of these.

| # | Rubric Item | Pts | Where in Code |
|---|-------------|-----|---------------|
| 1 | **Solo project credit** | 10 | N/A |
| 2 | **Collected/constructed original dataset through API integration** (`nfl_data_py` ingestion, multi-season join, feature build) | 10 | `src/data/` |
| 3 | **Compared multiple model architectures quantitatively** (Linear Reg vs Neural Net, controlled setup) | 7 | `src/models/`, `notebooks/experiments.ipynb` |
| 4 | **Error analysis with visualization and discussion of failure cases** | 7 | `notebooks/error_analysis.ipynb` |
| 5 | **Applied ML to time-series forecasting** (weekly player projections using rolling temporal features) | 7 | `src/features/`, `src/models/` |
| 6 | **Simulation-based evaluation** (season-long fantasy draft backtesting) | 7 | `src/evaluation/backtest.py` |
| 7 | **Feature engineering** (rolling averages, snap %, target share, matchup strength, etc.) | 5 | `src/features/engineer.py` |
| 8 | **Defined and trained a custom neural network architecture** (PyTorch) | 5 | `src/models/neural_net.py` |
| 9 | **Systematic hyperparameter tuning** (≥3 configs documented) | 5 | `notebooks/experiments.ipynb` |
| 10 | **Regularization** (L2 + dropout + early stopping — at least 2 required) | 5 | `src/models/neural_net.py`, `src/training/trainer.py` |
| 11 | **Modular code design** with reusable functions/classes | 3 | Entire `src/` package |
| 12 | **Train/validation/test split** with documented ratios | 3 | `src/data/split.py` |
| 13 | **Training curves** (loss and metrics over time) | 3 | `src/training/trainer.py` |
| 14 | **Baseline model** (constant prediction = player's season average) | 3 | `src/models/baseline.py` |
| 15 | **Used ≥3 distinct evaluation metrics** | 3 | `src/evaluation/metrics.py` |
| 16 | **Interpretable model design or explainability analysis** (Ridge coefficients + permutation importance for NN) | 7 | `notebooks/03_error_analysis.ipynb` |

**Total if all hit: 90 pts (capped at 73).** Only 15 of 16 items need to land to cap out, giving a 17-pt buffer. The rubric grades only the first 15 selected items, so list the strongest 15 in the self-assessment and keep the 16th as insurance.

### Rubric Overflow: Items Implemented But Not Counted in the 16

These items are naturally implemented by the project but are **not** claimed as part of the 15/16. They further demonstrate depth:

| Rubric Item | Pts | Where in Code |
|-------------|-----|---------------|
| Properly normalized/standardized input features | 3 | `src/models/linear.py` (StandardScaler) |
| Applied basic preprocessing appropriate to modality | 3 | `src/data/preprocessing.py` |
| Used appropriate data loading with batching and shuffling | 3 | `src/training/trainer.py` (DataLoader) |
| Used learning rate scheduling | 3 | `src/training/trainer.py` (ReduceLROnPlateau) |
| Applied batch normalization | 3 | `src/models/neural_net.py` (BatchNorm1d) |
| Documented design decision with technical tradeoffs | 3 | Section 6 of this doc + technical walkthrough |
| Substantive ATTRIBUTION.md on AI tool usage | 3 | `ATTRIBUTION.md` |

### One Piece of Work → One Item Rule

The rubric states each piece of work can claim at most one item (plus stackable add-ons). Here is the explicit mapping to avoid double-counting:

| Work | Claims Item # | Does NOT claim |
|------|--------------|----------------|
| Rolling window features + share stats + matchup features | #7 (Feature engineering, 5 pts) | — |
| Temporal split + rolling window modeling paradigm | #5 (Time-series forecasting, 7 pts) | — |
| PyTorch MLP architecture definition | #8 (Custom neural network, 5 pts) | — |
| Dropout + L2 weight decay + early stopping | #10 (Regularization, 5 pts) | — |
| Hyperparameter grid search across configs | #9 (Systematic tuning, 5 pts) | — |
| Ridge vs NN controlled comparison | #3 (Architecture comparison, 7 pts) | — |
| Ridge coefficients + permutation importance | #16 (Explainability, 7 pts) | — |

---

## 3. Repository Structure

```
fantasy-football-predictor/
├── README.md                  # What it does, quick start, video links, evaluation results
├── SETUP.md                   # Step-by-step installation
├── ATTRIBUTION.md             # AI tool usage + data source attribution
├── requirements.txt           # Pinned dependencies
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py              # All hyperparams, paths, constants, scoring weights
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # nfl_data_py ingestion + caching
│   │   ├── preprocessing.py   # Cleaning, missing values, filtering
│   │   └── split.py           # Temporal train/val/test split
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py        # All feature engineering logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py        # Constant-prediction baseline
│   │   ├── linear.py          # Scikit-learn Linear/Ridge Regression
│   │   └── neural_net.py      # PyTorch custom MLP
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Training loop, early stopping, LR scheduling, curve logging
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py         # MAE, RMSE, R², per-position breakdown
│       └── backtest.py        # Season-long fantasy draft simulation
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis + visualizations
│   ├── 02_experiments.ipynb   # Hyperparameter tuning + model comparison
│   └── 03_error_analysis.ipynb# Failure case deep-dive
│
├── data/                      # .gitignore this — generated at runtime
│   ├── raw/                   # Cached nfl_data_py pulls
│   ├── processed/             # Feature-engineered datasets
│   └── splits/                # Train/val/test CSVs
│
├── outputs/
│   ├── figures/               # Training curves, comparison charts, error plots
│   └── models/                # Saved model weights (.pt) and sklearn pickles
│
└── scripts/
    └── run_pipeline.py        # End-to-end: load → features → train → evaluate
```

---

## 4. Detailed Module Specifications

### 4.1 `src/config.py` — Central Configuration

All magic numbers live here. Nothing is hardcoded in other files.

```python
# Key constants to define:
SEASONS = list(range(2018, 2025))       # nfl_data_py seasons to pull (2018-2024)
POSITIONS = ["QB", "RB", "WR", "TE"]
SCORING = {"passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
           "rushing_yards": 0.1, "rushing_tds": 6,
           "receptions": 1, "receiving_yards": 0.1, "receiving_tds": 6,
           "fumbles_lost": -2}          # Standard PPR

# Temporal split boundaries (season-based, NOT random)
TRAIN_SEASONS = list(range(2018, 2023))  # 2018-2022
VAL_SEASONS   = [2023]                   # 2023
TEST_SEASONS  = [2024]                   # 2024

# Rolling feature windows
ROLLING_WINDOWS = [3, 5, 8]             # weeks lookback

# Neural net defaults
NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4                  # L2 regularization
NN_EPOCHS = 200
NN_BATCH_SIZE = 256
NN_PATIENCE = 15                        # early stopping
```

### 4.2 `src/data/loader.py` — Data Ingestion

**Rubric target: "Collected/constructed original dataset through API integration" (10 pts)**

```python
# Responsibilities:
# 1. Pull weekly player stats via nfl_data_py.import_weekly_data(SEASONS)  # 2018-2024
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
# 1. Filter to skill positions (QB, RB, WR, TE)
# 2. Remove bye weeks / players with 0 snaps (did not play)
# 3. Handle missing values:
#    - Statistical columns: fill with 0 (player didn't record that stat)
#    - Snap percentage: fill with position median for that week
# 4. Remove partial-season players (< 6 games in a season) to ensure
#    enough history for rolling features
# 5. Compute target variable: fantasy_points using config.SCORING
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
#   Train: 2018-2022 seasons (~5 seasons × ~17 weeks × ~300 players ≈ 25K+ rows)
#   Val:   2023 season (~5K rows)
#   Test:  2024 season (~5K rows)
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
# === PLAYER ROLLING FEATURES (per player, per window in ROLLING_WINDOWS) ===
# For each stat and window w, compute using ONLY past data (shift(1) before rolling):
#   - rolling_mean_{stat}_L{w}       (e.g., rolling_mean_fantasy_points_L3)
#   - rolling_std_{stat}_L{w}        (volatility / consistency)
#   - rolling_max_{stat}_L{w}        (ceiling indicator)
#
# Key stats to roll: fantasy_points, targets, receptions, carries,
#                     rushing_yards, receiving_yards, passing_yards, snap_pct
#
# === USAGE / OPPORTUNITY FEATURES (5-week rolling shares) ===
#   - target_share:      rolling_sum(player_targets, L5) / rolling_sum(team_targets, L5), shifted
#   - carry_share:       rolling_sum(player_carries, L5) / rolling_sum(team_carries, L5), shifted
#   - snap_pct:          from snap count data (direct feature, not derived)
#   - air_yards_share:   rolling_sum(player_air_yards, L5) / rolling_sum(team_air_yards, L5), shifted
#   - redzone_targets_share: same rolling pattern (if column available, else fill 0)
#
# === MATCHUP / OPPONENT FEATURES ===
#   - opp_fantasy_pts_allowed_to_pos: rolling average of fantasy points the
#     upcoming opponent has allowed to this position over last 5 weeks
#     (e.g., "how many PPR points have the Cowboys allowed to WRs recently?")
#   - opp_def_rank_vs_pos: rank version of the above (1 = worst D = best matchup)
#
# === CONTEXTUAL FEATURES ===
#   - is_home:           binary, home vs away
#   - week:              integer 1-18 (captures early/late season patterns)
#   - season_games_played: cumulative games for this player this season
#   - days_rest:         days since last game (bye week detection)
#
# === POSITION ENCODING ===
#   - One-hot encode position (QB, RB, WR, TE)
#
# === TEAM-LEVEL AGGREGATION (for share features) ===
# 1. Group by (recent_team, season, week) and sum targets, carries, air_yards
# 2. Merge team totals back to player rows
# 3. Compute 5-week rolling sums for BOTH player and team stats (shifted)
# 4. Share = player_rolling_sum / team_rolling_sum
# 5. IMPORTANT: Both numerator and denominator use shift(1) + rolling to
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
# 2. Rolling operations MUST stay within player groups. Use
#    df.groupby("player_id")[stat].transform(lambda x: x.shift(1).rolling(...))
#    NOT: groupby().shift() followed by bare .rolling() — the latter rolls
#    across adjacent players' rows, silently corrupting all features.
# 3. rolling std with min_periods=1 still returns NaN for single observations
#    (std of one value is undefined). First row per player per season will
#    have NaN for all 24 std features. Handled by NaN filling strategy.
# 4. The first few weeks of each season will have NaN rolling features.
#    Strategy: fill with the player's full prior season mean where available,
#    then fill remaining NaNs with position-level mean from the training set,
#    then fill any still-remaining NaNs with 0.
# 5. Group all rolling computations by player_id.
# 6. Rookies in test season: no prior season data → use position-level
#    training set averages for all rolling features.
#
# === EXPECTED FEATURE COUNT ===
# Rolling: 8 stats × 3 windows × 3 aggs (mean/std/max) = 72
# Share: target_share, carry_share, snap_pct, air_yards_share, redzone_targets_share = 5
# Matchup: opp_fantasy_pts_allowed_to_pos, opp_def_rank_vs_pos = 2
# Contextual: is_home, week, season_games_played, days_rest = 4
# Position: one-hot (QB, RB, WR, TE) = 4
# TOTAL: ~87 features
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

### 4.8 `src/models/neural_net.py` — Custom PyTorch MLP

**Rubric targets: "Custom neural network" (5 pts) + "Regularization" (5 pts) + "Compared multiple architectures" (7 pts, part 2)**

```python
# Architecture: Multi-Layer Perceptron for tabular regression
#
# class FantasyPointsNet(nn.Module):
#     def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout=0.3):
#         # Expected input_dim: ~87 (see feature count estimate in 4.5)
#         # This validates the layer sizing: 87 -> 128 -> 64 -> 32 -> 1
#         #
#         # For each hidden layer:
#         #   Linear -> BatchNorm1d -> ReLU -> Dropout
#         #   (BatchNorm BEFORE activation = pre-activation normalization)
#         # Final layer: Linear -> single output (predicted fantasy points)
#
#     def forward(self, x):
#         # Pass through hidden blocks, return single float prediction
#
# Regularization employed (need at least 2 for rubric):
#   1. Dropout (0.3) between hidden layers
#   2. L2 weight decay (1e-4) via AdamW optimizer
#   3. Early stopping (patience=15 on validation loss)
#
# Optionally also: BatchNorm acts as mild regularizer
#
# Loss function: MSELoss (standard for regression)
# Optimizer: AdamW (Adam with decoupled weight decay)
```

### 4.9 `src/training/trainer.py` — Training Loop

**Rubric targets: "Training curves" (3 pts) + "Regularization/early stopping" (5 pts)**

```python
# class Trainer:
#     def __init__(self, model, optimizer, scheduler, device, patience):
#         ...
#
#     def train(self, train_loader, val_loader, n_epochs):
#         # For each epoch:
#         #   1. Train pass: forward, loss, backward, optimizer step
#         #   2. Validation pass: compute val loss + metrics (no grad)
#         #   3. Log train_loss, val_loss, val_MAE, val_RMSE to history dict
#         #   4. Step the LR scheduler (ReduceLROnPlateau on val_loss)
#         #   5. Early stopping check: if val_loss hasn't improved in
#         #      `patience` epochs, restore best weights and stop
#         #   6. Save best model checkpoint
#         # Return: history dict with all logged metrics per epoch
#
#     def plot_training_curves(self, history, save_path):
#         # Two-panel figure:
#         #   Left: train loss vs val loss over epochs
#         #   Right: val MAE and val RMSE over epochs
#         # Save to outputs/figures/training_curves.png
#
# Also create standard PyTorch DataLoaders:
#   def make_dataloaders(X_train, y_train, X_val, y_val, batch_size):
#       # Convert to TensorDataset, wrap in DataLoader
#       # Train: shuffle=True; Val/Test: shuffle=False
#       # num_workers=0 (simplicity, CPU-only)
#       # pin_memory=False (no GPU)
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

### 4.11 `src/evaluation/backtest.py` — Fantasy Draft Simulation

**Rubric target: "Simulation-based evaluation" (7 pts)**

```python
# Simulate a weekly fantasy season using model predictions vs actuals.
#
# Format: Best-ball — pick best available per position per week from
# the full player pool. No roster limits, no waiver wire, no trades.
# This isolates prediction quality from roster management strategy.
#
# Approach: For each week in the test season (2024):
#   1. Use model to predict fantasy points for all players
#   2. "Draft" the optimal starting lineup based on predictions:
#      - 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (best remaining RB/WR/TE)
#      - Fill positions greedily in order: QB → RB → WR → TE → FLEX
#        (FLEX is filled last from remaining RB/WR/TE pool)
#   3. Score the drafted lineup using ACTUAL points
#   4. Compare against:
#      a. Oracle lineup (best possible lineup with perfect knowledge)
#      b. Baseline-drafted lineup (using season average predictions)
#      c. Linear regression-drafted lineup
#      d. Neural net-drafted lineup
#
# Missing predictions handling:
#   - After the 3-stage NaN filling strategy (player prior season mean →
#     position training set mean → zero), ALL feature columns are populated.
#     Ridge/NN models will produce predictions for every player.
#   - For players with mostly zero-filled features (e.g., rookies with no
#     prior data), model predictions may be unreliable but are still produced.
#   - No fallback mechanism is needed — all active players get a prediction.
#
# Output metrics:
#   - Total season points scored per strategy
#   - Weekly win rate vs baseline strategy
#   - "Optimal lineup capture rate" = % of oracle points achieved
#   - Visualization: cumulative points over the season for each strategy
#
# This is the most compelling evaluation for the demo video — it answers
# "would this model actually help you win your fantasy league?"
```

### 4.12 `notebooks/02_experiments.ipynb` — Hyperparameter Tuning

**Rubric target: "Systematic hyperparameter tuning" (5 pts)**

```python
# Document at least 3 configurations with results for EACH model:
#
# Linear Regression:
#   - Ridge alpha values: [0.01, 0.1, 1.0, 10.0, 100.0]
#   - Record val MAE/RMSE/R² for each
#
# Neural Network (vary one at a time from defaults):
#   Config 1 (baseline):  hidden=[128,64,32], dropout=0.3, lr=1e-3
#   Config 2 (wider):     hidden=[256,128,64], dropout=0.3, lr=1e-3
#   Config 3 (shallower): hidden=[128,64],     dropout=0.3, lr=1e-3
#   Config 4 (less dropout): hidden=[128,64,32], dropout=0.1, lr=1e-3
#   Config 5 (lower LR):    hidden=[128,64,32], dropout=0.3, lr=3e-4
#
# Present results in a comparison table and identify best config.
# Use validation set for selection, then report final test metrics only once.
#
# Visualizations:
#   - Training curves overlay for each NN config (val loss vs epoch)
#   - Bar chart of val MAE across all configs (Ridge + NN variants)
#   - Table: config → val MAE / val RMSE / val R²
```

### 4.13 `notebooks/03_error_analysis.ipynb` — Error Analysis + Explainability

**Rubric targets: "Error analysis with visualization and discussion" (7 pts) + "Interpretable model design or explainability analysis" (7 pts)**

```python
# === PART A: ERROR ANALYSIS ===
# Investigate WHERE and WHY the model fails:
#
# 1. Residual distribution plot (histogram of predicted - actual)
# 2. Scatter plot: predicted vs actual, color-coded by position
# 3. Worst predictions table: top 20 biggest misses, with context
#    (was the player injured mid-game? was it a blowout? first game back?)
# 4. Error by player archetype:
#    - Consistent high-volume players (e.g., top-20 WRs by targets)
#    - Boom/bust players (high variance)
#    - Backup/handcuff RBs (unpredictable usage)
# 5. Error by week-of-season (are early-season predictions worse due to
#    less rolling history?)
# 6. Discussion: What types of events does the model fundamentally cannot
#    predict? (injuries, game script, weather, coaching decisions)
#
# === PART B: EXPLAINABILITY ANALYSIS ===
# 7. Ridge coefficient analysis:
#    - Bar chart of top 20 features by |coef_| (absolute coefficient value)
#    - Discuss which features the linear model relies on most
# 8. Neural net permutation importance:
#    - Use sklearn.inspection.permutation_importance on validation set
#    - Bar chart of top 20 features by importance
# 9. Side-by-side comparison:
#    - Do both models agree on which features matter?
#    - Discussion of differences (e.g., NN may capture non-linear interactions)
```

### 4.14 `notebooks/01_eda.ipynb` — Exploratory Data Analysis

```python
# Visualizations and analysis to understand the dataset:
#
# 1. Fantasy points distribution by position (box plots or violin plots)
#    - Shows scoring range differences: QBs highest, TEs lowest
# 2. Correlation heatmap of key features vs fantasy_points
# 3. Target variable distribution (histogram + summary stats)
# 4. Seasonal trends: average fantasy points per position per season
# 5. Top player analysis: top-10 scorers per position per season
# 6. Feature distributions: snap_pct, target_share, rolling averages
# 7. Missing data summary: how many NaNs in each feature after engineering
# 8. Train/val/test split visualization: data volume per season
#
# This notebook is for understanding the data — no model training here.
# Can be developed anytime but polish last.
```

---

## 5. End-to-End Pipeline (`scripts/run_pipeline.py`)

```python
# This script runs the full pipeline in order:
#
# 1. LOAD:      loader.load_raw_data(SEASONS)
# 2. PREPROCESS: preprocessing.preprocess(raw_df)
# 3. FEATURES:  engineer.build_features(clean_df)
# 4. SPLIT:     split.temporal_split(featured_df, ...)
# 5. BASELINE:  Evaluate SeasonAverageBaseline and LastWeekBaseline
#               (baselines operate on unscaled DataFrames with fantasy_points column)
# 6. LINEAR:    Fit Ridge (internally fits StandardScaler), evaluate, save model
# 7. SCALE:     Extract scaler from Ridge (ridge_model.scaler), use it to
#               transform train/val/test feature arrays for the neural net
# 8. NEURAL:    Train FantasyPointsNet on pre-scaled arrays, evaluate, save model
# 9. COMPARE:   Side-by-side metrics table (all 4 approaches)
# 10. BACKTEST:  Run fantasy draft simulation
# 11. SAVE:      All figures, metrics, and model artifacts to outputs/
#
# Should be runnable with: python scripts/run_pipeline.py
# Expected runtime: ~5-10 minutes on a laptop (no GPU required for this scale)
```

---

## 6. Key Design Decisions to Document

These are decisions you should explain in the technical walkthrough video and README. They directly map to the rubric item "Documented a design decision where you chose between ML approaches based on technical tradeoffs" (3 pts — claim as a 16th item if room, or use as backup).

1. **Temporal split vs random split:** Random split would leak future information (a player's week 15 stats informing a week 10 prediction). Season-based split is the correct approach for time-series, even though it gives less training data.

2. **Ridge Regression vs plain Linear Regression:** Ridge adds L2 regularization which prevents overfitting on correlated features (many rolling stats are correlated). This is the fair comparison — "best linear model" vs "neural network."

3. **MLP vs RNN/LSTM:** An LSTM could model sequences more naturally, but an MLP with hand-crafted rolling features is simpler, more interpretable, and performs comparably on tabular data. The feature engineering essentially does what an RNN would learn. You could mention LSTM as future work.

4. **PPR scoring system:** Standard PPR is the most common fantasy format and makes receptions valuable, which increases the signal for WR/TE prediction.

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
5. Download/cache data: python scripts/run_pipeline.py --data-only
   (or: data is auto-downloaded on first pipeline run)
6. Run full pipeline: python scripts/run_pipeline.py
7. Run notebooks: jupyter notebook notebooks/
8. Expected runtime: ~5-10 minutes on a laptop (no GPU required)
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

- **What it Does:** A machine learning system that predicts weekly fantasy football PPR points for NFL skill-position players. Compares a Ridge Regression baseline against a custom PyTorch neural network using engineered temporal and matchup features from 7 seasons of NFL data. Includes a season-long fantasy draft backtesting simulation.
- **Quick Start:** Clone, install, run pipeline command. 3-5 lines max.
- **Video Links:** Direct links to demo video and technical walkthrough (YouTube or similar).
- **Evaluation:** Summary table of MAE, RMSE, R² for each model (baseline, Ridge, NN) overall and per position. Backtest results (total season points per strategy).

### 7.4 Video Content Plans

**Demo Video** (non-technical audience, no code shown):
- Open with the problem: "Every week, millions of fantasy football managers need to decide which players to start"
- Show what the system does: takes NFL stats → predicts weekly points → recommends lineups
- Show backtest results: cumulative points chart comparing strategies, highlight win rate
- Conclude with key finding: does the ML model actually help win fantasy matchups?
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
- **Evaluation** → Standard ML metrics *plus* a realistic fantasy draft simulation that answers "would this actually help you win?"

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
nfl_data_py>=0.3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.0
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
tqdm>=4.65
```

No GPU required. The dataset is ~35K rows and the MLP is small — trains in under a minute on CPU.

---

## 10. Implementation Order (Suggested for Claude Code)

Follow this order to ensure each step can be validated before moving on:

1. **`src/config.py`** — Set up all constants first
2. **`src/data/loader.py`** — Pull and cache data, verify shape and columns
3. **`src/data/preprocessing.py`** — Clean data, compute fantasy points target
4. **`src/data/split.py`** — Temporal split, print split sizes
5. **`src/features/engineer.py`** — Build all features, verify no leakage (assert no current-week stats in features)
6. **`src/evaluation/metrics.py`** — Metric functions (needed for all model evaluation)
7. **`src/models/baseline.py`** — Implement and evaluate baselines first
8. **`src/models/linear.py`** — Ridge regression, evaluate, compare to baselines
9. **`src/models/neural_net.py`** — Define architecture
10. **`src/training/trainer.py`** — Training loop with early stopping + curve logging
11. **Train + evaluate neural net**, compare to linear and baselines
12. **`notebooks/02_experiments.ipynb`** — Hyperparameter search
13. **`src/evaluation/backtest.py`** — Fantasy draft simulation
14. **`notebooks/03_error_analysis.ipynb`** — Deep-dive on failures
15. **`notebooks/01_eda.ipynb`** — Can be done anytime, but polish last
16. **`scripts/run_pipeline.py`** — Wire everything together
17. **Documentation** — README, SETUP.md, ATTRIBUTION.md

---

## 11. Potential Pitfalls & Edge Cases

- **nfl_data_py API changes:** Pin the version. If a function signature changes, the cache files in `data/raw/` will still work.
- **Players changing teams mid-season:** Use `recent_team` from the weekly data, not roster data, for that week's team context.
- **Bye weeks:** A player on bye has no row in weekly data — that's correct. But the *following* week's rolling features should NOT treat the bye as a zero-point game. Filter on `games_played > 0` or similar before rolling.
- **Injured players:** Players who are active but leave early will have low stats. This is inherently unpredictable and should appear in error analysis as a known failure mode.
- **Rookies in test season:** Rookies in 2024 have no prior-season data. Fill their rolling features with positional averages. Flag this in error analysis.
- **Week 1 of each season:** No in-season rolling features exist. Use prior season averages as fallback.
- **Players changing positions mid-career:** Some players (e.g., Taysom Hill) appear as QB in some weeks and TE in others. Use the roster-based position (from `import_rosters()`) which is the season-level designation. If a player's position differs between seasons, use the most recent season's roster position. Do not use weekly position labels.
- **Snap count data coverage:** `nfl_data_py.import_snap_counts()` may not cover all seasons (earlier seasons may lack data). If snap counts are unavailable for certain seasons, `snap_pct` will be NaN for those rows — handled by the position-median fill in preprocessing.

---

## 12. Self-Assessment Plan

The Gradescope self-assessment (3 pts) requires mapping each claimed rubric item to evidence. Template to fill in post-implementation:

| # | Rubric Item | Evidence Location | Description |
|---|-------------|-------------------|-------------|
| 1 | Solo project credit | N/A | Completed individually |
| 2 | Original dataset via API | `src/data/loader.py` | nfl_data_py multi-table ingestion + merge |
| 3 | Compared architectures | `notebooks/02_experiments.ipynb` | Controlled comparison table |
| 4 | Error analysis | `notebooks/03_error_analysis.ipynb` | Residual plots, failure cases |
| 5 | Time-series forecasting | `src/features/engineer.py` | Rolling temporal features + temporal split |
| 6 | Simulation-based eval | `src/evaluation/backtest.py` | Season-long fantasy draft backtest |
| 7 | Feature engineering | `src/features/engineer.py` | 87+ derived features |
| 8 | Custom neural network | `src/models/neural_net.py` | PyTorch MLP definition |
| 9 | Hyperparameter tuning | `notebooks/02_experiments.ipynb` | 3+ configs per model |
| 10 | Regularization | `src/models/neural_net.py`, `src/training/trainer.py` | Dropout + L2 + early stopping |
| 11 | Modular code design | `src/` package | Reusable modules and classes |
| 12 | Train/val/test split | `src/data/split.py` | Temporal split with documented ratios |
| 13 | Training curves | `src/training/trainer.py` | Loss + metrics over epochs |
| 14 | Baseline model | `src/models/baseline.py` | Season average + last week baselines |
| 15 | ≥3 evaluation metrics | `src/evaluation/metrics.py` | MAE, RMSE, R² |
| 16 | Explainability analysis | `notebooks/03_error_analysis.ipynb` | Ridge coefficients + permutation importance |

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

1. **Smoke test:** Run pipeline on a single season (e.g., 2023 only) to verify all modules execute without error
2. **Data leakage check:** Assert that no feature column contains current-week data:
   - After `build_features()`, verify rolling features for week W do NOT include week W stats
   - Check that `.shift(1)` is applied before every `.rolling()` call
3. **Shape checks:** At each pipeline step, print and verify DataFrame shape:
   - After load: ~35K rows × ~50 raw columns
   - After preprocessing: slightly fewer rows (filtered)
   - After features: same rows × ~87 feature columns + target + metadata
   - After split: train ~25K, val ~5K, test ~5K
4. **Baseline sanity checks:** Verify baseline MAE is in a reasonable range (roughly 5-10 PPR points) — if it's 0 or 50+, something is wrong
5. **Model sanity check:** Verify trained models produce predictions in a reasonable range (0-50 PPR points typically)

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

# Fantasy Football Predictor — Method Contracts & Wishlists

Companion to `DESIGN_DOC.md`. This document specifies concrete function signatures, data schemas, and implementation contracts for every module.

---

## 1. Data Layer

### 1.1 `src/data/loader.py`

#### `load_raw_data(seasons: list[int], cache_dir: str = "data/raw") -> pd.DataFrame`

**nfl_data_py API calls (in order):**

```python
import nfl_data_py as nfl

# 1. Weekly player stats — primary data source
weekly = nfl.import_weekly_data(seasons)
# Columns: player_id, player_name, position, recent_team, season, week,
#   completions, attempts, passing_yards, passing_tds, interceptions,
#   carries, rushing_yards, rushing_tds, receptions, targets,
#   receiving_yards, receiving_tds, sack_fumbles_lost, rushing_fumbles_lost,
#   target_share (may exist), air_yards_share (may exist), wopr (may exist)

# 2. Roster data — reliable position labels
rosters = nfl.import_rosters(seasons)
# Columns: player_id, season, position, team, ...
# Use to override weekly.position (which can be inconsistent)

# 3. Schedule data — for opponent info
schedules = nfl.import_schedules(seasons)
# Columns: season, week, home_team, away_team, ...
# Used to determine each player's opponent for matchup features

# 4. Snap counts — snap percentage feature
snap_counts = nfl.import_snap_counts(seasons)
# Columns: player (name), team, season, week, offense_snaps, offense_pct, ...
# Merge on player_id + season + week (may need name-based matching fallback)
```

**Merge strategy:**
1. `weekly` LEFT JOIN `rosters` on (`player_id`, `season`) → override position
2. `weekly` LEFT JOIN `snap_counts` on (`player_id`, `season`, `week`) → add snap_pct
3. Schedule data is merged later in feature engineering (for opponent lookup)

**Output schema (key columns):**
| Column | Type | Description |
|--------|------|-------------|
| player_id | str | GSIS ID, stable across seasons |
| player_name | str | Display name |
| position | str | QB/RB/WR/TE (from roster data) |
| recent_team | str | Team abbreviation for that week |
| season | int | 2018-2025 |
| week | int | 1-18 |
| passing_yards | float | Raw stat |
| rushing_yards | float | Raw stat |
| receiving_yards | float | Raw stat |
| receptions | float | Raw stat |
| targets | float | Raw stat |
| carries | float | Raw stat |
| passing_tds | float | Raw stat |
| rushing_tds | float | Raw stat |
| receiving_tds | float | Raw stat |
| interceptions | float | Raw stat |
| sack_fumbles_lost | float | Raw stat |
| rushing_fumbles_lost | float | Raw stat |
| snap_pct | float | Offensive snap percentage (0-100) |

**Caching:**
- `data/raw/weekly_2018_2025.parquet`
- `data/raw/rosters_2018_2025.parquet`
- `data/raw/schedules_2018_2025.parquet`
- `data/raw/snap_counts_2018_2025.parquet`
- Check `os.path.exists()` before pulling; delete file to force refresh

#### `compute_fantasy_points(df: pd.DataFrame, scoring: dict) -> pd.Series`

**Exact column mapping:**
```python
# Standard PPR scoring
fantasy_points = (
    df["passing_yards"].fillna(0) * 0.04 +
    df["passing_tds"].fillna(0) * 4 +
    df["interceptions"].fillna(0) * -2 +
    df["rushing_yards"].fillna(0) * 0.1 +
    df["rushing_tds"].fillna(0) * 6 +
    df["receptions"].fillna(0) * 1 +
    df["receiving_yards"].fillna(0) * 0.1 +
    df["receiving_tds"].fillna(0) * 6 +
    (df["sack_fumbles_lost"].fillna(0) + df["rushing_fumbles_lost"].fillna(0)) * -2
)
```

---

### 1.2 `src/data/preprocessing.py`

#### `preprocess(raw_df: pd.DataFrame) -> pd.DataFrame`

**Filter steps (in order):**

| Step | Action | Expected Impact |
|------|--------|-----------------|
| 1 | Filter to `position in ["QB", "RB", "WR", "TE"]` | Removes K, DEF, OL, etc. |
| 2 | Remove rows where player didn't play (0 snaps or NaN snap_pct AND all stats are 0) | Removes bye weeks, inactive players |
| 3 | Fill missing stat columns with 0 | No row removal |
| 4 | Fill missing `snap_pct` with position-week median | No row removal |
| 5 | Compute `fantasy_points` target via `compute_fantasy_points()` | Adds column |
| 6 | Remove players with < 6 games in a season | Removes low-sample players |

**Expected output:** ~30-35K rows with all stat columns + `fantasy_points` target.

---

### 1.3 `src/data/split.py`

#### `temporal_split(df, train_seasons, val_seasons, test_seasons) -> (train_df, val_df, test_df)`

| Split | Seasons | Expected Rows |
|-------|---------|---------------|
| Train | 2018-2022 | ~25,000 |
| Val | 2023 | ~5,000 |
| Test | 2024 | ~5,000 |

**Post-split actions:**
- Print split sizes to console
- Save split DataFrames to `data/splits/train.parquet`, `val.parquet`, `test.parquet`
- Assert no season overlap between splits

---

## 2. Feature Engineering

### 2.1 `src/features/engineer.py`

#### `build_features(df: pd.DataFrame) -> pd.DataFrame`

**Complete feature catalog:**

##### Rolling Features (72 features)

For each combination of stat × window × aggregation:

**Stats (8):** `fantasy_points`, `targets`, `receptions`, `carries`, `rushing_yards`, `receiving_yards`, `passing_yards`, `snap_pct`

**Windows (3):** 3, 5, 8 weeks

**Aggregations (3):** mean, std, max

**Naming convention:** `rolling_{agg}_{stat}_L{window}`
- Example: `rolling_mean_fantasy_points_L3`, `rolling_std_targets_L5`, `rolling_max_rushing_yards_L8`

**Code pattern (critical — prevents leakage):**
```python
# Group by player, shift(1) FIRST, then rolling
for stat in ROLL_STATS:
    for window in ROLLING_WINDOWS:
        shifted = df.groupby("player_id")[stat].shift(1)
        df[f"rolling_mean_{stat}_L{window}"] = shifted.rolling(window, min_periods=1).mean()
        df[f"rolling_std_{stat}_L{window}"]  = shifted.rolling(window, min_periods=1).std()
        df[f"rolling_max_{stat}_L{window}"]  = shifted.rolling(window, min_periods=1).max()
```

##### Share / Usage Features (5 features)

| Feature | Formula | Notes |
|---------|---------|-------|
| `target_share` | player_targets / team_targets (rolling, shifted) | Team totals from groupby(recent_team, season, week) |
| `carry_share` | player_carries / team_carries (rolling, shifted) | Same team groupby |
| `snap_pct` | Direct from data (already exists) | Not a derived feature |
| `air_yards_share` | player_air_yards / team_air_yards (rolling, shifted) | If column available |
| `redzone_targets_share` | player_rz_targets / team_rz_targets (rolling, shifted) | If column available; fill 0 if not |

**Team-level aggregation pattern:**
```python
# Compute team totals per week
team_totals = df.groupby(["recent_team", "season", "week"])["targets"].sum().reset_index()
team_totals.rename(columns={"targets": "team_targets"}, inplace=True)

# Merge back, then shift and compute share
df = df.merge(team_totals, on=["recent_team", "season", "week"], how="left")
df["team_targets_shifted"] = df.groupby("player_id")["team_targets"].shift(1)
df["player_targets_shifted"] = df.groupby("player_id")["targets"].shift(1)
df["target_share"] = df["player_targets_shifted"] / df["team_targets_shifted"]
```

##### Matchup / Opponent Features (2 features)

| Feature | Formula |
|---------|---------|
| `opp_fantasy_pts_allowed_to_pos` | 5-week rolling mean of fantasy points the opponent defense allowed to this position |
| `opp_def_rank_vs_pos` | Rank of opponent (1 = most points allowed = best matchup for offense) |

**Pipeline:**
1. From `schedules`, determine each team's opponent for each week
2. Merge opponent onto player rows: `df["opponent"] = lookup(recent_team, season, week)`
3. Compute defense stats: for each (team_as_defense, position, week), sum fantasy points scored against them
4. Rolling 5-week mean (shifted) of points allowed per position
5. Rank within each week (1 = most points allowed)
6. Join back to player rows on (opponent, position, week)

##### Contextual Features (4 features)

| Feature | Type | Source |
|---------|------|--------|
| `is_home` | int (0/1) | From schedule data: player's team == home_team |
| `week` | int (1-18) | Direct from data |
| `season_games_played` | int | Cumulative count of games for this player this season |
| `days_rest` | int | Days since player's last game (7 = normal, 14 = post-bye) |

##### Position Encoding (4 features)

One-hot: `pos_QB`, `pos_RB`, `pos_WR`, `pos_TE` (using `pd.get_dummies`)

##### Total: ~87 features

#### `get_feature_columns() -> list[str]`

Returns the ordered list of all 87 feature column names. Used by models to select input columns from the DataFrame.

#### NaN Filling Strategy

Applied AFTER all features are computed, BEFORE model training:

1. **Player prior season mean:** For each player + feature, fill NaN with that player's full prior-season mean for that feature
2. **Position-level training set mean:** Fill remaining NaNs with the mean of that feature across all players of the same position in the training set
3. **Zero fill:** Fill any still-remaining NaNs with 0

```python
# Pseudocode
for col in feature_columns:
    # Step 1: player prior season mean (computed separately)
    df[col] = df[col].fillna(player_prior_season_means)
    # Step 2: position-level training set mean
    df[col] = df.groupby("position")[col].transform(lambda x: x.fillna(x.mean()))
    # Step 3: zero
    df[col] = df[col].fillna(0)
```

---

## 3. Models

### 3.1 `src/models/baseline.py`

#### `class SeasonAverageBaseline`

```python
class SeasonAverageBaseline:
    """Predict each player's expanding season-to-date average fantasy points."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # For each row, compute the mean of fantasy_points for that player
        # in that season, using only PRIOR weeks (shift + expanding mean)
        # Returns: array of predictions, same length as df
        pass
```

#### `class LastWeekBaseline`

```python
class LastWeekBaseline:
    """Predict each player scored the same as last week."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # For each row, return the player's fantasy_points from the previous week
        # Handle week 1 / first appearance: use season average or 0
        # Returns: array of predictions, same length as df
        pass
```

### 3.2 `src/models/linear.py`

#### `class RidgeModel`

```python
class RidgeModel:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # 1. Fit scaler on X_train
        # 2. Transform X_train
        # 3. Fit Ridge on scaled X_train, y_train
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1. Transform X with fitted scaler
        # 2. Return Ridge predictions
        pass

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        # Return |coef_| sorted descending, indexed by feature name
        pass

    def save(self, model_dir: str = "outputs/models") -> None:
        # Save scaler to {model_dir}/scaler.pkl
        # Save model to {model_dir}/ridge_model.pkl
        pass

    def load(self, model_dir: str = "outputs/models") -> None:
        # Load scaler and model from disk
        pass
```

### 3.3 `src/models/neural_net.py`

#### `class FantasyPointsNet(nn.Module)`

```python
class FantasyPointsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int] = [128, 64, 32],
                 dropout: float = 0.3):
        super().__init__()
        # Build sequential blocks:
        # For each hidden layer size:
        #   nn.Linear(prev_dim, hidden_dim)
        #   nn.BatchNorm1d(hidden_dim)   # pre-activation normalization
        #   nn.ReLU()
        #   nn.Dropout(dropout)
        # Final: nn.Linear(last_hidden, 1)  # single regression output
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all hidden blocks
        # Return shape: (batch_size, 1) or (batch_size,)
        pass
```

**Architecture diagram:**
```
Input (~87) → Linear(87, 128) → BatchNorm(128) → ReLU → Dropout(0.3)
           → Linear(128, 64)  → BatchNorm(64)  → ReLU → Dropout(0.3)
           → Linear(64, 32)   → BatchNorm(32)  → ReLU → Dropout(0.3)
           → Linear(32, 1)    → Output (predicted fantasy points)
```

**Optimizer and loss:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)
```

---

## 4. Training

### 4.1 `src/training/trainer.py`

#### `class Trainer`

```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, patience=15):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

    def train(self, train_loader, val_loader, n_epochs) -> dict:
        """
        Returns history dict:
        {
            "train_loss": [float, ...],    # per epoch
            "val_loss": [float, ...],
            "val_mae": [float, ...],
            "val_rmse": [float, ...],
        }
        """
        # For each epoch:
        #   1. model.train() → iterate train_loader → forward, loss, backward, step
        #   2. model.eval() → iterate val_loader → compute val_loss, val_MAE, val_RMSE
        #   3. scheduler.step(val_loss)
        #   4. Early stopping check:
        #      if val_loss < best_val_loss:
        #          save model state, reset patience counter
        #      else:
        #          increment counter; if counter >= patience: restore best, break
        #   5. Log to history
        pass

    def plot_training_curves(self, history: dict, save_path: str) -> None:
        """Two-panel figure saved to outputs/figures/training_curves.png"""
        # Left panel:  train_loss and val_loss vs epoch
        # Right panel:  val_mae and val_rmse vs epoch
        # Mark early stopping point with vertical line
        pass
```

#### `make_dataloaders(X, y, batch_size, shuffle) -> DataLoader`

```python
def make_dataloaders(X_train, y_train, X_val, y_val, batch_size=256):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, val_loader
```

---

## 5. Evaluation

### 5.1 `src/evaluation/metrics.py`

#### `compute_metrics(y_true, y_pred) -> dict`

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2":   r2_score(y_true, y_pred),
    }
```

#### `compute_positional_metrics(df, pred_col, true_col) -> pd.DataFrame`

```python
def compute_positional_metrics(df, pred_col, true_col) -> pd.DataFrame:
    """Returns DataFrame with columns: position, mae, rmse, r2, n_samples"""
    results = []
    for pos in ["QB", "RB", "WR", "TE"]:
        mask = df["position"] == pos
        metrics = compute_metrics(df.loc[mask, true_col], df.loc[mask, pred_col])
        metrics["position"] = pos
        metrics["n_samples"] = mask.sum()
        results.append(metrics)
    return pd.DataFrame(results)
```

#### `print_comparison_table(results: dict) -> None`

```python
def print_comparison_table(results: dict) -> None:
    """
    results = {
        "Season Average": {"mae": ..., "rmse": ..., "r2": ...},
        "Last Week":      {"mae": ..., "rmse": ..., "r2": ...},
        "Ridge":          {"mae": ..., "rmse": ..., "r2": ...},
        "Neural Net":     {"mae": ..., "rmse": ..., "r2": ...},
    }
    Prints formatted comparison table to console.
    """
    pass
```

### 5.2 `src/evaluation/backtest.py`

#### `run_backtest(test_df, models_dict) -> dict`

```python
def run_backtest(test_df: pd.DataFrame, models_dict: dict) -> dict:
    """
    Simulate weekly fantasy season using different prediction strategies.

    Args:
        test_df: Test set with actual fantasy_points + model predictions
        models_dict: {"strategy_name": "prediction_column_name", ...}
            e.g., {"Season Avg": "pred_baseline", "Ridge": "pred_ridge", ...}
            Must also include "Oracle": "fantasy_points" (perfect knowledge)

    Lineup format (best-ball):
        1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (best remaining RB/WR/TE)
        Pick best available per position from FULL player pool each week.

    Returns:
        {
            "weekly_points": {strategy: [week1_pts, week2_pts, ...]},
            "total_points": {strategy: float},
            "weekly_wins_vs_baseline": {strategy: int},  # weeks beating baseline
            "oracle_capture_rate": {strategy: float},     # total_pts / oracle_pts
        }
    """
    pass
```

#### `plot_cumulative_points(backtest_results, save_path) -> None`

```python
def plot_cumulative_points(backtest_results: dict, save_path: str) -> None:
    """Line chart: cumulative fantasy points over the season for each strategy."""
    # X-axis: week number
    # Y-axis: cumulative points
    # One line per strategy, different colors
    # Legend with total season points
    pass
```

---

## 6. Explainability

### Location: `notebooks/03_error_analysis.ipynb` (Part B)

#### Ridge Coefficient Analysis

```python
# Get feature importance from Ridge model
importance = ridge_model.get_feature_importance(feature_names)

# Plot top 20 features
importance.head(20).plot(kind="barh", figsize=(10, 8))
plt.title("Ridge Regression: Top 20 Feature Importances (|coefficient|)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.savefig("outputs/figures/ridge_feature_importance.png")
```

#### Neural Net Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Wrap PyTorch model in sklearn-compatible interface
# (or use a simple wrapper that implements .predict())
perm_result = permutation_importance(
    model_wrapper, X_val, y_val,
    n_repeats=10, random_state=42, scoring="neg_mean_absolute_error"
)

# Plot top 20
perm_importance = pd.Series(perm_result.importances_mean, index=feature_names)
perm_importance.nlargest(20).plot(kind="barh", figsize=(10, 8))
plt.title("Neural Net: Top 20 Features by Permutation Importance")
plt.xlabel("Mean MAE Increase When Permuted")
plt.tight_layout()
plt.savefig("outputs/figures/nn_permutation_importance.png")
```

#### Side-by-Side Comparison

```python
# Combine Ridge and NN top features into one comparison
# Discuss: do both models agree on what matters?
# Expected finding: both likely value rolling_mean_fantasy_points_L3 highly,
# but NN may also pick up interaction effects (e.g., matchup × usage)
```

---

## 7. Config Constants (`src/config.py`)

Complete list of all configurable values:

```python
# === Data ===
SEASONS = list(range(2018, 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]
MIN_GAMES_PER_SEASON = 6
CACHE_DIR = "data/raw"
SPLITS_DIR = "data/splits"

# === Scoring (PPR) ===
SCORING = {
    "passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 0.1, "rushing_tds": 6,
    "receptions": 1, "receiving_yards": 0.1, "receiving_tds": 6,
    "fumbles_lost": -2,  # sum of sack_fumbles_lost + rushing_fumbles_lost
}

# === Split ===
TRAIN_SEASONS = list(range(2018, 2023))
VAL_SEASONS = [2023]
TEST_SEASONS = [2024]

# === Features ===
ROLLING_WINDOWS = [3, 5, 8]
ROLL_STATS = [
    "fantasy_points", "targets", "receptions", "carries",
    "rushing_yards", "receiving_yards", "passing_yards", "snap_pct",
]
ROLL_AGGS = ["mean", "std", "max"]
OPP_ROLLING_WINDOW = 5  # for matchup features

# === Ridge ===
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# === Neural Net ===
NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT = 0.3
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-4
NN_EPOCHS = 200
NN_BATCH_SIZE = 256
NN_PATIENCE = 15

# === LR Scheduler ===
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# === Backtest ===
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}  # FLEX = best RB/WR/TE
FLEX_POSITIONS = ["RB", "WR", "TE"]

# === Paths ===
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"
```

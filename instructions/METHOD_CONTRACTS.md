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
# Seasons: 2018-2024 (must match config.SEASONS)
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
# Columns: pfr_player_id, player, team, season, week, offense_snaps, offense_pct, ...
# WARNING: snap_counts uses pfr_player_id (Pro Football Reference ID), NOT
# the GSIS player_id used in weekly data. Merge strategy:
#   a. Try merge on (pfr_player_id) if rosters provide a pfr_player_id column
#   b. Fallback: merge on (player [name], team, season, week) — less reliable
#      due to name formatting differences (e.g., "Jr." vs "Jr")
#   c. Use fuzzy name matching or a manual mapping table as last resort
# After merge, rename offense_pct → snap_pct for consistency
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
- `data/raw/weekly_2018_2024.parquet`
- `data/raw/rosters_2018_2024.parquet`
- `data/raw/schedules_2018_2024.parquet`
- `data/raw/snap_counts_2018_2024.parquet`
- Check `os.path.exists()` before pulling; delete file to force refresh

#### `compute_fantasy_points(df: pd.DataFrame, scoring: dict = None) -> pd.Series`

**Uses `config.SCORING` dict — no hardcoded weights.**

```python
from src.config import SCORING

def compute_fantasy_points(df, scoring=None):
    if scoring is None:
        scoring = SCORING

    # Map config keys to DataFrame columns.
    # "fumbles_lost" is a virtual key: sum of sack_fumbles_lost + rushing_fumbles_lost
    col_map = {
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "interceptions": "interceptions",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "receptions": "receptions",
        "receiving_yards": "receiving_yards",
        "receiving_tds": "receiving_tds",
    }

    fantasy_points = pd.Series(0.0, index=df.index)
    for key, weight in scoring.items():
        if key == "fumbles_lost":
            # Special case: sum all *_fumbles_lost columns
            val = df["sack_fumbles_lost"].fillna(0) + df["rushing_fumbles_lost"].fillna(0)
        else:
            val = df[col_map[key]].fillna(0)
        fantasy_points += val * weight
    return fantasy_points


def compute_fantasy_points_floor(df) -> pd.Series:
    """Yardage + receptions only (no TDs, no turnovers).
    Separates the stable, predictable component of fantasy scoring
    from high-variance touchdown production."""
    return (
        df["passing_yards"].fillna(0) * 0.04 +
        df["rushing_yards"].fillna(0) * 0.1 +
        df["receiving_yards"].fillna(0) * 0.1 +
        df["receptions"].fillna(0) * 1.0
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
| 6 | Compute `fantasy_points_floor` (yardage + receptions only, no TDs) | Adds column |

**NOTE:** The min-games filter (< 6 games in a season) is intentionally NOT applied
here. It is applied in `build_features()` AFTER team totals are computed, so that
fringe players' targets/carries/air_yards contribute to correct team denominators
for share features. Filtering before team totals would inflate target_share and
carry_share for remaining players.

**Expected output:** ~35-40K rows with all stat columns + `fantasy_points` + `fantasy_points_floor` targets.
Rows are reduced to ~30-35K after the min-games filter in `build_features()`.

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

##### Rolling Features (93 features: 90 mean/std/max + 3 min)

For each combination of stat × window × aggregation:

**Stats (10):** `fantasy_points`, `fantasy_points_floor`, `targets`, `receptions`, `carries`, `rushing_yards`, `receiving_yards`, `passing_yards`, `attempts`, `snap_pct`

**Windows (3):** 3, 5, 8 weeks

**Aggregations (3):** mean, std, max — plus `min` for `fantasy_points` only (floor indicator)

**Naming convention:** `rolling_{agg}_{stat}_L{window}`
- Example: `rolling_mean_fantasy_points_L3`, `rolling_std_targets_L5`, `rolling_max_rushing_yards_L8`

**PREREQUISITE:** DataFrame must be sorted by `(player_id, season, week)` before any rolling operations. Unsorted data causes `shift(1)` to return the wrong row.

```python
df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
```

**Code pattern (critical — prevents leakage, cross-player bleed, AND cross-season contamination):**
```python
# Group by (player_id, season) — NOT just player_id — so rolling windows
# stay within a single season. Without the season boundary, shift(1) on
# Week 1 of 2020 returns Week 17 of 2019, and the L3/L5 windows for
# early-season weeks are entirely built from stale cross-offseason data.
# Players who changed teams/roles/schemes would carry misleading signal.
for stat in ROLL_STATS:
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{stat}_L{window}"] = df.groupby(
            ["player_id", "season"]
        )[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{stat}_L{window}"] = df.groupby(
            ["player_id", "season"]
        )[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"rolling_max_{stat}_L{window}"] = df.groupby(
            ["player_id", "season"]
        )[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
        # Rolling min for fantasy_points only (floor indicator)
        if stat == "fantasy_points":
            df[f"rolling_min_{stat}_L{window}"] = df.groupby(
                ["player_id", "season"]
            )[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
# NOTE: rolling std with min_periods=1 returns NaN for single observations
# (std of one value is undefined). The first valid row per player per season
# will have NaN for all std features. This is handled by fill_nans_safe().
```

##### Prior-Season Summary Features (24 features)

Since within-season rolling features are NaN for early weeks, we provide
explicit prior-season summaries as stable fallback signal.

```python
# Compute per-player, per-season aggregates
prior = df.groupby(["player_id", "season"]).agg(
    {stat: ["mean", "std", "max"] for stat in ROLL_STATS}
)
prior.columns = [f"prior_season_{agg}_{stat}" for stat, agg in prior.columns]

# Shift by one season: season S gets stats from season S-1
prior = prior.reset_index()
prior["season"] = prior["season"] + 1  # align S-1 stats with season S rows
df = df.merge(prior, on=["player_id", "season"], how="left")
# Rookies / first-year players will have NaN here → handled by fill_nans_safe()
```

##### EWMA Features (14 features)

Exponentially weighted moving averages weight recent games more heavily than
older games, capturing momentum better than equal-weight rolling means.

**Stats (7):** `fantasy_points`, `targets`, `carries`, `receiving_yards`, `rushing_yards`, `passing_yards`, `snap_pct`

**Spans (2):** 3, 5

**Naming convention:** `ewma_{stat}_L{span}`

```python
for stat in EWMA_STATS:
    for span in EWMA_SPANS:
        df[f"ewma_{stat}_L{span}"] = df.groupby(
            ["player_id", "season"]
        )[stat].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )
# EWMA with span=3 gives ~86% weight to the last 3 observations
# but with exponential decay, making it more responsive than rolling mean.
```

##### Trend / Momentum Features (4 features)

Captures whether a player is trending up or down. Positive trend = recent
performance exceeds longer-term average (breakout signal).

**Stats (4):** `fantasy_points`, `targets`, `carries`, `snap_pct`

**Formula:** `trend_{stat} = rolling_mean_{stat}_L3 - rolling_mean_{stat}_L8`

```python
for stat in TREND_STATS:
    short = df.groupby(["player_id", "season"])[stat].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    long = df.groupby(["player_id", "season"])[stat].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df[f"trend_{stat}"] = short - long
```

##### Share / Usage Features (6 features)

| Feature | Formula | Notes |
|---------|---------|-------|
| `target_share_L3` | rolling_sum(player_targets, L3) / rolling_sum(team_targets, L3), both shifted | Short-window: captures recent usage changes |
| `target_share_L5` | rolling_sum(player_targets, L5) / rolling_sum(team_targets, L5), both shifted | Longer-window: captures stable workload |
| `carry_share_L3` | rolling_sum(player_carries, L3) / rolling_sum(team_carries, L3), both shifted | Short-window carry share |
| `carry_share_L5` | rolling_sum(player_carries, L5) / rolling_sum(team_carries, L5), both shifted | Longer-window carry share |
| `snap_pct` | Direct from data (already exists) | Not a derived feature |
| `air_yards_share` | rolling_sum(player_air_yards, L5) / rolling_sum(team_air_yards, L5), both shifted | If column available; drop if not |

**NOTE:** `redzone_targets_share` is removed — nfl_data_py does not provide redzone
target data in weekly stats. An all-zero feature adds no signal.

**Team-level aggregation pattern (computed BEFORE min-games filter):**
```python
# Compute team totals per week — uses ALL players including fringe/filtered ones
team_totals = df.groupby(["recent_team", "season", "week"]).agg(
    team_targets=("targets", "sum"),
    team_carries=("carries", "sum"),
).reset_index()

# Merge back to player rows
df = df.merge(team_totals, on=["recent_team", "season", "week"], how="left")

# === Apply min-games filter HERE, after team totals are computed ===
games_per_season = df.groupby(["player_id", "season"])["week"].transform("count")
df = df[games_per_season >= MIN_GAMES_PER_SEASON].copy()
```

**Trade-safe share computation (uses stint_id to reset on team change):**
```python
# Detect team changes within a season
df = df.sort_values(["player_id", "season", "week"])
df["team_changed"] = (
    df.groupby(["player_id", "season"])["recent_team"].shift(1) != df["recent_team"]
).fillna(False)
df["stint_id"] = df.groupby(["player_id", "season"])["team_changed"].cumsum()

# Multi-window rolling shares with stint-aware grouping
for window in SHARE_WINDOWS:  # [3, 5]
    df[f"player_targets_roll_L{window}"] = df.groupby(
        ["player_id", "season", "stint_id"]
    )["targets"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    df[f"team_targets_roll_L{window}"] = df.groupby(
        ["player_id", "season", "stint_id"]
    )["team_targets"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    df[f"target_share_L{window}"] = (
        df[f"player_targets_roll_L{window}"] /
        df[f"team_targets_roll_L{window}"]
    ).fillna(0)

    # Same pattern for carry_share
    df[f"player_carries_roll_L{window}"] = df.groupby(
        ["player_id", "season", "stint_id"]
    )["carries"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    df[f"team_carries_roll_L{window}"] = df.groupby(
        ["player_id", "season", "stint_id"]
    )["team_carries"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).sum()
    )
    df[f"carry_share_L{window}"] = (
        df[f"player_carries_roll_L{window}"] /
        df[f"team_carries_roll_L{window}"]
    ).fillna(0)

# Clean up intermediate columns
drop_cols = [c for c in df.columns if c.startswith(("player_targets_roll", "team_targets",
             "player_carries_roll", "team_carries", "team_changed", "stint_id"))]
df.drop(columns=drop_cols, inplace=True)
```

##### Matchup / Opponent Features (4 features)

| Feature | Formula |
|---------|---------|
| `opp_fantasy_pts_allowed_to_pos` | 5-week rolling mean of total fantasy points the opponent defense allowed to this position |
| `opp_rush_pts_allowed_to_pos` | 5-week rolling mean of rushing fantasy points (rush yards + rush TDs) the opponent allowed to this position |
| `opp_recv_pts_allowed_to_pos` | 5-week rolling mean of receiving fantasy points (rec yards + rec TDs + receptions) the opponent allowed to this position |
| `opp_def_rank_vs_pos` | Rank of opponent (1 = most points allowed = best matchup for offense) |

**Splitting rush vs receiving points provides position-specific matchup signal:**
an RB facing a defense weak against the run but strong against dump-offs gets
a different (and more accurate) projection than one facing the reverse.

**Pipeline (leakage-safe — defense stats use only prior weeks):**
1. From `schedules`, determine each team's opponent for each week
2. Merge opponent onto player rows: `df["opponent"] = lookup(recent_team, season, week)`
3. Compute defense stats: for each (team_as_defense, position, week), sum the
   fantasy points scored AGAINST that defense by that position in that week.
   Compute total, rushing-only, and receiving-only fantasy points separately.
4. **CRITICAL:** shift(1) the defense stats per (team_as_defense, position) BEFORE
   rolling — this ensures week W's defense stat does NOT include week W's game.
   Then compute 5-week rolling mean of the shifted values.
5. Rank within each week across all teams (1 = most points allowed = best matchup)
6. Join back to player rows on (opponent, position, week)

```python
# Step 3-4 detail:
# Compute rushing and receiving fantasy point components per player-week
df["rush_fantasy"] = df["rushing_yards"] * 0.1 + df["rushing_tds"] * 6
df["recv_fantasy"] = df["receiving_yards"] * 0.1 + df["receiving_tds"] * 6 + df["receptions"] * 1

# def_pts = points scored AGAINST each defense, per position, per week
def_pts = df.groupby(["opponent", "position", "season", "week"]).agg(
    pts_allowed_to_pos=("fantasy_points", "sum"),
    rush_pts_allowed_to_pos=("rush_fantasy", "sum"),
    recv_pts_allowed_to_pos=("recv_fantasy", "sum"),
).reset_index()

# Shift FIRST (so week W doesn't include week W), then rolling mean
def_pts = def_pts.sort_values(["opponent", "position", "season", "week"])
for col in ["pts_allowed_to_pos", "rush_pts_allowed_to_pos", "recv_pts_allowed_to_pos"]:
    def_pts[f"opp_{col}"] = def_pts.groupby(
        ["opponent", "position"]
    )[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
# Rename to final feature names
def_pts.rename(columns={
    "opp_pts_allowed_to_pos": "opp_fantasy_pts_allowed_to_pos",
    "opp_rush_pts_allowed_to_pos": "opp_rush_pts_allowed_to_pos",
    "opp_recv_pts_allowed_to_pos": "opp_recv_pts_allowed_to_pos",
}, inplace=True)
```

##### Contextual Features (4 features)

| Feature | Type | Source |
|---------|------|--------|
| `is_home` | int (0/1) | From schedule data: player's team == home_team |
| `week` | int (1-18) | Direct from data |
| `is_returning_from_absence` | int (0/1) | Did the player miss 1+ games before this week? |
| `days_rest` | int | Days since player's last game (7 = normal, 14 = post-bye) |

**NOTE:** `season_games_played` was removed — it is nearly perfectly correlated
with `week` for healthy players (zero marginal signal). `is_returning_from_absence`
captures the actionable signal: players returning from injury/suspension often
have depressed first-game-back performance.

**`is_returning_from_absence` derivation:**
```python
# Detect gaps > 1 week between appearances
df["weeks_since_last_game"] = df.groupby(
    ["player_id", "season"]
)["week"].diff().fillna(1)
df["is_returning_from_absence"] = (df["weeks_since_last_game"] > 1).astype(int)
df.drop(columns=["weeks_since_last_game"], inplace=True)
```

**`days_rest` derivation (requires schedule data):**
```python
# 1. From schedules, extract game dates: (season, week, home_team, away_team, gameday)
# 2. Melt to one row per (team, season, week, gameday):
#    home rows + away rows → each team has a gameday per week
# 3. Merge onto player rows via (recent_team, season, week) → adds gameday
# 4. Sort by (player_id, season, week), then compute:
#    df["days_rest"] = df.groupby(["player_id", "season"])["gameday"].diff().dt.days
# 5. Fill NaN (week 1 / first appearance) with 7 (assume normal rest)
```

##### Position Encoding (4 features)

One-hot: `pos_QB`, `pos_RB`, `pos_WR`, `pos_TE` (using `pd.get_dummies`)

##### Total: ~144 features

#### `get_feature_columns() -> list[str]`

**Dynamically generates** the ordered list of all feature column names based on
config constants. Do NOT hardcode the list — it must stay in sync with
`build_features()` automatically.

```python
def get_feature_columns() -> list[str]:
    cols = []

    # Rolling features (mean/std/max for all stats, + min for fantasy_points)
    for stat in ROLL_STATS:
        for window in ROLLING_WINDOWS:
            for agg in ROLL_AGGS:  # ["mean", "std", "max"]
                cols.append(f"rolling_{agg}_{stat}_L{window}")
            if stat == "fantasy_points":
                cols.append(f"rolling_min_{stat}_L{window}")

    # Prior-season summary features
    for stat in ROLL_STATS:
        for agg in ROLL_AGGS:
            cols.append(f"prior_season_{agg}_{stat}")

    # EWMA features
    for stat in EWMA_STATS:
        for span in EWMA_SPANS:
            cols.append(f"ewma_{stat}_L{span}")

    # Trend features
    for stat in TREND_STATS:
        cols.append(f"trend_{stat}")

    # Share features (multi-window)
    for window in SHARE_WINDOWS:
        cols.append(f"target_share_L{window}")
        cols.append(f"carry_share_L{window}")
    cols += ["snap_pct", "air_yards_share"]

    # Matchup features
    cols += ["opp_fantasy_pts_allowed_to_pos", "opp_rush_pts_allowed_to_pos",
             "opp_recv_pts_allowed_to_pos", "opp_def_rank_vs_pos"]

    # Contextual features
    cols += ["is_home", "week", "is_returning_from_absence", "days_rest"]

    # Position encoding
    cols += ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    return cols
```

#### `fill_nans_safe(train_df, val_df, test_df, feature_cols) -> (train_df, val_df, test_df)`

**CRITICAL: Called AFTER `temporal_split()`, NOT inside `build_features()`.
Uses ONLY training set statistics for fill values to prevent data leakage.**

Steps:
1. **Player prior season mean:** For each player + feature, fill NaN with that player's full prior-season mean for that feature (safe: prior season is always known data)
2. **Position-level training set mean:** Fill remaining NaNs with the mean of that feature across all players of the same position **in the training set only**
3. **Zero fill:** Fill any still-remaining NaNs with 0

```python
def fill_nans_safe(train_df, val_df, test_df, feature_cols):
    """Fill NaNs using ONLY training set statistics. Must be called AFTER temporal_split."""

    # Step 1: Player prior-season mean (computed per player from train data)
    # This is safe for val/test too — prior-season data is always historical
    player_feature_means = train_df.groupby("player_id")[feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in feature_cols:
            mask = split_df[col].isna()
            split_df.loc[mask, col] = split_df.loc[mask, "player_id"].map(
                player_feature_means[col]
            )

    # Step 2: Position-level mean from TRAINING SET ONLY
    pos_means = train_df.groupby("position")[feature_cols].mean()
    for split_df in [train_df, val_df, test_df]:
        for col in feature_cols:
            for pos in ["QB", "RB", "WR", "TE"]:
                mask = (split_df[col].isna()) & (split_df["position"] == pos)
                split_df.loc[mask, col] = pos_means.loc[pos, col]

    # Step 3: Zero fill for any remaining NaNs (e.g., rookies with no history)
    for split_df in [train_df, val_df, test_df]:
        split_df[feature_cols] = split_df[feature_cols].fillna(0)

    return train_df, val_df, test_df
```

---

## 3. Models

### 3.1 `src/models/baseline.py`

#### `class SeasonAverageBaseline`

```python
class SeasonAverageBaseline:
    """Predict each player's expanding season-to-date average fantasy points.

    NOTE: This baseline requires the 'fantasy_points' column in df to compute
    predictions (it uses prior weeks' actual points). This is valid for
    backtesting — it simulates a fantasy manager who knows historical scores
    but not the current week's outcome. The predict() method uses only data
    available BEFORE each week (via shift + expanding mean).
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # For each row, compute the mean of fantasy_points for that player
        # in that season, using only PRIOR weeks (shift + expanding mean)
        # df MUST be sorted by (player_id, season, week)
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
        # Final linear outputs (batch_size, 1) — squeeze to (batch_size,)
        # MUST squeeze so shape matches y tensors in MSELoss.
        # return self.net(x).squeeze(-1)  # shape: (batch_size,)
        pass
```

**Architecture diagram:**
```
Input (~159) → Linear(159, 128) → BatchNorm(128) → ReLU → Dropout(0.3)
            → Linear(128, 64)   → BatchNorm(64)  → ReLU → Dropout(0.3)
            → Linear(64, 32)    → BatchNorm(32)  → ReLU → Dropout(0.3)
            → Linear(32, 1)     → squeeze(-1)    → Output shape: (batch_size,)
```
**NOTE:** input_dim should be set dynamically: `input_dim = len(get_feature_columns())`.
Consider testing `hidden_layers=[256, 128, 64]` during hyperparameter tuning given
the larger feature space.

**Output shape contract:** `forward()` returns shape `(batch_size,)` after squeezing, matching the shape of `y` tensors passed to `MSELoss`. Without squeezing, the `(batch_size, 1)` vs `(batch_size,)` mismatch causes silent broadcasting bugs in loss computation.

**Optimizer and loss:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)
# Interaction with early stopping: LR scheduler patience (5) < early stopping
# patience (15). The LR will halve up to ~3 times before early stopping fires.
# With factor=0.5 applied 3x, LR drops to 1/8 of initial (1e-3 → 1.25e-4).
# This is intentional: give the model a chance to escape plateaus via LR
# reduction before giving up entirely.
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
    """
    Expects PRE-SCALED numpy arrays. The pipeline (run_pipeline.py) is
    responsible for scaling using the scaler fitted by RidgeModel:
        scaler = ridge_model.scaler
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    Do NOT fit a second scaler here.
    """
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

#### `run_weekly_simulation(test_df, pred_columns, true_col="fantasy_points") -> dict`

```python
def run_weekly_simulation(test_df: pd.DataFrame, pred_columns: dict,
                          true_col: str = "fantasy_points") -> dict:
    """
    Week-by-week prediction accuracy simulation across the test season.
    Evaluates individual player projection quality — NO lineup construction.

    Args:
        test_df: Test set with actual fantasy_points + model prediction columns
        pred_columns: {"model_name": "prediction_column_name", ...}
            e.g., {"Season Avg": "pred_baseline", "Ridge": "pred_ridge",
                    "Neural Net": "pred_nn"}
        true_col: Column name for actual fantasy points

    Returns:
        {
            "weekly_metrics": {
                model_name: [{"week": int, "mae": float, "rmse": float, "r2": float}, ...]
            },
            "weekly_ranking": {
                model_name: [{
                    "week": int, "position": str,
                    "top12_hit_rate": float,   # precision@12
                    "spearman_corr": float,    # rank correlation
                }, ...]
            },
            "season_summary": {
                model_name: {"mae": float, "rmse": float, "r2": float}
            },
        }
    """
    pass
```

#### `plot_weekly_accuracy(sim_results, save_path) -> None`

```python
def plot_weekly_accuracy(sim_results: dict, save_path: str) -> None:
    """Two-panel figure showing prediction quality over the season.

    Left panel:  Per-week MAE for each model (line chart, weeks 1-18)
    Right panel: Per-week top-12 hit rate averaged across positions

    Highlights whether predictions improve as rolling features accumulate.
    """
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
SEASONS = list(range(2018, 2025))  # 2018-2024 only — must match union of split seasons
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

# === Features: Rolling ===
ROLLING_WINDOWS = [3, 5, 8]
ROLL_STATS = [
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "passing_yards",
    "attempts", "snap_pct",
]
# NOTE: "attempts" = passing_attempts from nfl_data_py (key QB volume signal)
# NOTE: "fantasy_points_floor" = yardage + receptions only (no TDs), computed
#        in preprocessing.py. Separates stable yardage floor from volatile TD upside.
ROLL_AGGS = ["mean", "std", "max"]
# rolling_min is computed only for fantasy_points (floor indicator)

# === Features: EWMA (exponentially weighted moving averages) ===
EWMA_STATS = ["fantasy_points", "targets", "carries", "receiving_yards",
              "rushing_yards", "passing_yards", "snap_pct"]
EWMA_SPANS = [3, 5]

# === Features: Trend/Momentum ===
TREND_STATS = ["fantasy_points", "targets", "carries", "snap_pct"]
# trend = rolling_mean_L3 - rolling_mean_L8 (positive = trending up)

# === Features: Share ===
SHARE_WINDOWS = [3, 5]  # multi-window for target_share and carry_share

# === Features: Opponent/Matchup ===
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
TOP_K_RANKING = 12  # Evaluate precision@K for per-position ranking accuracy

# === Paths ===
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"
```

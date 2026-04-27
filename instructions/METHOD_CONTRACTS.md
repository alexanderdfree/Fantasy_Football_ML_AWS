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
# Seasons: 2012-2025 (must match config.SEASONS)
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
| position | str | QB/RB/WR/TE/K/DST (from roster data) |
| recent_team | str | Team abbreviation for that week |
| season | int | 2012-2025 |
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
- `data/raw/weekly_2012_2025.parquet`
- `data/raw/rosters_2012_2025.parquet`
- `data/raw/schedules_2012_2025.parquet`
- `data/raw/snap_counts_2012_2025.parquet`
- Check `os.path.exists()` before pulling; delete file to force refresh

#### `compute_fantasy_points(df: pd.DataFrame, scoring: dict = None) -> pd.Series`

**Uses `config.SCORING` dict — no hardcoded weights.**

```python
from src.config import SCORING

def compute_fantasy_points(df, scoring=None):
    if scoring is None:
        scoring = SCORING

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
            val = df["sack_fumbles_lost"].fillna(0) + df["rushing_fumbles_lost"].fillna(0)
        else:
            val = df[col_map[key]].fillna(0)
        fantasy_points += val * weight
    return fantasy_points
```

#### `compute_all_scoring_formats(df: pd.DataFrame) -> pd.DataFrame`

Computes fantasy points for all three scoring formats. Adds columns:
`fantasy_points_standard`, `fantasy_points_half_ppr`, `fantasy_points` (full PPR).

```python
def compute_all_scoring_formats(df):
    df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)
    df["fantasy_points"] = compute_fantasy_points(df, SCORING_PPR)
    return df
```

#### `compute_fantasy_points_floor(df: pd.DataFrame, ppr_weight: float = 1.0) -> pd.Series`

Yardage + receptions only (no TDs, no turnovers). The `ppr_weight` parameter
controls the reception value (0.0 for standard, 0.5 for half-PPR, 1.0 for full PPR).

```python
def compute_fantasy_points_floor(df, ppr_weight=1.0):
    return (
        df["passing_yards"].fillna(0) * 0.04 +
        df["rushing_yards"].fillna(0) * 0.1 +
        df["receiving_yards"].fillna(0) * 0.1 +
        df["receptions"].fillna(0) * ppr_weight
    )
```

#### `compute_all_floor_formats(df: pd.DataFrame) -> pd.DataFrame`

Computes floor for all three formats. Adds columns:
`fantasy_points_floor_standard`, `fantasy_points_floor_half_ppr`, `fantasy_points_floor`.

```python
def compute_all_floor_formats(df):
    df["fantasy_points_floor_standard"] = compute_fantasy_points_floor(df, ppr_weight=0.0)
    df["fantasy_points_floor_half_ppr"] = compute_fantasy_points_floor(df, ppr_weight=0.5)
    df["fantasy_points_floor"] = compute_fantasy_points_floor(df, ppr_weight=1.0)
    return df
```

---

### 1.2 `src/data/preprocessing.py`

#### `preprocess(raw_df: pd.DataFrame) -> pd.DataFrame`

**Filter steps (in order):**

| Step | Action | Expected Impact |
|------|--------|-----------------|
| 1 | Filter to `position in ["QB", "RB", "WR", "TE", "K", "DST"]` | Removes OL, etc. |
| 2 | Remove rows where player didn't play (0 snaps or NaN snap_pct AND all stats are 0) | Removes bye weeks, inactive players |
| 3 | Fill missing stat columns with 0 | No row removal |
| 4 | Fill missing `snap_pct` with position-week median | No row removal |
| 5 | Compute fantasy points for all scoring formats via `compute_all_scoring_formats()` | Adds `fantasy_points_standard`, `fantasy_points_half_ppr`, `fantasy_points` |
| 6 | Compute fantasy points floor for all formats via `compute_all_floor_formats()` | Adds `fantasy_points_floor_standard`, `fantasy_points_floor_half_ppr`, `fantasy_points_floor` |

**NOTE:** The min-games filter (< 6 games in a season) is intentionally NOT applied
here. It is applied in `build_features()` AFTER team totals are computed, so that
fringe players' targets/carries/air_yards contribute to correct team denominators
for share features. Filtering before team totals would inflate target_share and
carry_share for remaining players.

**Expected output:** ~35-40K rows with all stat columns + fantasy points columns for all three scoring formats (standard, half-PPR, full PPR) and their corresponding floor columns.
Rows are reduced to ~30-35K after the min-games filter in `build_features()`.

---

### 1.3 `src/data/split.py`

#### `temporal_split(df, train_seasons, val_seasons, test_seasons) -> (train_df, val_df, test_df)`

| Split | Seasons | Expected Rows |
|-------|---------|---------------|
| Train | 2012-2023 | ~12 seasons of data |
| Val | 2024 | ~1 season |
| Test | 2025 | ~1 season |

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

##### Prior-Season Summary Features (30 features)

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

##### Total: ~155 general features (before position-specific additions/drops)

Each position then adds its own specific features (varying count per position) and drops
irrelevant ones (e.g., QBs drop receiver-centric features, RBs drop passing features).
Weather/Vegas features (implied_team_total, implied_opp_total, is_dome, etc.) are added
per-position from `src/shared/weather_features.py`.
K and DST bypass the general feature pipeline entirely and use custom features.

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
            for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
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
        # in that season, using only PRIOR weeks (shift + expanding mean).
        # Sort is handled internally; predictions are returned in the caller's
        # row order.
        # Returns: array of predictions, same length as df
        pass
```

#### `class LastWeekBaseline`

```python
class LastWeekBaseline:
    """Predict each player scored the same as last week."""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # For each row, return the player's fantasy_points from the previous week.
        # Handle week 1 / first appearance: use season average or 0.
        # Sort is handled internally; predictions are returned in the caller's
        # row order.
        # Returns: array of predictions, same length as df
        pass
```

### 3.2 `src/shared/models.py`

#### `class RidgeMultiTarget`

```python
class RidgeMultiTarget:
    """Wrapper around scikit-learn Ridge for multi-target prediction.
    Each target gets an independent Ridge model with separate alpha."""

    def __init__(self, target_names: list[str], alphas: dict[str, float]):
        # Creates one Ridge model per target with its own alpha
        pass

    def fit(self, X_train: np.ndarray, y_dict: dict[str, np.ndarray]) -> None:
        # Fit each Ridge model on its target
        pass

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        # Returns dict of per-target predictions (clamped >= 0) + "total" (sum)
        pass

    def get_feature_importance(self, target_name: str) -> pd.Series:
        # Return |coef_| for a specific target
        pass
```

### 3.3 `src/shared/neural_net.py`

#### `class MultiHeadNet(nn.Module)`

```python
class MultiHeadNet(nn.Module):
    def __init__(self, input_dim: int, target_names: list[str],
                 backbone_layers: list[int], head_hidden: int = 32,
                 dropout: float = 0.3, head_hidden_overrides: dict = None):
        # Shared backbone: [Linear → BatchNorm → ReLU → Dropout] × len(backbone_layers)
        # Per-target heads: Linear(backbone_out, head_hidden) → ReLU → Linear(head_hidden, 1)
        pass

    def forward(self, x: torch.Tensor) -> dict:
        # Returns dict with per-target predictions + "total"
        # Each head output passes through clamp(min=0) for non-negativity
        # (configurable per-target via non_negative_targets parameter)
        pass

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        # Convenience method: numpy in → numpy out
        pass
```

**Architecture diagram (example: RB with backbone=[128, 64] and raw-stat targets):**
```
Input (N features) → Linear(N, 128) → BatchNorm(128) → ReLU → Dropout(0.15)
                   → Linear(128, 64) → BatchNorm(64) → ReLU → Dropout(0.15)
                   ├→ rushing_tds head:     GatedTDHead(64, gate_hidden=16, value_hidden=64)
                   ├→ receiving_tds head:   GatedTDHead(64, gate_hidden=16, value_hidden=64)
                   ├→ rushing_yards head:   Linear(64, 48) → ReLU → Linear(48, 1) → clamp(min=0)
                   ├→ receiving_yards head: Linear(64, 48) → ReLU → Linear(48, 1) → clamp(min=0)
                   ├→ receptions head:      Linear(64, 48) → ReLU → Linear(48, 1) → clamp(min=0)
                   └→ fumbles_lost head:    Linear(64, 48) → ReLU → Linear(48, 1) → clamp(min=0)
                   Total = fantasy points aggregated post-hoc via shared.aggregate_targets
```

**Note:** `non_negative_targets` controls which heads are clamped (per-head, not global).
By default all targets are clamped to >= 0. DST overrides this to leave `pts_allowed_bonus`
unconstrained (range: -4 to +10).

**Note:** RB's TD heads (`rushing_tds`, `receiving_tds`) are wrapped by `GatedTDHead`,
a zero-inflated hurdle head: a sigmoid gate predicts `P(TD > 0)` and a Softplus branch
predicts `E[TD | TD > 0]`; the product is the expected TD count. WR/TE use one gated head
(`receiving_tds`); QB/K/DST do not use gated TD heads.

**Optimizer and loss (RB raw-stat example, from `src/RB/rb_config.py`):**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5)
criterion = MultiTargetLoss(
    # Representative subset — full target list is RB_TARGETS
    target_names=["rushing_yards", "receiving_yards", "receptions", "rushing_tds"],
    # RB_LOSS_WEIGHTS entries (~2.0 / huber_delta)
    loss_weights={"rushing_yards": 0.133, "receiving_yards": 0.133,
                  "receptions": 1.0, "rushing_tds": 4.0},
    # RB_HUBER_DELTAS entries (raw-stat units)
    huber_deltas={"rushing_yards": 15.0, "receiving_yards": 15.0,
                  "receptions": 2.0, "rushing_tds": 0.5},
    w_total=1.0,  # RB_LOSS_W_TOTAL
)
# Scheduler varies by position: OneCycleLR (TE, K) or CosineWarmRestarts (QB, RB, WR, DST)
```

---

## 4. Training

### 4.1 `src/shared/training.py`

#### `class MultiTargetLoss(nn.Module)`

```python
class MultiTargetLoss(nn.Module):
    """Combined Huber loss for multi-head network.
    Loss = sum(weight[t] * Huber(pred[t], target[t])) + w_total * Huber(total_pred, total_actual)
    Per-target Huber deltas allow different MSE-to-MAE thresholds.
    """
    def __init__(self, target_names, loss_weights, huber_deltas=None, w_total=0.5):
        pass

    def forward(self, preds: dict, targets: dict) -> tuple:
        # Returns (combined_loss, components_dict)
        pass
```

#### `class MultiHeadTrainer`

```python
class MultiHeadTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device,
                 target_names, patience=15, scheduler_per_batch=False):
        pass

    def train(self, train_loader, val_loader, n_epochs) -> dict:
        """
        Returns history dict:
        {
            "train_loss": [float, ...],
            "val_loss": [float, ...],
            "val_loss_{target}": [float, ...],   # per target
            "val_mae_{target}": [float, ...],    # per target
            "val_mae_total": [float, ...],
            "val_rmse_total": [float, ...],
        }
        """
        # For each epoch:
        #   1. Train: forward → loss → backward → gradient clipping (max_norm=1.0) → step
        #   2. Val: per-target + total MAE/RMSE
        #   3. LR scheduler step (supports ReduceLROnPlateau, OneCycleLR, CosineWarmRestarts)
        #   4. Early stopping: restore best weights if no improvement in `patience` epochs
        pass
```

#### `make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size)`

```python
def make_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    """Creates DataLoaders for multi-target training.
    y_train_dict/y_val_dict: {target_name: np.ndarray, "total": np.ndarray}
    Train: shuffle=True, drop_last=True; Val: shuffle=False
    """
    pass
```

#### `plot_training_curves(history, target_names, save_path)`

```python
def plot_training_curves(history, target_names, save_path):
    """Four-panel figure:
    Top-left: overall train/val loss
    Top-right: per-target validation losses
    Bottom-left: per-target MAE
    Bottom-right: total MAE and RMSE
    """
    pass
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
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
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

### Ridge Coefficient Analysis

Per-target feature importance is available via `RidgeMultiTarget.get_feature_importance(target_name)`,
which returns `|coef_|` sorted descending for any target.

### Neural Net Permutation Importance

`MultiHeadNet.predict_numpy()` provides a convenient numpy interface for permutation importance analysis
using `sklearn.inspection.permutation_importance`.

---

## 7. Config Constants (`src/config.py`)

Complete list of all configurable values:

```python
# === Data ===
SEASONS = list(range(2012, 2026))  # 2012-2025 (snap counts available from 2012+)
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
MIN_GAMES_PER_SEASON = 6
CACHE_DIR = "data/raw"
SPLITS_DIR = "data/splits"

# === Scoring ===
_BASE_SCORING = {
    "passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
    "rushing_yards": 0.1, "rushing_tds": 6,
    "receiving_yards": 0.1, "receiving_tds": 6,
    "fumbles_lost": -2,  # sum of sack_fumbles_lost + rushing_fumbles_lost
}
PPR_FORMATS = {"standard": 0.0, "half_ppr": 0.5, "ppr": 1.0}
SCORING_STANDARD = {**_BASE_SCORING, "receptions": 0.0}
SCORING_HALF_PPR = {**_BASE_SCORING, "receptions": 0.5}
SCORING_PPR      = {**_BASE_SCORING, "receptions": 1.0}
SCORING = SCORING_PPR  # Default (full PPR)

# === Split ===
TRAIN_SEASONS = list(range(2012, 2024))  # 2012-2023
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]

# === Cross-Validation (expanding window) ===
CV_VAL_SEASONS = [2021, 2022, 2023, 2024]

# === Features: Rolling ===
ROLLING_WINDOWS = [3, 5, 8]
ROLL_STATS = [
    "fantasy_points", "fantasy_points_floor", "targets", "receptions",
    "carries", "rushing_yards", "receiving_yards", "passing_yards",
    "attempts", "snap_pct",
]
ROLL_AGGS = ["mean", "std", "max"]

# === Features: EWMA ===
EWMA_STATS = ["fantasy_points", "targets", "carries", "receiving_yards",
              "rushing_yards", "passing_yards", "snap_pct"]
EWMA_SPANS = [3, 5]

# === Features: Trend/Momentum ===
TREND_STATS = ["fantasy_points", "targets", "carries", "snap_pct"]

# === Features: Share ===
SHARE_WINDOWS = [3, 5]

# === Features: Opponent/Matchup ===
OPP_ROLLING_WINDOW = 5

# === Ridge ===
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# === Neural Net (global defaults; overridden per-position in {POS}/{pos}_config.py) ===
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
TOP_K_RANKING = 12

# === Paths ===
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"
```

**Position-specific configs** are in `{POS}/{pos}_config.py` and override the global defaults.
Each position config defines:
- Target decomposition (`{POS}_TARGETS`)
- Position-specific features (`{POS}_SPECIFIC_FEATURES`)
- Features to drop (`{POS}_DROP_FEATURES`)
- Ridge alpha grids per target
- NN architecture (backbone, head hidden, dropout, learning rate, etc.)
- Loss weights and Huber deltas per target
- LR scheduler type and parameters

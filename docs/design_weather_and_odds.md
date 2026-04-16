# Design Document 1: Weather & Implied-Odds Features for the Existing Neural Network

> **Status: IMPLEMENTED.** Weather and Vegas features are now computed in `shared/weather_features.py`
> and used by all position pipelines. Position-specific subsets are configured in each `{pos}_config.py`
> under the `weather_vegas` key in the include-features dict.

## Motivation

Your schedule data (loaded in `src/data/loader.py`) already contains `roof`, `surface`, `temp`, `wind`, `spread_line`, and `total_line` columns but none of these flow into the feature engineering pipeline. Multiple papers (Landers & Duperrouzel 2019, IBM Watson 2018/2019) found venue/context features to be among the highest-correlated predictors. The Vegas implied team total is widely regarded as the single strongest publicly available prior for expected offensive output.

## Data Availability

Already present in your schedule data (via `nfl_data_py`):

| Column | Coverage | Example Values |
|--------|----------|----------------|
| `roof` | ~100% | outdoors, dome, closed, open |
| `surface` | ~98% | grass, fieldturf, a_turf, sportturf |
| `temp` | ~67% | -6 to 97 (Fahrenheit) |
| `wind` | ~67% | 0-35 (mph) |
| `spread_line` | ~95% (2012+) | -7.5 to +14 (negative = favored) |
| `total_line` | ~95% (2012+) | 33.5 to 56.5 |
| `home_rest` / `away_rest` | ~100% | days since last game |
| `div_game` | ~100% | 0 or 1 |

**Key point**: `temp` and `wind` are missing for ~33% of games. These are dome games and games where weather wasn't recorded. Dome games have no meaningful weather, so "missing = dome" is actually informative, not a gap.

## New Features (12 total)

### Vegas-Derived Features (3) — highest expected impact

```
implied_team_total = (total_line / 2) + (spread_line / 2)    # if home team
implied_team_total = (total_line / 2) - (spread_line / 2)    # if away team
```

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `implied_team_total` | See above | Best public prior for expected team offense. A team implied for 27 points will produce more fantasy output than one implied for 17. |
| `implied_opp_total` | `total_line - implied_team_total` | Proxy for expected game script: if opponent is expected to score a lot, the team may need to pass more (helps WR/pass-catching RBs). |
| `total_line` | Raw over/under | Game-level pace proxy. High totals = fast-paced, high-scoring environments that lift all fantasy-relevant players. |

**Why not `spread_line` directly?** The spread is already encoded in `implied_team_total` and `implied_opp_total`. Including it separately adds multicollinearity without new information.

### Venue Features (5)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `is_dome` | 1 if `roof` in {dome, closed}, else 0 | Dome games have higher passing stats (no wind/rain). Binary is cleaner than one-hot for 4 roof types. |
| `is_grass` | 1 if `surface` == 'grass', else 0 | Grass vs. turf affects injury risk and RB rushing efficiency. |
| `temp_adjusted` | `temp` if outdoors, else 65.0 (imputed neutral) | Extreme cold (< 32F) suppresses passing. Dome games get a neutral value rather than NaN. |
| `wind_adjusted` | `wind` if outdoors, else 0.0 | High wind (> 15 mph) suppresses deep passing and kicking. |
| `is_divisional` | `div_game` column | Divisional games have different scoring patterns (more familiarity, sometimes lower scoring). |

### Rest/Schedule Features (2)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `days_rest_improved` | `home_rest` if home, `away_rest` if away | More granular than your current `days_rest` (which uses week gaps × 7). Uses actual rest days from schedule data. |
| `rest_advantage` | `player_rest - opponent_rest` | Relative rest edge. A team on 10 days rest vs. opponent on 4 has a meaningful advantage. |

### Interaction Feature (2)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `implied_total_x_dome` | `implied_team_total × is_dome` | Dome games with high implied totals are premium passing environments. Captures the compounding effect. |
| `implied_total_x_wind` | `implied_team_total × (1 - wind_adjusted/40)` | High wind reduces the realized upside of high-implied-total games. |

## Files to Modify

### 1. `src/data/loader.py` — Merge schedule columns into player data

Currently, schedule data is loaded but only `is_home` and `opponent_team` make it into the main DataFrame. The schedule contains `spread_line`, `total_line`, `roof`, `surface`, `temp`, `wind`, `home_rest`, `away_rest`, and `div_game`.

**Change**: In `load_raw_data()`, after the existing schedule merge (~line 85-95), also carry forward the columns listed above. The merge key is already `(season, week, recent_team)` — just add the columns to the merge selection.

```python
# Existing merge already joins on season/week/team
# Add these columns to the list of schedule fields carried forward:
schedule_cols = [
    "spread_line", "total_line", "roof", "surface",
    "temp", "wind", "home_rest", "away_rest", "div_game"
]
```

**Leakage check**: All of these are known *before* the game starts (Vegas lines close at kickoff, weather is forecast, venue is fixed). No leakage risk.

### 2. `src/features/engineer.py` — Build the 12 new features

Add a new function `_build_vegas_and_venue_features(df)` called from `build_features()`.

```python
def _build_vegas_and_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Vegas ---
    home_mask = df["is_home"] == 1
    df["implied_team_total"] = np.where(
        home_mask,
        df["total_line"] / 2 + df["spread_line"] / 2,   # spread is from home perspective
        df["total_line"] / 2 - df["spread_line"] / 2,
    )
    df["implied_opp_total"] = df["total_line"] - df["implied_team_total"]
    # total_line stays as-is

    # --- Venue ---
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["is_grass"] = (df["surface"] == "grass").astype(int)
    df["temp_adjusted"] = np.where(df["is_dome"] == 1, 65.0, df["temp"].fillna(65.0))
    df["wind_adjusted"] = np.where(df["is_dome"] == 1, 0.0, df["wind"].fillna(0.0))
    df["is_divisional"] = df["div_game"].fillna(0).astype(int)

    # --- Rest ---
    df["days_rest_improved"] = np.where(home_mask, df["home_rest"], df["away_rest"]).clip(4, 21)
    opp_rest = np.where(home_mask, df["away_rest"], df["home_rest"])
    df["rest_advantage"] = df["days_rest_improved"] - opp_rest

    # --- Interactions ---
    df["implied_total_x_dome"] = df["implied_team_total"] * df["is_dome"]
    df["implied_total_x_wind"] = df["implied_team_total"] * (1 - df["wind_adjusted"] / 40).clip(0, 1)

    return df
```

Then update `get_feature_columns()` to include the 12 new column names.

### 3. `src/features/engineer.py` — Update `get_feature_columns()`

Append the 12 new feature names to the returned list:

```python
vegas_venue_features = [
    "implied_team_total", "implied_opp_total", "total_line",
    "is_dome", "is_grass", "temp_adjusted", "wind_adjusted", "is_divisional",
    "days_rest_improved", "rest_advantage",
    "implied_total_x_dome", "implied_total_x_wind",
]
```

### 4. Position config files — Decide which features to keep per position

Not all 12 features matter equally for every position:

| Feature | QB | RB | WR | TE | K |
|---------|----|----|----|----|---|
| `implied_team_total` | Y | Y | Y | Y | Y |
| `implied_opp_total` | Y | Y | Y | Y | N |
| `total_line` | Y | Y | Y | Y | Y |
| `is_dome` | Y | N | Y | Y | Y |
| `is_grass` | N | Y | N | N | N |
| `temp_adjusted` | Y | N | Y | N | Y |
| `wind_adjusted` | Y | N | Y | N | Y |
| `is_divisional` | Y | Y | Y | Y | N |
| `days_rest_improved` | Y | Y | Y | Y | Y |
| `rest_advantage` | Y | Y | Y | Y | N |
| `implied_total_x_dome` | Y | N | Y | Y | N |
| `implied_total_x_wind` | Y | N | Y | N | Y |

- RBs: Grass/turf matters for rushing efficiency; wind/dome less important since their production is run-heavy
- QBs/WRs: Weather and dome status directly impact passing volume
- Kickers: Wind and dome are critical for field goals

However, these 12 general features go into the *shared* `get_feature_columns()` list (not the position-specific 8). The per-position `drop_features` lists in config files should exclude irrelevant ones for each position.

### 5. `src/features/engineer.py` — Update `fill_nans_safe()`

The existing NaN fill logic (player mean → position mean → zero) should work for most new features. However:
- `implied_team_total`, `implied_opp_total`, `total_line`: fill with position-week median from training set (these are game-level, not player-level)
- `temp_adjusted`, `wind_adjusted`: already handled by the imputation in `_build_vegas_and_venue_features`
- Binary features (`is_dome`, `is_grass`, `is_divisional`): fill with 0

No new fill logic is needed — the existing step 3 (zero fill) handles all remaining NaNs, which is correct behavior for binary features and reasonable for Vegas lines (0 = no information).

### 6. No changes to `shared/neural_net.py` or `shared/training.py`

The neural network and training infrastructure are feature-count-agnostic. `input_dim` is inferred from the feature array at pipeline runtime. Adding 12 features just increases `input_dim` from ~70 to ~82. No architectural changes needed.

## Handling Missing Vegas Data

For games before ~2012 or rare missing lines:
- `implied_team_total` / `implied_opp_total` / `total_line`: will be NaN → filled to 0 by `fill_nans_safe` step 3
- Since your training data starts at 2012 and Vegas data has ~95% coverage from 2012+, this affects <5% of rows
- Alternative: fill with position-season median implied total (more informative than 0). This could be added as a step in `fill_nans_safe`.

## Leakage Audit

| Feature | Known pre-game? | Safe? |
|---------|----------------|-------|
| `implied_team_total` | Yes (lines set before kickoff) | Yes |
| `implied_opp_total` | Yes | Yes |
| `total_line` | Yes | Yes |
| `is_dome` / `is_grass` | Yes (fixed venue) | Yes |
| `temp_adjusted` / `wind_adjusted` | Yes (pregame weather) | Yes |
| `is_divisional` | Yes (schedule) | Yes |
| `days_rest_improved` | Yes (schedule) | Yes |
| `rest_advantage` | Yes (schedule) | Yes |
| Interaction features | Derived from safe inputs | Yes |

All features are available before kickoff. No target leakage.

## Verification Plan

1. Run the data loader and confirm the schedule columns merge correctly — print `df[schedule_cols].describe()` to verify coverage
2. Run `build_features()` and confirm 12 new columns appear with reasonable value ranges
3. Run the RB pipeline (fastest, ~30s) and compare test MAE before/after:
   - Expect `implied_team_total` to be the highest-impact new feature
   - Check Ridge feature importance to see if Vegas features rank highly
4. Run all position pipelines and compare `benchmark_results.json` before/after
5. Spot-check: for a known dome game (e.g., Rams at SoFi), verify `is_dome=1`, `wind_adjusted=0`, `temp_adjusted=65`

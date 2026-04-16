# RB Position Model — Detailed Implementation Document

## Companion to `instructions/DESIGN_DOC.md` and `instructions/METHOD_CONTRACTS.md`

This document specifies the complete implementation plan for an RB-specific fantasy football points predictor. It specializes the general all-position pipeline into a targeted Running Back model with multi-target decomposition.

---

## 1. Overview and Objectives

### 1.1 Purpose

Build an RB-only fantasy football points predictor that lives as a self-contained subfolder within the general project. The RB model specializes the general pipeline by:

1. Filtering all data to RB-only rows
2. Decomposing the prediction target into three sub-targets (rushing_floor, receiving_floor, td_points) plus a fumble penalty term
3. Adding 15 RB-specific engineered features beyond the general features
4. Using a multi-head neural network architecture where a shared backbone feeds three separate output heads
5. Training 3 separate Ridge models (one per sub-target) for the linear baseline

### 1.2 Core Hypothesis

Decomposing RB fantasy points into rushing floor, receiving floor, and touchdown points will improve predictions because:

- **Rushing floor** (yards-based) is the most volume-predictable component
- **Receiving floor** (receptions + yards) captures the PPR-specific value of pass-catching backs
- **TD points** are high-variance and harder to predict, but isolating them prevents TD noise from corrupting floor predictions

### 1.3 Success Criteria

- Total fantasy point MAE on RB test rows should be competitive with or better than the general model's RB-filtered MAE
- Per-target MAE should reveal that rushing_floor is easiest to predict, receiving_floor is intermediate, and td_points is hardest
- Top-12 RB hit rate (precision@12) should exceed 40% weekly

---

## 2. Data Pipeline (RB-Specific Filtering and Dataset Stats)

### 2.1 Data Source

The RB model reuses the general pipeline's data loading and preprocessing. It consumes the output of `src/data/preprocessing.py` (`preprocess()`) and `src/features/engineer.py` (`build_features()`), then applies an RB filter.

**Critical ordering**: The RB filter must be applied AFTER `build_features()` completes, because:
- Team-level aggregations (team_targets, team_carries) need ALL position players
- Opponent defense stats (`opp_fantasy_pts_allowed_to_pos`) are computed by position, using all players of that position
- Share features (target_share, carry_share) require the full team denominator

### 2.1.1 nflverse Column Availability for RBs

`nfl.import_weekly_data()` provides ~115 columns. The general pipeline uses a subset; the RB model exploits additional RB-relevant columns that are already present in the raw data but unused by the general pipeline. **No additional API calls are required.**

**Columns already used by the general pipeline:**
`carries`, `rushing_yards`, `rushing_tds`, `rushing_fumbles_lost`, `receptions`, `targets`, `receiving_yards`, `receiving_tds`, `sack_fumbles_lost`, `snap_pct`

**Additional columns available for RB feature engineering (confirmed in nflverse data dictionary):**

| Column | Type | RB Relevance |
|--------|------|-------------|
| `rushing_first_downs` | int | First downs gained on rush attempts — chain-moving signal |
| `receiving_first_downs` | int | First downs on receptions — pass-catching value |
| `rushing_epa` | float | Total expected points added on rush plays — context-adjusted efficiency |
| `receiving_epa` | float | Total EPA on targets — context-adjusted receiving value |
| `receiving_air_yards` | float | Total air yards on targets — average depth of target proxy |
| `receiving_yards_after_catch` | float | YAC — RBs generate most receiving yards after catch |
| `rushing_2pt_conversions` | int | 2-point conversion rushes — rare but contributes to scoring |
| `receiving_2pt_conversions` | int | 2-point conversion receptions — rare but contributes to scoring |
| `opponent_team` | str | Opponent abbreviation — directly available, no schedule merge needed for basic opponent lookup |
| `fantasy_points_ppr` | float | Pre-computed PPR fantasy points by nflverse — use as validation check against our computation |
| `target_share` | float | Weekly player targets / team targets — pre-computed snapshot (our rolling shares differ intentionally) |
| `special_teams_tds` | int | Kick/punt return TDs — rare for RBs but nonzero for some (e.g., Cordarrelle Patterson) |

**Columns NOT available in `import_weekly_data()` (common misconceptions):**
- Red zone carries/targets — must be derived from play-by-play data (`import_pbp_data()`), which is ~700MB/season and out of scope for this project
- Route participation rate — not tracked in summary stats
- Yards before contact — only in Next Gen Stats (`import_ngs_data("rushing")`), limited season availability
- Offensive line grades — external data source (PFF), not in nflverse

### 2.1.2 Snap Count Merge Strategy (Corrected)

Snap counts use `pfr_player_id` (Pro Football Reference), NOT the `player_id` (GSIS ID) used in weekly data. The correct merge path uses `nfl.import_ids()` as a bridge table:

```python
# Bridge table: maps between ID systems
ids = nfl.import_ids()
# Columns include: gsis_id, pfr_id, espn_id, name, ...

# Create mapping: pfr_id -> gsis_id
pfr_to_gsis = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()

# Merge snap_counts with bridge table first
snap_counts = snap_counts.merge(pfr_to_gsis, left_on="pfr_player_id", right_on="pfr_id", how="left")

# Now merge with weekly data on gsis_id (= player_id)
weekly = weekly.merge(
    snap_counts[["gsis_id", "season", "week", "offense_pct"]],
    left_on=["player_id", "season", "week"],
    right_on=["gsis_id", "season", "week"],
    how="left"
)
weekly.rename(columns={"offense_pct": "snap_pct"}, inplace=True)
```

**Fallback for unmatched rows** (~2-5% typically): fill snap_pct with position-week median from successfully matched rows.

### 2.1.3 Data Filtering Requirements

```python
# Filter to regular season only — nflverse includes playoff weeks
df = df[df["season_type"] == "REG"]

# Validate fantasy points computation against nflverse
if "fantasy_points_ppr" in df.columns:
    our_pts = compute_fantasy_points(df)
    discrepancy = (df["fantasy_points_ppr"] - our_pts).abs()
    n_mismatch = (discrepancy > 0.5).sum()
    if n_mismatch > 0:
        print(f"WARNING: {n_mismatch} rows differ from nflverse PPR points by > 0.5")
        # Common cause: 2pt conversions not included in our scoring dict
```

### 2.2 RB Filtering

```python
# In RB/rb_data.py

def filter_to_rb(df: pd.DataFrame) -> pd.DataFrame:
    """Filter featured DataFrame to RB rows only.

    Must be called AFTER build_features() and AFTER temporal_split()
    so all team-level and opponent-level features are correctly computed
    from the full-position dataset.
    """
    rb_df = df[df["position"] == "RB"].copy()

    # Drop position encoding columns (all RB, no variance)
    pos_cols = ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
    rb_df.drop(columns=[c for c in pos_cols if c in rb_df.columns], inplace=True)

    return rb_df
```

### 2.3 Expected Dataset Sizes

Based on general estimates (~30-35K total rows after min-games filter, RBs ~25-28% of skill position players):

| Split | General (all positions) | RB-only (estimated) |
|-------|------------------------|---------------------|
| Train (2012-2023) | ~12 seasons of data | Expanded with 2012+ data |
| Val (2024) | ~1 season | ~1 season |
| Test (2025) | ~1 season | ~1 season |

### 2.4 Pipeline Integration Point

The RB pipeline hooks into the general pipeline at step 5 (after `fill_nans_safe()`):

```
General pipeline steps 1-5 (load -> preprocess -> features -> split -> NaN fill)
    |
    v
filter_to_rb(train_df), filter_to_rb(val_df), filter_to_rb(test_df)
    |
    v
compute_rb_targets(rb_train_df), compute_rb_targets(rb_val_df), compute_rb_targets(rb_test_df)
    |
    v
add_rb_specific_features(rb_train_df, rb_val_df, rb_test_df)
    |
    v
RB-specific model training and evaluation
```

---

## 3. Target Variable Construction

### 3.1 The Four Components

Each RB game-week row has four target components derived from raw stats:

| Target | Formula | Scoring Logic | Nature |
|--------|---------|---------------|--------|
| `rushing_floor` | `rushing_yards * 0.1` | Yards-only rushing production (same across all formats) | Low variance, volume-driven |
| `receiving_floor` | `receptions * ppr_weight + receiving_yards * 0.1` | Reception value varies by format (0/0.5/1.0) | Medium variance, role-dependent |
| `td_points` | `rushing_tds * 6 + receiving_tds * 6` | Touchdown scoring plays only (same across all formats). 2pt conversions excluded to align with SCORING_PPR. | High variance, hard to predict |
| `fumble_penalty` | `(sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost) * -2` | Turnover penalty (same across all formats) | Very low frequency, negative |

**Multi-format receiving floor:** The receiving_floor is the only component that varies across scoring formats. The system computes:
- `receiving_floor_standard`: `receiving_yards * 0.1` (no reception value)
- `receiving_floor_half_ppr`: `receptions * 0.5 + receiving_yards * 0.1`
- `receiving_floor`: `receptions * 1.0 + receiving_yards * 0.1` (full PPR, default)

**Total fantasy points** = rushing_floor + receiving_floor + td_points + fumble_penalty

**Scoring completeness note:** The `td_points` target intentionally excludes 2-point conversions to align with the `SCORING_PPR` dict used by `compute_fantasy_points()`. While `rushing_2pt_conversions` and `receiving_2pt_conversions` are available in `import_weekly_data()`, they are rare (~0.5% of RB game-weeks) and their omission is consistent with the scoring system. The small discrepancy vs nflverse's `fantasy_points_ppr` is validated by the sanity check in `compute_rb_targets()`.

**`special_teams_tds` exclusion:** Some RBs score on kick/punt returns. Most standard leagues do NOT award these to the individual player (they go to D/ST scoring). We exclude them from our target to match standard rules. If league rules differ, add `special_teams_tds * 6` to td_points.

### 3.2 Implementation

```python
# In RB/rb_targets.py

from src.config import PPR_FORMATS

def compute_rb_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 prediction targets + fumble penalty for RB rows.

    Computes targets for all scoring formats (standard, half_ppr, ppr).
    The receiving_floor varies by format; rushing_floor and td_points are the same.

    Returns:
        df with added columns per format:
          - rushing_floor (same across formats)
          - receiving_floor_standard, receiving_floor_half_ppr, receiving_floor
          - td_points (same across formats)
          - fumble_penalty (same across formats)
          - fantasy_points_check
    """
    df = df.copy()

    df["rushing_floor"] = df["rushing_yards"].fillna(0) * 0.1

    # Receiving floor varies by PPR format (reception weight differs)
    rec_yards_pts = df["receiving_yards"].fillna(0) * 0.1
    receptions = df["receptions"].fillna(0)
    for fmt, weight in PPR_FORMATS.items():
        suffix = "" if fmt == "ppr" else f"_{fmt}"
        df[f"receiving_floor{suffix}"] = receptions * weight + rec_yards_pts

    df["td_points"] = (
        df["rushing_tds"].fillna(0) * 6 +
        df["receiving_tds"].fillna(0) * 6 +
        df["rushing_2pt_conversions"].fillna(0) * 2 +
        df["receiving_2pt_conversions"].fillna(0) * 2
    )

    df["fumble_penalty"] = (
        df["sack_fumbles_lost"].fillna(0) +
        df["rushing_fumbles_lost"].fillna(0) +
        df["receiving_fumbles_lost"].fillna(0)
    ) * -2

    # Sanity check: sum should match fantasy_points minus passing component (full PPR)
    df["fantasy_points_check"] = (
        df["rushing_floor"] + df["receiving_floor"] +
        df["td_points"] + df["fumble_penalty"]
    )
    ...
    return df
```

### 3.3 Fumble Handling Decision

Fumbles are extremely low-frequency events for RBs (most games have 0 fumbles lost). Three options were considered:

- **Option A**: Include fumble_penalty as a 4th prediction head — Rejected: too sparse for meaningful prediction
- **Option B**: Fold fumble_penalty into rushing_floor (since most fumbles happen on rushing plays) — Rejected: contaminates the clean rushing yards signal
- **Option C (CHOSEN)**: Use historical fumble rate as a constant adjustment. Compute each player's trailing fumble rate, multiply by -2, and subtract from the final prediction as a post-hoc correction.

```python
def compute_fumble_adjustment(df: pd.DataFrame) -> pd.Series:
    """Compute per-player historical fumble rate for post-prediction adjustment.

    Uses the rolling mean of total fumbles over L8 window (shifted, no leakage).
    This gives a stable estimate of each player's fumble propensity.
    """
    total_fumbles = df["sack_fumbles_lost"].fillna(0) + df["rushing_fumbles_lost"].fillna(0)
    fumble_rate = df.groupby(["player_id", "season"])[
        total_fumbles.name if hasattr(total_fumbles, 'name') else "total_fumbles"
    ].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())

    return fumble_rate * -2  # Convert to fantasy point penalty
```

In practice, the fumble adjustment is small (typically -0.1 to -0.3 points per game) and can be approximated as a constant per-player offset derived from training data. The implementation will compute `rolling_mean_fumble_penalty_L8` as part of RB-specific features and add it to the sum of the 3 heads.

### 3.4 Expected Target Distributions (Approximate)

| Target | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| rushing_floor | ~4.5 | ~3.8 | ~3.5 | 0.0 | 25+ |
| receiving_floor | ~3.0 | ~3.5 | ~1.8 | 0.0 | 25+ |
| td_points | ~2.0 | ~3.5 | 0.0 | 0.0 | 24+ |
| fumble_penalty | ~-0.15 | ~0.55 | 0.0 | -4.0 | 0.0 |
| **total fantasy_points** | **~9.5** | **~7.5** | **~7.5** | **0.0** | **~50** |

Note: td_points has a median of 0 because many RB games produce 0 TDs. This zero-inflated distribution is the core prediction challenge.

---

## 4. Feature Engineering

### 4.1 General Features: Inherited and Pruned (~114 features)

The RB model inherits general features from `build_features()`, then **prunes position-irrelevant features** to improve signal-to-noise ratio. With only ~7K training samples, every noise feature hurts generalization.

#### Feature Selection: Whitelist Approach

The RB model uses an explicit **whitelist** (`RB_INCLUDE_FEATURES` in `rb_config.py`) rather than a blacklist. New columns must be opted in, preventing silent leakage. Key pruning decisions:

- **L5 rolling windows dropped**: >0.97 correlation with L3/L8 combinations. Only L3 and L8 retained (except `rolling_min_fantasy_points_L5`).
- **All EWMA features dropped**: >0.98 correlation with rolling means.
- **Passing-specific features dropped**: `passing_yards` and `attempts` rolling/prior-season features are near-zero for RBs.
- **Position encoding dropped**: All rows are RB — zero variance.
- **Defense stats added**: `opp_def_sacks_L5`, `opp_def_pass_yds_allowed_L5`, etc. (6 features).
- **Weather/Vegas features added**: `implied_team_total`, `implied_opp_total`, `is_dome`, `rest_advantage`.
- **Contextual features expanded**: `practice_status`, `game_status`, `depth_chart_rank` added alongside original contextual features.

#### Features RETAINED (~114 features)

| Category | Count | Key Examples |
|----------|-------|---------|
| Rolling mean/std/max (8 RB-relevant stats) | 72 | `rolling_mean_fantasy_points_L3`, `rolling_std_carries_L5`, `rolling_max_rushing_yards_L8` |
| Rolling min (fantasy_points only) | 3 | `rolling_min_fantasy_points_L3/L5/L8` |
| Prior-season summaries (8 stats) | 24 | `prior_season_mean_rushing_yards`, `prior_season_std_targets` |
| EWMA (6 RB-relevant stats) | 12 | `ewma_fantasy_points_L3`, `ewma_carries_L5`, `ewma_rushing_yards_L3` |
| Trend/momentum | 4 | `trend_fantasy_points`, `trend_carries`, `trend_targets`, `trend_snap_pct` |
| Share features | 6 | `target_share_L3/L5`, `carry_share_L3/L5`, `snap_pct`, `air_yards_share` |
| Matchup/opponent | 4 | `opp_fantasy_pts_allowed_to_pos`, `opp_rush_pts_allowed_to_pos` |
| Contextual | 4 | `is_home`, `week`, `is_returning_from_absence`, `days_rest` |

**Rolling stats retained (8):** `fantasy_points`, `fantasy_points_floor`, `targets`, `receptions`, `carries`, `rushing_yards`, `receiving_yards`, `snap_pct`

**Note on `air_yards_share`:** Confirmed available in `import_weekly_data()` as a pre-computed weekly column. For RBs, this captures how much of the passing game is directed at the RB. Low values are typical (RBs get short targets), but week-to-week variation is predictive of receiving workload changes.

**Note on matchup features**: `opp_rush_pts_allowed_to_pos` and `opp_recv_pts_allowed_to_pos` are especially valuable for RBs — an RB facing a defense weak against the run but strong against dump-offs gets a different (and more accurate) projection than one facing the reverse.

```python
# In RB/rb_features.py

# Features to drop from the general pipeline for RB model
RB_DROP_FEATURES = []

# Passing-specific rolling features (noise for RBs)
for stat in ["passing_yards", "attempts"]:
    for window in [3, 5, 8]:
        for agg in ["mean", "std", "max"]:
            RB_DROP_FEATURES.append(f"rolling_{agg}_{stat}_L{window}")

# Passing-specific EWMA features
for span in [3, 5]:
    RB_DROP_FEATURES.append(f"ewma_passing_yards_L{span}")

# Passing-specific prior-season features
for stat in ["passing_yards", "attempts"]:
    for agg in ["mean", "std", "max"]:
        RB_DROP_FEATURES.append(f"prior_season_{agg}_{stat}")

# Position encoding (zero variance for RB-only data)
RB_DROP_FEATURES += ["pos_QB", "pos_RB", "pos_WR", "pos_TE"]
```

### 4.2 RB-Specific Features (15 features)

These features are computed AFTER the general `build_features()` and AFTER filtering to RB rows. They exploit RB-specific efficiency metrics, workload context, and nflverse columns not used by the general pipeline.

All features use L3 (3-week) rolling windows for responsiveness to recent form. All use `.shift(1)` within `(player_id, season)` groups to prevent leakage.

**Current feature list** (from `rb_config.py:RB_SPECIFIC_FEATURES`):
1. `yards_per_carry_L3` — rushing efficiency
2. `reception_rate_L3` — catch rate
3. `weighted_opportunities_L3` — PPR-adjusted volume (carries + 2×targets)
4. `team_rb_carry_share_L3` — bellcow vs committee signal
5. `team_rb_target_share_L3` — pass-catching back identification
6. `rushing_epa_per_attempt_L3` — context-adjusted rushing efficiency
7. `rushing_first_down_rate_L3` — rushing chain-moving ability
8. `receiving_first_down_rate_L3` — receiving chain-moving ability
9. `yac_per_reception_L3` — post-catch creation
10. `receiving_epa_per_target_L3` — context-adjusted receiving value
11. `air_yards_per_target_L3` — average depth of target
12. `career_carries` — experience/durability proxy
13. `team_rb_carry_hhi_L3` — backfield carry concentration (Herfindahl index)
14. `team_rb_target_hhi_L3` — backfield target concentration
15. `opportunity_index_L3` — composite opportunity metric

**Design principles for feature selection:**
1. **Incremental signal**: Each feature must capture something the inherited features do NOT. Ratio features (efficiency) are incremental because the inherited features only have raw volume rolling stats, not their ratios.
2. **Data grounded**: Only use columns confirmed available in `import_weekly_data()` (see §2.1.1).
3. **Stable denominators**: Ratio features require sufficient denominator volume. L3 windows aggregate 3 games, reducing single-game noise in low-volume stats (e.g., targets for early-down backs).
4. **Individual player projection focus**: Features prioritize predicting a specific player's next-week output over league-wide ranking. This means efficiency and usage share features matter more than raw counting stats (which are already captured in inherited rolling features).

---

#### Feature 1: `yards_per_carry_L3`

**Rationale**: Captures rushing efficiency independent of volume. The inherited features have `rolling_mean_rushing_yards_L3` and `rolling_mean_carries_L3` separately, but their *ratio* is not captured. A back averaging 5.0 YPC on 15 carries projects very differently from one averaging 3.2 YPC on 15 carries — same volume, different efficiency.

**Formula**: `rolling_sum(rushing_yards, L3) / rolling_sum(carries, L3)` (both shifted)

**Why ratio of sums, not mean of ratios**: `sum(yards)/sum(carries)` correctly weights high-volume games. Mean of per-game YPC would let a 2-carry, 30-yard game (15.0 YPC) dominate.

```python
rush_yds_roll = df.groupby(["player_id", "season"])["rushing_yards"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
carries_roll = df.groupby(["player_id", "season"])["carries"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["yards_per_carry_L3"] = (rush_yds_roll / carries_roll).fillna(0)
df.loc[carries_roll == 0, "yards_per_carry_L3"] = 0
```

---

#### Feature 2: `reception_rate_L3`

**Rationale**: Catch rate distinguishes reliable pass-catching backs from early-down grinders. A back with an 85% catch rate converts targets to PPR points more efficiently than one at 60%. This is especially important for the `receiving_floor` sub-target, where each reception = 1 PPR point.

**Formula**: `rolling_sum(receptions, L3) / rolling_sum(targets, L3)` (both shifted)

```python
rec_roll = df.groupby(["player_id", "season"])["receptions"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
tgt_roll = df.groupby(["player_id", "season"])["targets"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["reception_rate_L3"] = (rec_roll / tgt_roll).fillna(0)
df.loc[tgt_roll == 0, "reception_rate_L3"] = 0
```

---

#### Feature 3: `weighted_opportunities_L3`

**Rationale**: In PPR scoring, a target is worth more than a carry because a reception alone = 1 point. `carries + 2×targets` is a well-established fantasy analytics metric (popularized by analysts like JJ Zachariason) that captures PPR-adjusted volume. This is NOT captured by inherited features which track carries and targets separately without the PPR weighting.

**Formula**: `rolling_mean(carries + 2*targets, L3)` (shifted)

```python
df["_raw_weighted_opps"] = df["carries"].fillna(0) + 2 * df["targets"].fillna(0)
df["weighted_opportunities_L3"] = df.groupby(["player_id", "season"])[
    "_raw_weighted_opps"
].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df.drop(columns=["_raw_weighted_opps"], inplace=True)
```

---

#### Feature 4: `team_rb_carry_share_L3`

**Rationale**: Detects committee vs. bellcow situations. A back with 70% of his team's RB carries is a bellcow; one with 30% is in a committee. This differs from the inherited `carry_share_L{w}` which divides by ALL team carries (including QB scrambles, WR end-arounds); this divides by RB-only carries for a cleaner position-specific signal.

**Formula**: `rolling_sum(player_carries, L3) / rolling_sum(team_rb_carries, L3)` (both shifted)

**Important**: `team_rb_carries` must be computed BEFORE the min-games filter removes low-snap RBs.

```python
# Step 1: Compute team-level RB totals from the full RB dataset (pre min-games filter)
def compute_team_rb_totals(full_rb_df: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level RB totals for share features.

    Args:
        full_rb_df: All RB rows (before min-games filter), from the general pipeline.
    """
    team_rb_totals = full_rb_df.groupby(["recent_team", "season", "week"]).agg(
        team_rb_carries=("carries", "sum"),
        team_rb_targets=("targets", "sum"),
    ).reset_index()
    return team_rb_totals

# Step 2: Merge team totals back and compute rolling share
player_carries_roll = df.groupby(["player_id", "season"])["carries"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
team_rb_carries_roll = df.groupby(["player_id", "season"])["team_rb_carries"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["team_rb_carry_share_L3"] = (player_carries_roll / team_rb_carries_roll).fillna(0)
df.loc[team_rb_carries_roll == 0, "team_rb_carry_share_L3"] = 0
```

---

#### Feature 5: `team_rb_target_share_L3`

**Rationale**: Identifies the primary pass-catching back on each team. A back commanding 50%+ of team RB targets is the PPR-valuable receiving back. This is especially important because in PPR, the difference between a 3-target-per-game and a 6-target-per-game back is massive (~3+ expected PPR points per game from receptions alone).

**Formula**: `rolling_sum(player_targets, L3) / rolling_sum(team_rb_targets, L3)` (both shifted)

```python
player_targets_roll = df.groupby(["player_id", "season"])["targets"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
team_rb_targets_roll = df.groupby(["player_id", "season"])["team_rb_targets"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["team_rb_target_share_L3"] = (player_targets_roll / team_rb_targets_roll).fillna(0)
df.loc[team_rb_targets_roll == 0, "team_rb_target_share_L3"] = 0
```

---

#### Feature 6: `rushing_epa_per_attempt_L3` *(NEW — replaces `yards_per_touch_L3`)*

**Rationale**: Expected Points Added per rush attempt is a strictly better efficiency metric than YPC. EPA captures game context that raw yardage cannot: 3 yards on 3rd-and-2 (first down, ~+1.5 EPA) is far more valuable than 3 yards on 1st-and-10 (~-0.2 EPA). A back with positive EPA/attempt is creating value above expectation. This is available directly from `import_weekly_data()` as `rushing_epa` (total EPA on all rush plays).

**Formula**: `rolling_sum(rushing_epa, L3) / rolling_sum(carries, L3)` (both shifted)

**Why this replaces `yards_per_touch_L3`**: The removed feature was a composite of YPC and receiving efficiency — redundant with Feature 1 (YPC) and the inherited receiving_yards rolling features. EPA/attempt adds genuinely orthogonal signal by capturing play context.

**Correlation with `yards_per_carry_L3`**: Expected ~0.5-0.7 correlation. The residual captures down-and-distance efficiency, game-script leverage, and field-position value that YPC misses.

```python
rushing_epa_roll = df.groupby(["player_id", "season"])["rushing_epa"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
carries_roll = df.groupby(["player_id", "season"])["carries"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["rushing_epa_per_attempt_L3"] = (rushing_epa_roll / carries_roll).fillna(0)
df.loc[carries_roll == 0, "rushing_epa_per_attempt_L3"] = 0
```

---

#### Feature 7: `first_down_rate_L3` *(NEW)*

**Rationale**: First down rate = chain-moving ability = staying on the field. An RB who converts first downs sustains drives, which leads to more snaps, more touches, and more scoring opportunities in the same game. `rushing_first_downs` and `receiving_first_downs` are directly available in `import_weekly_data()`. This captures drive-sustaining value that yardage alone misses — a back who grinds 4-yard runs on 3rd-and-3 is more valuable to sustained drives (and therefore to continued usage) than one who rips 8-yard runs on 1st-and-10 but can't convert.

**Formula**: `rolling_sum(rushing_first_downs + receiving_first_downs, L3) / rolling_sum(carries + receptions, L3)` (both shifted)

```python
df["_total_first_downs"] = (
    df["rushing_first_downs"].fillna(0) + df["receiving_first_downs"].fillna(0)
)
df["_total_touches"] = df["carries"].fillna(0) + df["receptions"].fillna(0)

first_downs_roll = df.groupby(["player_id", "season"])["_total_first_downs"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
touches_roll = df.groupby(["player_id", "season"])["_total_touches"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["first_down_rate_L3"] = (first_downs_roll / touches_roll).fillna(0)
df.loc[touches_roll == 0, "first_down_rate_L3"] = 0

df.drop(columns=["_total_first_downs", "_total_touches"], inplace=True)
```

---

#### Feature 8: `yac_per_reception_L3` *(NEW)*

**Rationale**: Yards After Catch per reception isolates an RB's ability to create yardage after the catch — the dominant component of RB receiving production. Most RB targets are short (screens, checkdowns, dump-offs with aDOT < 3 yards), so the difference between a productive and unproductive receiving back is almost entirely post-catch creation. `receiving_yards_after_catch` is directly available in `import_weekly_data()`.

**Why this matters for projection**: Two backs with identical target volumes and catch rates can have wildly different receiving_floor projections based on YAC ability. A back averaging 8 YAC/reception (e.g., Austin Ekeler, Alvin Kamara) produces ~0.8 more receiving_floor points per catch than one averaging 3 YAC/reception.

**Formula**: `rolling_sum(receiving_yards_after_catch, L3) / rolling_sum(receptions, L3)` (both shifted)

```python
yac_roll = df.groupby(["player_id", "season"])["receiving_yards_after_catch"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
rec_roll = df.groupby(["player_id", "season"])["receptions"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).sum()
)
df["yac_per_reception_L3"] = (yac_roll / rec_roll).fillna(0)
df.loc[rec_roll == 0, "yac_per_reception_L3"] = 0
```

---

#### Feature Redundancy Analysis

| Feature | Correlated Inherited Features | Why It's Still Incremental |
|---------|------------------------------|---------------------------|
| `yards_per_carry_L3` | `rolling_mean_rushing_yards`, `rolling_mean_carries` | Ratio captures efficiency; volume features don't |
| `reception_rate_L3` | `rolling_mean_receptions`, `rolling_mean_targets` | Catch rate is a player skill; raw counts are opportunity |
| `weighted_opportunities_L3` | `rolling_mean_carries`, `rolling_mean_targets` | PPR-adjusted weighting (2×targets) changes the signal |
| `team_rb_carry_share_L3` | `carry_share_L3` (general) | RB-only denominator vs all-position team denominator |
| `team_rb_target_share_L3` | `target_share_L3` (general) | RB-only denominator vs all-position team denominator |
| `rushing_epa_per_attempt_L3` | `yards_per_carry_L3` (~0.6 corr) | Context-adjusted: captures down/distance/field-position value |
| `first_down_rate_L3` | `rolling_mean_rushing_yards` (~0.4 corr) | Conversion-specific signal; yards don't distinguish 1st-down vs not |
| `yac_per_reception_L3` | `rolling_mean_receiving_yards` (~0.5 corr) | Post-catch creation; total rec yards includes air yards component |

---

### 4.3 RB Feature Column Registry

```python
# In RB/rb_config.py

RB_SPECIFIC_FEATURES = [
    "yards_per_carry_L3",
    "reception_rate_L3",
    "weighted_opportunities_L3",
    "team_rb_carry_share_L3",
    "team_rb_target_share_L3",
    "rushing_epa_per_attempt_L3",
    "rushing_first_down_rate_L3",
    "receiving_first_down_rate_L3",
    "yac_per_reception_L3",
    "receiving_epa_per_target_L3",
    "air_yards_per_target_L3",
    "career_carries",
    "team_rb_carry_hhi_L3",
    "team_rb_target_hhi_L3",
    "opportunity_index_L3",
]

# The RB model uses an explicit WHITELIST (RB_INCLUDE_FEATURES) rather than
# a blacklist. Features are grouped by category: rolling, prior_season, ewma,
# trend, share, matchup, defense, contextual, weather_vegas, specific.
# New columns must be explicitly added to the whitelist to prevent silent leakage.
```

### 4.4 Total Feature Count

The RB model uses an explicit whitelist (`RB_INCLUDE_FEATURES` in `rb_config.py`) that groups features by category:

| Category | Count | Notes |
|----------|-------|-------|
| Rolling (L3/L8 only, L5 dropped) | ~53 | 8 stats × 2 windows × 3 aggs + min for fp |
| Prior-season summaries | 24 | 8 RB-relevant stats × 3 aggs |
| EWMA | 0 | All dropped (>0.98 corr with rolling means) |
| Trend/momentum | 4 | fantasy_points, targets, carries, snap_pct |
| Share features | 6 | target_share, carry_share (L3/L5), snap_pct, air_yards_share |
| Matchup features | 4 | Standard opponent matchup features |
| Defense stats | 6 | opp_def_sacks_L5, rush/pass yds allowed, etc. |
| Contextual | 7 | is_home, week, absence, days_rest, practice/game status, depth chart |
| Weather/Vegas | 4 | implied_team_total, implied_opp_total, is_dome, rest_advantage |
| RB-specific features | 15 | Efficiency ratios, HHI, career carries, etc. |
| **Total** | **~123 features** | |

**Key changes from original design:**
- L5 rolling windows removed (>0.97 corr with L3/L8)
- All EWMA features removed (>0.98 corr with rolling means)
- Defense stats, weather/Vegas, and expanded contextual features added
- RB-specific features expanded from 8 to 15 (added HHI, career carries, split first-down rates, receiving EPA, air yards per target)

### 4.5 NaN Handling for RB-Specific Features

RB-specific features are ratio-based and may produce NaN or inf from division by zero. All are guarded with `.fillna(0)` and explicit zero-denominator checks at computation time. A separate RB-specific NaN fill handles the 8 new columns after train/val/test split:

```python
def fill_rb_nans(train_df, val_df, test_df, rb_feature_cols):
    """Fill NaNs in RB-specific feature columns using training set statistics.

    Called AFTER temporal_split() and AFTER add_rb_specific_features().
    Uses ONLY training set statistics to prevent leakage.
    """
    # Replace any inf values with NaN first (from div-by-zero edge cases)
    for split_df in [train_df, val_df, test_df]:
        split_df[rb_feature_cols] = split_df[rb_feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    # Compute training set means for RB-specific features
    train_means = train_df[rb_feature_cols].mean()

    for split_df in [train_df, val_df, test_df]:
        for col in rb_feature_cols:
            split_df[col] = split_df[col].fillna(train_means[col])

    return train_df, val_df, test_df
```

**NaN sources for RB-specific features:**
| Feature | NaN When | Fill Value |
|---------|----------|------------|
| `yards_per_carry_L3` | 0 carries in L3 window | Train mean (~4.2) |
| `reception_rate_L3` | 0 targets in L3 window | Train mean (~0.75) |
| `weighted_opportunities_L3` | Week 1, first season appearance | Train mean (~18) |
| `team_rb_carry_share_L3` | 0 team RB carries (bye-adjacent) | Train mean (~0.45) |
| `team_rb_target_share_L3` | 0 team RB targets | Train mean (~0.35) |
| `rushing_epa_per_attempt_L3` | 0 carries in L3 window | Train mean (~0.0) |
| `first_down_rate_L3` | 0 touches in L3 window | Train mean (~0.22) |
| `yac_per_reception_L3` | 0 receptions in L3 window | Train mean (~5.5) |

---

## 5. Model Architecture

### 5.1 Ridge Multi-Target (Baseline)

Three separate Ridge regression models, one per target. Each has its own StandardScaler and Ridge instance.

```python
# In RB/rb_models.py

from src.models.linear import RidgeModel

class RBRidgeMultiTarget:
    """Three separate Ridge models predicting rushing_floor, receiving_floor, td_points."""

    def __init__(self, alpha: float = 1.0):
        self.rushing_model = RidgeModel(alpha=alpha)
        self.receiving_model = RidgeModel(alpha=alpha)
        self.td_model = RidgeModel(alpha=alpha)
        self.target_names = ["rushing_floor", "receiving_floor", "td_points"]

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        """
        Args:
            X_train: Feature array, shape (n_samples, n_features)
            y_train_dict: {
                "rushing_floor": np.ndarray shape (n_samples,),
                "receiving_floor": np.ndarray shape (n_samples,),
                "td_points": np.ndarray shape (n_samples,),
            }
        """
        self.rushing_model.fit(X_train, y_train_dict["rushing_floor"])
        self.receiving_model.fit(X_train, y_train_dict["receiving_floor"])
        self.td_model.fit(X_train, y_train_dict["td_points"])

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict of per-target predictions plus total."""
        preds = {
            "rushing_floor": self.rushing_model.predict(X),
            "receiving_floor": self.receiving_model.predict(X),
            "td_points": self.td_model.predict(X),
        }
        preds["total"] = preds["rushing_floor"] + preds["receiving_floor"] + preds["td_points"]
        return preds

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        """Convenience: returns just the total fantasy points prediction."""
        preds = self.predict(X)
        return preds["total"]

    def get_feature_importance(self, feature_names: list) -> dict:
        """Per-target feature importance from Ridge coefficients."""
        return {
            "rushing_floor": self.rushing_model.get_feature_importance(feature_names),
            "receiving_floor": self.receiving_model.get_feature_importance(feature_names),
            "td_points": self.td_model.get_feature_importance(feature_names),
        }

    def save(self, model_dir: str = "RB/outputs/models") -> None:
        self.rushing_model.save(f"{model_dir}/rushing")
        self.receiving_model.save(f"{model_dir}/receiving")
        self.td_model.save(f"{model_dir}/td")

    def load(self, model_dir: str = "RB/outputs/models") -> None:
        self.rushing_model.load(f"{model_dir}/rushing")
        self.receiving_model.load(f"{model_dir}/receiving")
        self.td_model.load(f"{model_dir}/td")
```

**Scaler note**: Each sub-model has its own StandardScaler internally. Since all three use the same feature matrix `X`, the scalers will be identical. This is slightly redundant but keeps the interface clean and reuses the existing `RidgeModel` class without modification.

### 5.2 Multi-Head Neural Network

```python
# In RB/rb_neural_net.py

import torch
import torch.nn as nn
import numpy as np

class RBMultiHeadNet(nn.Module):
    """Multi-head neural network for RB fantasy point decomposition.

    Architecture:
        Input (N features)
            -> Shared backbone [Linear(N, 128) -> BN -> ReLU -> Dropout(0.15)]
            -> Shared backbone [Linear(128, 64) -> BN -> ReLU -> Dropout(0.15)]
            -> Head 1 (rushing_floor):  Linear(64, 48) -> ReLU -> Linear(48, 1) -> clamp(min=0)
            -> Head 2 (receiving_floor): Linear(64, 48) -> ReLU -> Linear(48, 1) -> clamp(min=0)
            -> Head 3 (td_points):      Linear(64, 64) -> ReLU -> Linear(64, 1) -> clamp(min=0)

    Total prediction = head1 + head2 + head3 + fumble_adjustment
    """

    def __init__(
        self,
        input_dim: int,
        backbone_layers: list = [128, 64],
        head_hidden: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # === Shared Backbone ===
        backbone_blocks = []
        prev_dim = input_dim
        for hidden_dim in backbone_layers:
            backbone_blocks.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_blocks)

        backbone_out_dim = backbone_layers[-1]  # 64

        # === Output Heads ===
        # Each head: Linear -> ReLU -> Linear -> squeeze
        self.rushing_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.receiving_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self.td_head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor, shape (batch_size, input_dim)

        Returns:
            Dictionary with keys:
                "rushing_floor": shape (batch_size,)
                "receiving_floor": shape (batch_size,)
                "td_points": shape (batch_size,)
                "total": shape (batch_size,) — sum of the 3 heads
        """
        shared = self.backbone(x)  # (batch_size, 64)

        rushing = self.rushing_head(shared).squeeze(-1)      # (batch_size,)
        receiving = self.receiving_head(shared).squeeze(-1)   # (batch_size,)
        td = self.td_head(shared).squeeze(-1)                 # (batch_size,)

        return {
            "rushing_floor": rushing,
            "receiving_floor": receiving,
            "td_points": td,
            "total": rushing + receiving + td,
        }

    def predict_numpy(self, X: np.ndarray, device: torch.device) -> dict:
        """Convenience method for inference from numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            preds = self.forward(x_tensor)
            return {k: v.cpu().numpy() for k, v in preds.items()}
```

### 5.3 Architecture Diagram

```
Input (batch_size, N)
    |
    v
[Linear(N, 128)] -> [BatchNorm1d(128)] -> [ReLU] -> [Dropout(0.15)]
    |
    v
[Linear(128, 64)] -> [BatchNorm1d(64)] -> [ReLU] -> [Dropout(0.15)]
    |
    +---> Shared representation (batch_size, 64)
    |
    |--- Head 1 (rushing_floor) ---+
    |   [Linear(64, 48)] -> [ReLU] -> [Linear(48, 1)] -> clamp(min=0) -> (batch_size,)
    |
    |--- Head 2 (receiving_floor) ---+
    |   [Linear(64, 48)] -> [ReLU] -> [Linear(48, 1)] -> clamp(min=0) -> (batch_size,)
    |
    |--- Head 3 (td_points) ---+
        [Linear(64, 64)] -> [ReLU] -> [Linear(64, 1)] -> clamp(min=0) -> (batch_size,)

Total = Head1 + Head2 + Head3    (shape: batch_size,)
```

### 5.4 Parameter Count Estimate

| Component | Parameters |
|-----------|-----------|
| Backbone Linear(122, 128) + bias | 122*128 + 128 = 15,744 |
| Backbone BN(128) | 256 (gamma + beta) |
| Backbone Linear(128, 64) + bias | 128*64 + 64 = 8,256 |
| Backbone BN(64) | 128 |
| Head 1: Linear(64, 32) + Linear(32, 1) | 64*32+32 + 32*1+1 = 2,113 |
| Head 2: same | 2,113 |
| Head 3: same | 2,113 |
| **Total** | **~30,700** |

This is a modest network suitable for ~7K training samples.

---

## 6. Training Pipeline

### 6.1 Multi-Target Loss Function

```python
# In RB/rb_training.py

class MultiTargetLoss(nn.Module):
    """Combined Huber loss for multi-head RB network.

    Loss = sum(weight[t] * Huber(pred[t], target[t], delta[t]))
           + w_total * Huber(total_pred, total_actual)

    Per-target Huber deltas allow different MSE-to-MAE transition thresholds.
    The auxiliary total loss keeps the sum calibrated even if individual heads drift.
    """

    def __init__(
        self,
        target_names: list[str],
        loss_weights: dict[str, float],
        huber_deltas: dict[str, float] = None,
        w_total: float = 0.25,
    ):
        super().__init__()
        self.target_names = target_names
        self.loss_weights = loss_weights
        self.huber_deltas = huber_deltas or {}
        self.w_total = w_total

    def forward(self, preds: dict, targets: dict) -> tuple:
        """
        Args:
            preds: from RBMultiHeadNet.forward()
                keys: rushing_floor, receiving_floor, td_points, total
            targets: same keys, each shape (batch_size,)

        Returns:
            (total_loss, loss_components_dict)
        """
        loss_rushing = self.mse(preds["rushing_floor"], targets["rushing_floor"])
        loss_receiving = self.mse(preds["receiving_floor"], targets["receiving_floor"])
        loss_td = self.mse(preds["td_points"], targets["td_points"])
        loss_total = self.mse(preds["total"], targets["total"])

        combined = (
            self.w_rushing * loss_rushing +
            self.w_receiving * loss_receiving +
            self.w_td * loss_td +
            self.w_total * loss_total
        )

        components = {
            "loss_rushing": loss_rushing.item(),
            "loss_receiving": loss_receiving.item(),
            "loss_td": loss_td.item(),
            "loss_total_aux": loss_total.item(),
            "loss_combined": combined.item(),
        }

        return combined, components
```

### 6.2 Multi-Target DataLoader

```python
# In RB/rb_training.py

class RBMultiTargetDataset(torch.utils.data.Dataset):
    """Dataset that returns features + dict of targets."""

    def __init__(self, X: np.ndarray, y_dict: dict):
        """
        Args:
            X: Feature array, shape (n_samples, n_features)
            y_dict: {
                "rushing_floor": shape (n_samples,),
                "receiving_floor": shape (n_samples,),
                "td_points": shape (n_samples,),
                "total": shape (n_samples,),  # = sum of above + fumble_penalty
            }
        """
        self.X = torch.FloatTensor(X)
        self.targets = {k: torch.FloatTensor(v) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {k: v[idx] for k, v in self.targets.items()}
        return x, y


def make_rb_dataloaders(X_train, y_train_dict, X_val, y_val_dict, batch_size=256):
    """Create DataLoaders for multi-target RB training.

    Args:
        X_train, X_val: Pre-scaled numpy feature arrays
        y_train_dict, y_val_dict: Target dicts with keys
            rushing_floor, receiving_floor, td_points, total
        batch_size: Batch size (default 256)
    """
    train_ds = RBMultiTargetDataset(X_train, y_train_dict)
    val_ds = RBMultiTargetDataset(X_val, y_val_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader
```

### 6.3 Multi-Head Trainer

```python
# In RB/rb_training.py

class RBMultiHeadTrainer:
    """Training loop for multi-head RB network.

    Extends the general Trainer pattern but handles multi-target loss
    and per-head metric logging.
    """

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
            "train_loss": [float, ...],
            "val_loss": [float, ...],
            "val_loss_rushing": [float, ...],
            "val_loss_receiving": [float, ...],
            "val_loss_td": [float, ...],
            "val_mae_total": [float, ...],
            "val_mae_rushing": [float, ...],
            "val_mae_receiving": [float, ...],
            "val_mae_td": [float, ...],
            "val_rmse_total": [float, ...],
        }
        """
        history = {k: [] for k in [
            "train_loss", "val_loss",
            "val_loss_rushing", "val_loss_receiving", "val_loss_td",
            "val_mae_total", "val_mae_rushing", "val_mae_receiving", "val_mae_td",
            "val_rmse_total",
        ]}

        for epoch in range(n_epochs):
            # --- Training pass ---
            self.model.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = {k: v.to(self.device) for k, v in y_batch.items()}

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss, _ = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = epoch_train_loss / n_train_batches
            history["train_loss"].append(avg_train_loss)

            # --- Validation pass ---
            self.model.eval()
            all_preds = {k: [] for k in ["rushing_floor", "receiving_floor", "td_points", "total"]}
            all_targets = {k: [] for k in ["rushing_floor", "receiving_floor", "td_points", "total"]}
            epoch_val_loss = 0.0
            val_components_accum = {}
            n_val_batches = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = {k: v.to(self.device) for k, v in y_batch.items()}

                    preds = self.model(X_batch)
                    loss, components = self.criterion(preds, y_batch)

                    epoch_val_loss += loss.item()
                    for k in components:
                        val_components_accum[k] = val_components_accum.get(k, 0) + components[k]
                    n_val_batches += 1

                    for k in all_preds:
                        all_preds[k].append(preds[k].cpu().numpy())
                        all_targets[k].append(y_batch[k].cpu().numpy())

            avg_val_loss = epoch_val_loss / n_val_batches
            history["val_loss"].append(avg_val_loss)

            # Per-target val losses
            history["val_loss_rushing"].append(
                val_components_accum.get("loss_rushing", 0) / n_val_batches
            )
            history["val_loss_receiving"].append(
                val_components_accum.get("loss_receiving", 0) / n_val_batches
            )
            history["val_loss_td"].append(
                val_components_accum.get("loss_td", 0) / n_val_batches
            )

            # Compute MAE per target
            for k in ["rushing_floor", "receiving_floor", "td_points", "total"]:
                y_pred_all = np.concatenate(all_preds[k])
                y_true_all = np.concatenate(all_targets[k])
                if k == "rushing_floor":
                    history["val_mae_rushing"].append(np.mean(np.abs(y_pred_all - y_true_all)))
                elif k == "receiving_floor":
                    history["val_mae_receiving"].append(np.mean(np.abs(y_pred_all - y_true_all)))
                elif k == "td_points":
                    history["val_mae_td"].append(np.mean(np.abs(y_pred_all - y_true_all)))
                elif k == "total":
                    history["val_mae_total"].append(np.mean(np.abs(y_pred_all - y_true_all)))
                    history["val_rmse_total"].append(
                        np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
                    )

            # --- LR Scheduler ---
            self.scheduler.step(avg_val_loss)

            # --- Early Stopping ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(self.best_model_state)
                    break

            # --- Logging ---
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {avg_val_loss:.4f} | "
                    f"MAE total: {history['val_mae_total'][-1]:.3f} | "
                    f"rush: {history['val_mae_rushing'][-1]:.3f} | "
                    f"recv: {history['val_mae_receiving'][-1]:.3f} | "
                    f"td: {history['val_mae_td'][-1]:.3f}"
                )

        return history
```

### 6.4 Hyperparameters

```python
# In RB/rb_config.py

# === RB Model Config ===
RB_TARGETS = ["rushing_floor", "receiving_floor", "td_points"]

# Ridge (per-target alpha grids, tuned via logspace search)
RB_RIDGE_ALPHA_GRIDS = {
    "rushing_floor":   np.logspace(-2, 3, 15),
    "receiving_floor": np.logspace(-2, 2.5, 20),
    "td_points":       np.logspace(-1, 4, 15),
}
RB_RIDGE_PCA_COMPONENTS = 80  # PCR: retains 99.8% variance, drops condition number

# TD model: "ridge" | "two_stage" | "ordinal" | "gated_ordinal"
RB_TD_MODEL_TYPE = "gated_ordinal"

# Neural Net
RB_NN_BACKBONE_LAYERS = [128, 64]
RB_NN_HEAD_HIDDEN = 48
RB_NN_HEAD_HIDDEN_OVERRIDES = {"td_points": 64}  # Larger head for sparse TD signal
RB_NN_DROPOUT = 0.15
RB_NN_LR = 1e-3
RB_NN_WEIGHT_DECAY = 5e-5
RB_NN_EPOCHS = 300
RB_NN_BATCH_SIZE = 256
RB_NN_PATIENCE = 30

# Loss weights
RB_LOSS_WEIGHTS = {"rushing_floor": 1.2, "receiving_floor": 1.0, "td_points": 2.0}
RB_LOSS_W_TOTAL = 0.25

# Huber deltas (per-target MSE-to-MAE transition thresholds)
RB_HUBER_DELTAS = {"rushing_floor": 2.0, "receiving_floor": 2.5, "td_points": 2.0}

# LR Scheduler (CosineWarmRestarts)
RB_SCHEDULER_TYPE = "cosine_warm_restarts"
RB_COSINE_T0 = 40
RB_COSINE_T_MULT = 2
RB_COSINE_ETA_MIN = 1e-5

# Attention NN (game history variant)
RB_TRAIN_ATTENTION_NN = True
RB_ATTN_D_MODEL = 32
RB_ATTN_N_HEADS = 2
RB_ATTN_GATED_TD = True  # Sigmoid gate P(TD>0) × Softplus E[TD|TD>0]

# LightGBM (Optuna-tuned, 50 trials, CV MAE 4.51)
RB_TRAIN_LIGHTGBM = True
```

### 6.5 Training Script Initialization

```python
# In RB/run_rb_pipeline.py (initialization section)

device = torch.device("cpu")  # No GPU required for this scale

input_dim = len(feature_cols)  # ~123

model = MultiHeadNet(
    input_dim=input_dim,
    target_names=RB_TARGETS,
    backbone_layers=RB_NN_BACKBONE_LAYERS,      # [128, 64]
    head_hidden=RB_NN_HEAD_HIDDEN,               # 48
    dropout=RB_NN_DROPOUT,                       # 0.15
    head_hidden_overrides=RB_NN_HEAD_HIDDEN_OVERRIDES,  # {"td_points": 64}
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=RB_NN_LR, weight_decay=RB_NN_WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=RB_COSINE_T0, T_mult=RB_COSINE_T_MULT, eta_min=RB_COSINE_ETA_MIN,
)

criterion = MultiTargetLoss(
    target_names=RB_TARGETS,
    loss_weights=RB_LOSS_WEIGHTS,
    huber_deltas=RB_HUBER_DELTAS,
    w_total=RB_LOSS_W_TOTAL,
)

trainer = MultiHeadTrainer(
    model=model, optimizer=optimizer, scheduler=scheduler,
    criterion=criterion, device=device, target_names=RB_TARGETS,
    patience=RB_NN_PATIENCE,
)
```

---

## 7. Evaluation Framework

### 7.1 Primary Metrics (Total Fantasy Points)

Computed on the **total** predicted fantasy points (sum of 3 heads) vs actual:

| Metric | What It Measures |
|--------|-----------------|
| MAE | Average absolute error in fantasy points (most interpretable) |
| RMSE | Penalizes large misses more than MAE |
| R-squared | Proportion of variance explained |
| Top-12 RB Hit Rate | Of the actual top-12 RBs each week, how many did the model rank in its top-12? |
| Spearman Correlation | Rank correlation between predicted and actual RB rankings per week |

### 7.2 Per-Target Diagnostic Metrics

These reveal where the model succeeds and fails:

| Target | Expected MAE (rough) | Interpretation |
|--------|---------------------|----------------|
| rushing_floor | ~2.5 pts | Easiest: driven by volume (carries x YPC) |
| receiving_floor | ~2.0 pts | Moderate: depends on target share and catch rate |
| td_points | ~3.0 pts | Hardest: TDs are inherently high-variance |

### 7.3 Implementation

```python
# In RB/rb_evaluation.py

from src.evaluation.metrics import compute_metrics

def compute_rb_metrics(y_true_dict: dict, y_pred_dict: dict) -> dict:
    """Compute per-target and total metrics for RB model.

    Args:
        y_true_dict: {"rushing_floor": ..., "receiving_floor": ..., "td_points": ..., "total": ...}
        y_pred_dict: same structure

    Returns:
        {
            "total": {"mae": float, "rmse": float, "r2": float},
            "rushing_floor": {"mae": float, "rmse": float, "r2": float},
            "receiving_floor": {"mae": float, "rmse": float, "r2": float},
            "td_points": {"mae": float, "rmse": float, "r2": float},
        }
    """
    results = {}
    for target in ["total", "rushing_floor", "receiving_floor", "td_points"]:
        results[target] = compute_metrics(y_true_dict[target], y_pred_dict[target])
    return results


def compute_rb_ranking_metrics(test_df, pred_col="pred_total", true_col="fantasy_points", top_k=12):
    """Per-week ranking quality metrics for RB model.

    Args:
        test_df: RB test DataFrame with prediction and actual columns
        pred_col: Column name for predicted total fantasy points
        true_col: Column name for actual fantasy points
        top_k: Number of top RBs to evaluate (default 12)

    Returns:
        {
            "weekly": [{"week": int, "top_k_hit_rate": float, "spearman": float}, ...],
            "season_avg_hit_rate": float,
            "season_avg_spearman": float,
        }
    """
    from scipy.stats import spearmanr

    weekly_results = []
    for week in sorted(test_df["week"].unique()):
        week_df = test_df[test_df["week"] == week]

        if len(week_df) < top_k:
            continue

        # Actual top-K
        actual_top_k = set(week_df.nlargest(top_k, true_col)["player_id"])
        # Predicted top-K
        pred_top_k = set(week_df.nlargest(top_k, pred_col)["player_id"])

        hit_rate = len(actual_top_k & pred_top_k) / top_k

        # Spearman rank correlation
        corr, _ = spearmanr(week_df[pred_col], week_df[true_col])

        weekly_results.append({
            "week": week,
            "top_k_hit_rate": hit_rate,
            "spearman": corr,
        })

    avg_hit_rate = np.mean([r["top_k_hit_rate"] for r in weekly_results])
    avg_spearman = np.mean([r["spearman"] for r in weekly_results])

    return {
        "weekly": weekly_results,
        "season_avg_hit_rate": avg_hit_rate,
        "season_avg_spearman": avg_spearman,
    }


def print_rb_comparison_table(results: dict) -> None:
    """Pretty-print comparison of all RB models.

    results = {
        "Season Average Baseline": {"total": {"mae": ..., ...}, ...},
        "RB Ridge Multi-Target": {"total": {"mae": ..., ...}, ...},
        "RB Multi-Head NN": {"total": {"mae": ..., ...}, ...},
    }
    """
    print("\n" + "=" * 80)
    print("RB Model Comparison -- Total Fantasy Points")
    print("=" * 80)
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 56)
    for model_name, metrics in results.items():
        m = metrics["total"]
        print(f"{model_name:<30} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

    print("\n" + "=" * 80)
    print("RB Model Comparison -- Per-Target MAE")
    print("=" * 80)
    print(f"{'Model':<30} {'Rush Floor':>12} {'Recv Floor':>12} {'TD Pts':>12}")
    print("-" * 68)
    for model_name, metrics in results.items():
        if "rushing_floor" in metrics:
            print(
                f"{model_name:<30} "
                f"{metrics['rushing_floor']['mae']:>12.3f} "
                f"{metrics['receiving_floor']['mae']:>12.3f} "
                f"{metrics['td_points']['mae']:>12.3f}"
            )
```

### 7.4 Weekly Simulation (RB Backtest)

```python
# In RB/rb_backtest.py

def run_rb_weekly_simulation(test_df, pred_columns, true_col="fantasy_points", top_k=12):
    """Week-by-week simulation for RB models across the 2024 test season.

    Args:
        test_df: RB-only test DataFrame
        pred_columns: {"model_name": "pred_column_name", ...}
        true_col: Actual fantasy points column
        top_k: Top-K for ranking metrics

    Returns:
        {
            "weekly_metrics": {
                model_name: [{"week": int, "mae": float, "rmse": float, "r2": float}, ...]
            },
            "weekly_ranking": {
                model_name: [{"week": int, "top_k_hit_rate": float, "spearman": float}, ...]
            },
            "season_summary": {
                model_name: {"mae": float, "rmse": float, "r2": float}
            },
        }
    """
    # Same structure as src/evaluation/backtest.py run_weekly_simulation
    # but operates on RB-only data without position filtering
    pass  # Implementation follows general backtest pattern
```

---

## 8. File Structure

### 8.1 RB Subfolder Layout

```
RB/
├── __init__.py
├── RB_IMPLEMENTATION.md         # This document
├── rb_config.py                 # RB-specific constants and hyperparameters
├── rb_data.py                   # filter_to_rb(), compute_team_rb_totals()
├── rb_targets.py                # compute_rb_targets(), compute_fumble_adjustment()
├── rb_features.py               # add_rb_specific_features(), get_rb_feature_columns(), fill_rb_nans()
├── rb_models.py                 # RBRidgeMultiTarget class
├── rb_neural_net.py             # RBMultiHeadNet (nn.Module)
├── rb_training.py               # MultiTargetLoss, RBMultiTargetDataset, RBMultiHeadTrainer
├── rb_evaluation.py             # compute_rb_metrics(), compute_rb_ranking_metrics()
├── rb_backtest.py               # run_rb_weekly_simulation()
├── run_rb_pipeline.py           # End-to-end RB pipeline script
│
├── data/                        # RB-specific cached data (gitignored)
│   └── splits/
│       ├── rb_train.parquet
│       ├── rb_val.parquet
│       └── rb_test.parquet
│
├── outputs/
│   ├── figures/
│   │   ├── rb_training_curves.png
│   │   ├── rb_per_target_loss.png
│   │   ├── rb_weekly_mae.png
│   │   ├── rb_top12_hit_rate.png
│   │   ├── rb_ridge_feature_importance.png
│   │   └── rb_pred_vs_actual_scatter.png
│   └── models/
│       ├── rushing/              # Ridge sub-model for rushing_floor
│       │   ├── scaler.pkl
│       │   └── ridge_model.pkl
│       ├── receiving/            # Ridge sub-model for receiving_floor
│       │   ├── scaler.pkl
│       │   └── ridge_model.pkl
│       ├── td/                   # Ridge sub-model for td_points
│       │   ├── scaler.pkl
│       │   └── ridge_model.pkl
│       └── rb_multihead_nn.pt   # PyTorch multi-head model state dict
│
└── notebooks/
    └── rb_analysis.ipynb         # RB-specific EDA, error analysis, feature importance
```

### 8.2 Naming Conventions

- All RB-specific Python files use `rb_` prefix
- All RB-specific output files use `rb_` prefix
- Configuration constants use `RB_` prefix
- Target columns: `rushing_floor`, `receiving_floor`, `td_points`, `fumble_penalty`
- Prediction columns: `pred_rushing_floor`, `pred_receiving_floor`, `pred_td_points`, `pred_total`
- RB-specific features: descriptive names with `_L3` suffix for rolling window

---

## 9. Integration with the General Pipeline

### 9.1 Dependency Chain

The RB pipeline depends on the general pipeline completing steps 1-5. It does NOT modify any general pipeline files.

```
scripts/run_pipeline.py                    RB/run_rb_pipeline.py
========================                   =========================
1. load_raw_data()
2. preprocess()
3. build_features()
4. temporal_split()
5. fill_nans_safe()
   |
   +-- Save featured splits to data/splits/
       |
       +---------------------------------> 6. Load featured splits from data/splits/
                                           7. filter_to_rb() on each split
                                           8. compute_rb_targets()
                                           9. add_rb_specific_features()
                                           10. fill_rb_nans()
                                           11. Train RB Ridge multi-target
                                           12. Train RB multi-head NN
                                           13. Evaluate (per-target + total)
                                           14. Run RB backtest
                                           15. Save outputs
```

### 9.2 Shared Module Usage

The RB pipeline imports from the general `src/` package:

| Import | From | Used For |
|--------|------|----------|
| `get_feature_columns()` | `src.features.engineer` | Base feature list |
| `RidgeModel` | `src.models.linear` | Reused as sub-model in RBRidgeMultiTarget |
| `compute_metrics()` | `src.evaluation.metrics` | Per-target and total metric computation |
| `SeasonAverageBaseline` | `src.models.baseline` | RB-filtered baseline comparison |
| `SCORING`, `SEASONS`, etc. | `src.config` | Shared constants |

### 9.3 How to Run

```bash
# Run the general pipeline first (if not already run)
python scripts/run_pipeline.py

# Then run the RB-specific pipeline
python RB/run_rb_pipeline.py
```

Or, the RB pipeline can be called from the general pipeline's main script as an optional step:

```python
# At the end of scripts/run_pipeline.py
if __name__ == "__main__":
    # ... general pipeline ...

    # Optional: run RB position model
    print("\n=== Running RB Position Model ===")
    from RB.run_rb_pipeline import run_rb_pipeline
    run_rb_pipeline(train_df, val_df, test_df)
```

---

## 10. Full Pipeline Script

### `RB/run_rb_pipeline.py`

```python
"""
End-to-end RB position model pipeline.

Can be run standalone (loads general splits from disk) or called from
the general run_pipeline.py with DataFrames passed directly.
"""

import os
import numpy as np
import pandas as pd
import torch

from src.config import SPLITS_DIR
from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonAverageBaseline

from RB.rb_config import (
    RB_TARGETS, RB_RIDGE_ALPHAS, RB_SPECIFIC_FEATURES,
    RB_NN_BACKBONE_LAYERS, RB_NN_HEAD_HIDDEN, RB_NN_DROPOUT,
    RB_NN_LR, RB_NN_WEIGHT_DECAY, RB_NN_EPOCHS, RB_NN_BATCH_SIZE,
    RB_NN_PATIENCE, RB_SCHEDULER_PATIENCE, RB_SCHEDULER_FACTOR,
    RB_LOSS_W_RUSHING, RB_LOSS_W_RECEIVING, RB_LOSS_W_TD, RB_LOSS_W_TOTAL,
)
from RB.rb_data import filter_to_rb
from RB.rb_targets import compute_rb_targets
from RB.rb_features import add_rb_specific_features, get_rb_feature_columns, fill_rb_nans
from RB.rb_models import RBRidgeMultiTarget
from RB.rb_neural_net import RBMultiHeadNet
from RB.rb_training import MultiTargetLoss, RBMultiHeadTrainer, make_rb_dataloaders
from RB.rb_evaluation import compute_rb_metrics, compute_rb_ranking_metrics, print_rb_comparison_table


def run_rb_pipeline(train_df=None, val_df=None, test_df=None):
    """Run the full RB position model pipeline."""

    # --- Step 1: Load data ---
    if train_df is None:
        print("Loading general splits from disk...")
        train_df = pd.read_parquet(f"{SPLITS_DIR}/train.parquet")
        val_df = pd.read_parquet(f"{SPLITS_DIR}/val.parquet")
        test_df = pd.read_parquet(f"{SPLITS_DIR}/test.parquet")

    # --- Step 2: Filter to RB ---
    print("Filtering to RB...")
    rb_train = filter_to_rb(train_df)
    rb_val = filter_to_rb(val_df)
    rb_test = filter_to_rb(test_df)
    print(f"  RB splits: train={len(rb_train)}, val={len(rb_val)}, test={len(rb_test)}")

    # --- Step 3: Compute targets ---
    print("Computing RB targets...")
    rb_train = compute_rb_targets(rb_train)
    rb_val = compute_rb_targets(rb_val)
    rb_test = compute_rb_targets(rb_test)

    # --- Step 4: Add RB-specific features ---
    print("Adding RB-specific features...")
    rb_train, rb_val, rb_test = add_rb_specific_features(rb_train, rb_val, rb_test)

    # --- Step 5: Fill NaNs for RB features ---
    rb_train, rb_val, rb_test = fill_rb_nans(rb_train, rb_val, rb_test, RB_SPECIFIC_FEATURES)

    # --- Step 6: Prepare feature arrays ---
    feature_cols = get_rb_feature_columns()
    X_train = rb_train[feature_cols].values.astype(np.float32)
    X_val = rb_val[feature_cols].values.astype(np.float32)
    X_test = rb_test[feature_cols].values.astype(np.float32)

    y_train_dict = {t: rb_train[t].values for t in RB_TARGETS}
    y_train_dict["total"] = rb_train["fantasy_points"].values
    y_val_dict = {t: rb_val[t].values for t in RB_TARGETS}
    y_val_dict["total"] = rb_val["fantasy_points"].values
    y_test_dict = {t: rb_test[t].values for t in RB_TARGETS}
    y_test_dict["total"] = rb_test["fantasy_points"].values

    # --- Step 7: Baseline ---
    print("\n=== RB Baseline ===")
    baseline = SeasonAverageBaseline()
    baseline_preds = baseline.predict(rb_test)
    baseline_metrics = {"total": compute_metrics(y_test_dict["total"], baseline_preds)}

    # --- Step 8: Ridge multi-target ---
    print("\n=== RB Ridge Multi-Target ===")
    best_alpha = None
    best_val_mae = float("inf")
    for alpha in RB_RIDGE_ALPHAS:
        ridge = RBRidgeMultiTarget(alpha=alpha)
        ridge.fit(X_train, y_train_dict)
        val_preds = ridge.predict(X_val)
        val_mae = np.mean(np.abs(val_preds["total"] - y_val_dict["total"]))
        print(f"  Alpha={alpha:.2f}: Val MAE={val_mae:.3f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    print(f"  Best alpha: {best_alpha}")
    ridge_model = RBRidgeMultiTarget(alpha=best_alpha)
    ridge_model.fit(X_train, y_train_dict)
    ridge_test_preds = ridge_model.predict(X_test)
    ridge_metrics = compute_rb_metrics(y_test_dict, ridge_test_preds)

    # --- Step 9: Multi-head NN ---
    print("\n=== RB Multi-Head Neural Net ===")
    # Scale features using the rushing Ridge model's scaler
    scaler = ridge_model.rushing_model.scaler
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create dataloaders
    train_loader, val_loader = make_rb_dataloaders(
        X_train_scaled, y_train_dict, X_val_scaled, y_val_dict,
        batch_size=RB_NN_BATCH_SIZE,
    )

    # Initialize model
    device = torch.device("cpu")
    input_dim = X_train_scaled.shape[1]
    model = RBMultiHeadNet(
        input_dim=input_dim,
        backbone_layers=RB_NN_BACKBONE_LAYERS,
        head_hidden=RB_NN_HEAD_HIDDEN,
        dropout=RB_NN_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=RB_NN_LR, weight_decay=RB_NN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=RB_SCHEDULER_PATIENCE, factor=RB_SCHEDULER_FACTOR,
    )
    criterion = MultiTargetLoss(
        w_rushing=RB_LOSS_W_RUSHING, w_receiving=RB_LOSS_W_RECEIVING,
        w_td=RB_LOSS_W_TD, w_total=RB_LOSS_W_TOTAL,
    )

    trainer = RBMultiHeadTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, device=device, patience=RB_NN_PATIENCE,
    )

    history = trainer.train(train_loader, val_loader, n_epochs=RB_NN_EPOCHS)

    # Test predictions
    nn_test_preds = model.predict_numpy(X_test_scaled, device)
    nn_metrics = compute_rb_metrics(y_test_dict, nn_test_preds)

    # --- Step 10: Comparison ---
    print_rb_comparison_table({
        "Season Average Baseline": baseline_metrics,
        "RB Ridge Multi-Target": ridge_metrics,
        "RB Multi-Head NN": nn_metrics,
    })

    # --- Step 11: Ranking metrics ---
    rb_test["pred_ridge_total"] = ridge_test_preds["total"]
    rb_test["pred_nn_total"] = nn_test_preds["total"]

    ridge_ranking = compute_rb_ranking_metrics(rb_test, pred_col="pred_ridge_total")
    nn_ranking = compute_rb_ranking_metrics(rb_test, pred_col="pred_nn_total")

    print(f"\nRidge Top-12 Hit Rate: {ridge_ranking['season_avg_hit_rate']:.3f}")
    print(f"NN Top-12 Hit Rate:    {nn_ranking['season_avg_hit_rate']:.3f}")
    print(f"Ridge Spearman:        {ridge_ranking['season_avg_spearman']:.3f}")
    print(f"NN Spearman:           {nn_ranking['season_avg_spearman']:.3f}")

    # --- Step 12: Save ---
    os.makedirs("RB/outputs/models", exist_ok=True)
    os.makedirs("RB/outputs/figures", exist_ok=True)

    ridge_model.save("RB/outputs/models")
    torch.save(model.state_dict(), "RB/outputs/models/rb_multihead_nn.pt")

    return {
        "ridge_metrics": ridge_metrics,
        "nn_metrics": nn_metrics,
        "ridge_ranking": ridge_ranking,
        "nn_ranking": nn_ranking,
        "history": history,
    }


if __name__ == "__main__":
    results = run_rb_pipeline()
```

---

## Data Shape Summary

| Stage | Shape | Description |
|-------|-------|-------------|
| General splits loaded | ~25K / ~5K / ~5K rows, ~155 cols | Full position data |
| After `filter_to_rb()` | ~7K / ~1.4K / ~1.4K rows | RB only, pos encoding dropped |
| After `compute_rb_targets()` | same rows, +4 cols | Added target decomposition (with 2pt conversions) |
| After `add_rb_specific_features()` | same rows, +15 cols | Added 15 RB features |
| After whitelist filtering | same rows | Only whitelisted features retained (~123 cols) |
| Feature matrix X | (n_samples, ~123) | Ready for models |
| Target dict y | 4 arrays each (n_samples,) | rushing_floor, receiving_floor, td_points, total |
| NN output | dict of 4 tensors each (batch,) | Per-head + sum predictions |

---

## Design Notes

### On the Fumble Adjustment in Final Predictions

The 3-head model predicts rushing_floor + receiving_floor + td_points. The fumble_penalty is not directly predicted. Two options:

1. **Ignore fumble_penalty**: Accept a small systematic upward bias (~0.15 pts per game). This is within the model's MAE.
2. **Subtract historical fumble rate (recommended)**: Use `rolling_mean_fumble_penalty_L8` and subtract it from the total prediction as a post-hoc correction.

```python
total_pred = rushing_pred + receiving_pred + td_pred + fumble_adjustment
```

### On Loss Weight Tuning

The current loss weights (rushing: 1.2, receiving: 1.0, td: 2.0, total: 0.25) reflect tuning on the validation set:

- Rushing floor gets a slight boost (1.2) as the primary RB floor component
- TD weight elevated (2.0) for the zero-inflated, discrete nature of the target
- Total auxiliary weight reduced (0.25) to prevent it from overwhelming per-target gradients
- Huber deltas widened (2.0-2.5) from initial tight values — tight deltas caused flat gradient plateaus encouraging mean-clustering

### On Potential Future Improvements (Out of Scope)

- **Game script features**: Implied pass/rush ratio from Vegas lines (external data, not in nflverse)
- **Red zone carry/target share**: Derivable from `nfl.import_pbp_data()` by filtering `yardline_100 <= 20`, but PBP data is ~700MB/season. Strong predictor of td_points if compute budget allows.
- **Next Gen Stats rushing metrics**: `nfl.import_ngs_data("rushing")` provides rush yards over expected (RYOE), time behind line of scrimmage, etc. Limited to recent seasons and fewer players.
- **Injury report integration**: `nfl.import_injuries()` exists in nfl_data_py — could model availability likelihood and return-from-injury snap ramps
- **Depth chart position**: `nfl.import_depth_charts()` provides weekly starter/backup designation — a leading indicator of workload changes before they show up in stats
- **Zero-inflated regression for td_points**: Distribution is heavily zero-inflated (median = 0). A two-stage model (predict P(TD>0), then predict TD count | TD>0) may outperform MSE regression on this target

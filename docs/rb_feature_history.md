# RB Feature Engineering History

Authoritative record of every feature change to the RB model — what was added, removed, or moved between feature pathways, and why. Sources are git log on `src/rb/config.py` + `src/rb/features.py` (and their pre-rename predecessors `RB/rb_config.py`, `RB/rb_features.py`), [TODO.md](../TODO.md)'s **Fixed archive**, and inline comments in the configs.

Companion artifact: [`analysis_output/rb_feature_audit.json`](../analysis_output/rb_feature_audit.json) and the audit script [`src/analysis/analysis_rb_feature_audit.py`](../src/analysis/analysis_rb_feature_audit.py) for empirical multicollinearity evidence on the current feature set.

## How to read this doc

- **Pathways**: the model has three feature pathways and every feature lives in exactly one.
  - `INCLUDE_FEATURES` — the flat opt-in whitelist used by Ridge, LightGBM, ElasticNet, and the base `MultiHeadNet`. Organised into 9 categories (rolling, prior_season, ewma, trend, share, matchup, defense, contextual, weather_vegas, specific).
  - `ATTN_STATIC_FEATURES` — derived from `INCLUDE_FEATURES` via `ATTN_STATIC_CATEGORIES` (currently: prior_season + matchup + defense + contextual + weather_vegas). Feeds the attention NN's static branch.
  - `ATTN_HISTORY_STATS` — per-game raw stats fed as a sequence to the attention NN's history branch (independent of `INCLUDE_FEATURES`).
- **Targets** are listed separately. They moved from fantasy-point components to raw NFL stats in the target-migration (PR #15).
- Drops and additions are timestamped to PR/SHA so you can `git show <sha>` for the full diff.

---

## Current state (snapshot at SHA `7318c72`)

### Targets (6, raw NFL stats)

| Target | Loss family | Huber δ / Loss weight |
|---|---|---|
| `rushing_tds` | Poisson NLL + BCE gate | w=1.0 |
| `receiving_tds` | Poisson NLL + BCE gate | w=1.0 |
| `rushing_yards` | Huber | δ=15, w=0.133 (≈ 2.0/δ) |
| `receiving_yards` | Huber | δ=15, w=0.133 |
| `receptions` | Hurdle ZTNB-2 + BCE gate | w=1.0 |
| `fumbles_lost` | Poisson NLL | w=1.0 |

Aggregator `predictions_to_fantasy_points("RB", preds)` converts the raw-stat predictions to fantasy points post-prediction. There is no `total` head.

### `INCLUDE_FEATURES` (the model-agnostic whitelist)

Counts reflect what the audit materialises (some prior_season aggregates show up under aliases; see [analysis_output/rb_feature_audit.json](../analysis_output/rb_feature_audit.json) for the exact materialised list).

#### `rolling` — 28 columns
For each of `[fantasy_points, targets, receptions, carries, rushing_yards, receiving_yards, snap_pct]`, four aggregates `[mean, std, max]` × two windows `[L3, L8]`. `fantasy_points` additionally has `rolling_min` at L3, L5, L8 (the only L5 retained; see "L5 dropped" below).

#### `prior_season` — 21 columns
Same 7 stats × `[mean, std, max]` aggregated over the prior NFL season.

#### `ewma` — 0 columns (intentionally empty)
Dropped in [config.py:60-61](../src/rb/config.py:60): "All EWMA dropped (>0.98 corr with rolling means)".

#### `trend` — 4 columns
`trend_fantasy_points`, `trend_targets`, `trend_carries`, `trend_snap_pct`. Slope-style features capturing recent-vs-baseline change.

#### `share` — 6 columns
`target_share_L3/L5`, `carry_share_L3/L5`, `snap_pct`, `air_yards_share`. The team-total denominators are full-team (incl. QB scrambles in carries, all positions in air_yards).

#### `matchup` — 4 columns
Opponent points-allowed-to-position rolling means: `opp_fantasy_pts_allowed_to_pos`, `opp_rush_pts_allowed_to_pos`, `opp_recv_pts_allowed_to_pos`, plus the per-week rank `opp_def_rank_vs_pos`.

#### `defense` — 6 columns
L5 rolling means of the opponent defense's allowed stats: `opp_def_sacks_L5`, `opp_def_pass_yds_allowed_L5`, `opp_def_pass_td_allowed_L5`, `opp_def_ints_L5`, `opp_def_rush_yds_allowed_L5`, `opp_def_pts_allowed_L5`. Built by [`_build_defense_matchup_features`](../src/features/engineer.py:512).

#### `contextual` — 7 columns
`is_home`, `week`, `is_returning_from_absence`, `days_rest`, `practice_status`, `game_status`, `depth_chart_rank`.

#### `weather_vegas` — 4 columns
`implied_team_total`, `implied_opp_total`, `is_dome`, `rest_advantage`. (Other weather features dropped per-position; see [src/shared/weather_features.py](../src/shared/weather_features.py) `WEATHER_DROPS_BY_POSITION["RB"]`.)

#### `specific` — 15 columns (RB-only engineered features)
Defined in `SPECIFIC_FEATURES` at [config.py:14-30](../src/rb/config.py:14):

| Feature | What it measures |
|---|---|
| `yards_per_carry_L3` | Rushing efficiency, last 3 games |
| `reception_rate_L3` | Catch rate, last 3 games |
| `weighted_opportunities_L3` | Raw `carries + 2*targets`, L3 mean |
| `team_rb_carry_share_L3` | Player carries / RB-only team carries |
| `team_rb_target_share_L3` | Player targets / RB-only team targets |
| `rushing_epa_per_attempt_L3` | EPA value per carry |
| `rushing_first_down_rate_L3` | First-down conversion on carries |
| `receiving_first_down_rate_L3` | First-down conversion on receptions |
| `yac_per_reception_L3` | YAC efficiency |
| `receiving_epa_per_target_L3` | EPA value per target |
| `air_yards_per_target_L3` | Intended depth on targets |
| `career_carries` | Cumulative pre-game career carries |
| `team_rb_carry_hhi_L3` | Team-level concentration of carry shares |
| `team_rb_target_hhi_L3` | Team-level concentration of target shares |
| `opportunity_index_L3` | Player weighted opps / team weighted opps |

### `ATTN_STATIC_CATEGORIES` (5 categories → ATTN_STATIC_FEATURES)

`prior_season + matchup + defense + contextual + weather_vegas` ≈ **42 columns** that feed the attention NN's static branch (full list materialised at runtime).

**Excluded by design**: `rolling`, `ewma`, `trend`, `share`, `specific`. The attention sequence already learns temporal patterns from `ATTN_HISTORY_STATS`; routing rolling/share/specific into the static branch would double-count signal ([config.py:281-283](../src/rb/config.py:281)).

### `ATTN_HISTORY_STATS` (15 per-game stats fed as a sequence)

```python
[
    "rushing_yards", "receiving_yards", "rushing_tds", "receiving_tds",
    "carries", "targets", "receptions", "fumbles_lost", "snap_pct",
    "rushing_first_downs", "receiving_first_downs",
    "game_carry_share", "game_target_share",
    "game_carry_hhi", "game_target_hhi",
]
```

`max_seq_len = 17`. Padding=0, mask flag tracks real vs padded games.

**Excluded by design**: `fantasy_points` ([config.py:258-262](../src/rb/config.py:258)) — deterministic linear combination of the represented components.

---

## Change log (reverse chronological)

### 2026-04 — `7318c72` chore: explicit `non_negative_targets` and `aggregate_fn`
Made the per-head non-negativity clamp explicit in the RB config (`NN_NON_NEGATIVE_TARGETS = set(TARGETS)`) instead of relying on the `MultiHeadNet` default (which clamps every head if the kwarg is `None`). Same effect, written out for clarity. Important for future refactors where a head might want to allow negative outputs (e.g., a bonus that can subtract).

### 2026-04 — `ec1564d` / PR #163 feat(attn): add `fumbles_lost` as attention history channel
**Added** `fumbles_lost` to `ATTN_HISTORY_STATS` (RB went 14 → 15 channels). `fumbles_lost` was already an RB target but wasn't in any attention sequence. The model had to recover the lagged-fumble signal indirectly through `snap_pct` + scoring shocks. Now the attention NN sees the lagged event directly. Side-effect: with `fumbles_lost` in the sequence, `fantasy_points` is now a full linear combination of the represented channels (yards + TDs + receptions + fumbles_lost), which strengthened the case for the next change.

### 2026-04 — `e5c97db` / PR #161 refactor(rb): drop `fantasy_points` from attention history
**Removed** `fantasy_points` from `ATTN_HISTORY_STATS`. Justified as deterministic linear combination of the other channels (yards + TDs + receptions + fumbles_lost). Lagged signal still reaches Ridge / LightGBM / base NN through the `rolling` / `prior_season` / `trend` categories.

### 2026-04 — `aa3e168` / PR #113 feat(nn): restore TD-head gate (Variant C)
**Re-added** BCE gate to `rushing_tds` and `receiving_tds`: `GATED_TARGETS = ["receptions", "rushing_tds", "receiving_tds"]`. Triggered by the `src/tuning/ablate_rb_gate.py` results (run 24813558434):

| Variant | FP MAE | Rush TD MAE | Rec TD MAE | Rec MAE |
|---|---|---|---|---|
| A (Huber+gate, pre-PR-#96) | 4.453 | 0.277 | 0.077 | 1.034 |
| B (Poisson/no gate, PR-#96) | 4.258 | 0.329 | 0.064 | 0.989 |
| C (Poisson+gate, this PR) | 4.239 | 0.304 | 0.099 | 0.983 |

Variant C edges B by 0.019 pt/game — under the 0.05 decision threshold, so the ablation script reported "drop the gate" and PR #96 shipped B. But B's per-target rushing_tds MAE regressed +0.052 vs A; Variant C closes half that gap at zero cost to FP MAE. Documented in extensive detail in [config.py:292-307](../src/rb/config.py:292).

### 2026-04 — `f11dd89` / PR #108 feat(lgbm): RB retuned on huber objective
**Changed** `LGBM_OBJECTIVE` from `"fair"` to `"huber"`. 50 Optuna trials (run 24823926033) on the huber objective gave:

| Metric | Old fair → new huber | Δ |
|---|---|---|
| Total MAE (holdout) | 4.479 → 4.155 | -0.325 |
| Rushing yards MAE | 21.3 → 17.6 | -3.7 |
| Receiving yards MAE | 9.15 → 8.68 | -0.47 |
| Rushing TDs MAE | 0.299 → 0.321 | +0.022 |
| Receiving TDs MAE | 0.093 → 0.108 | +0.015 |
| Top-12 hit rate | 0.476 → 0.508 | +0.032 |
| Spearman ρ | 0.577 → 0.733 | +0.156 |

CV MAE moved only +0.0095 (4.5149 → 4.5244, within the ±0.05 tolerance), so the holdout -0.325 is the better hyperparams paying off, not a lucky CV→holdout draw. **No feature changes** — pure hyperparameter retune. Listed here because it changed the *production* model's reliance on each feature substantially.

### 2026-04 — `5d5f606` / PR #105 feat(models): ElasticNet baseline + SHAP diagnostic
**Added** `TRAIN_ELASTICNET = False` (off by default), `ENET_L1_RATIOS = [0.3, 0.5, 0.7]`. Optional L1+L2 baseline alongside Ridge. Skips PCA explicitly: "L1 on a rotated basis zeros components, not features". SHAP script (`analysis_shap_lgbm.py`) loads trained LightGBM models and emits per-target summary plots backed by a new `build_train_matrix` helper that replays pipeline setup so diagnostics don't drift from training (the failure mode that deprecated the prior `analysis_rb_feature_signal.py`).

### 2026-04 — `6857cf1` / PR #96 feat(nn): retarget hurdle head from TDs to receptions; Poisson NLL on counts
**Changed** loss families per head:
- TDs / fumbles: Huber → **Poisson NLL** (empirical dispersion 1.03–1.17 with negligible zero-excess; the prior BCE gate on `(TD > 0)` was unmotivated)
- Receptions: Huber + Poisson hybrid → **Hurdle ZTNB-2** (variance/mean ~2.0 with zero-excess up to +0.13 — textbook hurdle fit). Gate BCE added via the existing `GATED_TARGETS` path.
- Yards: stayed on Huber.

`HEAD_LOSSES` dict introduced as the canonical per-target loss mapping. `MultiTargetLoss` got a `head_losses` dispatcher. Per-target value heads gained `value_log_alpha` for hurdle-NegBin's per-sample dispersion. Validation catches the common misconfig (a `hurdle_negbin` target not in `gated_targets`).

### 2026-04 — `9ead4f9` / PR #93 refactor: shared feature-build helpers
**Refactored** every position's `*_features.py` (including RB's) to route through three shared helpers:
- `safe_divide` — replaces hand-rolled `(num/denom).fillna(0) + .loc[denom==0] = 0`
- `rolling_agg` — replaces hand-rolled shifted-rolling closures
- `fill_nans_with_train_means` — replaces hand-rolled inf→NaN + train-mean backfill loop

Behaviour-preserving for RB; just reduces duplication and enforces the leak-prevention `shift(1)` consistently.

### 2026-04 — `bb0d046` / PR #82 chore: finish raw-stat migration cleanup
**Broadened** `fumbles_lost` target across QB/RB/WR/TE to sum all three categories (`sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost`) so the aggregator lands exactly on `nflverse fantasy_points`. **Deleted** `fantasy_points_floor` feature entirely from the loader, configs, attention history, and `GAME_HISTORY_STATS` — raw-stat rolling means already carry that signal. **Deleted** `RB/RB_IMPLEMENTATION.md` (superseded by [docs/ARCHITECTURE.md](ARCHITECTURE.md)). **Added** DEPRECATED headers to `analysis_rb_feature_signal.py`, `tune_rb_gate.py`, `analysis_weather_vegas_correlation.py` — these scripts reference pre-migration targets (`rushing_floor` / `receiving_floor` / `td_points`) and don't run on current data.

### 2026-04 — `2500ecc` / PR #55 refactor(attn): per-position whitelist for attention static features
**Replaced** the prefix/suffix blacklist in `get_attn_static_columns` with an explicit per-position whitelist. The old blacklist was missing every entry in `*_SPECIFIC_FEATURES` (because those columns don't match the blacklist shape), so RB's `yards_per_carry_L3`, `team_rb_carry_share_L3`, `opportunity_index_L3`, `career_carries`, etc. were all leaking into the attention static branch — **duplicating signal that the attention branch already learns from `ATTN_HISTORY_STATS`**. After this PR:

- `RB_ATTN_STATIC_CATEGORIES = ["prior_season", "matchup", "defense", "contextual", "weather_vegas"]`
- `RB_ATTN_STATIC_FEATURES = [c for cat in CATEGORIES for c in INCLUDE_FEATURES[cat]]`
- `rolling`, `ewma`, `trend`, `share`, `specific` are explicitly excluded.

**This is the single most important attention-NN feature-pathway change in the history.** Before this PR, the static branch was effectively double-fed.

### 2026-04 — `53ce08f` / PR #41 fix(rb): use `np.divide(..., where=)` for `opportunity_index`
Behaviour-preserving cleanup of `opportunity_index_L3`'s per-game ratio computation. `np.where` evaluates both branches unconditionally → division-by-zero `RuntimeWarning` even when results were discarded. Switch to `np.divide(..., out=zeros, where=denom>0)`, matching the pattern used a few lines above for `game_carry_share` / `game_target_share`. Output values bit-identical.

### 2026-04 — `d229830` / PR #21 tune(rb): rebalance `LOSS_WEIGHTS` inverse-to-delta
**Changed** loss weights to ≈ `2.0 / huber_delta` per head. Post-target-migration NN had count-target heads collapsing to the mean: yards-dominated Huber loss was ~δ²/2 = 112 per sample for yards vs 0.12 for counts — a 1000× imbalance. Yards gradients drowned out count-head gradients. Post-fix:
- yards: w = 0.133 (= 2/15)
- TDs / fumbles / receptions: w = 1.0

Encoded the `2.0/δ` invariant inline in [config.py:200-216](../src/rb/config.py:200) so future contributors who change a Huber δ re-derive the matching weight. NN FP MAE recovered from 5.21 → 4.23 (-0.98 pt/game). Ridge / LGBM unaffected (per-target, independent).

### 2026-04 — `51cb2e3` / PR #15 target-migration: RB raw-stat targets + dual-gate TD
**This is the foundational target migration.** Switched RB predictions from fantasy-point components (`rushing_floor`, `receiving_floor`, `td_points`) to raw NFL stats:

```python
TARGETS = ["rushing_tds", "receiving_tds", "rushing_yards",
           "receiving_yards", "receptions", "fumbles_lost"]
```

- `fumbles_lost` = `rushing_fumbles_lost + receiving_fumbles_lost` (skill positions don't have sack fumbles).
- `compute_fumble_adjustment` deleted; `fumbles_lost` is now a direct target instead of a post-prediction adjustment.
- `predictions_to_fantasy_points("RB", preds)` converts to PPR fantasy points post-prediction. Scoring-format flexibility: change the aggregator, not the model.
- TD modeling: two parallel TD stacks (one per TD type) with `class_values=[0,1,2,3]` (raw counts) instead of one combined stack with `[0,6,12,18]` (point equivalents). Two `GatedTDHead` heads, one per TD target.
- `POSITION_INFO["RB"]` rewritten with 6 target entries.

This PR was the trigger for the loss-weight rebalance (#21), the per-position attention static whitelist (#55), the raw-stat cleanup (#82), and the loss-family rework (#96). All later work flows from here.

### 2026-04 — `9a631ac` / PR #8 feat(attention): per-target queries + 2-layer game encoder
**Architectural change** to the RB attention NN (no feature-list change, but it changes how features are consumed).

Before: a shared `AttentionPool` query forced all RB targets to share a single pooled summary through the backbone, pushing the learned queries toward broadly-useful averages — exactly the smoothed-summary behavior that made the attention model underperform the plain multi-head NN. With only 2 shared queries, the pool had no capacity to extract target-specific slices.

After: `AttentionPool` carries `[n_targets, n_heads, d_model]` queries and returns `[batch, n_targets, n_heads * d_model]`. Each head consumes `concat(shared_static, its_own_history_vec)`. Also bumped `ATTN_ENCODER_HIDDEN_DIM` 0 → 32 so each game becomes a richer nonlinear event embedding (Linear→ReLU→LayerNorm→Linear→ReLU) before attention.

### Pre-PR (early/exploratory commits)

The pre-restructure history is a sequence of terse-message commits (`weather`, `binary gate on NN, added features to attention`, `do it`, `more`, `goat`, `a`) that landed the original feature scaffolding before commit-message hygiene was enforced. The relevant adds visible in the diff trail:

- **Weather/Vegas** (`7c4e560` "added weather to all", `1d501fe` "weather"): first introduction of `is_dome`, `is_grass`, `temp_adjusted`, `wind_adjusted`, `implied_team_total`, `implied_opp_total`, `total_line`, `implied_total_x_wind`, `is_divisional`, `days_rest_improved`, `rest_advantage`. Subsequently pruned per-position (see [docs/design_weather_and_odds.md](design_weather_and_odds.md) and [src/shared/weather_features.py:WEATHER_DROPS_BY_POSITION](../src/shared/weather_features.py)).
- **Attention features** (`18170a6` "binary gate on NN, added features to attention"): introduced `ATTN_HISTORY_STATS` for RB.
- **PCA** (`39349da` "PCA on WR, RB"): introduced `RIDGE_PCA_COMPONENTS = 80` for RB. Dropped Ridge condition number from 1.8e8 (after `is_home` removal) to 49.8; both yards targets improved by ~0.002 MAE ([config.py:155-157](../src/rb/config.py:155)).
- **Depth chart + injuries** (`c399c12`): introduced `depth_chart_rank`, `is_returning_from_absence`, `practice_status`, `game_status`, `days_rest`.
- **L5 dropped** (pre-restructure cleanup): every `rolling_*_L5` column dropped because >0.97 correlated with `L3` or `L8` neighbour. Exception: `rolling_min_fantasy_points_L5` retained ([config.py:56](../src/rb/config.py:56)).
- **EWMA dropped** (pre-restructure cleanup): all EWMA columns dropped because >0.98 correlated with their `rolling_mean` counterparts ([config.py:60-61](../src/rb/config.py:60)).
- **`spread_line` excluded** ([config.py:95-97](../src/rb/config.py:95)): "implied_team + implied_opp encodes both game total and spread direction without the perfect collinearity of keeping total_line alongside either."
- **`first_down_rate_L3` split** ([features.py:125-130](../src/rb/features.py:125)): combined first-down rate replaced with split `rushing_first_down_rate_L3` and `receiving_first_down_rate_L3` to avoid collinearity (the combined rate ≈ weighted average of the two).

---

## Patterns that emerged (design rules)

These rules show up repeatedly in commit messages and inline comments. They function as the standing "before you change feature X, read this" guardrails.

### 1. Raw-stat targets, never fantasy-point targets

Every position predicts raw NFL stats. Fantasy points are computed *after* prediction via `aggregate_fn`. Source: PR #15, PR #82, [CLAUDE.md](../CLAUDE.md). Violation cost (when this rule was first broken): the QB ~1.9 pt/game double-count fix in TODO.md archive.

### 2. Feature whitelist is explicit, not inferred

`INCLUDE_FEATURES` is opt-in. New columns must be added explicitly. Source: [CLAUDE.md](../CLAUDE.md). Why: prevents silent feature leakage; makes feature-list diffs readable.

### 3. Attention static branch has its own, smaller whitelist

`ATTN_STATIC_CATEGORIES` is a subset of `INCLUDE_FEATURES` keys, not the full list. Source: PR #55. Why: the attention history branch already extracts temporal patterns from `ATTN_HISTORY_STATS`; routing rolling/share/specific into the static branch would double-count.

### 4. Loss weights are tuned inverse-to-Huber-delta

`LOSS_WEIGHTS[t] ≈ 2.0 / HUBER_DELTAS[t]`. Source: PR #21, [CLAUDE.md](../CLAUDE.md), [config.py:200-216](../src/rb/config.py:200). Why: without this, yards heads (δ=15) dominate count heads (δ=0.5) by ~2500× per sample.

### 5. Always diff training vs inference paths

Training pipeline (`src/shared/pipeline.py`) and serving (`src/serving/app.py`) both build features. They have drifted before. Source: [CLAUDE.md](../CLAUDE.md), [TODO.md](../TODO.md) archive (weather/Vegas merge in training but not serving; scaler clip in one path but not the other).

### 6. Use torch ops in NN training paths, not numpy

Anything inside the forward pass / loss / `aggregate_fn` callback must stay in `torch` — `np.digitize` / `np.clip` / `np.where` on tensors silently breaks autograd. Source: [CLAUDE.md](../CLAUDE.md).

### 7. L3/L8 only — `*_L5` was uniformly redundant

All `rolling_*_L5` columns dropped at the >0.97-corr threshold. Source: [config.py:46](../src/rb/config.py:46). Exception: `rolling_min_fantasy_points_L5` kept (the only L5 that wasn't redundant). The `target_share_L5` and `carry_share_L5` columns in the `share` category were never re-audited under this rule and the recent audit ([analysis_output/rb_feature_audit.json](../analysis_output/rb_feature_audit.json)) confirms they trip the threshold (r=0.966 and 0.984 respectively) — pending follow-up.

### 8. No EWMA — uniformly redundant with rolling means

Source: [config.py:60-61](../src/rb/config.py:60). All EWMA columns dropped at >0.98 corr.

### 9. `fantasy_points` excluded from attention sequence

Linear combination of represented components. Source: PR #161 + [config.py:258-262](../src/rb/config.py:258).

### 10. RB-specific constants are explicit, not defaults

Per `7318c72`: `NN_NON_NEGATIVE_TARGETS = set(TARGETS)` is written out instead of relying on the `MultiHeadNet` default. Why: makes future diffs (e.g., introducing a signed head) audit-able without reading the default.

---

## Pending recommendations (from the latest audit)

See [`/Users/alex/.claude/plans/stringently-analyze-all-my-wild-pony.md`](../../.claude/plans/stringently-analyze-all-my-wild-pony.md) for the full audit and recommendation list. Headline:

**Strong drops** (empirically confirmed, low-risk):

| Feature | Keep instead | Evidence |
|---|---|---|
| `opp_def_rank_vs_pos` | `opp_fantasy_pts_allowed_to_pos` | Spearman -0.937, by-construction rank |
| `target_share_L5` | `target_share_L3` | r = 0.966 (matches the L5-drop rule) |
| `carry_share_L5` | `carry_share_L3` | r = 0.984 |
| `carry_share_L3` | `team_rb_carry_share_L3` | r = 0.982 (denominator differs only by QB scrambles) |
| `weighted_opportunities_L3` | `opportunity_index_L3` | r = 0.940 (count vs share of same quantity) |
| `is_returning_from_absence` | `days_rest` | r = 0.934 (indicator is essentially `days_rest > 13`) |

**Architectural follow-up** (separate PR): RB lacks `OPP_ATTN_HISTORY_STATS`, the parallel opp-defense attention branch QB/WR/TE all have. Adding it lets the 6 `opp_def_*_L5` features migrate from a fixed L5 mean in static to a per-game sequence the attention pool can recency-weight. Infra ([build_opp_defense_history_arrays at engineer.py:340](../src/features/engineer.py:340)) is fully built and wired through pipeline + serving. Mirror WR's stanza in [src/rb/config.py](../src/rb/config.py).

**Probable drops** (pending ablation): see the audit recommendations table for prior_season-block redundancies (12 columns at VIF 20–55), the matchup VIF blow-up (193/119/78), and the `prior_season_max` ↔ `prior_season_std` overlap for skewed stats.

---

## How to extend this doc

When you change a feature list, append an entry to **Change log** with:
- PR #, SHA, date.
- Which feature(s) added / removed / moved.
- Why (cite an ablation, a benchmark diff, or a TODO.md archive entry).
- Side-effects (e.g., "fumbles_lost in history made fantasy_points a linear combo, justifying its later removal").

When a pattern emerges across multiple feature changes, promote it to **Patterns that emerged** with citations.

# Bugs & Potential Issues

Tracking all known issues, from confirmed bugs to speculative concerns.
Older fixed issues are kept for reference — they show patterns worth watching for.

---

## Fixed

### [FIXED] Total aux loss double-counted adjustments
- **Files:** `shared/pipeline.py:208-211`
- **What:** Training total target was `fantasy_points` (includes INT/fumble penalties), but the model predicts `sum(heads)` (clean targets only). The total aux loss trained heads to absorb penalties. Then at inference, adjustments were added *again* via `adj.values`. Net effect: ~1.9 pts/game double-counted penalty for QB.
- **Fix:** Changed total target to `sum(pos_train[t].values for t in targets)`.
- **Lesson:** When a loss term compares a derived quantity (sum of heads) to a label, the label must match the derivation exactly. Any mismatch between what the model produces and what it's trained against will leak into predictions.

### [FIXED] Softplus floor inflated low-scoring predictions
- **Files:** `shared/neural_net.py:62-70`
- **What:** `softplus(0) ~ 0.693` per head created a ~2.08 point floor across 3 heads. No player could be predicted below ~2 points. Also created a scale mismatch with Ridge (`np.maximum(..., 0)` allows exact zeros), biasing the ensemble.
- **Fix:** Replaced with `torch.clamp(val, min=0.0)`.
- **Lesson:** Non-negativity constraints on output layers must allow exact zeros. Softplus is good in hidden layers (smooth gradient), but on outputs its floor compounds across heads. Clamp/ReLU is better when you need a hard boundary.

### [FIXED] No feature clipping after StandardScaler
- **Files:** `shared/pipeline.py:311-313`, `app.py:320`
- **What:** Test features could produce z-scores up to 19.5 — far outside the training distribution. NN predictions were unpredictable for these inputs.
- **Fix:** Added `np.clip(..., -4, 4)` after all `StandardScaler.transform()` calls.
- **Lesson:** Always clip scaled features. StandardScaler assumes train/test distributions are similar, but outliers in test data can produce extreme z-scores. Clip at +/-4 (catches 0.3-0.4% of values, prevents catastrophic extrapolation).

### [FIXED] `kicker_week_split` does not exist — app.py crashed on import
- **File:** `app.py:62, 262`
- **What:** Imported `kicker_week_split` from `K.k_data`, but the function was renamed to `kicker_season_split`. App crashed immediately with `ImportError`.
- **Fix:** Changed import and call site to `kicker_season_split`.
- **Lesson:** When renaming functions, grep for all call sites across the project — not just the file where the function is defined.

### [FIXED] DST `pts_allowed_bonus` clamped to >= 0, but target ranges from -4 to +10
- **Files:** `shared/neural_net.py`, `DST/dst_config.py`, `DST/run_dst_pipeline.py`, `app.py`
- **What:** The softplus-to-clamp fix applied `clamp(min=0)` globally to all heads. But DST's `pts_allowed_bonus` ranges from -4 (35+ points allowed) to +10 (shutout). The model couldn't predict negative tiers.
- **Fix:** Added `non_negative_targets` parameter to `MultiHeadNet.__init__` (defaults to all targets). DST config specifies `{"defensive_scoring", "td_points"}`, leaving `pts_allowed_bonus` unconstrained.
- **Lesson:** Output constraints must be per-head when targets have different valid ranges. A global clamp works for most positions but breaks any target that can legitimately be negative.

### [FIXED] Pipeline evaluation added adjustment to predictions but not to the total target
- **Files:** `shared/pipeline.py:305, 349` (and equivalent in `run_cv_pipeline`)
- **What:** After fixing the total aux loss target to `sum(targets)`, evaluation still added `adj_test.values` to Ridge and NN total predictions. This compared `sum(preds) + adj` against `sum(targets)` — the adjustment inflated the evaluation error.
- **Fix:** Removed `+ adj_test.values` from evaluation totals. Adjustment is only applied at inference in `app.py`.
- **Lesson:** When changing a training target, trace all downstream consumers — evaluation metrics, ensemble computation, and plotting all need to stay consistent. This was a cascading side-effect of Fix 1 that we caught.

---

## Open — Moderate Issues

### [FIXED] Predictions are always PPR but API serves multiple scoring formats
- **File:** `app.py:505-561`
- **What:** `/api/predictions` accepts a `scoring` param (standard, half_ppr, ppr). It selects the correct *actual* column, but `ridge_pred` and `nn_pred` are always trained on PPR targets. When a user selects "standard" scoring, actuals change but predictions don't — the comparison is apples-to-oranges.
- **Fix:** Added `scoring_note` field to API response when scoring != ppr. UI already displays a "PPR Scoring" badge and has no scoring selector, so users aren't misled. Training separate models per format is out of scope.
- **Impact:** API consumers now get a clear warning.

### [ACKNOWLEDGED] K features use cross-season rolling windows
- **File:** `K/k_features.py:25`
- **What:** Kicker features group by `["player_id"]` only (no season reset), so rolling windows span across seasons. A kicker's late-season 2024 stats influence early-2025 predictions.
- **Rationale:** Kickers have stable multi-year careers and small per-season sample sizes (~17 games), so cross-season windows provide more signal. Comment expanded with this rationale.
- **Risk:** If a kicker changes teams or role between seasons, stale cross-season features could mislead the model. Likely small impact.

### [FIXED] RB test fixture missing `receiving_epa` and `receiving_air_yards` columns
- **File:** `RB/tests/test_rb_features.py:10-52`
- **What:** `_make_player_games()` fixture didn't include `receiving_epa` or `receiving_air_yards` columns added with features 10-11. The feature list `RB_FEATURE_COLS` also had stale names (`first_down_rate_L3` instead of split `rushing_first_down_rate_L3` / `receiving_first_down_rate_L3`). 11 tests failed with `KeyError: 'receiving_epa'`.
- **Fix:** Added missing columns to the fixture and updated `RB_FEATURE_COLS` to match the current 11 features in `rb_config.py`.
- **Lesson:** When adding new features to `*_features.py`, update the corresponding test fixture and expected feature list.

### [FIXED] DST prior-season feature alignment used `.values`
- **File:** `DST/dst_features.py:74-86`
- **What:** Prior-season features were merged via `season+1` then assigned back using `.values` (strips index). If the merge reordered rows, assignments would be silently misaligned.
- **Fix:** Changed to index-preserving merge: `reset_index()` → merge → `set_index("index")` → loc-based assignment using the original index.

---

## Open — Low / Uncertain Issues

### [FIXED] Feature column filtering could silently drop features at inference
- **File:** `app.py:308-311`
- **What:** `feature_cols = [c for c in feature_cols if c in pos_train.columns]` filters to available columns. If a feature from training is missing, the model gets fewer features than expected → dimension mismatch → crash. The crash would happen, but the error message wouldn't identify which feature is missing.
- **Fix:** Added count comparison and warning log when columns are dropped.

### [FIXED] No API error handling
- **Files:** `app.py:85-90`
- **What:** All API routes lacked try/except. If `_get_data()` or model loading failed, the user saw a generic 500 with no useful message.
- **Fix:** Added Flask `@app.errorhandler(Exception)` that returns JSON `{"error": ...}` for `/api/` routes. Logs full traceback to console.

### [LOW] `_cache` dict grows without eviction
- **File:** `app.py:359`
- **What:** `_get_data()` caches results in a module-level dict. The cache is never cleared. Not a real problem for a class project (server restarts frequently), but worth noting.

### [LOW] `drop_last=True` silently discards training samples
- **File:** `shared/training.py:77`
- **What:** Last incomplete batch is dropped. With batch_size=512 (WR) and 32,521 training rows, 121 rows (~0.4%) are never seen. Standard practice, but combined with early stopping means those rows never contribute.

### [LOW] K targets overwrite `fantasy_points` column
- **File:** `K/k_targets.py:37`
- **What:** `df["fantasy_points"] = df["fg_points"] + df["pat_points"] + df["miss_penalty"]` overwrites the original column. Safe if called once, but calling twice would use the already-computed value. Not a current bug, just fragile.

### [LOW] Redundant NaN handling in feature engineering
- **Files:** All `*_features.py` files
- **What:** Pattern like `(a / b).fillna(0)` followed by `df.loc[b == 0, col] = 0` is redundant — fillna already handled the division-by-zero case. Not wrong, just noisy.

### [UNCERTAIN] K/DST index collision in `_get_data()`
- **File:** `app.py:386-403`
- **What:** K/DST test rows are appended to `results` with `offset = results.index.max() + 1`. Assumes the general test data has a well-behaved index. If the general test parquet has gaps, K/DST indices could collide. Probably safe in practice since parquet preserves sequential indices.

### [UNCERTAIN] Huber delta asymmetry across targets
- **Files:** Position config files (`*_config.py`)
- **What:** DST `pts_allowed_bonus` has Huber delta 3.0 with target range [-4, 10] (delta covers 21% of range). QB `td_points` also has delta 3.0 but range [0, 72+] (delta covers 4% of range). This means the loss is much more robust to outliers for `pts_allowed_bonus` than for `td_points`. May be intentional tuning, or may be an oversight.

### [UNCERTAIN] Team share features computed per-split
- **Files:** `RB/rb_features.py:69`, `WR/wr_features.py:61`, `TE/te_features.py:54`
- **What:** Team carry/target shares are computed within each split independently. A player's share could differ between train and test if their teammates are distributed differently across splits. By design (prevents leakage), but the share values won't be globally consistent.

# Design Document 3: XGBoost as a 4th Ensemble Model

> **Status: historical artifact (2026-04-21).** Rationale was folded into [ADR-001 §D3](ARCHITECTURE.md#d3-three-way-model-comparison-no-ensemble). Kept for provenance, not updated after the three-way independent comparison was chosen instead.

## Motivation

The current ensemble averages Ridge regression (linear, well-calibrated) and a multi-head neural network (nonlinear, better per-target precision). Both operate on the same StandardScaler-normalized features.

Gradient boosted trees occupy a fundamentally different region of the model-class space: they learn axis-aligned splits, handle feature interactions natively without needing them in the input, are invariant to monotonic feature transformations (no scaling required), and are robust to irrelevant features. On tabular datasets of this size (3,000-6,500 samples per position), tree-based models consistently match or beat neural networks (Grinsztajn et al., NeurIPS 2022). Adding XGBoost should improve ensemble diversity — the primary driver of ensemble accuracy — because it makes errors in different places than Ridge or the NN.

XGBoost specifically (vs. LightGBM) is preferred here because its level-wise tree growth is more conservative on small datasets, reducing overfitting risk without requiring aggressive `num_leaves` constraints.

## Architecture Overview

```
Feature Matrix X (same features as Ridge/NN)
        │
        ├──► Ridge Multi-Target     ──► per-target preds ──┐
        │                                                   │
        ├──► Multi-Head NN          ──► per-target preds ──┤
        │    (StandardScaler → NN)                          │
        │                                                   ├──► Equal-weight average
        ├──► Weather NN (optional)  ──► per-target preds ──┤    per target
        │    (StandardScaler → NN)                          │
        │                                                   │
        └──► XGBoost Multi-Target   ──► per-target preds ──┘
             (no scaling needed)
```

XGBoost trains one `XGBRegressor` per target (mirroring Ridge's per-target structure), predicts per-target values clamped >= 0, and sums them for the total. No scaler is needed — tree splits are invariant to monotonic transformations.

## New File: `src/shared/xgboost_model.py`

```python
"""XGBoost multi-target model for fantasy point decomposition."""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


class XGBoostMultiTarget:
    """Separate XGBoost regressors for each target in a multi-target decomposition.

    Mirrors the RidgeMultiTarget interface: fit(), predict(), save(), load(),
    get_feature_importance().
    """

    def __init__(
        self,
        target_names: list[str],
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 2.0,
        reg_alpha: float = 0.1,
        min_child_weight: int = 5,
        seed: int = 42,
    ):
        self.target_names = target_names
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            objective="reg:pseudohuberror",  # Huber loss, matching Ridge/NN philosophy
            tree_method="hist",
            random_state=seed,
            verbosity=0,
        )
        self._models = {
            name: XGBRegressor(**self._params) for name in target_names
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train_dict: dict,
        X_val: np.ndarray = None,
        y_val_dict: dict = None,
    ) -> dict:
        """Train one XGBRegressor per target with optional early stopping.

        Args:
            X_train: (n_train, n_features)
            y_train_dict: {"target_name": np.ndarray (n_train,), ...}
            X_val: (n_val, n_features) — enables early stopping if provided
            y_val_dict: {"target_name": np.ndarray (n_val,), ...}

        Returns:
            best_iterations: {"target_name": int, ...} — rounds used per target
        """
        best_iterations = {}
        for name, model in self._models.items():
            fit_kwargs = {}
            if X_val is not None and y_val_dict is not None:
                model.set_params(early_stopping_rounds=30)
                fit_kwargs["eval_set"] = [(X_val, y_val_dict[name])]
                fit_kwargs["verbose"] = False

            model.fit(X_train, y_train_dict[name], **fit_kwargs)
            best_iterations[name] = getattr(model, "best_iteration", model.n_estimators)
        return best_iterations

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict of per-target predictions (clamped >= 0) plus total."""
        preds = {}
        for name, model in self._models.items():
            raw = model.predict(X)
            preds[name] = np.maximum(raw, 0.0)
        preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def get_feature_importance(self, feature_names: list[str]) -> dict:
        """Returns {target: pd.Series of feature importance (gain-based)}."""
        result = {}
        for name, model in self._models.items():
            importance = model.feature_importances_  # gain-based by default
            result[name] = pd.Series(
                importance, index=feature_names
            ).sort_values(ascending=False)
        return result

    def save(self, model_dir: str) -> None:
        """Saves each per-target model to {model_dir}/xgboost/{target_name}.json."""
        xgb_dir = os.path.join(model_dir, "xgboost")
        os.makedirs(xgb_dir, exist_ok=True)
        for name, model in self._models.items():
            model.save_model(os.path.join(xgb_dir, f"{name}.json"))

    def load(self, model_dir: str) -> None:
        """Loads from {model_dir}/xgboost/{target_name}.json."""
        xgb_dir = os.path.join(model_dir, "xgboost")
        for name, model in self._models.items():
            model.load_model(os.path.join(xgb_dir, f"{name}.json"))
```

### Key design decisions

1. **`reg:pseudohuberror` objective** — XGBoost's built-in Pseudo-Huber loss is robust to outliers like the Huber loss used in Ridge/NN, keeping the loss philosophy consistent across the ensemble.

2. **No scaler** — Trees split on rank order, not magnitude. Skipping StandardScaler avoids an unnecessary artifact and one fewer file to save/load.

3. **Per-target models (not multi-output)** — Mirrors `RidgeMultiTarget`. Each target gets its own tree ensemble that can learn different interaction patterns (rushing_floor cares about carries/YPC interactions; td_points cares about red zone/goalline interactions). XGBoost doesn't natively support multi-output regression, so this is also the simplest approach.

4. **Early stopping on validation set** — Uses the same train/val split as the NN. Stops adding trees when validation loss plateaus for 30 rounds, preventing overfitting.

5. **`save_model` / `load_model` with JSON** — XGBoost's native JSON format is human-readable and forward-compatible across XGBoost versions, unlike pickle.

6. **Clamping `>= 0`** — Same as Ridge. Fantasy sub-scores can't be negative (except DST `pts_allowed_bonus`, which would need the same `non_negative_targets` override as the NN if DST is added later).

## Config Additions

Add these keys to each position's config dict. Example for RB (`src/rb/config.py`):

```python
# === XGBoost ===
RB_XGB_N_ESTIMATORS = 500
RB_XGB_MAX_DEPTH = 6
RB_XGB_LEARNING_RATE = 0.05
RB_XGB_SUBSAMPLE = 0.8
RB_XGB_COLSAMPLE_BYTREE = 0.8
RB_XGB_REG_LAMBDA = 2.0        # L2 regularization on leaf weights
RB_XGB_REG_ALPHA = 0.1         # L1 regularization on leaf weights
RB_XGB_MIN_CHILD_WEIGHT = 5    # min samples per leaf (prevents overfitting on rare patterns)
```

### Recommended starting hyperparameters by position

| Parameter | RB | QB | WR | TE | K | DST |
|-----------|----|----|----|----|---|-----|
| `n_estimators` | 500 | 500 | 500 | 500 | 300 | 500 |
| `max_depth` | 6 | 5 | 6 | 5 | 4 | 5 |
| `learning_rate` | 0.05 | 0.03 | 0.05 | 0.03 | 0.05 | 0.03 |
| `subsample` | 0.8 | 0.8 | 0.8 | 0.75 | 0.8 | 0.75 |
| `colsample_bytree` | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| `reg_lambda` | 2.0 | 3.0 | 2.0 | 3.0 | 2.0 | 3.0 |
| `reg_alpha` | 0.1 | 0.1 | 0.1 | 0.2 | 0.1 | 0.2 |
| `min_child_weight` | 5 | 8 | 5 | 8 | 10 | 8 |

Rationale: QB, TE, and DST have fewer training samples (~3,000-3,500), so use shallower trees (`max_depth` 5), slower learning rate (0.03), and higher regularization (`reg_lambda` 3.0, `min_child_weight` 8). RB and WR have more data (~5,500-6,500) and can tolerate deeper trees. K has the smallest effective dataset (2015-2025 only) so uses the shallowest trees.

## Pipeline Integration

### New function: `_train_xgboost()` in `src/shared/pipeline.py`

```python
def _train_xgboost(X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
                   cfg, targets, seed):
    """Train XGBoost multi-target model. Returns (model, test_preds, metrics, importance).

    No scaling needed — tree models are invariant to monotonic transformations.
    """
    from shared.xgboost_model import XGBoostMultiTarget

    model = XGBoostMultiTarget(
        target_names=targets,
        n_estimators=cfg["xgb_n_estimators"],
        max_depth=cfg["xgb_max_depth"],
        learning_rate=cfg["xgb_learning_rate"],
        subsample=cfg["xgb_subsample"],
        colsample_bytree=cfg["xgb_colsample_bytree"],
        reg_lambda=cfg["xgb_reg_lambda"],
        reg_alpha=cfg["xgb_reg_alpha"],
        min_child_weight=cfg["xgb_min_child_weight"],
        seed=seed,
    )

    best_iters = model.fit(X_train, y_train_dict, X_val, y_val_dict)
    print(f"  XGBoost best iterations: {best_iters}")

    test_preds = model.predict(X_test)
    metrics = compute_target_metrics(y_test_dict, test_preds, targets)

    importance = model.get_feature_importance(
        feature_names=list(range(X_train.shape[1]))  # replaced with actual names in caller
    )

    return model, test_preds, metrics, importance
```

### Changes to `run_pipeline()` in `src/shared/pipeline.py`

Insert XGBoost training after the NN block (~line 370) and before the ensemble block (~line 400):

```python
    # --- XGBoost Multi-Target ---
    print("\n--- XGBoost Multi-Target ---")
    xgb_model, xgb_test_preds, xgb_metrics, xgb_importance = _train_xgboost(
        X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict,
        cfg, targets, seed,
    )
```

Update the ensemble block to include XGBoost:

```python
    # --- Ensemble: average Ridge + NN + XGBoost ---
    n_models = 3
    ensemble_preds = {}
    for t in targets:
        ensemble_preds[t] = (
            ridge_test_preds[t] + nn_test_preds[t] + xgb_test_preds[t]
        ) / n_models
    ensemble_preds["total"] = (
        ridge_test_preds["total"] + nn_test_preds["total"] + xgb_test_preds["total"]
    ) / n_models
```

Update the comparison table:

```python
    comparison = {
        "Season Average Baseline": baseline_metrics,
        f"{pos} Ridge Multi-Target": ridge_metrics,
        f"{pos} Multi-Head NN": nn_metrics,
        f"{pos} XGBoost Multi-Target": xgb_metrics,
        f"{pos} Ensemble (Ridge+NN+XGB)": ensemble_metrics,
    }
```

Update model saving:

```python
    # Save XGBoost models
    xgb_model.save(f"{output_dir}/models")
    # Creates: {position}/outputs/models/xgboost/{target_name}.json
```

Update ranking metrics and backtest:

```python
    pos_test["pred_xgb_total"] = xgb_test_preds["total"]
    xgb_ranking = compute_ranking_metrics(pos_test, pred_col="pred_xgb_total")
    print(f"XGBoost Top-12 Hit Rate: {xgb_ranking['season_avg_hit_rate']:.3f}")

    backtest_pred_columns = {
        "Season Avg": "pred_baseline",
        "Ridge": "pred_ridge_total",
        "Neural Net": "pred_nn_total",
        "XGBoost": "pred_xgb_total",
        "Ensemble": "pred_ensemble",
    }
```

### Changes to `_apply_position_models()` in `src/serving/app.py`

Add XGBoost inference after the NN block (~line 346):

```python
    # XGBoost predictions
    xgb_file = f"{model_dir}/xgboost/{targets[0]}.json"
    xgb_total = None
    xgb_preds = None
    if os.path.exists(xgb_file):
        from shared.xgboost_model import XGBoostMultiTarget
        xgb_model = XGBoostMultiTarget(target_names=targets)
        xgb_model.load(model_dir)
        xgb_preds = xgb_model.predict(X_test_pos)  # no scaling needed
        xgb_total = sum(xgb_preds[t] for t in targets) + adj.values

    # Write into results
    if xgb_total is not None:
        results.loc[pos_index, "xgb_pred"] = np.round(xgb_total, 2).astype(np.float32)
```

Add `"xgb_pred"` column and per-target MAE tracking alongside Ridge/NN.

### Changes to `POSITION_REGISTRY` in `src/serving/app.py`

Add XGBoost config to each position entry:

```python
    "QB": {
        ...existing keys...
        "xgb_kwargs": dict(
            n_estimators=QB_XGB_N_ESTIMATORS,
            max_depth=QB_XGB_MAX_DEPTH,
            ...
        ),
    },
```

## Model Save Structure

After training, each position directory gains an `xgboost/` subdirectory:

```
src/rb/outputs/models/
├── rushing_floor/
│   ├── ridge_model.pkl
│   └── scaler.pkl
├── receiving_floor/
│   ├── ridge_model.pkl
│   └── scaler.pkl
├── td_points/
│   ├── ridge_model.pkl
│   └── scaler.pkl
├── xgboost/                     # NEW
│   ├── rushing_floor.json       # XGBoost native JSON format
│   ├── receiving_floor.json
│   └── td_points.json
├── rb_multihead_nn.pt
├── nn_scaler.pkl
├── rb_weather_nn.pt
└── weather_nn_scaler.pkl
```

## Benchmark Integration

### Changes to `src/benchmarking/benchmark.py`

Update `summarize()` to include XGBoost metrics:

```python
def summarize(position, result):
    ridge = result["ridge_metrics"]["total"]
    nn = result["nn_metrics"]["total"]
    xgb = result["xgb_metrics"]["total"]
    summary = {
        ...existing keys...
        "xgb_mae":   round(xgb["mae"], 3),
        "xgb_r2":    round(xgb["r2"], 3),
        "xgb_per_target": {
            t: round(result["xgb_metrics"][t]["mae"], 3)
            for t in result["xgb_metrics"] if t != "total"
        },
        "xgb_top12": round(result["xgb_ranking"]["season_avg_hit_rate"], 3),
    }
    return summary
```

Update `print_table()` to add the XGBoost column.

### Updated return dict from `run_pipeline()`

```python
{
    "ridge_metrics": {...},
    "nn_metrics": {...},
    "xgb_metrics": {...},           # NEW
    "ridge_ranking": {...},
    "nn_ranking": {...},
    "xgb_ranking": {...},           # NEW
    "history": {...},
    "sim_results": {...},
    "weather_nn_metrics": {...},
    "weather_nn_ranking": {...},
    "xgb_feature_importance": {...}, # NEW — gain-based importance per target
}
```

## Ensemble Weight Discussion

The initial implementation uses equal weights (1/3 each for Ridge, NN, XGBoost). This is the safest default — it requires no additional tuning and equal-weight ensembles are surprisingly hard to beat.

If XGBoost consistently outperforms on certain targets (likely `td_points`, where tree splits on red-zone features can capture threshold effects), a follow-up could tune per-target ensemble weights on the validation set. But start with equal weights and measure first.

## Dependency

Add `xgboost` to requirements:

```
pip install xgboost
```

XGBoost has no dependency conflicts with the existing stack (numpy, pandas, scikit-learn, torch).

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/shared/xgboost_model.py` | **New** | `XGBoostMultiTarget` class |
| `src/shared/pipeline.py` | **Modify** | Add `_train_xgboost()`, update ensemble, update comparison/ranking/saving |
| `{POS}/{pos}_config.py` | **Modify** | Add 8 XGBoost hyperparameters per position (6 files) |
| `src/serving/app.py` | **Modify** | Load and run XGBoost at inference, add `xgb_pred` column |
| `src/benchmarking/benchmark.py` | **Modify** | Track XGBoost metrics in summary and print table |
| `requirements.txt` | **Modify** | Add `xgboost` dependency |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting on small datasets (QB/TE ~3K samples) | Conservative defaults: `max_depth=5`, `min_child_weight=8`, `reg_lambda=3.0`, early stopping with patience=30 |
| Correlated errors with Ridge (both use raw features) | Trees learn interactions Ridge can't; subsample + colsample add stochasticity. Empirically, tree/linear correlations are lower than NN/linear. |
| Training time increase | XGBoost with 500 trees on 5K samples trains in <5 seconds per target. Negligible vs. NN training (30-60s). |
| `xgboost` dependency not installed | Import inside `_train_xgboost()` so the rest of the pipeline still works without it. Clear error message if missing. |
| DST `pts_allowed_bonus` can be negative | Same issue as NN — don't clamp that target. Add a `non_negative_targets` parameter to `XGBoostMultiTarget` mirroring the NN approach. |

## Verification Plan

1. **Unit test `XGBoostMultiTarget`**: Verify `fit()`, `predict()`, `save()`, `load()` round-trip produces identical predictions. Verify predictions are clamped >= 0.
2. **Train on RB first** (most data, fastest iteration): Compare XGBoost-alone MAE vs. Ridge MAE vs. NN MAE.
3. **Compare 2-way vs. 3-way ensemble**: Does adding XGBoost to the Ridge+NN ensemble reduce total MAE? Check both total and per-target.
4. **Inspect feature importance**: XGBoost's gain-based importance should highlight different features than Ridge's coefficient magnitudes. If they're identical, the ensemble isn't gaining diversity.
5. **Check per-target improvements**: XGBoost should especially help `td_points` (discrete, threshold-driven) and targets where feature interactions matter (e.g., carry share x matchup quality).
6. **Run full benchmark** across all positions: `python -m src.benchmarking.benchmark RB QB WR TE --note "added XGBoost"`. Compare against prior benchmark results under `benchmark_history/`.
7. **Validate app inference**: Run `python -m src.serving.app`, hit the API, confirm `xgb_pred` column appears with reasonable values.

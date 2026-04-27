"""Shared pytest config for tests/.

Provides the session-scoped Flask/API fixtures used by the API contract suite
(Unit 8) and cross-position pipeline E2E/reproducibility suites (Unit 10).

The strategy doc (swift-roaming-bumblebee) describes a `/predict_json` endpoint
as an aspirational API surface. The current app.py exposes a set of read-only
`/api/*` GET endpoints that lazily build cached predictions from on-disk
parquet + trained model artifacts. These fixtures codify the *current* contract
while keeping the `tiny_qb_model` scaffold ready for when `/predict_json`
lands — new tests can consume it without touching conftest.

Project-root sys.path wiring and pytest-marker registration live in the root
``conftest.py`` so this file doesn't duplicate them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic in-memory results — exercised via monkeypatching app._cache so we
# don't have to load the real trained models or parquet splits in CI.
# ---------------------------------------------------------------------------
def _synthetic_results(seed: int = 42, n_per_position: int = 4) -> pd.DataFrame:
    """Build a minimal results DataFrame matching the shape app.py's cache uses.

    Columns mirror what `_get_data()` produces: player identifiers, weekly
    actuals, scoring format breakouts, and model predictions.
    """
    rng = np.random.default_rng(seed)
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    rows = []
    for pos in positions:
        for i in range(n_per_position):
            for week in (1, 2, 3, 4, 5, 6, 7):
                actual = float(rng.uniform(5, 30))
                # Attention NN and LightGBM aren't trained for K/DST, so those
                # rows mirror production by leaving the cells NaN.
                attn_pred = float(actual + rng.normal(0, 2)) if pos not in ("K", "DST") else np.nan
                lgbm_pred = float(actual + rng.normal(0, 2)) if pos not in ("K", "DST") else np.nan
                rows.append(
                    {
                        "player_id": f"{pos}{i:03d}",
                        "player_display_name": f"{pos} Player {i}",
                        "position": pos,
                        "recent_team": "KC",
                        "season": 2025,
                        "week": week,
                        "headshot_url": "",
                        "fantasy_points": actual,
                        "fantasy_points_standard": actual * 0.9,
                        "fantasy_points_half_ppr": actual * 0.95,
                        "ridge_pred": float(actual + rng.normal(0, 2)),
                        "nn_pred": float(actual + rng.normal(0, 2)),
                        "attn_nn_pred": attn_pred,
                        "lgbm_pred": lgbm_pred,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_metrics() -> dict:
    """Metrics payload matching the shape `_get_data()` returns."""
    return {
        "Ridge Regression": {
            "overall": {"mae": 4.23, "rmse": 6.1, "r2": 0.45},
            "by_position": [
                {"position": "QB", "mae": 5.1, "rmse": 7.0, "n": 100},
                {"position": "RB", "mae": 4.2, "rmse": 5.8, "n": 200},
            ],
        },
        "Neural Network": {
            "overall": {"mae": 4.05, "rmse": 5.9, "r2": 0.48},
            "by_position": [
                {"position": "QB", "mae": 4.9, "rmse": 6.8, "n": 100},
                {"position": "RB", "mae": 4.0, "rmse": 5.6, "n": 200},
            ],
        },
        "Attention NN": {
            "overall": {"mae": 3.95, "rmse": 5.75, "r2": 0.50},
            "by_position": [
                {"position": "QB", "mae": 4.8, "rmse": 6.7, "n": 100},
                {"position": "RB", "mae": 3.9, "rmse": 5.5, "n": 200},
            ],
        },
        "LightGBM": {
            "overall": {"mae": 4.00, "rmse": 5.80, "r2": 0.49},
            "by_position": [
                {"position": "QB", "mae": 4.85, "rmse": 6.75, "n": 100},
                {"position": "RB", "mae": 3.95, "rmse": 5.55, "n": 200},
            ],
        },
    }


@pytest.fixture
def synthetic_cache():
    """Return the dict that can be spliced into `app._cache` for tests."""
    return {
        "results": _synthetic_results(),
        "metrics": _synthetic_metrics(),
        "position_details": {
            pos: {
                "n_features": 42,
                "n_samples_test": 100,
                "target_metrics": {"total": {"ridge_mae": 5.0, "nn_mae": 4.8}},
            }
            for pos in ["QB", "RB", "WR", "TE", "K", "DST"]
        },
    }


@pytest.fixture
def app_module(monkeypatch):
    """Import app.py with a clean `_cache` per test.

    monkeypatch restores the original `_cache` attribute at teardown, so
    cross-test contamination through the module-global cache is prevented.
    """
    import src.serving.app as app_mod

    monkeypatch.setattr(app_mod, "_cache", {})
    return app_mod


@pytest.fixture
def client(app_module):
    """Flask test client over a freshly-cached `app`."""
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def client_with_data(app_module, synthetic_cache):
    """Flask test client with synthetic data pre-loaded into `_cache`.

    Most `/api/*` endpoints call `_get_data()`, which reads parquet files and
    loads trained models. Pre-populating the cache short-circuits that load so
    tests don't depend on on-disk artifacts.
    """
    app_module._cache.update(synthetic_cache)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Tiny QB model artifact — exercised for future /predict_json tests and for
# the graceful-degradation test that monkeypatches joblib.load to raise.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_qb_model(tmp_path_factory):
    """Session-scoped: train a tiny Ridge model on 50-row synthetic QB data.

    Writes joblib artifacts + scaler + feature-column JSON to a tmp dir shaped
    the way app.py's `_apply_position_models` expects
    (`{model_dir}/{target}/ridge_model.pkl`, `{model_dir}/nn_scaler.pkl`,
    `{model_dir}/qb_multihead_nn.pt`).

    The fixture trains deterministically from seed=42. It is session-scoped so
    the 50-row fit runs once per test session, not per-test.
    """
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    model_dir = tmp_path_factory.mktemp("QB_tiny_models")
    rng = np.random.default_rng(42)

    # Synthetic 50-row QB training data (features arbitrary — purpose is shape)
    n, n_features = 50, 8
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    targets = {
        "passing_yards": rng.uniform(150, 400, size=n),
        "rushing_yards": rng.uniform(0, 60, size=n),
        "passing_tds": rng.uniform(0, 4, size=n),
        "rushing_tds": rng.uniform(0, 2, size=n),
        "interceptions": rng.uniform(0, 3, size=n),
        "fumbles_lost": rng.uniform(0, 2, size=n),
    }

    # Per-target ridge models (matches RidgeMultiTarget layout)
    for target, y in targets.items():
        target_dir = model_dir / target
        target_dir.mkdir(exist_ok=True)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        # Mirror src/models/linear.py RidgeModel save format (scaler + model)
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, str(target_dir / "scaler.pkl"))
        joblib.dump(ridge, str(target_dir / "ridge_model.pkl"))

    # Scaler for the NN head
    nn_scaler = StandardScaler()
    nn_scaler.fit(X)
    joblib.dump(nn_scaler, str(model_dir / "nn_scaler.pkl"))

    # Feature-column manifest — mirrors get_qb_feature_columns() shape
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols))

    return model_dir


@pytest.fixture
def valid_qb_payload():
    """Minimal valid POST body for a hypothetical /predict_json QB request.

    Shaped per the reference strategy doc (swift-roaming-bumblebee). Retained
    for forward-compatibility so `/predict_json` tests can consume it directly
    once the endpoint is implemented.
    """
    return {
        "players": [
            {
                "player_id": "00-0034796",
                "position": "QB",
                "week": 5,
                "season": 2024,
                "scoring_format": "HALF_PPR",
            }
        ]
    }

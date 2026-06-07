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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic schedule data — only patched in when the real parquet is absent.
#
# ``src/shared/weather_features._load_schedules`` reads
# ``data/raw/schedules_2012_2025.parquet`` (a 28MB file fetched at CI runtime
# via ``nfl_data_py``). Local checkouts and worktrees that haven't pulled the
# raw data folder don't have it on disk, so the QB/RB/TE/WR e2e + CV pipeline
# tests crash at the schedule-merge step.
#
# This fixture detects the missing file and monkeypatches ``_load_schedules``
# (plus the module-level ``_schedule_cache``) to return a synthetic schedule
# carrying the columns ``_build_team_schedule_lookup`` reads. When CI runs
# and the real parquet exists, the fixture is a no-op — CI behaviour is
# unchanged.
# ---------------------------------------------------------------------------

# Mirrors the team set on the real parquet so synthetic schedules can serve
# every position's e2e/CV test regardless of which TEAMS subset it picks.
_NFL_TEAMS: tuple[str, ...] = (
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",
    "LAC",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
)


def _build_synthetic_schedules() -> pd.DataFrame:
    """Build a minimal full-season schedule frame for tests.

    Covers every (season, week, team) combination 2012-2025 weeks 1-18 with
    REG game_type, so `_build_team_schedule_lookup` always finds a match for
    any synthetic player frame the e2e/CV tests generate.

    Pairs teams sequentially into games (16 games per week with 32 teams).
    Carries the schedule columns read by ``_build_team_schedule_lookup``
    (weather/Vegas merge) plus ``home_score``/``away_score`` for
    ``_build_defense_matchup_features`` (points-allowed L5). Constant values
    sit well inside the real distribution (spread ~0, total ~45, dome=false,
    grass surface, 65F/5mph, 7 days rest, non-divisional, 21-point games).
    """
    rows = []
    teams = list(_NFL_TEAMS)
    for season in range(2012, 2026):
        for week in range(1, 19):  # NFL added week 18 in 2021; harmless for earlier seasons
            # Pair teams (0,1), (2,3), ..., (30,31). Rotate the pairing per
            # week so every team faces several opponents — the merge only
            # cares about the (season, week, recent_team) key, but realistic
            # rotation keeps the synthetic data closer to true distribution.
            offset = week % len(teams)
            rotated = teams[offset:] + teams[:offset]
            for i in range(0, len(rotated), 2):
                home = rotated[i]
                away = rotated[i + 1]
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "home_team": home,
                        "away_team": away,
                        "home_score": 21,
                        "away_score": 21,
                        "spread_line": 0.0,
                        "total_line": 45.0,
                        "roof": "outdoors",
                        "surface": "grass",
                        "temp": 65.0,
                        "wind": 5.0,
                        "home_rest": 7,
                        "away_rest": 7,
                        "div_game": 0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session", autouse=True)
def _patch_schedules_if_missing():
    """Session-scoped autouse: stub ``_load_schedules`` when the parquet
    isn't on disk.

    Only fires when ``data/raw/schedules_2012_2025.parquet`` is absent —
    CI ships with the real parquet and the fixture short-circuits, so the
    production code path is unchanged on the CI runner.

    Tests that already patch ``_load_schedules`` themselves (e.g. the unit
    suite in ``tests/shared/test_weather_features.py``, the DST E2E
    fixtures) layer their per-function or per-module patches on top of this
    session patch — ``pytest.MonkeyPatch`` + ``unittest.mock.patch`` both
    snapshot+restore correctly, so no cross-test leakage.
    """
    project_root = Path(__file__).resolve().parents[1]
    parquet_path = project_root / "data" / "raw" / "schedules_2012_2025.parquet"
    if parquet_path.exists():
        # Real data is on disk — let the production loader run.
        yield
        return

    synthetic = _build_synthetic_schedules()
    mp = pytest.MonkeyPatch()
    from src.shared import weather_features as _wf

    mp.setattr(_wf, "_schedule_cache", synthetic)
    mp.setattr(_wf, "_load_schedules", lambda: synthetic)
    try:
        yield
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# Synthetic in-memory results — exercised via monkeypatching app._cache so we
# don't have to load the real trained models or parquet splits in CI.
# ---------------------------------------------------------------------------
def _synthetic_results(seed: int = 42, n_per_position: int = 4) -> pd.DataFrame:
    """Build a minimal results DataFrame matching the shape app.py's cache uses.

    Columns mirror what `_get_data()` produces: player identifiers, weekly
    actuals, scoring format breakouts (PPR / half / standard for both actuals
    and per-model predictions), and the legacy unsuffixed prediction columns
    that older tests still read.
    """
    rng = np.random.default_rng(seed)
    positions = ["QB", "RB", "WR", "TE", "K", "DST"]
    # Mirror the actuals' format multipliers so tests can predict expected
    # values for the format-aware endpoints.
    fmt_multipliers = {"ppr": 1.0, "half_ppr": 0.95, "standard": 0.9}
    # Per-target raw-stat columns for the breakdown endpoint. Sourced from app so
    # the fixture tracks the real schema (POSITION_INFO targets + the union the
    # results frame pre-declares).
    from src.serving.app import _ALL_TARGETS, _MODEL_PRED_PREFIXES, POSITION_INFO

    pos_target_keys = {p: [t["key"] for t in POSITION_INFO[p]["targets"]] for p in positions}
    rows = []
    for pos in positions:
        for i in range(n_per_position):
            for week in (1, 2, 3, 4, 5, 6, 7):
                actual = float(rng.uniform(5, 30))
                base_ridge = float(actual + rng.normal(0, 2))
                base_nn = float(actual + rng.normal(0, 2))
                # Attention NN and LightGBM aren't trained for K/DST, so those
                # rows mirror production by leaving the cells NaN.
                base_attn = float(actual + rng.normal(0, 2)) if pos not in ("K", "DST") else np.nan
                base_lgbm = float(actual + rng.normal(0, 2)) if pos not in ("K", "DST") else np.nan
                row = {
                    "player_id": f"{pos}{i:03d}",
                    "player_display_name": f"{pos} Player {i}",
                    "position": pos,
                    "recent_team": "KC",
                    "season": 2025,
                    "week": week,
                    "headshot_url": "",
                    "fantasy_points": actual,
                    "fantasy_points_standard": actual * fmt_multipliers["standard"],
                    "fantasy_points_half_ppr": actual * fmt_multipliers["half_ppr"],
                    # Legacy unsuffixed pred columns kept as PPR aliases.
                    "ridge_pred": base_ridge,
                    "nn_pred": base_nn,
                    "attn_nn_pred": base_attn,
                    "lgbm_pred": base_lgbm,
                }
                # Per-format pred columns. NaN preds (K/DST attn/lgbm) stay NaN
                # across all three formats — multiplying by a constant preserves
                # NaN under numpy / float arithmetic.
                for fmt, m in fmt_multipliers.items():
                    row[f"ridge_pred_{fmt}"] = base_ridge * m
                    row[f"nn_pred_{fmt}"] = base_nn * m
                    row[f"attn_nn_pred_{fmt}"] = (
                        base_attn * m if not np.isnan(base_attn) else np.nan
                    )
                    row[f"lgbm_pred_{fmt}"] = base_lgbm * m if not np.isnan(base_lgbm) else np.nan
                # Per-target raw-stat columns: NaN everywhere, then fill this
                # position's own targets (sparse, mirrors _load_base_data_locked).
                # lgbm stays NaN for K/DST (no LightGBM trained there).
                for t in _ALL_TARGETS:
                    row[f"actual_{t}"] = np.nan
                    for prefix in _MODEL_PRED_PREFIXES:
                        row[f"pred_{prefix}_{t}"] = np.nan
                for t in pos_target_keys[pos]:
                    stat = float(rng.uniform(0, 50))
                    row[f"actual_{t}"] = stat
                    row[f"pred_ridge_{t}"] = stat + float(rng.normal(0, 2))
                    row[f"pred_nn_{t}"] = stat + float(rng.normal(0, 2))
                    if pos not in ("K", "DST"):
                        row[f"pred_attn_nn_{t}"] = stat + float(rng.normal(0, 2))
                        row[f"pred_lgbm_{t}"] = stat + float(rng.normal(0, 2))
                    else:
                        # attn_nn IS trained for K/DST in production; keep it real
                        # so the breakdown shows attn_nn for these positions.
                        row[f"pred_attn_nn_{t}"] = stat + float(rng.normal(0, 2))
                rows.append(row)
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
    metrics = _synthetic_metrics()
    # Per-format metrics: each format scales actuals by a different multiplier
    # so the endpoints can verify they pick up the right cache slot.
    metrics_by_format = {
        "ppr": metrics,
        "half_ppr": _scale_metrics(metrics, 0.95),
        "standard": _scale_metrics(metrics, 0.9),
    }
    return {
        "results": _synthetic_results(),
        "metrics": metrics,
        "metrics_by_format": metrics_by_format,
        "position_details": {
            pos: {
                "n_features": 42,
                "n_samples_test": 100,
                "target_metrics": {
                    "total": {"ridge_mae": 5.0, "nn_mae": 4.8},
                    "total_by_format": {
                        "ppr": {"ridge_mae": 5.0, "nn_mae": 4.8},
                        "half_ppr": {"ridge_mae": 4.75, "nn_mae": 4.56},
                        "standard": {"ridge_mae": 4.5, "nn_mae": 4.32},
                    },
                },
            }
            for pos in ["QB", "RB", "WR", "TE", "K", "DST"]
        },
    }


def _scale_metrics(metrics: dict, multiplier: float) -> dict:
    """Return a copy of metrics with overall + by_position MAE/RMSE scaled."""
    scaled = {}
    for model, m in metrics.items():
        overall = m.get("overall")
        scaled_overall = (
            {k: round(v * multiplier, 4) if k in ("mae", "rmse") else v for k, v in overall.items()}
            if overall
            else None
        )
        scaled_by_pos = [
            {k: (round(v * multiplier, 4) if k in ("mae", "rmse") else v) for k, v in row.items()}
            for row in m.get("by_position", [])
        ]
        scaled[model] = {"overall": scaled_overall, "by_position": scaled_by_pos}
    return scaled


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """Import app.py with a clean `_cache` per test.

    monkeypatch restores the original `_cache` attribute at teardown, so
    cross-test contamination through the module-global cache is prevented.

    Also redirects ``_PREDICTIONS_CACHE_DIR`` to a per-test ``tmp_path``
    subdir so the serving disk cache (written by ``_compute_metrics_locked``
    on the production path) can't leak into the repo's ``data/serving_cache/``
    and accidentally hydrate later tests.
    """
    import src.serving.app as app_mod
    import src.serving.core as core

    monkeypatch.setattr(app_mod, "_cache", {})
    monkeypatch.setattr(core, "_PREDICTIONS_CACHE_DIR", str(tmp_path / "serving_cache"))
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
        # Mirror src/shared/models.py RidgeModel save format (scaler + model)
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, str(target_dir / "scaler.pkl"))
        joblib.dump(ridge, str(target_dir / "ridge_model.pkl"))

    # Scaler for the NN head
    nn_scaler = StandardScaler()
    nn_scaler.fit(X)
    joblib.dump(nn_scaler, str(model_dir / "nn_scaler.pkl"))

    # Feature-column manifest — mirrors get_feature_columns() shape
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

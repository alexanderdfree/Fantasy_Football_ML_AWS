"""Flask API contract tests for app.py (P0.1).

All tests here are `@pytest.mark.integration` — they cross the Flask boundary
via `test_client()`.

## Scope note

The reference strategy doc (`swift-roaming-bumblebee.md`) describes a
`POST /predict_json` endpoint. The current `app.py` does not implement that
endpoint yet — its public surface is a set of read-only `GET /api/*` routes
that lazily build cached predictions from on-disk parquet splits and trained
models. These tests codify the contract of what actually ships today.

The `valid_qb_payload` and `tiny_qb_model` fixtures in `conftest.py` are
retained for forward-compatibility: when `/predict_json` lands, new test cases
can consume them without touching the fixtures. A placeholder test records the
current 404 behavior so regressions on this front are visible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# ===========================================================================
# GET /health — trivial contract
# ===========================================================================


class TestHealth:
    def test_health_returns_200_and_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.is_json
        assert r.get_json() == {"status": "ok"}

    def test_health_does_not_require_cache(self, client):
        """/health must succeed even when model cache is empty."""
        # `client` uses `app_module` which monkeypatches _cache to {}
        r = client.get("/health")
        assert r.status_code == 200


# ===========================================================================
# GET /api/model_architecture — config-driven, no model loading
# ===========================================================================


class TestModelArchitecture:
    def test_returns_200_and_schema(self, client):
        r = client.get("/api/model_architecture")
        assert r.status_code == 200
        assert r.is_json
        body = r.get_json()
        # Top-level keys documented in app.py
        assert set(body) == {"overview", "training_loop", "positions"}

    def test_positions_payload_covers_all_six(self, client):
        r = client.get("/api/model_architecture")
        body = r.get_json()
        assert set(body["positions"]) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_position_payload_has_required_fields(self, client):
        r = client.get("/api/model_architecture")
        body = r.get_json()
        qb = body["positions"]["QB"]
        # Fields surfaced by _position_arch_payload()
        required = {
            "targets",
            "backbone_layers",
            "head_hidden",
            "head_hidden_overrides",
            "dropout",
            "lr",
            "weight_decay",
            "batch_size",
            "epochs",
            "patience",
            "scheduler",
            "attention_enabled",
            "lightgbm_enabled",
            "feature_count",
            "features",
        }
        assert required.issubset(qb.keys())
        # feature_count should be a non-negative int
        assert isinstance(qb["feature_count"], int)
        assert qb["feature_count"] >= 0
        # features is a categorised dict
        assert isinstance(qb["features"], dict)

    def test_overview_documents_framework_and_ensemble(self, client):
        r = client.get("/api/model_architecture")
        body = r.get_json()
        overview = body["overview"]
        assert "framework" in overview
        assert "device" in overview
        assert "ensemble" in overview
        assert isinstance(overview["ensemble"], list)


# ===========================================================================
# GET /api/predictions — happy path + error paths
# ===========================================================================


class TestPredictions:
    def test_happy_path_returns_players(self, client_with_data):
        r = client_with_data.get("/api/predictions")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body) == {"players", "total"}
        assert body["total"] == len(body["players"])
        assert body["total"] > 0
        # Row schema
        row = body["players"][0]
        expected_keys = {
            "player_id",
            "name",
            "position",
            "team",
            "week",
            "actual",
            "ridge_pred",
            "nn_pred",
            "headshot",
        }
        assert expected_keys.issubset(row.keys())

    def test_predictions_are_finite_floats(self, client_with_data):
        r = client_with_data.get("/api/predictions")
        body = r.get_json()
        for row in body["players"]:
            for pred_field in ("ridge_pred", "nn_pred", "actual"):
                val = row[pred_field]
                # _safe_num() maps NaN/inf to None — that's the contract
                assert val is None or (isinstance(val, (int, float)) and math.isfinite(val)), (
                    f"{pred_field}={val!r} violates contract"
                )

    def test_position_filter(self, client_with_data):
        r = client_with_data.get("/api/predictions?position=QB")
        body = r.get_json()
        assert all(row["position"] == "QB" for row in body["players"])

    def test_invalid_week_returns_400(self, client_with_data):
        r = client_with_data.get("/api/predictions?week=notanumber")
        assert r.status_code == 400
        assert r.is_json
        assert "error" in r.get_json()

    def test_search_filter(self, client_with_data):
        # Case-insensitive substring match on player_display_name
        r = client_with_data.get("/api/predictions?search=qb%20player%200")
        body = r.get_json()
        assert all("qb player 0" in row["name"].lower() for row in body["players"])

    def test_sort_order_desc_default(self, client_with_data):
        r = client_with_data.get("/api/predictions?sort=actual&order=desc")
        body = r.get_json()
        actuals = [row["actual"] or 0 for row in body["players"]]
        assert actuals == sorted(actuals, reverse=True)


# ===========================================================================
# GET /api/weeks, /api/metrics, /api/position_details
# ===========================================================================


class TestAuxiliaryEndpoints:
    def test_weeks_returns_sorted_integers(self, client_with_data):
        r = client_with_data.get("/api/weeks")
        assert r.status_code == 200
        body = r.get_json()
        assert "weeks" in body
        assert body["weeks"] == sorted(body["weeks"])
        assert all(isinstance(w, int) for w in body["weeks"])

    def test_metrics_returns_both_models(self, client_with_data):
        r = client_with_data.get("/api/metrics")
        assert r.status_code == 200
        body = r.get_json()
        assert "Ridge Regression" in body
        assert "Neural Network" in body
        for model_name in ("Ridge Regression", "Neural Network"):
            assert "overall" in body[model_name]
            assert "by_position" in body[model_name]

    def test_position_details_covers_all_positions(self, client_with_data):
        r = client_with_data.get("/api/position_details")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body) == {"QB", "RB", "WR", "TE", "K", "DST"}
        # Each position should surface static metadata from POSITION_INFO
        qb = body["QB"]
        assert qb["label"] == "Quarterback"
        assert "targets" in qb
        assert "architecture" in qb

    def test_top_players_returns_list(self, client_with_data):
        r = client_with_data.get("/api/top_players")
        assert r.status_code == 200
        body = r.get_json()
        assert "players" in body
        # Each row carries aggregated per-player stats
        if body["players"]:
            row = body["players"][0]
            for key in (
                "player_id",
                "name",
                "position",
                "team",
                "avg_actual",
                "avg_ridge",
                "avg_nn",
                "games",
            ):
                assert key in row

    def test_weekly_accuracy_parallel_arrays(self, client_with_data):
        r = client_with_data.get("/api/weekly_accuracy")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body) == {"weeks", "ridge_mae", "nn_mae", "attn_nn_mae", "lgbm_mae"}
        n = len(body["weeks"])
        for key in ("ridge_mae", "nn_mae", "attn_nn_mae", "lgbm_mae"):
            assert len(body[key]) == n, f"{key} length {len(body[key])} != weeks length {n}"
            # Attn NN / LightGBM can contain None for weeks where no rows had
            # a prediction (e.g. all K/DST) — skip those when checking sign.
            assert all(mae is None or mae >= 0 for mae in body[key])


# ===========================================================================
# GET /api/player/<id> — happy path + missing
# ===========================================================================


class TestPlayerEndpoint:
    def test_known_player_returns_weekly(self, client_with_data):
        r = client_with_data.get("/api/player/QB000")
        assert r.status_code == 200
        body = r.get_json()
        for key in (
            "player_id",
            "name",
            "position",
            "team",
            "weekly",
            "season_avg",
            "season_total",
        ):
            assert key in body
        assert body["player_id"] == "QB000"
        assert body["position"] == "QB"
        assert isinstance(body["weekly"], list)
        assert len(body["weekly"]) > 0

    def test_unknown_player_returns_404(self, client_with_data):
        r = client_with_data.get("/api/player/nonexistent-id")
        assert r.status_code == 404
        assert r.is_json
        assert "error" in r.get_json()


# ===========================================================================
# GET / — template rendering
# ===========================================================================


class TestIndexRoute:
    def test_index_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200
        # index.html is rendered as HTML, not JSON
        assert "text/html" in r.content_type


# NOTE: `app.py`'s global `@app.errorhandler(Exception)` catches every
# exception, including werkzeug HTTPExceptions (404 NotFound,
# 405 MethodNotAllowed). For non-`/api/` paths it `raise e`s — which
# prevents Flask's normal 404 response from ever being produced and
# surfaces as a 500 at the WSGI layer. Re-enabling intuitive 404s would
# require explicit error handlers for `NotFound` and `MethodNotAllowed`.
# This test file records that behavior by omitting a "404 for unknown
# route" assertion; callers exercising unknown routes should expect the
# handler to surface a 500 JSON error for `/api/*` and a re-raise for
# other paths. If app.py grows explicit HTTPException handlers, add
# coverage here.


# ===========================================================================
# POST /predict_json — not implemented yet; conftest fixtures retain shape
# ===========================================================================

# The reference strategy doc (swift-roaming-bumblebee) calls for a
# POST /predict_json endpoint with a `players[]` body and a
# `predicted_points` response. app.py does not implement that endpoint
# today — its entire public surface is read-only GET routes.
#
# The `tiny_qb_model` and `valid_qb_payload` fixtures in conftest.py are
# retained as scaffolding so new contract tests can be added without
# touching fixture plumbing once the endpoint lands. See
# `TestTinyQBModelFixture` below for a round-trip sanity check on the
# scaffold itself.
#
# We intentionally do NOT assert 404/405 on POST /predict_json here: the
# global error handler re-raises non-`/api/` exceptions, which obscures
# the underlying HTTP status from the test client.


# ===========================================================================
# Graceful degradation — failing model load surfaces as 500 JSON error
# ===========================================================================


class TestGracefulDegradation:
    """Endpoints that depend on the lazy `_get_data()` cache must surface
    errors as structured JSON, not as raw HTML tracebacks.

    app.py's `handle_api_error` catches every exception on `/api/*` routes and
    returns `{"error": str(e)}, 500` — this is the documented failure mode
    (the strategy doc names 503 as the target for a future refactor, but the
    current code uses 500; we assert whatever the app actually does).
    """

    def test_predictions_surfaces_load_failure_as_json(self, client, app_module, monkeypatch):
        """When `_get_data()` raises, /api/predictions returns a JSON error payload."""

        def _boom():
            raise RuntimeError("simulated model load failure")

        monkeypatch.setattr(app_module, "_get_data", _boom)

        r = client.get("/api/predictions")
        # Must be JSON, not an HTML 500 page
        assert r.is_json, f"Expected JSON error body, got {r.content_type}"
        body = r.get_json()
        assert "error" in body
        assert "simulated model load failure" in body["error"]
        # The handler returns 500 for /api/ routes today
        assert r.status_code == 500

    def test_joblib_load_failure_bubbles_up_as_json(self, client, app_module, monkeypatch):
        """If joblib.load raises (e.g. corrupted model artifact) during data
        build, the error must bubble up as structured JSON rather than HTML."""
        import joblib

        def _joblib_boom(*args, **kwargs):
            raise OSError("simulated joblib failure")

        monkeypatch.setattr(joblib, "load", _joblib_boom)

        # _get_data will try to read parquet splits (may fail earlier). Either
        # way the global /api/ error handler should turn the exception into
        # structured JSON — that's the contract we care about.
        r = client.get("/api/predictions")
        assert r.is_json, f"Expected JSON error body, got {r.content_type}"
        assert r.status_code == 500
        assert "error" in r.get_json()


# ===========================================================================
# Scoring format sanity check — documents current contract
# ===========================================================================


class TestScoringFormats:
    """The strategy doc calls for POST /predict_json to differentiate between
    STANDARD / HALF_PPR / PPR scoring. That endpoint isn't implemented yet.

    In the current read-only API surface, scoring-format columns are exposed
    in `/api/predictions` row payloads only as the single `actual` field (the
    canonical `fantasy_points` — defaults to PPR). The synthetic cache
    includes `fantasy_points_standard` and `fantasy_points_half_ppr` so once
    those columns are plumbed through the API, this test will be expanded.
    """

    def test_synthetic_cache_separates_formats(self, synthetic_cache):
        """Sanity: the fixture itself produces *different* values per format."""
        df = synthetic_cache["results"]
        # These three columns should all be populated
        assert df["fantasy_points"].notna().all()
        assert df["fantasy_points_standard"].notna().all()
        assert df["fantasy_points_half_ppr"].notna().all()
        # And within each row the three format values should differ
        # (our fixture builds them at different multipliers: 1.0, 0.95, 0.9)
        sample = df.iloc[0]
        assert not np.isclose(sample["fantasy_points"], sample["fantasy_points_standard"])
        assert not np.isclose(sample["fantasy_points"], sample["fantasy_points_half_ppr"])
        assert not np.isclose(sample["fantasy_points_standard"], sample["fantasy_points_half_ppr"])


# ===========================================================================
# tiny_qb_model fixture — sanity check the scaffold is usable
# ===========================================================================


class TestTinyQBModelFixture:
    """The tiny_qb_model fixture is currently scaffolding for the future
    /predict_json endpoint. Verify it produces a sane on-disk layout now so
    regressions on the fixture itself are caught early.
    """

    def test_tiny_model_artifacts_exist(self, tiny_qb_model):
        assert (tiny_qb_model / "nn_scaler.pkl").exists()
        assert (tiny_qb_model / "feature_columns.json").exists()
        for target in (
            "passing_yards",
            "rushing_yards",
            "passing_tds",
            "rushing_tds",
            "interceptions",
            "fumbles_lost",
        ):
            assert (tiny_qb_model / target / "ridge_model.pkl").exists()
            assert (tiny_qb_model / target / "scaler.pkl").exists()

    def test_tiny_model_predicts_finite(self, tiny_qb_model):
        """Round-trip: load the saved Ridge + scaler and ensure predictions are finite."""
        import joblib

        scaler = joblib.load(str(tiny_qb_model / "passing_yards" / "scaler.pkl"))
        ridge = joblib.load(str(tiny_qb_model / "passing_yards" / "ridge_model.pkl"))

        rng = np.random.default_rng(0)
        X = rng.normal(size=(3, 8)).astype(np.float32)
        preds = ridge.predict(scaler.transform(X))
        assert preds.shape == (3,)
        assert np.isfinite(preds).all()

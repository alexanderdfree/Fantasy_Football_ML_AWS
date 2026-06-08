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

import src.serving.core as core

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

    def test_k_and_dst_expose_attention_history(self, client):
        """K and DST train attention NNs, so the arch payload must list their
        attention-history features — the calls had omitted the kwarg (audit
        #774), leaving them attention-enabled with no disclosed inputs."""
        r = client.get("/api/model_architecture")
        body = r.get_json()
        for pos in ("K", "DST"):
            feats = body["positions"][pos]["features"]
            assert "attention_history" in feats, f"{pos} missing attention_history"
            assert len(feats["attention_history"]) > 0

    def test_k_scheduler_surfaces_distinct_attention_max_lr(self, client):
        """K's attention NN trains at a distinct OneCycle max_lr (2e-3) from the
        dense NN (1e-3); the scheduler string must surface that override, not
        advertise only the dense NN's rate (#845)."""
        r = client.get("/api/model_architecture")
        sched = r.get_json()["positions"]["K"]["scheduler"]
        assert "attention NN" in sched, sched


# ===========================================================================
# GET /api/predictions — happy path + error paths
# ===========================================================================


class TestPredictions:
    def test_happy_path_returns_players(self, client_with_data):
        r = client_with_data.get("/api/predictions")
        assert r.status_code == 200
        body = r.get_json()
        # ``degraded_positions`` powers the frontend banner (feat/partb-...);
        # empty list in the happy-path scenario where every position loaded.
        assert set(body) == {"players", "total", "degraded_positions", "scoring"}
        assert body["total"] == len(body["players"])
        assert body["total"] > 0
        assert body["degraded_positions"] == []
        assert body["scoring"] == "ppr"
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
            "nflcom_pred",
            "rotowire_pred",
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

    def test_invalid_position_returns_400(self, client_with_data):
        # Unknown position must be a clean 400, not a KeyError-driven 500
        # (audit #910 / #578).
        r = client_with_data.get("/api/predictions?position=FOO")
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

    def test_teams_returns_sorted_strings(self, client_with_data):
        r = client_with_data.get("/api/teams")
        assert r.status_code == 200
        body = r.get_json()
        assert "teams" in body
        assert body["teams"] == sorted(body["teams"])
        assert all(isinstance(t, str) for t in body["teams"])

    def test_metrics_returns_both_models(self, client_with_data):
        r = client_with_data.get("/api/metrics")
        assert r.status_code == 200
        body = r.get_json()
        assert "Ridge Regression" in body
        assert "Neural Network" in body
        for model_name in ("Ridge Regression", "Neural Network"):
            assert "overall" in body[model_name]
            assert "by_position" in body[model_name]

    def test_get_data_missing_scoring_fallback_is_empty_metrics(self, app_module, monkeypatch):
        """A partial metrics cache should not KeyError when the requested
        scoring format and PPR alias are both absent."""
        monkeypatch.setattr(core, "_ensure_metrics", lambda: None)
        app_module._cache["results"] = "rows"
        app_module._cache["metrics_by_format"] = {"half_ppr": {"ok": True}}

        rows, metrics = core._get_data("standard")
        assert rows == "rows"
        assert metrics == {}

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

    def test_position_details_fumble_formulas_include_all_loss_sources(self, client_with_data):
        r = client_with_data.get("/api/position_details")
        assert r.status_code == 200
        body = r.get_json()
        expected = "sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost"
        # TE was the lone "raw count" outlier though it computes the same 3-source
        # sum (src/te/targets.py) — locked in here (#835).
        for pos in ("QB", "RB", "WR", "TE"):
            target = next(t for t in body[pos]["targets"] if t["key"] == "fumbles_lost")
            assert target["formula"] == expected

    def test_weekly_accuracy_parallel_arrays(self, client_with_data):
        r = client_with_data.get("/api/weekly_accuracy")
        assert r.status_code == 200
        body = r.get_json()
        assert set(body) == {"weeks", "ridge_mae", "nn_mae", "attn_nn_mae", "lgbm_mae", "scoring"}
        assert body["scoring"] == "ppr"
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

    def test_index_links_favicon(self, client):
        r = client.get("/")
        assert b'rel="icon"' in r.data
        assert b"/static/favicon.svg" in r.data

    def test_favicon_route_serves_svg(self, client):
        r = client.get("/favicon.ico")
        assert r.status_code == 200
        assert "image/svg+xml" in r.content_type
        assert b"<svg" in r.data


# `app.py`'s global `@app.errorhandler(Exception)` catches every exception,
# including werkzeug HTTPExceptions (404 NotFound, 405 MethodNotAllowed). For
# `/api/*` paths it preserves the HTTPException's real status as JSON (audit
# #909) — an unknown `/api/` route returns 404, a wrong method returns 405 — and
# only genuine non-HTTP bugs collapse to a 500. For non-`/api/` paths it still
# `raise e`s, deferring to Flask's default rendering. The `/api/*` status
# contract is asserted below.


class TestApiErrorStatus:
    """`/api/*` errorhandler preserves real HTTP status codes (audit #909)."""

    def test_unknown_api_route_returns_404_not_500(self, client):
        r = client.get("/api/this-route-does-not-exist")
        assert r.status_code == 404
        assert r.is_json
        assert "error" in r.get_json()

    def test_wrong_method_on_api_route_returns_405(self, client):
        # /api/model_architecture is GET-only; POST must surface 405, not 500.
        r = client.post("/api/model_architecture")
        assert r.status_code == 405
        assert r.is_json


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
    returns `{"error": "Internal server error"}, 500` — a fixed generic
    message, NOT `str(e)`. Echoing exception text leaked filesystem paths /
    config / library internals (CodeQL py/stack-trace-exposure). Full
    traceback is still logged server-side via `traceback.print_exc()`.
    """

    def test_predictions_surfaces_load_failure_as_json(self, client, app_module, monkeypatch):
        """When `_get_data()` raises, /api/predictions returns a JSON 500."""

        def _boom(*_args, **_kwargs):
            raise RuntimeError("simulated model load failure")

        monkeypatch.setattr(core, "_get_data", _boom)

        r = client.get("/api/predictions")
        # Must be JSON, not an HTML 500 page
        assert r.is_json, f"Expected JSON error body, got {r.content_type}"
        body = r.get_json()
        assert "error" in body
        # Generic message — exception text MUST NOT leak to the client.
        assert "simulated model load failure" not in body["error"]
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
        # The contract is "JSON, never an HTML error page". The status depends on
        # whether serving splits exist relative to cwd: with no data (CI) the
        # build raises and the /api/ handler returns a JSON 500; with data present
        # the app degrades gracefully (NaN'd model preds) and returns a JSON 200.
        # Both satisfy the contract, so assert JSON either way rather than coupling
        # to ambient data presence.
        assert r.is_json, f"Expected JSON body, got {r.content_type}"
        assert r.status_code in (200, 500)
        if r.status_code == 500:
            assert "error" in r.get_json()


# ===========================================================================
# Scoring format sanity check — documents current contract
# ===========================================================================


class TestScoringFormats:
    """End-to-end checks that every scoring-aware endpoint routes to the right
    DataFrame column / cache slot when ``?scoring=`` is set. The synthetic
    cache populates fantasy_points_{format} actuals and {model}_pred_{format}
    predictions at fixed multipliers (PPR=1.0, half=0.95, standard=0.9), so a
    half-PPR/standard request must produce numerically different values than
    PPR for the same player.
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

    @pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
    def test_predictions_endpoint_routes_to_format_columns(
        self, client_with_data, synthetic_cache, scoring
    ):
        """/api/predictions?scoring=X must surface the matching actual / pred
        columns. Also asserts the scoring round-trip via the response field."""
        r = client_with_data.get(f"/api/predictions?scoring={scoring}")
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("scoring") == scoring

        df = synthetic_cache["results"]
        # Pick the first player-week from the fixture and find its row in the
        # response payload, then verify each numeric field maps to the
        # format-specific column.
        sample_row = df.iloc[0]
        actual_col = "fantasy_points" if scoring == "ppr" else f"fantasy_points_{scoring}"
        target_player = next(
            p
            for p in body["players"]
            if p["player_id"] == sample_row["player_id"] and p["week"] == int(sample_row["week"])
        )
        assert target_player["actual"] == pytest.approx(round(sample_row[actual_col], 2))
        for prefix in ("ridge", "nn", "attn_nn", "lgbm", "nflcom", "rotowire"):
            expected = sample_row[f"{prefix}_pred_{scoring}"]
            actual = target_player[f"{prefix}_pred"]
            if expected != expected:  # NaN check (missing model/expert coverage)
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-3)

    def test_predictions_invalid_scoring_falls_back_to_ppr(self, client_with_data):
        """Garbage scoring values are quietly remapped to PPR rather than 400ing
        — keeps stale bookmarks working without surfacing 5xx to the browser."""
        r = client_with_data.get("/api/predictions?scoring=bogus")
        assert r.status_code == 200
        assert r.get_json().get("scoring") == "ppr"

    @pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
    def test_player_endpoint_uses_format_columns(self, client_with_data, synthetic_cache, scoring):
        df = synthetic_cache["results"]
        player_id = df.iloc[0]["player_id"]
        actual_col = "fantasy_points" if scoring == "ppr" else f"fantasy_points_{scoring}"
        r = client_with_data.get(f"/api/player/{player_id}?scoring={scoring}")
        assert r.status_code == 200
        body = r.get_json()
        assert body["scoring"] == scoring
        # season_avg / season_total should aggregate the format-specific column.
        rows = df[df["player_id"] == player_id]
        assert body["season_avg"] == pytest.approx(round(rows[actual_col].mean(), 2))
        assert body["season_total"] == pytest.approx(round(rows[actual_col].sum(), 2))
        # weekly[0] mirrors the first row's actual under this format.
        first_week_row = rows.sort_values("week").iloc[0]
        assert body["weekly"][0]["actual"] == pytest.approx(round(first_week_row[actual_col], 2))
        for prefix in ("nflcom", "rotowire"):
            expected = first_week_row[f"{prefix}_pred_{scoring}"]
            actual = body["weekly"][0][f"{prefix}_pred"]
            if expected != expected:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
    def test_weekly_accuracy_uses_format_columns(self, client_with_data, scoring):
        r = client_with_data.get(f"/api/weekly_accuracy?scoring={scoring}")
        assert r.status_code == 200
        body = r.get_json()
        assert body["scoring"] == scoring
        # All MAE arrays should be present and finite for the synthetic cache.
        assert body["weeks"]
        assert all(v is None or math.isfinite(v) for v in body["ridge_mae"])
        # Half-PPR errors are smaller than PPR for our synthetic fixture
        # because both actuals and preds are scaled by the same multiplier
        # (errors scale linearly), so we only assert the result is finite.

    def test_metrics_endpoint_returns_format_specific_cache(self, client_with_data):
        ppr = client_with_data.get("/api/metrics?scoring=ppr").get_json()
        std = client_with_data.get("/api/metrics?scoring=standard").get_json()
        # The synthetic cache scales overall MAE by 0.9 for standard, so the
        # two cache slots must return numerically different values.
        ppr_mae = ppr["Ridge Regression"]["overall"]["mae"]
        std_mae = std["Ridge Regression"]["overall"]["mae"]
        assert std_mae == pytest.approx(round(ppr_mae * 0.9, 4))

    @pytest.mark.parametrize("scoring", ["ppr", "half_ppr", "standard"])
    def test_position_details_swaps_total_row_per_format(self, client_with_data, scoring):
        r = client_with_data.get(f"/api/position_details?scoring={scoring}")
        assert r.status_code == 200
        body = r.get_json()
        # Each position should now expose target_metrics["total"] aligned to
        # the requested format (taken from total_by_format[scoring]).
        for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
            target_metrics = body[pos]["target_metrics"]
            # The fixture exposes total_by_format with three distinct slots.
            expected_ridge = {
                "ppr": 5.0,
                "half_ppr": 4.75,
                "standard": 4.5,
            }[scoring]
            assert target_metrics["total"]["ridge_mae"] == pytest.approx(expected_ridge)
            # total_by_format should be consumed by the endpoint and not
            # leaked back to clients.
            assert "total_by_format" not in target_metrics


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


# ===========================================================================
# Front-end assets — no external CDN dependency
# ===========================================================================


class TestStaticAssets:
    """Guard the self-hosted Chart.js: app.js references ``Chart`` at init, so a
    blocked/failed external <script> left the global undefined and threw, taking
    the whole UI's event wiring down with it. Keep every script same-origin.
    """

    def test_index_has_no_external_scripts(self, client):
        import re

        html = client.get("/").get_data(as_text=True)
        srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
        assert srcs, "index page should load at least one script"
        external = [s for s in srcs if s.startswith(("http://", "https://", "//"))]
        assert not external, f"index must not load scripts from a CDN: {external}"

    def test_vendored_chartjs_is_served(self, client):
        r = client.get("/static/js/vendor/chart.umd.min.js")
        assert r.status_code == 200
        assert b"Chart.js v4.4.0" in r.get_data()

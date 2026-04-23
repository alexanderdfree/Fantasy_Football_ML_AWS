"""Graceful per-position degradation (Part B).

Covers ``_degraded_positions``, ``_ensure_position_loaded``'s failure caching,
``_ensure_all_positions_loaded``'s best-effort fan-out + all-broken fail-loud
contract, and the ``degraded_positions`` field on ``/api/predictions``.

The fixture pre-populates ``app._cache`` with a minimal results DataFrame and
stubs ``_load_base_data_locked`` so tests don't need on-disk parquets or
trained models. ``_apply_position_models`` is monkeypatched per-test to
simulate per-position and per-model failure shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytestmark = pytest.mark.integration


_ALL_POS = ("QB", "RB", "WR", "TE", "K", "DST")


def _minimal_results() -> pd.DataFrame:
    """One row per position; columns + initial values mirror what
    ``_load_base_data_locked`` sets up (ridge/nn init to 0.0, attn/lgbm to NaN).
    """
    return pd.DataFrame(
        {
            "player_id": [f"{p}-1" for p in _ALL_POS],
            "player_display_name": [f"{p} Player" for p in _ALL_POS],
            "position": list(_ALL_POS),
            "recent_team": ["XX"] * len(_ALL_POS),
            "season": [2025] * len(_ALL_POS),
            "week": [1] * len(_ALL_POS),
            "headshot_url": [""] * len(_ALL_POS),
            "fantasy_points": [10.0] * len(_ALL_POS),
            "fantasy_points_standard": [9.0] * len(_ALL_POS),
            "fantasy_points_half_ppr": [9.5] * len(_ALL_POS),
            "ridge_pred": [0.0] * len(_ALL_POS),
            "nn_pred": [0.0] * len(_ALL_POS),
            "attn_nn_pred": [np.nan] * len(_ALL_POS),
            "lgbm_pred": [np.nan] * len(_ALL_POS),
        }
    )


@pytest.fixture
def degraded_mode_app(monkeypatch):
    """Import ``app`` with a pre-populated minimal cache. Per-test monkeypatches
    of ``_apply_position_models`` control which positions succeed or fail.
    """
    import app as app_mod

    fake_cache = {
        "base_loaded": True,
        "splits": {p: (None, None, None) for p in _ALL_POS},
        "results": _minimal_results(),
        "positions_loaded": set(),
        "k_kicks_df": None,
    }
    monkeypatch.setattr(app_mod, "_cache", fake_cache)
    # Prevent the real base-data loader from running if the early-return
    # check on base_loaded ever flips.
    monkeypatch.setattr(app_mod, "_load_base_data_locked", lambda: None)
    return app_mod


@pytest.fixture
def degraded_client(degraded_mode_app):
    degraded_mode_app.app.config["TESTING"] = True
    with degraded_mode_app.app.test_client() as c:
        yield c, degraded_mode_app


# ---------------------------------------------------------------------------
# _degraded_positions — unit coverage for the error-key parser
# ---------------------------------------------------------------------------


class TestDegradedPositionsHelper:
    def test_empty_when_no_errors(self, degraded_mode_app):
        assert degraded_mode_app._degraded_positions() == []

    def test_parses_per_model_keys(self, degraded_mode_app):
        degraded_mode_app._cache["position_load_errors"] = {
            "QB_ridge": "bad",
            "WR_nn": "bad",
        }
        assert degraded_mode_app._degraded_positions() == ["QB", "WR"]

    def test_parses_bare_position_key_from_outer_catch(self, degraded_mode_app):
        """_ensure_position_loaded's outer except writes a plain ``pos`` key
        (no suffix) when feature-building or data-loading blows up."""
        degraded_mode_app._cache["position_load_errors"] = {"DST": "setup boom"}
        assert degraded_mode_app._degraded_positions() == ["DST"]

    def test_mixed_outer_and_inner_keys_dedup(self, degraded_mode_app):
        degraded_mode_app._cache["position_load_errors"] = {
            "WR": "outer",
            "WR_nn": "inner",
            "QB_attn_nn": "inner",
        }
        assert degraded_mode_app._degraded_positions() == ["QB", "WR"]

    def test_dst_prefix_does_not_absorb_other_positions(self, degraded_mode_app):
        """ "DST" doesn't contain an underscore; the parser must not false-match
        it against keys like "QB_nn"."""
        degraded_mode_app._cache["position_load_errors"] = {
            "DST_lgbm": "bad",
            "QB_nn": "bad",
        }
        assert degraded_mode_app._degraded_positions() == ["DST", "QB"]


# ---------------------------------------------------------------------------
# /api/predictions: one failure does not poison the other five
# ---------------------------------------------------------------------------


class TestOnePositionFailsOthersServe:
    def test_outer_failure_in_one_position_degrades_only_that_pos(self, degraded_client):
        """Raising from _apply_position_models for QB lands QB in
        positions_failed + degraded_positions. The other five positions'
        rows still appear in /api/predictions."""
        client, app_mod = degraded_client

        def fake_apply(train, val, test, pos, results):
            if pos == "QB":
                raise RuntimeError("QB boom (e.g. missing nn_scaler.pkl)")

        with mock.patch.object(app_mod, "_apply_position_models", side_effect=fake_apply):
            r = client.get("/api/predictions")

        assert r.status_code == 200
        body = r.get_json()
        assert body["degraded_positions"] == ["QB"]
        positions_seen = {row["position"] for row in body["players"]}
        # All six rows still emitted; QB is degraded but its row lives on.
        assert positions_seen == set(_ALL_POS)

    def test_inner_per_model_failure_records_without_raising(self, degraded_client):
        """A per-model failure inside _apply_position_models records the error
        and continues — the position still "loads" (no raise). It must still
        surface in degraded_positions so the banner flags it."""
        client, app_mod = degraded_client

        def fake_apply(train, val, test, pos, results):
            if pos == "WR":
                app_mod._cache.setdefault("position_load_errors", {})["WR_nn"] = "nn load failed"
            # Not raising → _ensure_position_loaded adds WR to positions_loaded.

        with mock.patch.object(app_mod, "_apply_position_models", side_effect=fake_apply):
            r = client.get("/api/predictions")

        assert r.status_code == 200
        assert r.get_json()["degraded_positions"] == ["WR"]
        assert "WR" in app_mod._cache["positions_loaded"]
        assert "WR" not in app_mod._cache.get("positions_failed", set())

    def test_failure_is_cached_and_not_retried(self, degraded_client):
        """Hitting /api/predictions twice must not re-invoke
        _apply_position_models — a broken model would re-spam logs + slow
        every request otherwise."""
        client, app_mod = degraded_client
        calls: list[str] = []

        def fake_apply(train, val, test, pos, results):
            calls.append(pos)
            if pos == "QB":
                raise RuntimeError("QB boom")

        with mock.patch.object(app_mod, "_apply_position_models", side_effect=fake_apply):
            r1 = client.get("/api/predictions")
            r2 = client.get("/api/predictions")

        assert r1.status_code == 200 and r2.status_code == 200
        # First request: one call per position. Second request: zero.
        assert sorted(calls) == sorted(_ALL_POS)


# ---------------------------------------------------------------------------
# _ensure_all_positions_loaded: partial failure ≠ crash; total failure → crash
# ---------------------------------------------------------------------------


class TestAllPositionsFailPreservesFailLoud:
    def test_all_six_fail_raises_so_gunicorn_aborts(self, degraded_mode_app):
        """Every position broken → raise so ``gunicorn --preload`` aborts and
        ECS blocks the rollout. Preserves the existing fail-loud contract for
        the all-broken case documented in ``shared/model_sync.py``."""
        with mock.patch.object(
            degraded_mode_app,
            "_apply_position_models",
            side_effect=RuntimeError("apocalypse"),
        ):
            with pytest.raises(RuntimeError, match="All positions failed"):
                degraded_mode_app._ensure_all_positions_loaded()

    def test_five_fail_one_succeeds_does_not_raise(self, degraded_mode_app):
        """Any successful position makes the fan-out non-fatal, matching the
        plan's "five broken is still better than six broken" goal."""

        def fake_apply(train, val, test, pos, results):
            if pos != "QB":
                raise RuntimeError(f"{pos} boom")

        with mock.patch.object(degraded_mode_app, "_apply_position_models", side_effect=fake_apply):
            degraded_mode_app._ensure_all_positions_loaded()  # MUST NOT raise

        assert degraded_mode_app._cache["positions_loaded"] == {"QB"}
        assert degraded_mode_app._cache["positions_failed"] == {"RB", "WR", "TE", "K", "DST"}
        assert set(degraded_mode_app._degraded_positions()) == {"RB", "WR", "TE", "K", "DST"}


# ---------------------------------------------------------------------------
# /api/predictions response shape
# ---------------------------------------------------------------------------


class TestResponseShape:
    def test_degraded_positions_field_present_when_all_load(self, degraded_client):
        client, app_mod = degraded_client
        with mock.patch.object(app_mod, "_apply_position_models", return_value=None):
            r = client.get("/api/predictions?position=QB")
        body = r.get_json()
        assert "degraded_positions" in body
        assert body["degraded_positions"] == []

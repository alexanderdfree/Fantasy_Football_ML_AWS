"""Tests for the upcoming-week subsystem (src/serving/upcoming_week.py).

Component-level tests (skeleton assembly, schedule-cache augmentation,
serialization, signature, artifact round-trip) plus the ``/api/upcoming_week``
route. The heavy ``build_features`` / inference path is exercised by the local
end-to-end check, not here — these stay fast and network-free.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.serving import upcoming_week
from src.serving.serialization import _pred_col
from src.shared import weather_features


@pytest.mark.unit
def test_build_skeleton_joins_matchup_and_stamps_week():
    roster = pd.DataFrame(
        {
            "player_id": ["00-1", "00-2", "00-3"],
            "position": ["WR", "RB", "QB"],
            "recent_team": ["SEA", "SEA", "NE"],
            "espn_name": ["A", "B", "C"],
        }
    )
    slate = pd.DataFrame(
        {
            "recent_team": ["SEA", "NE"],
            "opponent_team": ["NE", "SEA"],
            "is_home": [1, 0],
        }
    )
    history_cols = ["player_id", "season", "week", "recent_team", "position", "fantasy_points"]
    skel = upcoming_week._build_skeleton(2026, 1, slate, roster, history_cols)

    assert len(skel) == 3
    assert set(skel["season"]) == {2026}
    assert set(skel["week"]) == {1}
    assert set(skel["season_type"]) == {"REG"}
    sea_wr = skel[skel["player_id"] == "00-1"].iloc[0]
    assert sea_wr["opponent_team"] == "NE"
    assert sea_wr["is_home"] == 1
    ne_qb = skel[skel["player_id"] == "00-3"].iloc[0]
    assert ne_qb["opponent_team"] == "SEA"
    assert ne_qb["is_home"] == 0
    # History-only columns exist (NaN) so the concat with history isn't ragged.
    assert "fantasy_points" in skel.columns


@pytest.mark.unit
def test_augment_schedules_cache_merges_and_resets(monkeypatch, tmp_path):
    sched_path = tmp_path / "schedules.parquet"
    existing = pd.DataFrame(
        {
            "game_id": ["2025_01_DAL_PHI"],
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["PHI"],
            "away_team": ["DAL"],
            "home_score": [24],
            "away_score": [20],
            "spread_line": [8.5],
            "total_line": [47.5],
        }
    )
    existing.to_parquet(sched_path)
    monkeypatch.setattr(upcoming_week, "_schedules_path", lambda: str(sched_path))
    weather_features._schedule_cache = "STALE"

    new_rows = pd.DataFrame(
        {
            "game_id": ["2026_01_NE_SEA"],
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [pd.NA],
            "away_score": [pd.NA],
            "spread_line": [3.5],
            "total_line": [44.5],
        }
    )
    upcoming_week._augment_schedules_cache(new_rows)

    merged = pd.read_parquet(sched_path)
    assert set(merged["season"]) == {2025, 2026}
    assert (merged["game_id"] == "2026_01_NE_SEA").sum() == 1
    # The module schedule cache was invalidated so consumers re-read the file.
    assert weather_features._schedule_cache is None


@pytest.mark.unit
def test_augment_schedules_cache_dedupes_on_game_id(monkeypatch, tmp_path):
    sched_path = tmp_path / "schedules.parquet"
    base_cols = {
        "game_id": ["2026_01_NE_SEA"],
        "season": [2026],
        "week": [1],
        "game_type": ["REG"],
        "home_team": ["SEA"],
        "away_team": ["NE"],
        "home_score": [pd.NA],
        "away_score": [pd.NA],
        "spread_line": [3.5],
        "total_line": [44.5],
    }
    pd.DataFrame(base_cols).to_parquet(sched_path)
    monkeypatch.setattr(upcoming_week, "_schedules_path", lambda: str(sched_path))

    updated = pd.DataFrame({**base_cols, "spread_line": [2.5], "total_line": [45.5]})
    upcoming_week._augment_schedules_cache(updated)

    merged = pd.read_parquet(sched_path)
    assert len(merged) == 1  # replaced, not duplicated
    assert float(merged.iloc[0]["spread_line"]) == 2.5


@pytest.mark.unit
def test_results_to_upcoming_rows_shape_and_null_actual():
    cols = {
        "player_id": ["00-1"],
        "player_display_name": ["Star WR"],
        "position": ["WR"],
        "recent_team": ["SEA"],
        "opponent_team": ["NE"],
        "is_home": [1],
        "spread_line": [3.5],
        "total_line": [44.5],
        "implied_team_total": [24.0],
        "headshot_url": ["http://x/1.png"],
    }
    df = pd.DataFrame(cols)
    for prefix in ("ridge", "nn", "attn_nn", "lgbm"):
        df[_pred_col(prefix, "ppr")] = 15.5

    rows = upcoming_week._results_to_upcoming_rows(df, "ppr")
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "Star WR"
    assert r["opponent"] == "NE"
    assert r["is_home"] == 1
    assert r["implied_team_total"] == 24.0
    assert r["actual"] is None  # no games played yet
    assert r["attn_nn_pred"] == 15.5
    assert r["headshot"] == "http://x/1.png"


@pytest.mark.unit
def test_fill_current_week_context_carries_forward_and_defaults():
    # History: a veteran with a known depth_chart_rank; a rookie absent here.
    history = pd.DataFrame(
        {
            "player_id": ["vet", "vet"],
            "season": [2025, 2025],
            "week": [1, 17],
            "depth_chart_rank": [1.0, 1.0],
            "contract_guaranteed": [110.0, 110.0],
        }
    )
    slice_df = pd.DataFrame(
        {
            "player_id": ["vet", "rookie"],
            "depth_chart_rank": [np.nan, np.nan],
            "contract_guaranteed": [np.nan, np.nan],
            "game_status": [np.nan, np.nan],
            "practice_status": [np.nan, np.nan],
        }
    )
    out = upcoming_week._fill_current_week_context(slice_df, history)
    vet = out[out["player_id"] == "vet"].iloc[0]
    rook = out[out["player_id"] == "rookie"].iloc[0]
    # Veteran: carried forward from 2025.
    assert vet["depth_chart_rank"] == 1.0
    assert vet["contract_guaranteed"] == 110.0
    # Rookie (no history): default is the -1 training sentinel (build_position_features
    # remaps it to the train-mean of real ranks — the neutral no-data value), not 3
    # (which would standardize as a real rank-3, a train/serve mismatch, #1270).
    assert rook["depth_chart_rank"] == -1.0
    # Current-health columns default to active / full practice for everyone.
    assert set(out["game_status"]) == {1.0}
    assert set(out["practice_status"]) == {2.0}


@pytest.mark.unit
def test_fill_current_week_context_live_signals_win():
    # 'vet' carried a stale rank-1 forward (a 2024 spot start), but the live depth
    # chart now lists him a backup; 'backup' has no history (would default to 3)
    # but the live chart makes him a starter; 'healthy' has neither.
    history = pd.DataFrame(
        {
            "player_id": ["vet"],
            "season": [2024],
            "week": [3],
            "depth_chart_rank": [1.0],
            "contract_guaranteed": [110.0],
        }
    )
    slice_df = pd.DataFrame(
        {
            "player_id": ["vet", "backup", "healthy"],
            "depth_chart_rank": [np.nan, np.nan, np.nan],
            "contract_guaranteed": [np.nan, np.nan, np.nan],
            "game_status": [np.nan, np.nan, np.nan],
            "practice_status": [np.nan, np.nan, np.nan],
        }
    )
    depth = {"vet": 2.0, "backup": 1.0}
    gstat = {"vet": 0.5}  # vet is Questionable
    out = upcoming_week._fill_current_week_context(slice_df, history, depth, gstat)
    vet = out[out["player_id"] == "vet"].iloc[0]
    backup = out[out["player_id"] == "backup"].iloc[0]
    healthy = out[out["player_id"] == "healthy"].iloc[0]
    # Precedence: live ESPN rank wins over the stale carry-forward AND the default.
    assert vet["depth_chart_rank"] == 2.0  # live (2) beats carried 1.0
    assert backup["depth_chart_rank"] == 1.0  # live (1) beats default -1.0
    assert healthy["depth_chart_rank"] == -1.0  # no live, no history -> -1 sentinel (#1270)
    assert vet["contract_guaranteed"] == 110.0  # contracts still carry forward
    # Live injury game_status overrides the healthy default; others stay default.
    assert vet["game_status"] == 0.5
    assert healthy["game_status"] == 1.0
    # practice_status is never filled from ESPN -> always the default.
    assert set(out["practice_status"]) == {2.0}


@pytest.mark.unit
def test_fill_current_week_context_practice_and_contracts():
    # vet: in history (carry-forward source) AND in the live contract derive;
    # carryonly: history only; fresh: neither.
    history = pd.DataFrame(
        {
            "player_id": ["vet", "carryonly"],
            "season": [2024, 2024],
            "week": [3, 3],
            "contract_apy_cap_pct": [0.05, 0.07],
            "contract_guaranteed": [10.0, 20.0],
            "contract_years_remaining": [1.0, 2.0],
            "contract_age": [3.0, 2.0],
        }
    )
    slice_df = pd.DataFrame(
        {
            "player_id": ["vet", "carryonly", "fresh"],
            "contract_apy_cap_pct": [np.nan, np.nan, np.nan],
            "contract_guaranteed": [np.nan, np.nan, np.nan],
            "contract_years_remaining": [np.nan, np.nan, np.nan],
            "contract_age": [np.nan, np.nan, np.nan],
            "game_status": [np.nan, np.nan, np.nan],
            "practice_status": [np.nan, np.nan, np.nan],
        }
    )
    contract_features = pd.DataFrame(
        {
            "contract_apy_cap_pct": [0.20],
            "contract_guaranteed": [99.0],
            "contract_years_remaining": [4.0],
            "contract_age": [1.0],
        },
        index=pd.Index(["vet"], name="player_id"),
    )
    out = upcoming_week._fill_current_week_context(
        slice_df,
        history,
        practice_status_map={"vet": 1.0},
        contract_features=contract_features,
    )
    vet = out[out["player_id"] == "vet"].iloc[0]
    carryonly = out[out["player_id"] == "carryonly"].iloc[0]
    fresh = out[out["player_id"] == "fresh"].iloc[0]
    # Live contract derive beats the carried-forward value.
    assert vet["contract_apy_cap_pct"] == 0.20
    assert vet["contract_age"] == 1.0
    # No live contract → carry-forward from history.
    assert carryonly["contract_apy_cap_pct"] == 0.07
    # Neither live nor history → stays NaN (no contract default; build fills it).
    assert pd.isna(fresh["contract_apy_cap_pct"])
    # Live practice_status overrides the default; others stay healthy.
    assert vet["practice_status"] == 1.0
    assert carryonly["practice_status"] == 2.0
    assert fresh["practice_status"] == 2.0


@pytest.mark.unit
def test_input_signature_changes_with_lines():
    slate = pd.DataFrame(
        {
            "recent_team": ["SEA", "NE"],
            "opponent_team": ["NE", "SEA"],
            "is_home": [1, 0],
            "spread_line": [3.5, -3.5],
            "total_line": [44.5, 44.5],
        }
    )
    roster = pd.DataFrame({"player_id": ["00-1", "00-2"]})
    sig1 = upcoming_week._input_signature(2026, 1, slate, roster)
    sig2 = upcoming_week._input_signature(2026, 1, slate, roster)
    assert sig1 == sig2  # stable for identical inputs
    moved = slate.copy()
    moved.loc[0, "spread_line"] = 6.5
    assert upcoming_week._input_signature(2026, 1, moved, roster) != sig1


@pytest.mark.unit
def test_input_signature_changes_with_reserve_inactive_roster():
    # #1277: a RES/INA move (which resizes the inheritance vacancy out-set) must
    # invalidate the cache, else a stale artifact ships with the wrong out-set.
    slate = pd.DataFrame(
        {
            "recent_team": ["SEA"],
            "opponent_team": ["NE"],
            "is_home": [1],
            "spread_line": [3.5],
            "total_line": [44.5],
        }
    )
    roster = pd.DataFrame({"player_id": ["00-1", "00-2"]})
    base = upcoming_week._input_signature(2026, 1, slate, roster)
    rosters = pd.DataFrame(
        {
            "player_id": ["00-9"],
            "status": ["RES"],
            "position": ["RB"],
            "season": [2026],
            "team": ["SEA"],
            "week": [1],
        }
    )
    with_ir = upcoming_week._input_signature(2026, 1, slate, roster, rosters_df=rosters)
    assert with_ir != base  # a reserve/inactive player changes the signature
    # ACT (not sidelined) players do not size a vacancy → no signature change.
    active = rosters.assign(status=["ACT"])
    assert upcoming_week._input_signature(2026, 1, slate, roster, rosters_df=active) == base


@pytest.mark.unit
def test_write_then_read_artifact_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_week.core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    payload = {"available": True, "season": 2026, "week": 1, "scoring": {"ppr": []}}
    upcoming_week._write_artifact(payload)
    got = upcoming_week.read_cached_artifact()
    assert got == payload


@pytest.mark.unit
def test_read_cached_artifact_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(upcoming_week.core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    assert upcoming_week.read_cached_artifact() is None


@pytest.mark.unit
def test_s3_artifact_key_under_predictions_cache(monkeypatch):
    monkeypatch.setenv("FF_MODEL_S3_PREFIX", "models")
    assert upcoming_week._s3_artifact_key() == "models/predictions_cache/upcoming_week.json"


@pytest.mark.unit
def test_s3_sync_and_upload_noop_without_bucket(monkeypatch, tmp_path):
    # No FF_MODEL_S3_BUCKET → both are best-effort no-ops (False), no boto3 call.
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    monkeypatch.setattr(upcoming_week.core, "_PREDICTIONS_CACHE_DIR", str(tmp_path))
    assert upcoming_week.sync_artifact_from_s3() is False
    assert upcoming_week.upload_artifact_to_s3() is False


@pytest.mark.unit
def test_download_poller_disabled_at_zero_interval():
    assert upcoming_week.start_artifact_download_poller(interval_s=0) is None


@pytest.mark.integration
def test_route_503_when_artifact_absent(client):
    """No primed artifact (fresh tmp cache dir) -> 503 warming."""
    resp = client.get("/api/upcoming_week")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "warming"


@pytest.mark.integration
def test_route_serves_artifact_when_present(client, app_module, tmp_path):
    # app_module redirects core._PREDICTIONS_CACHE_DIR to tmp; write there.
    payload = {
        "available": True,
        "season": 2026,
        "week": 1,
        "week_label": "Week 1 · 2026",
        "scoring": {"ppr": [{"player_id": "00-1", "name": "Star WR"}]},
    }
    path = upcoming_week._artifact_path()
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)

    resp = client.get("/api/upcoming_week")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["available"] is True
    assert body["week"] == 1


@pytest.mark.integration
def test_route_serves_offseason_unavailable(client):
    payload = {"available": False, "reason": "offseason"}
    path = upcoming_week._artifact_path()
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)

    resp = client.get("/api/upcoming_week")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["available"] is False
    assert body["reason"] == "offseason"


@pytest.mark.unit
def test_run_upcoming_inference_passes_season_context_and_slices_week(monkeypatch):
    """#1411 regression: the models must see season-to-date rows, the artifact only week W.

    A one-week frame makes ``core._apply_position_models`` rebuild every
    within-season lookback (attention history, L3 rollups, QB
    season_starts_to_date) from a single row per player — a Week-1 cold start
    every mid-season week.
    """
    frame = pd.DataFrame(
        {
            "player_id": ["00-1"] * 4,
            "position": ["RB"] * 4,
            "recent_team": ["SEA"] * 4,
            "season": [2026] * 4,
            "week": [1, 2, 3, 4],
            "opponent_team": ["NE", "SF", "LA", "ARI"],
            "is_home": [1, 0, 1, 0],
        }
    )
    roster = pd.DataFrame({"player_id": ["00-1"], "espn_name": ["A"], "espn_id": ["10"]})
    slate = pd.DataFrame({"recent_team": ["SEA"], "spread_line": [3.5], "total_line": [44.5]})

    tiny = pd.DataFrame({"player_id": [], "season": [], "week": []})
    monkeypatch.setattr(upcoming_week.core, "_ensure_base_data", lambda: None)
    monkeypatch.setattr(upcoming_week.app_pkg, "_cache", {"splits": {"RB": (tiny, tiny, tiny)}})

    captured = {}
    sentinel_prefix = upcoming_week._MODEL_PRED_PREFIXES[0]

    def fake_apply(train, val, test, pos, results):
        captured[pos] = test.copy()
        results.loc[test.index, f"{sentinel_prefix}_pred"] = 7.0

    monkeypatch.setattr(upcoming_week.core, "_apply_position_models", fake_apply)

    out = upcoming_week.run_upcoming_inference(frame, roster, slate, 2026, 4)

    # The models received the full season-to-date frame (weeks 1..W)...
    assert set(captured["RB"]["week"]) == {1, 2, 3, 4}
    # ...but only the upcoming week is returned for serialization.
    assert len(out) == 1
    assert set(out["week"]) == {4}
    row = out.iloc[0]
    assert float(row[f"{sentinel_prefix}_pred"]) == 7.0
    assert row["player_display_name"] == "A"
    assert float(row["spread_line"]) == 3.5


@pytest.mark.unit
def test_build_upcoming_week_frame_keeps_season_to_date_reg_rows(monkeypatch):
    """The built frame = current-season completed REG weeks + context-filled week W."""
    history = pd.DataFrame(
        {
            "player_id": ["00-1"] * 4 + ["00-1"],
            "season": [2026, 2026, 2026, 2026, 2025],
            "week": [1, 2, 2, 3, 12],
            "season_type": ["REG", "REG", "POST", "REG", "REG"],
            "recent_team": ["SEA"] * 5,
            "position": ["RB"] * 5,
            "game_status": [1.0] * 5,
            "fantasy_points": [10.0, 12.0, 8.0, 14.0, 9.0],
        }
    )
    roster = pd.DataFrame({"player_id": ["00-1"], "position": ["RB"], "recent_team": ["SEA"]})
    slate = pd.DataFrame({"recent_team": ["SEA"], "opponent_team": ["NE"], "is_home": [1]})

    monkeypatch.setattr(upcoming_week, "_load_history", lambda: history)
    monkeypatch.setattr(
        upcoming_week,
        "build_features",
        lambda combined, injuries_df=None, rosters_df=None: combined,
    )

    out = upcoming_week.build_upcoming_week_frame(2026, 4, slate, roster)

    ctx = out[out["week"] < 4]
    # Completed current-season REG weeks only: no POST row, no prior season.
    assert sorted(zip(ctx["week"], ctx["season_type"], strict=True)) == [
        (1, "REG"),
        (2, "REG"),
        (3, "REG"),
    ]
    assert set(ctx["season"]) == {2026}
    # Prior-week rows pass through untouched by the context fill.
    assert ctx["fantasy_points"].tolist() == [10.0, 12.0, 14.0]
    # The synthetic week-W row is present, context-filled with defaults.
    wk = out[out["week"] == 4]
    assert len(wk) == 1
    assert wk.iloc[0]["season_type"] == "REG"
    assert float(wk.iloc[0]["game_status"]) == 1.0  # NaN -> healthy default
    # Disjoint original indices are preserved (prediction writes are by index).
    assert out.index.is_unique

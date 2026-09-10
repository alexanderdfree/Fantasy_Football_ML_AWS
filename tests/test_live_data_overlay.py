"""Live feature consumers mutate copies while historical release bytes stay pinned."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.release import DataReleaseError, assert_source_fetch_allowed
from src.serving import upcoming_week as live
from src.serving.live_build import HISTORICAL_CACHE_ENV, run_in_live_overlay

pytestmark = pytest.mark.unit
SUFFIX = "2012_2025"


def _schedule(season=2025):
    return pd.DataFrame(
        {
            "season": [season],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["SEA"],
            "away_team": ["NE"],
            "home_score": [27.0],
            "away_score": [20.0],
        }
    )


def _snapshot(tmp_path):
    original = tmp_path / "snapshot"
    original.mkdir()
    _schedule().to_parquet(original / f"schedules_{SUFFIX}.parquet")
    weekly = pd.DataFrame(
        {
            "player_id": ["old", "sea", "ne", "future"],
            "season": [2024, 2025, 2025, 2025],
            "week": [1, 1, 1, 2],
            "recent_team": ["SEA", "SEA", "NE", "SEA"],
        }
    )
    weekly.to_parquet(original / f"weekly_{SUFFIX}.parquet")
    weekly.drop(columns="player_id").rename(columns={"recent_team": "team"}).to_parquet(
        original / f"team_stats_{SUFFIX}.parquet"
    )
    (original / ".release.json").write_text(json.dumps({"release_id": "unit-pinned-release"}))
    return original


def _hashes(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }


def test_child_uses_overlay_for_real_schedule_and_team_writers_and_preserves_outputs(
    tmp_path, monkeypatch
):
    original = _snapshot(tmp_path)
    before = _hashes(original)
    evidence = tmp_path / "child-evidence.json"
    monkeypatch.setenv("TEST_LIVE_EVIDENCE", str(evidence))
    monkeypatch.setenv("TEST_EXPECTED_CWD", os.getcwd())
    command = r"""
import json, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from src.serving import upcoming_week as live
from src.data.release import DataReleaseError, assert_source_fetch_allowed, live_source_cache
root = Path(live.CACHE_DIR)
original = Path(os.environ["FF_LIVE_HISTORICAL_CACHE_DIR"])
assert root != original and not (root / ".release.json").exists()
assert os.environ["FF_DATA_RELEASE"] == "unit-pinned-release"
assert os.getcwd() == os.environ["TEST_EXPECTED_CWD"]
assert Path(live._artifact_path()).parent == Path(live.core._REPO_ROOT) / "data/serving_cache"
schedule = pd.read_parquet(root / "schedules_2012_2025.parquet")
live._augment_schedules_cache(schedule.assign(season=2026))
live.load_team_week_stats = lambda seasons, cache_dir=None: pd.read_parquet(Path(cache_dir) / "team_stats_2012_2025.parquet")
live._merge_live_team_stats(pd.DataFrame({"season":[2026],"week":[1],"team":["SEA"]}))
for directory in [original, root]:
    try:
        assert_source_fetch_allowed(directory / "missing.parquet")
    except DataReleaseError:
        pass
    else:
        raise AssertionError("unscoped fetch was allowed")
fresh = root.parent / "fresh"
fresh.mkdir()
with live_source_cache(fresh):
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(assert_source_fetch_allowed, fresh / "current.parquet").result()
    try:
        assert_source_fetch_allowed(original / "missing.parquet")
    except DataReleaseError:
        pass
    else:
        raise AssertionError("live scope unlocked original snapshot")
output = Path(os.environ["TEST_LIVE_EVIDENCE"])
live.core._PREDICTIONS_CACHE_DIR = str(output.parent / "artifact")
live._write_artifact({"available": True})
output.write_text(json.dumps({"schedule_seasons": sorted(pd.read_parquet(root / "schedules_2012_2025.parquet").season.unique().tolist()), "team_seasons": sorted(pd.read_parquet(root / "team_stats_2012_2025.parquet").season.unique().tolist()), "artifact": live._artifact_path()}))
"""
    assert run_in_live_overlay(str(original), [sys.executable, "-c", command]) == 0
    assert _hashes(original) == before
    result = json.loads(evidence.read_text())
    assert result["schedule_seasons"] == [2025, 2026]
    assert result["team_seasons"] == [2024, 2025, 2026]
    assert json.loads(Path(result["artifact"]).read_text())["available"] is True


def test_child_failure_exit_status_is_preserved(tmp_path):
    original = _snapshot(tmp_path)
    before = _hashes(original)
    assert run_in_live_overlay(str(original), [sys.executable, "-c", "raise SystemExit(7)"]) == 7
    assert _hashes(original) == before


def test_direct_refresh_rejects_mutating_original_snapshot(tmp_path, monkeypatch):
    original = _snapshot(tmp_path)
    monkeypatch.setattr(live, "CACHE_DIR", str(original))
    monkeypatch.setattr(
        live.espn_live, "next_unplayed_week", lambda *a: pytest.fail("source reached")
    )
    with pytest.raises(DataReleaseError, match="separate mutable cache"):
        live.refresh_upcoming_week_cache(force=True)


def test_cli_child_runs_main_once_without_relaunch(tmp_path, monkeypatch):
    monkeypatch.setenv(HISTORICAL_CACHE_ENV, str(tmp_path / "original"))
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path / "overlay"))
    calls = []
    monkeypatch.setattr(live, "main", lambda: calls.append("built"))
    monkeypatch.setattr(live, "run_in_live_overlay", lambda *a: pytest.fail("recursive child"))
    live.cli()
    assert calls == ["built"]


def test_cli_parent_preserves_failed_child_status(monkeypatch):
    monkeypatch.delenv(HISTORICAL_CACHE_ENV, raising=False)
    monkeypatch.setattr(live, "run_in_live_overlay", lambda *a: 7)
    with pytest.raises(SystemExit) as exc:
        live.cli()
    assert exc.value.code == 7


def _history_boundaries(tmp_path, monkeypatch):
    original = _snapshot(tmp_path)
    overlay = tmp_path / "overlay"
    shutil.copytree(original, overlay, ignore=shutil.ignore_patterns(".release.json"))
    monkeypatch.setenv(HISTORICAL_CACHE_ENV, str(original))
    monkeypatch.setenv("FF_DATA_RELEASE", "unit-pinned-release")
    monkeypatch.setattr(live, "CACHE_DIR", str(overlay))
    monkeypatch.setattr(live, "_history_cache", None)
    monkeypatch.setattr(live, "_history_seasons", None)
    monkeypatch.setattr(live, "preprocess", lambda frame: frame)
    monkeypatch.setattr(live.live_qbr, "recover_qbr", lambda frame, *a: (frame, {}))
    return original, overlay


def test_historical_subset_loads_full_snapshot_and_never_queries_sources(tmp_path, monkeypatch):
    original, _ = _history_boundaries(tmp_path, monkeypatch)
    before = _hashes(original)
    calls = []

    def load(seasons, cache_dir=None):
        calls.append((seasons, cache_dir))
        assert seasons == live.SEASONS and Path(cache_dir) == original
        return pd.read_parquet(original / f"weekly_{SUFFIX}.parquet")

    monkeypatch.setattr(live, "load_raw_data", load)
    monkeypatch.setattr(
        live,
        "load_team_week_stats",
        lambda seasons, cache_dir=None: pd.read_parquet(
            Path(cache_dir) / f"team_stats_{SUFFIX}.parquet"
        ),
    )
    monkeypatch.setattr(
        live.nfl_source, "snap_counts", lambda *a: pytest.fail("historical source fetch")
    )
    monkeypatch.setattr(
        live.live_qbr, "recover_qbr", lambda *a: pytest.fail("historical QBR fetch")
    )
    result = live._load_history(2025, 2, _schedule())
    assert set(result.player_id) == {"old", "sea", "ne"}
    assert len(calls) == 2  # archived subset and played current year, both same pinned range
    assert _hashes(original) == before


@pytest.mark.parametrize("year", [2026, 2027])
def test_only_uncovered_years_can_fetch_in_separate_live_cache(tmp_path, monkeypatch, year):
    original, overlay = _history_boundaries(tmp_path, monkeypatch)
    before = _hashes(original)
    new = pd.DataFrame(
        {
            "player_id": ["new-sea", "new-ne"],
            "season": [2026, 2026],
            "week": [1, 1],
            "recent_team": ["SEA", "NE"],
        }
    )
    calls = []

    def load(seasons, cache_dir=None):
        if seasons == live.SEASONS:
            assert Path(cache_dir) == original
            return pd.read_parquet(original / f"weekly_{SUFFIX}.parquet")
        assert seasons == [2026]
        assert Path(cache_dir) not in {original, overlay}
        assert_source_fetch_allowed(Path(cache_dir) / "weekly_2026_2026.parquet")
        _schedule(2026).to_parquet(Path(cache_dir) / "schedules_2026_2026.parquet")
        calls.append(Path(cache_dir))
        return new

    def teams(seasons, cache_dir=None):
        if seasons == live.SEASONS:
            return pd.read_parquet(Path(cache_dir) / f"team_stats_{SUFFIX}.parquet")
        assert seasons == [2026]
        assert_source_fetch_allowed(Path(cache_dir) / "team_stats_2026_2026.parquet")
        return new.drop(columns="player_id").rename(columns={"recent_team": "team"})

    monkeypatch.setattr(live, "load_raw_data", load)
    monkeypatch.setattr(live, "load_team_week_stats", teams)
    monkeypatch.setattr(
        live.nfl_source,
        "snap_counts",
        lambda *a: pd.DataFrame(columns=["pfr_player_id", "season", "week", "offense_pct"]),
    )
    schedule = _schedule(year)
    if year == 2027:
        schedule[["home_score", "away_score"]] = float("nan")
    result = live._load_history(year, 1, schedule)
    assert {"new-sea", "new-ne"} <= set(result.player_id)
    assert len(calls) == 1
    assert 2026 in pd.read_parquet(overlay / f"team_stats_{SUFFIX}.parquet").season.unique()
    assert _hashes(original) == before

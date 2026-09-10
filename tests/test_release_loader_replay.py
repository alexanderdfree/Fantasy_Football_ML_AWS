"""The seal must reject transient fallbacks that pinned historical readers cannot replay."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import external_sources as external
from src.data import loader, nfl_source, release
from src.data.redzone_pbp import _REQUIRED_RZ_PBP_COLUMNS

pytestmark = pytest.mark.unit
SEASONS = [2012, 2013]
SIGNATURE = "2012_2013"
DEPENDENCIES = (
    f"weekly_{SIGNATURE}.parquet",
    f"rosters_{SIGNATURE}.parquet",
    f"rosters_weekly_{SIGNATURE}.parquet",
    f"schedules_{SIGNATURE}.parquet",
    f"snap_counts_{SIGNATURE}.parquet",
    f"injuries_{SIGNATURE}.parquet",
    f"depth_charts_v3_{SIGNATURE}.parquet",
    f"redzone_pbp_v2_{SIGNATURE}.parquet",
    f"ff_opportunity_{SIGNATURE}.parquet",
    f"qbr_weekly_v2_{SIGNATURE}.parquet",
    f"contracts_{SIGNATURE}.parquet",
    "player_id_bridge_v2.parquet",
)


def _empty(columns):
    return pd.DataFrame(
        {
            name: pd.Series(dtype="int64" if name in {"season", "week"} else "object")
            for name in columns
        }
    )


@pytest.fixture
def cached_inputs(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "SEASONS", SEASONS)
    monkeypatch.setenv("FF_DATA_RELEASE", "")
    raw, splits = tmp_path / "raw", tmp_path / "splits"
    raw.mkdir()
    splits.mkdir()
    base = pd.DataFrame(
        {
            "player_id": ["player"] * 2,
            "season": SEASONS,
            "week": [1, 1],
            "position": ["QB"] * 2,
            "recent_team": ["KC"] * 2,
            "season_type": ["REG"] * 2,
            "interceptions": [0, 0],
            "sacks": [0, 0],
            "sack_yards": [0, 0],
            "_weekly_modern_schema_v2": [True, True],
        }
    )
    base.to_parquet(raw / f"weekly_{SIGNATURE}.parquet")
    roster = base[["player_id", "season", "week", "position"]].assign(team="KC", status="ACT")
    for name in ("rosters", "rosters_weekly"):
        roster.to_parquet(raw / f"{name}_{SIGNATURE}.parquet")
    pd.DataFrame({"season": SEASONS, "week": [1, 1], "home_team": ["KC"] * 2}).to_parquet(
        raw / f"schedules_{SIGNATURE}.parquet"
    )
    # 2012 is explicitly absent from this available source, as in production.
    pd.DataFrame(
        {
            "pfr_player_id": ["pfr"],
            "season": [2013],
            "week": [1],
            "position": ["QB"],
            "offense_snaps": [5],
            "offense_pct": [1.0],
        }
    ).to_parquet(raw / f"snap_counts_{SIGNATURE}.parquet")
    pd.DataFrame({"pfr_id": ["pfr"], "gsis_id": ["player"]}).to_parquet(
        raw / "player_id_bridge_v2.parquet"
    )
    _empty(["gsis_id", "season", "week", "practice_status", "report_status"]).to_parquet(
        raw / f"injuries_{SIGNATURE}.parquet"
    )
    pd.DataFrame(
        {
            "gsis_id": ["player"] * 2,
            "season": SEASONS,
            "week": [1, 1],
            "formation": ["Offense"] * 2,
            "depth_team": ["1"] * 2,
        }
    ).to_parquet(raw / f"depth_charts_v3_{SIGNATURE}.parquet")
    _empty(_REQUIRED_RZ_PBP_COLUMNS).to_parquet(raw / f"redzone_pbp_v2_{SIGNATURE}.parquet")
    _empty(
        ["player_id", "season", "week", *external.FF_OPP_FEATURE_COLUMNS, "_ff_opportunity_v2"]
    ).to_parquet(raw / f"ff_opportunity_{SIGNATURE}.parquet")
    _empty(["player_id", "season", "week", *external.QBR_FEATURE_COLUMNS]).to_parquet(
        raw / f"qbr_weekly_v2_{SIGNATURE}.parquet"
    )
    _empty(
        [
            "player_id",
            "season",
            *external.CONTRACT_FEATURE_COLUMNS,
            external._CONTRACT_TIEBREAK_SENTINEL,
        ]
    ).to_parquet(raw / f"contracts_{SIGNATURE}.parquet")
    for name in release.SPLIT_NAMES:
        base.to_parquet(splits / name)
    calls = []

    def unavailable(*args, **kwargs):
        calls.append("network")
        raise OSError("source outage")

    for name in (
        "ff_opportunity",
        "weekly_data",
        "rosters",
        "rosters_weekly",
        "schedules",
        "snap_counts",
        "player_ids",
        "player_metadata",
        "injuries",
        "depth_charts",
        "contracts",
        "pbp_data",
    ):
        monkeypatch.setattr(nfl_source, name, unavailable)
    monkeypatch.setattr(external, "_fetch_qbr_weekly_raw", unavailable)
    return raw, splits, calls


def test_complete_cached_empty_sources_seal_and_replay_without_changing_bytes(
    cached_inputs, monkeypatch
):
    raw, splits, calls = cached_inputs
    before = loader.load_raw_data(SEASONS, cache_dir=str(raw))
    hashes = {path.name: release._hash(path) for path in raw.iterdir()}
    manifest = release.seal_inputs(raw_dir=raw, splits_dir=splits)
    assert (splits / release.SEAL_NAME).is_file()
    assert manifest["coverage"][f"raw/snap_counts_{SIGNATURE}.parquet"][
        "global_seasons_absent"
    ] == [2012]
    assert manifest["coverage"][f"raw/ff_opportunity_{SIGNATURE}.parquet"]["rows"] == 0
    monkeypatch.setenv("FF_DATA_RELEASE", "a" * 64)
    after = loader.load_raw_data(SEASONS, cache_dir=str(raw))
    pd.testing.assert_frame_equal(before, after)
    assert hashes == {path.name: release._hash(path) for path in raw.iterdir()}
    assert calls == []


@pytest.mark.parametrize("name", DEPENDENCIES)
def test_every_missing_unconditional_loader_cache_blocks_seal(cached_inputs, name):
    raw, splits, calls = cached_inputs
    (raw / name).unlink()
    with pytest.raises(release.DataReleaseError, match="missing or incompatible"):
        release.seal_inputs(raw_dir=raw, splits_dir=splits)
    assert not (splits / release.SEAL_NAME).exists()
    assert calls == []
    # The directory-scoped gate cleans up after failure.
    release.assert_source_fetch_allowed(raw / name)


@pytest.mark.parametrize(
    "name",
    [
        f"ff_opportunity_{SIGNATURE}.parquet",
        f"qbr_weekly_v2_{SIGNATURE}.parquet",
        f"contracts_{SIGNATURE}.parquet",
        f"redzone_pbp_v2_{SIGNATURE}.parquet",
    ],
)
def test_stale_optional_schema_cannot_be_sealed(cached_inputs, name):
    raw, splits, calls = cached_inputs
    pd.DataFrame({"old_schema": [1]}).to_parquet(raw / name)
    with pytest.raises(release.DataReleaseError):
        release.seal_inputs(raw_dir=raw, splits_dir=splits)
    assert not (splits / release.SEAL_NAME).exists()
    assert calls == []


def test_uncached_optional_outage_blocks_prewarm_before_other_producers(cached_inputs, monkeypatch):
    from src import config
    from src.data import identity

    raw, _, calls = cached_inputs
    (raw / f"ff_opportunity_{SIGNATURE}.parquet").unlink()
    assert external.load_ff_opportunity(SEASONS, cache_dir=str(raw)).empty
    assert calls == ["network"]
    calls.clear()
    monkeypatch.setattr(config, "CACHE_DIR", str(raw))
    monkeypatch.setattr(
        identity,
        "load_player_id_bridge",
        lambda *a: pytest.fail("prewarm retried a producer before replay validation"),
    )
    with pytest.raises(release.DataReleaseError):
        release.prewarm_training_dependencies()
    assert calls == []


def test_conditional_identity_metadata_is_verified_when_needed(cached_inputs):
    raw, splits, calls = cached_inputs
    pd.DataFrame({"pfr_id": [], "gsis_id": []}).to_parquet(raw / "player_id_bridge_v2.parquet")
    with pytest.raises(release.DataReleaseError, match="player_metadata_v1"):
        release.seal_inputs(raw_dir=raw, splits_dir=splits)
    assert not (splits / release.SEAL_NAME).exists()
    assert calls == []


def test_replay_context_is_directory_scoped_and_overrides_legacy_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_DATA_RELEASE", "legacy")
    with release.require_cached_sources(tmp_path / "historical"):
        with pytest.raises(release.DataReleaseError):
            release.assert_source_fetch_allowed(tmp_path / "historical/missing.parquet")
        release.assert_source_fetch_allowed(tmp_path / "live/missing.parquet")
    release.assert_source_fetch_allowed(tmp_path / "historical/missing.parquet")

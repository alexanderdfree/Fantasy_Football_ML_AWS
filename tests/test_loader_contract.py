"""Schema / contract tests for ``src.data.loader.load_raw_data`` output.

This is the gate that catches silent schema drift from ``nfl_data_py`` or the
nflverse GitHub fallback before it propagates through every downstream
feature. It runs against a checked-in fixture parquet
(``tests/fixtures/weekly_2023_w1.parquet``) so it needs no network access.

Markers
-------
* ``@pytest.mark.unit`` — pure fixture-parquet inspection.
* ``@pytest.mark.integration`` — tests that exercise ``load_raw_data`` itself
  (mocked so they don't hit the network).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
import pytest

# sys.path wiring for ``src.*`` imports is handled in ``tests/conftest.py``.

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "weekly_2023_w1.parquet"

# Columns that every downstream feature in this repo expects. Extra columns
# are allowed (logged, not failed). Missing columns are a contract break.
REQUIRED_COLUMNS: set[str] = {
    "player_id",
    "season",
    "week",
    "position",
    "recent_team",
    "opponent_team",
    "fantasy_points",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "targets",
    "attempts",
    "carries",
}


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weekly_fixture() -> pd.DataFrame:
    """Load the checked-in weekly-stats fixture.

    Scoped at module level so every test reuses the same frame — these tests
    are read-only inspections.
    """
    assert FIXTURE_PATH.exists(), (
        f"Fixture missing at {FIXTURE_PATH}. See tests/fixtures/README.md "
        "for regeneration instructions."
    )
    return pd.read_parquet(FIXTURE_PATH)


# ---------------------------------------------------------------------------
# Unit tests — pure fixture inspection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fixture_loads_and_nonempty(weekly_fixture: pd.DataFrame) -> None:
    assert len(weekly_fixture) > 0, "Fixture is empty"
    assert len(weekly_fixture.columns) > 0, "Fixture has no columns"


@pytest.mark.unit
def test_required_columns_present(weekly_fixture: pd.DataFrame) -> None:
    """All whitelisted required columns must appear. Extras are allowed —
    we only want to fail on contract breaks, not on ``nfl_data_py`` adding
    new stats columns.
    """
    actual = set(weekly_fixture.columns)
    missing = REQUIRED_COLUMNS - actual
    assert not missing, (
        f"Loader output is missing {len(missing)} required columns: "
        f"{sorted(missing)}. This is a schema break — a downstream feature "
        "will silently NaN-fill or KeyError."
    )


@pytest.mark.unit
def test_key_dtypes(weekly_fixture: pd.DataFrame) -> None:
    """Pin the dtypes that feature engineering depends on.

    We assert *kind* rather than exact width (any int / any float) because
    ``nfl_data_py`` has churned between ``int32`` and ``int64`` historically
    and we don't want to wedge CI on width changes that are semantically
    identical.
    """
    df = weekly_fixture

    # player_id: string-like (pandas stores as object or pyarrow-backed str)
    assert ptypes.is_object_dtype(df["player_id"]) or ptypes.is_string_dtype(df["player_id"]), (
        f"player_id should be str/object, got {df['player_id'].dtype}"
    )

    # season / week: any int width
    assert ptypes.is_integer_dtype(df["season"]), (
        f"season should be integer, got {df['season'].dtype}"
    )
    assert ptypes.is_integer_dtype(df["week"]), f"week should be integer, got {df['week'].dtype}"

    # fantasy_points: any float width
    assert ptypes.is_float_dtype(df["fantasy_points"]), (
        f"fantasy_points should be float, got {df['fantasy_points'].dtype}"
    )

    # position: string-like
    assert ptypes.is_object_dtype(df["position"]) or ptypes.is_string_dtype(df["position"]), (
        f"position should be str/object, got {df['position'].dtype}"
    )


@pytest.mark.unit
def test_value_ranges(weekly_fixture: pd.DataFrame) -> None:
    """Bound the domains of the keys used by the train/val/test split and
    sanity-check the fantasy-points target.

    If ``nfl_data_py`` ever starts emitting e.g. week=0 for playoff byes or
    season=9999 as a sentinel, this test is what tells us.
    """
    df = weekly_fixture

    weeks = df["week"]
    assert weeks.min() >= 1 and weeks.max() <= 22, (
        f"week out of range: [{weeks.min()}, {weeks.max()}] not in [1, 22]"
    )

    seasons = df["season"]
    assert seasons.min() >= 2000 and seasons.max() <= 2100, (
        f"season out of range: [{seasons.min()}, {seasons.max()}] not in [2000, 2100]"
    )

    fp = df["fantasy_points"]
    assert fp.min() >= -10 and fp.max() <= 80, (
        f"fantasy_points out of range: [{fp.min()}, {fp.max()}] not in [-10, 80]"
    )


@pytest.mark.unit
def test_primary_key_unique(weekly_fixture: pd.DataFrame) -> None:
    """(player_id, season, week) must be unique — it's the join key used
    everywhere downstream (rolling features, opponent merges, etc).
    """
    dupes = weekly_fixture.duplicated(subset=["player_id", "season", "week"], keep=False)
    assert not dupes.any(), (
        f"Primary key (player_id, season, week) is not unique: "
        f"{dupes.sum()} duplicate rows. Offending keys:\n"
        f"{weekly_fixture.loc[dupes, ['player_id', 'season', 'week']].head()}"
    )


@pytest.mark.unit
def test_no_null_primary_key_or_position(weekly_fixture: pd.DataFrame) -> None:
    """Null player_id / season / week / position would silently drop rows
    from any groupby-based feature.
    """
    for col in ["player_id", "season", "week", "position"]:
        nulls = weekly_fixture[col].isna().sum()
        assert nulls == 0, f"{col} has {nulls} null values (should be 0)"


@pytest.mark.unit
def test_parquet_roundtrip(weekly_fixture: pd.DataFrame, tmp_path: Path) -> None:
    """Writing the fixture back out and re-reading must give an identical
    frame. This guards the cache code path in ``load_raw_data``, which
    reuses parquet between runs.
    """
    out = tmp_path / "roundtrip.parquet"
    weekly_fixture.to_parquet(out, compression="snappy")
    loaded = pd.read_parquet(out)

    assert list(loaded.columns) == list(weekly_fixture.columns)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        weekly_fixture.reset_index(drop=True),
        check_dtype=True,
    )


# ---------------------------------------------------------------------------
# Integration tests — exercise load_raw_data itself
# ---------------------------------------------------------------------------


# Columns the loader adds via its merge steps (not present in
# ``nfl.import_weekly_data`` output).
_LOADER_MERGE_ADDED = {"snap_pct", "practice_status", "game_status", "depth_chart_rank"}


def _empty_frame(columns: dict[str, str]) -> pd.DataFrame:
    """Build an empty DataFrame with typed columns for cache seeding."""
    return pd.DataFrame({name: pd.Series([], dtype=dtype) for name, dtype in columns.items()})


@pytest.mark.integration
def test_load_raw_data_uses_cache_and_returns_fixture_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise ``load_raw_data`` end-to-end against a cache-only setup so
    no network calls happen, and assert the returned frame obeys the same
    schema contract the fixture does.

    Every parquet cache is pre-seeded in ``tmp_path``. The only nfl_data_py
    call that still happens (``nfl.import_ids``, used for the snap-count
    merge bridge) is stubbed.
    """
    from src.data import loader as loader_module

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    seasons = [2023]
    suffix = f"{seasons[0]}_{seasons[-1]}"
    fixture = pd.read_parquet(FIXTURE_PATH)

    # Pre-merge weekly cache: drop the columns the loader *adds* during its
    # own merge steps so we don't collide on re-merge.
    weekly_pre = fixture.drop(columns=[c for c in _LOADER_MERGE_ADDED if c in fixture.columns])
    weekly_pre.to_parquet(cache_dir / f"weekly_{suffix}.parquet")

    # Each supporting cache is empty — we only need the loader's merge
    # steps to no-op so the weekly frame passes through cleanly. Dtypes
    # matter: snap_counts.pfr_player_id must be object to avoid a merge
    # dtype mismatch against the id-bridge frame.
    cache_shapes: dict[str, dict[str, str]] = {
        f"rosters_{suffix}": {"player_id": "object", "season": "int32", "position": "object"},
        f"schedules_{suffix}": {
            "season": "int32",
            "week": "int32",
            "home_team": "object",
            "away_team": "object",
        },
        f"snap_counts_{suffix}": {
            "pfr_player_id": "object",
            "season": "int32",
            "week": "int32",
            "offense_pct": "float64",
        },
        f"injuries_{suffix}": {
            "gsis_id": "object",
            "season": "int32",
            "week": "int32",
            "practice_status": "object",
            "report_status": "object",
        },
        f"depth_charts_{suffix}": {
            "gsis_id": "object",
            "season": "int32",
            "week": "int32",
            "formation": "object",
            "depth_team": "object",
        },
    }
    for stem, columns in cache_shapes.items():
        _empty_frame(columns).to_parquet(cache_dir / f"{stem}.parquet")

    # The one remaining nfl_data_py call when all caches exist.
    monkeypatch.setattr(
        loader_module.nfl,
        "import_ids",
        lambda: _empty_frame({"pfr_id": "object", "gsis_id": "object"}),
    )

    df = loader_module.load_raw_data(seasons=seasons, cache_dir=str(cache_dir))

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"loader output missing required columns: {missing}"

    for col in _LOADER_MERGE_ADDED:
        assert col in df.columns, f"loader didn't add {col!r} during merge"

    assert ptypes.is_integer_dtype(df["season"])
    assert ptypes.is_integer_dtype(df["week"])
    assert ptypes.is_float_dtype(df["fantasy_points"])

    assert df["week"].between(1, 22).all()
    assert df["season"].between(2000, 2100).all()

    assert not df.duplicated(["player_id", "season", "week"]).any()


@pytest.mark.integration
def test_nflverse_github_fallback_schema_matches_primary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify loader's 2025+ fallback path (nflverse GitHub release parquet)
    returns a frame matching the same required-column contract as the
    nfl_data_py primary path.

    Mocking strategy: ``loader.pd.read_parquet`` is the single seam used both
    for the URL fetch (string path) and for cache reads (Path / .parquet).
    The fake dispatches on ``"stats_player_week_"`` substring — URL fetches
    return the 2025 schema (with pre-rename column names), cache reads
    delegate to the real ``pd.read_parquet`` so previously-written caches
    keep working.
    """
    from src.data import loader as loader_module

    # Pre-write every supporting cache so the only network-shaped read is the
    # 2025 nflverse parquet URL.
    seasons = [2025]
    suffix = f"{seasons[0]}_{seasons[-1]}"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    cache_shapes: dict[str, dict[str, str]] = {
        f"rosters_{suffix}": {"player_id": "object", "season": "int32", "position": "object"},
        f"schedules_{suffix}": {
            "season": "int32",
            "week": "int32",
            "home_team": "object",
            "away_team": "object",
        },
        f"snap_counts_{suffix}": {
            "pfr_player_id": "object",
            "season": "int32",
            "week": "int32",
            "offense_pct": "float64",
        },
        f"injuries_{suffix}": {
            "gsis_id": "object",
            "season": "int32",
            "week": "int32",
            "practice_status": "object",
            "report_status": "object",
        },
        f"depth_charts_{suffix}": {
            "gsis_id": "object",
            "season": "int32",
            "week": "int32",
            "formation": "object",
            "depth_team": "object",
        },
    }
    for stem, columns in cache_shapes.items():
        _empty_frame(columns).to_parquet(cache_dir / f"{stem}.parquet")

    real_read_parquet = pd.read_parquet
    url_calls: list[str] = []

    def _fake_read_parquet(path, *args, **kwargs):
        s = str(path)
        if "stats_player_week_" in s:
            url_calls.append(s)
            # Why: minimal frame matching every column the URL branch needs
            # to rename, plus enough required-contract columns to assert the
            # loader output passes the same gate as the primary path.
            return pd.DataFrame(
                {
                    "player_id": ["P00", "P01"],
                    "season": [2025, 2025],
                    "week": [1, 2],
                    "position": ["QB", "RB"],
                    "team": ["KC", "SF"],  # → recent_team
                    "opponent_team": ["SF", "KC"],
                    "fantasy_points": [22.5, 14.0],
                    "passing_yards": [300.0, 0.0],
                    "passing_tds": [2, 0],
                    "passing_interceptions": [1, 0],  # → interceptions
                    "sacks_suffered": [2, 0],  # → sacks
                    "sack_yards_lost": [14, 0],  # → sack_yards
                    "rushing_yards": [10.0, 80.0],
                    "rushing_tds": [0, 1],
                    "receiving_yards": [0.0, 20.0],
                    "receiving_tds": [0, 0],
                    "receptions": [0, 3],
                    "targets": [0, 4],
                    "attempts": [40, 0],
                    "carries": [3, 18],
                }
            )
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(loader_module.pd, "read_parquet", _fake_read_parquet)
    monkeypatch.setattr(
        loader_module.nfl,
        "import_ids",
        lambda: _empty_frame({"pfr_id": "object", "gsis_id": "object"}),
    )

    df = loader_module.load_raw_data(seasons=seasons, cache_dir=str(cache_dir))

    # The URL branch must have fired (otherwise we proved nothing).
    assert any("stats_player_week_2025" in url for url in url_calls), (
        f"2025 URL branch never executed; only saw: {url_calls}"
    )

    # Same schema gate the primary-path test asserts.
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"2025 fallback path missing required columns: {missing}"

    # Renames must have happened.
    assert "team" not in df.columns, "pre-rename column 'team' leaked through"
    assert "passing_interceptions" not in df.columns
    assert df["recent_team"].iloc[0] == "KC"

    # Every loader-merge-added column lands.
    for col in _LOADER_MERGE_ADDED:
        assert col in df.columns, f"2025 path didn't add {col!r} during merge"

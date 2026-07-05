"""Unit tests for src.data.nflcom_loader.

All tests run offline — the upstream NFL.com fetch is mocked via the loader's
injectable ``reader=`` kwarg, and ``import_seasonal_rosters`` is bypassed by
passing ``rosters=`` directly to ``load_nflcom_with_gsis_id``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import pytest

from src.data.nflcom_loader import (
    _CACHE_VERSION,
    _FUM_LOST_RATIO,
    NFLCOM_DEFAULT_WEEKS,
    NFLCOM_POSITIONS,
    _build_roster_lookup,
    _team_abbr_normalize,
    load_nflcom_projections,
    load_nflcom_with_gsis_id,
    normalize_player_name,
)

FIXTURE_QB = Path(__file__).parent / "fixtures" / "nflcom_QB_projected_sample.csv"

pytestmark = pytest.mark.unit


# ---------- Pure-function tests --------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Josh Allen", "josh allen"),
        ("Patrick Mahomes II", "patrick mahomes"),
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Robert Griffin III", "robert griffin"),
        ("Ja'Marr Chase", "jamarr chase"),
        ("A.J. Brown", "aj brown"),
        ("D.J. Moore", "dj moore"),
        ("DK Metcalf", "dk metcalf"),
        ("Foo  Bar", "foo bar"),  # double-space collapsed
        ("  Spaced  ", "spaced"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_player_name(raw, expected):
    assert normalize_player_name(raw) == expected


def test_normalize_player_name_handles_nan():
    assert normalize_player_name(float("nan")) == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("OAK", "LV"),
        ("SD", "LAC"),
        ("STL", "LAR"),
        ("WSH", "WAS"),
        ("WAS", "WAS"),
        ("JAX", "JAX"),
        ("JAC", "JAX"),
        ("LA", "LAR"),
        ("@KC", "KC"),  # opponent prefix stripped
        ("kc", "KC"),  # case-normalized
        ("", ""),
        (None, ""),
    ],
)
def test_team_abbr_normalize(raw, expected):
    assert _team_abbr_normalize(raw) == expected


def test_schedule_team_code_normalization_is_nflverse_variant():
    """F104: the schedule-join variant is derived from the canonical base map
    but uses the Rams' nflverse code (STL -> LA, not LAR), and leaves the
    modern ``LA`` untouched (no LA -> LAR rewrite that would break the join)."""
    from src.data.nflcom_loader import schedule_team_code_normalization

    sched_map = schedule_team_code_normalization()
    # Exactly the three relocated franchises, with the nflverse Rams code.
    assert sched_map == {"OAK": "LV", "SD": "LAC", "STL": "LA"}
    # The base NFL.com/roster map keeps STL -> LAR and rewrites LA -> LAR;
    # the schedule variant must NOT, or the Rams join against player rows
    # (which carry "LA") silently misses.
    assert "LA" not in sched_map


# ---------- Loader (parser) tests -----------------------------------------------


def _fixture_reader_qb_only(url: str) -> pd.DataFrame:
    """Reader that returns the QB fixture for a specific URL only.

    Any other URL (other positions, other weeks) raises 404, simulating the
    upstream where most (year, week, pos) combos don't have data.
    """
    if "/2024/1/projected/QB_projected.csv" in url:
        return pd.read_csv(FIXTURE_QB)
    raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)


def test_normalize_one_position_handles_nan_team():
    """A NaN Team or PlayerOpponent must normalize to '' (empty), not the literal
    string 'NAN' (which would happen if we cast to str before normalizing)."""
    from src.data.nflcom_loader import _normalize_one_position

    raw = pd.DataFrame(
        [
            {
                "PlayerName": "Test",
                "PlayerId": 999,
                "Pos": "QB",
                "Team": float("nan"),
                "PlayerOpponent": float("nan"),
                "PassingYDS": 200.0,
                "PassingTD": 1.0,
                "PassingInt": 0.0,
                "RushingYDS": 0.0,
                "RushingTD": 0.0,
                "Fum": 0.0,
                "PlayerWeekProjectedPts": 8.0,
                "ProjectedRank": 1,
                "season": 2025,
                "week": 1,
                "position": "QB",
            }
        ]
    )
    out = _normalize_one_position(raw, "QB")
    assert out["team"].iloc[0] == ""
    assert out["opponent"].iloc[0] == ""


def test_load_projections_parses_fixture_schema(tmp_path):
    df = load_nflcom_projections(
        seasons=[2024],
        weeks=[1],
        cache_dir=str(tmp_path),
        reader=_fixture_reader_qb_only,
    )
    # All target columns present and numeric.
    expected_cols = {
        "season",
        "week",
        "position",
        "nflcom_player_id",
        "player_name",
        "team",
        "opponent",
        "nflcom_projected_pts",
        "nflcom_projected_rank",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "fumbles_lost",
    }
    assert expected_cols.issubset(df.columns)
    # Five fixture rows, all QB.
    assert len(df) == 5
    assert (df["position"] == "QB").all()
    # Spot-check Josh Allen's mapped values.
    allen = df[df["player_name"] == "Josh Allen"].iloc[0]
    assert allen["passing_yards"] == pytest.approx(256.85)
    assert allen["passing_tds"] == pytest.approx(1.72)
    assert allen["interceptions"] == pytest.approx(0.89)
    assert allen["rushing_yards"] == pytest.approx(34.9)
    assert allen["rushing_tds"] == pytest.approx(0.57)
    # Fum=0.57 → fumbles_lost = 0.57 × 0.5 = 0.285
    assert allen["fumbles_lost"] == pytest.approx(0.57 * _FUM_LOST_RATIO)
    # PlayerWeekProjectedPts preserved.
    assert allen["nflcom_projected_pts"] == pytest.approx(21.14)
    # QB rows have receiving_* = 0.
    assert (df["receiving_yards"] == 0).all()
    assert (df["receptions"] == 0).all()


def test_load_projections_handles_404_gracefully(tmp_path):
    """Reader that raises 404 on every position; loader should fail loud only
    when *no* rows are fetched (else degrade to whatever subset succeeded)."""

    def stub(url: str):
        # Succeed only for QB week 1; 404 everything else.
        if "/2024/1/projected/QB_projected.csv" in url:
            return pd.read_csv(FIXTURE_QB)
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    df = load_nflcom_projections(
        seasons=[2024],
        weeks=[1, 2],
        cache_dir=str(tmp_path),
        reader=stub,
    )
    # Only the QB-week-1 stub succeeded; 404s were logged + skipped.
    assert len(df) == 5
    assert df["week"].unique().tolist() == [1]
    assert df["position"].unique().tolist() == ["QB"]


def test_load_projections_raises_when_zero_rows(tmp_path):
    """If every fetch 404s, raise so the caller doesn't silently get an empty frame."""

    def all_404(url: str):
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with pytest.raises(RuntimeError, match="No NFL.com projection rows"):
        load_nflcom_projections(seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=all_404)


def test_load_projections_uses_cache(tmp_path):
    # First call: write cache.
    load_nflcom_projections(
        seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=_fixture_reader_qb_only
    )
    # Cache key includes a weeks signature (F73): weeks=[1] -> "w1-1".
    cache_file = tmp_path / f"nflcom_projections_{_CACHE_VERSION}_2024_2024_w1-1.parquet"
    assert cache_file.exists()

    # Second call: reader raises if invoked. Cache hit must skip it entirely.
    def boom(url: str):
        raise AssertionError(f"reader should not be called on cache hit; got {url}")

    df = load_nflcom_projections(seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=boom)
    assert len(df) == 5


def test_load_projections_cache_key_varies_with_weeks(tmp_path):
    """F73: a different ``weeks`` range must not collide on the earlier cache.

    A first call for weeks=[1] then a second for weeks=[1, 2] must write
    distinct cache files (keyed by the weeks signature) so the second call
    re-fetches rather than silently returning the narrower frame's coverage.
    """
    load_nflcom_projections(
        seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=_fixture_reader_qb_only
    )
    # Distinct week range -> distinct cache key, so the reader is consulted
    # again (a stale-key collision would instead boom-skip the network).
    load_nflcom_projections(
        seasons=[2024], weeks=[1, 2], cache_dir=str(tmp_path), reader=_fixture_reader_qb_only
    )
    files = sorted(p.name for p in tmp_path.glob("nflcom_projections_v1_*.parquet"))
    assert files == [
        "nflcom_projections_v1_2024_2024_w1-1.parquet",
        "nflcom_projections_v1_2024_2024_w1-2.parquet",
    ], files


def test_load_projections_cache_key_disambiguates_sparse_seasons(tmp_path):
    """#1398: a sparse season list must not collide on the contiguous
    superset's cache file (the #488 defect class). ``[2023, 2025]`` and the
    contiguous ``[2023, 2024, 2025]`` share min/max but must write distinct
    cache files — otherwise a later full-range call silently returns the
    2-season partial frame.
    """

    def sparse_reader(url: str) -> pd.DataFrame:
        # Serve the QB fixture for week 1 of any of the three seasons; 404 else.
        if "/1/projected/QB_projected.csv" in url and any(
            f"/{y}/1/" in url for y in (2023, 2024, 2025)
        ):
            return pd.read_csv(FIXTURE_QB)
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    # Sparse [2023, 2025] -> hashed signature (NOT the bare 2023_2025).
    load_nflcom_projections(
        seasons=[2023, 2025], weeks=[1], cache_dir=str(tmp_path), reader=sparse_reader
    )
    # Contiguous superset [2023, 2024, 2025] -> plain 2023_2025 signature.
    load_nflcom_projections(
        seasons=[2023, 2024, 2025], weeks=[1], cache_dir=str(tmp_path), reader=sparse_reader
    )
    files = sorted(p.name for p in tmp_path.glob("nflcom_projections_v1_*.parquet"))
    # Two distinct files: the contiguous key is the legacy {min}_{max}; the
    # sparse key carries the disambiguating {len}_{hash} suffix.
    assert len(files) == 2, files
    assert f"nflcom_projections_{_CACHE_VERSION}_2023_2025_w1-1.parquet" in files
    sparse = [f for f in files if f.startswith(f"nflcom_projections_{_CACHE_VERSION}_2023_2025_2_")]
    assert len(sparse) == 1, files


def test_load_projections_contiguous_cache_key_is_legacy_min_max(tmp_path):
    """#1398: the contiguous-range fix must be byte-identical to the legacy
    ``{min}_{max}`` key so existing caches (and the filename pins below) stay
    valid — the fix only *adds* a suffix for sparse lists."""
    load_nflcom_projections(
        seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=_fixture_reader_qb_only
    )
    assert (tmp_path / f"nflcom_projections_{_CACHE_VERSION}_2024_2024_w1-1.parquet").exists()


def test_load_projections_force_refresh_bypasses_cache(tmp_path):
    load_nflcom_projections(
        seasons=[2024], weeks=[1], cache_dir=str(tmp_path), reader=_fixture_reader_qb_only
    )
    # Thread-safe counter — the loader fan-outs fetches across a thread pool.
    import threading

    lock = threading.Lock()
    counter = {"n": 0}

    def counting_reader(url: str):
        with lock:
            counter["n"] += 1
        return _fixture_reader_qb_only(url)

    load_nflcom_projections(
        seasons=[2024],
        weeks=[1],
        cache_dir=str(tmp_path),
        reader=counting_reader,
        force_refresh=True,
    )
    # Reader was called once per (year, week, position) attempt = 1 × 1 × 5 positions.
    assert counter["n"] == len(NFLCOM_POSITIONS)


def test_read_one_projection_retries_transient_then_succeeds():
    """A non-404 HTTPError on first attempt should be retried; second attempt wins."""
    from src.data.nflcom_loader import _read_one_projection

    state = {"n": 0}

    def stub(url: str):
        state["n"] += 1
        if state["n"] == 1:
            raise HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=None)
        return pd.read_csv(FIXTURE_QB)

    df = _read_one_projection(2024, 1, "QB", reader=stub, backoff_s=0.0)
    assert df is not None
    assert state["n"] == 2  # retried once
    assert len(df) == 5


def test_read_one_projection_does_not_retry_404():
    """404 means 'this week doesn't exist' — must not retry, must return None."""
    from src.data.nflcom_loader import _read_one_projection

    state = {"n": 0}

    def stub(url: str):
        state["n"] += 1
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    df = _read_one_projection(2024, 1, "QB", reader=stub, backoff_s=0.0)
    assert df is None
    assert state["n"] == 1  # NOT retried


def test_read_one_projection_gives_up_after_max_retries():
    """Persistent transient errors should ultimately give up and return None."""
    from src.data.nflcom_loader import _read_one_projection

    state = {"n": 0}

    def stub(url: str):
        state["n"] += 1
        raise HTTPError(url, 503, "Service Unavailable", hdrs=None, fp=None)

    df = _read_one_projection(2024, 1, "QB", reader=stub, max_retries=1, backoff_s=0.0)
    assert df is None
    assert state["n"] == 2  # initial + 1 retry


def test_load_projections_empty_seasons_raises():
    with pytest.raises(ValueError, match="non-empty"):
        load_nflcom_projections(seasons=[])


def test_default_weeks_cover_full_regular_season():
    # NFL regular season is 18 weeks since 2021.
    assert tuple(range(1, 19)) == NFLCOM_DEFAULT_WEEKS


# ---------- Roster lookup + gsis_id join tests ----------------------------------


def _make_rosters(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a synthetic rosters-shaped frame.

    nflverse's import_seasonal_rosters columns we rely on: player_id (gsis_id),
    player_name, team, position, season.
    """
    return pd.DataFrame(rows)


def test_build_roster_lookup_normalizes_and_dedups():
    rosters = _make_rosters(
        [
            {
                "player_id": "00-0034796",
                "player_name": "Patrick Mahomes II",  # suffix gets stripped
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-0034796",  # same player, dup row
                "player_name": "Patrick Mahomes II",
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-0036345",
                "player_name": "Ja'Marr Chase",
                "team": "CIN",
                "position": "WR",
                "season": 2024,
            },
        ]
    )
    lookup = _build_roster_lookup(rosters)
    # Dedup'd: one Mahomes row.
    assert len(lookup) == 2
    mahomes = lookup[lookup["norm_name"] == "patrick mahomes"].iloc[0]
    assert mahomes["player_id"] == "00-0034796"
    chase = lookup[lookup["norm_name"] == "jamarr chase"].iloc[0]
    assert chase["player_id"] == "00-0036345"


def test_build_roster_lookup_accepts_team_abbr_column():
    """nflverse seasonal_rosters uses team_abbr; we tolerate that too."""
    rosters = pd.DataFrame(
        [
            {
                "player_id": "00-001",
                "player_name": "Test Guy",
                "team_abbr": "BUF",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    lookup = _build_roster_lookup(rosters)
    assert lookup.iloc[0]["team"] == "BUF"


def test_build_roster_lookup_drops_blank_names_and_ids():
    rosters = pd.DataFrame(
        [
            {
                "player_id": "",
                "player_name": "Nobody",
                "team": "X",
                "position": "QB",
                "season": 2024,
            },
            {"player_id": "id1", "player_name": "", "team": "X", "position": "QB", "season": 2024},
            {
                "player_id": "id2",
                "player_name": "Real Player",
                "team": "X",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    lookup = _build_roster_lookup(rosters)
    assert len(lookup) == 1
    assert lookup.iloc[0]["player_id"] == "id2"


def test_load_with_gsis_id_team_match(tmp_path):
    rosters = _make_rosters(
        [
            {
                "player_id": "00-AALLEN",
                "player_name": "Josh Allen",
                "team": "BUF",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-MAHOMES",
                "player_name": "Patrick Mahomes II",
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-LAMAR",
                "player_name": "Lamar Jackson",
                "team": "BAL",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-AJM",
                "player_name": "AJ McCarron",  # rosters lacks the dots
                "team": "CIN",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-WINSTON",
                "player_name": "Jameis Winston",
                "team": "CLE",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    df = load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
    )
    # All five fixture rows match.
    assert df["player_id"].notna().sum() == 5
    # Spot-check normalized join.
    mahomes = df[df["player_name"] == "Patrick Mahomes II"].iloc[0]
    assert mahomes["player_id"] == "00-MAHOMES"
    aj = df[df["player_name"] == "A.J. McCarron"].iloc[0]
    assert aj["player_id"] == "00-AJM"


def test_load_with_gsis_id_position_fallback(tmp_path):
    """Player on a different team than NFL.com lists (mid-season trade) — should
    still match via (norm_name, season, position) fallback."""
    rosters = _make_rosters(
        [
            # Allen is on KC in our rosters but BUF in NFL.com — primary join misses.
            {
                "player_id": "00-AALLEN",
                "player_name": "Josh Allen",
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-MAHOMES",
                "player_name": "Patrick Mahomes II",
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-LAMAR",
                "player_name": "Lamar Jackson",
                "team": "BAL",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-AJM",
                "player_name": "AJ McCarron",
                "team": "CIN",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-WINSTON",
                "player_name": "Jameis Winston",
                "team": "CLE",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    df = load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
    )
    allen = df[df["player_name"] == "Josh Allen"].iloc[0]
    # Matched by (norm_name, season, position) fallback.
    assert allen["player_id"] == "00-AALLEN"
    # All five still match.
    assert df["player_id"].notna().sum() == 5


def test_load_with_gsis_id_position_fallback_skips_collisions(tmp_path):
    """Two distinct gsis_ids share the same (norm_name, season, position).
    The fallback must NOT silently pick one — both rows stay unmatched.
    """
    # NFL.com fixture has 'Josh Allen' on BUF; rosters give us TWO Josh Allens
    # (different gsis_ids), neither on BUF. Primary join misses; unique-only
    # fallback should refuse to pick.
    rosters = _make_rosters(
        [
            {
                "player_id": "00-AALLEN-1",
                "player_name": "Josh Allen",
                "team": "JAX",  # different team
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-AALLEN-2",
                "player_name": "Josh Allen",  # same normalized name
                "team": "TEN",  # different team
                "position": "QB",
                "season": 2024,
            },
            # Other fixture players match cleanly via primary or fallback.
            {
                "player_id": "00-MAHOMES",
                "player_name": "Patrick Mahomes II",
                "team": "KC",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-LAMAR",
                "player_name": "Lamar Jackson",
                "team": "BAL",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-AJM",
                "player_name": "AJ McCarron",
                "team": "CIN",
                "position": "QB",
                "season": 2024,
            },
            {
                "player_id": "00-WINSTON",
                "player_name": "Jameis Winston",
                "team": "CLE",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    df = load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
        # Lower threshold for this test: only 4/5 will match (Allen stays
        # unmatched), giving us 80%.
        min_match_rate=0.5,
    )
    allen = df[df["player_name"] == "Josh Allen"].iloc[0]
    # Collision should leave Allen unmatched, not silently pick AALLEN-1 or -2.
    assert pd.isna(allen["player_id"])
    # The other 4 still match.
    assert df["player_id"].notna().sum() == 4


def test_load_with_gsis_id_below_threshold_raises(tmp_path):
    """Only one of five fixture players has a roster entry → match rate = 20%
    → should raise."""
    rosters = _make_rosters(
        [
            {
                "player_id": "00-AALLEN",
                "player_name": "Josh Allen",
                "team": "BUF",
                "position": "QB",
                "season": 2024,
            },
        ]
    )
    with pytest.raises(RuntimeError, match="match rate"):
        load_nflcom_with_gsis_id(
            seasons=[2024],
            cache_dir=str(tmp_path),
            rosters=rosters,
            reader=_fixture_reader_qb_only,
        )


def test_load_with_gsis_id_writes_cache(tmp_path):
    rosters = _make_rosters(
        [
            {"player_id": f"00-{i}", "player_name": n, "team": t, "position": "QB", "season": 2024}
            for i, (n, t) in enumerate(
                [
                    ("Josh Allen", "BUF"),
                    ("Patrick Mahomes II", "KC"),
                    ("Lamar Jackson", "BAL"),
                    ("AJ McCarron", "CIN"),
                    ("Jameis Winston", "CLE"),
                ]
            )
        ]
    )
    load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
    )
    # Joined cache key includes min_match_rate (F105): default 0.90 -> "mr90".
    joined_path = tmp_path / f"nflcom_projections_joined_{_CACHE_VERSION}_2024_2024_mr90.parquet"
    assert joined_path.exists()

    # Second call hits the joined cache; reader / rosters not consulted.
    def boom(url):
        raise AssertionError("should not be called")

    df = load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=None,  # would normally call nfl_data_py — but cache short-circuits
        reader=boom,
    )
    assert len(df) == 5


def test_load_with_gsis_id_cache_key_varies_with_min_match_rate(tmp_path):
    """F105: a stricter ``min_match_rate`` must not silently hit an earlier,
    looser-threshold cache. Distinct thresholds -> distinct cache files."""
    rosters = _make_rosters(
        [
            {"player_id": f"00-{i}", "player_name": n, "team": t, "position": "QB", "season": 2024}
            for i, (n, t) in enumerate(
                [
                    ("Josh Allen", "BUF"),
                    ("Patrick Mahomes II", "KC"),
                    ("Lamar Jackson", "BAL"),
                    ("AJ McCarron", "CIN"),
                    ("Jameis Winston", "CLE"),
                ]
            )
        ]
    )
    # Full-match roster, so both thresholds pass — we are checking the cache
    # KEY varies, not the RuntimeError path.
    load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
        min_match_rate=0.80,
    )
    load_nflcom_with_gsis_id(
        seasons=[2024],
        cache_dir=str(tmp_path),
        rosters=rosters,
        reader=_fixture_reader_qb_only,
        min_match_rate=0.99,
    )
    files = sorted(p.name for p in tmp_path.glob("nflcom_projections_joined_v1_*.parquet"))
    assert files == [
        "nflcom_projections_joined_v1_2024_2024_mr80.parquet",
        "nflcom_projections_joined_v1_2024_2024_mr99.parquet",
    ], files


def test_normalize_one_position_missing_optional_columns_no_crash():
    """F27: a CSV missing the projected-pts / rank / stat columns must degrade
    to NaN/0 columns rather than crashing on a scalar-NaN ``.fillna()``."""
    from src.data.nflcom_loader import _normalize_one_position

    # Minimal frame with only the always-required identity columns.
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [1, 1],
            "position": ["QB", "QB"],
            "PlayerId": ["1", "2"],
            "PlayerName": ["Josh Allen", "Patrick Mahomes"],
            "Team": ["BUF", "KC"],
            "PlayerOpponent": ["MIA", "DEN"],
        }
    )
    out = _normalize_one_position(df, "QB")
    # Missing PlayerWeekProjectedPts -> 0.0 (NaN coerced by the .fillna chain).
    assert list(out["nflcom_projected_pts"]) == [0.0, 0.0]
    # Missing ProjectedRank -> all NaN (no fillna applied to rank).
    assert out["nflcom_projected_rank"].isna().all()
    # Missing QB stat columns (PassingYDS, ...) -> 0.0.
    assert list(out["passing_yards"]) == [0.0, 0.0]

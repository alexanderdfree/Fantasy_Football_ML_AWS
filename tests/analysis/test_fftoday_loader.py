"""Unit tests for src.analysis.fftoday_loader.

All tests run offline: the FFToday HTTP fetch is mocked via the loader's
injectable ``reader=`` kwarg (returning crafted FFToday-shaped HTML), and the
nflverse roster fetch is bypassed by passing ``rosters=`` directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import analysis_expert_comparison as aec
from src.analysis.fftoday_loader import (
    FFTODAY_POS_IDS,
    _parse_projection_html,
    load_fftoday_projections,
    load_fftoday_with_gsis_id,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("joined", [False, True])
def test_transient_partial_fftoday_cache_recovers(tmp_path, monkeypatch, joined):
    from urllib.error import HTTPError

    from src.analysis import fftoday_loader as mod

    failing = True

    def reader(url):
        if failing and "GameWeek=2&PosID=30" in url:
            raise HTTPError(url, 503, "temporary outage", {}, None)
        return _fake_reader(url)

    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "FFTODAY_DEFAULT_WEEKS", (1, 2))
    monkeypatch.setattr(mod.nfl_source, "rosters", lambda _: _rosters())

    def load():
        if joined:
            return mod.load_fftoday_with_gsis_id(
                [2013], str(tmp_path), reader=reader, min_match_rate=0.4
            )
        return mod.load_fftoday_projections([2013], cache_dir=str(tmp_path), reader=reader)

    partial = load()
    assert not (partial.position.eq("WR") & partial.week.eq(2)).any()
    assert partial.attrs[mod._FETCH_COMPLETE_ATTR] is False
    assert not list(tmp_path.glob("*.parquet"))
    failing = False
    healed = load()
    assert (healed.position.eq("WR") & healed.week.eq(2)).any()
    assert healed.attrs[mod._FETCH_COMPLETE_ATTR] is True


def test_legacy_partial_fftoday_cache_is_refetched(tmp_path):
    full = load_fftoday_projections(
        [2013], weeks=(1, 2), cache_dir=str(tmp_path), reader=_fake_reader
    )
    path = next(tmp_path.glob("*.parquet"))
    legacy = full.loc[full.week.eq(1)].copy()
    legacy.attrs.clear()
    legacy.to_parquet(path)
    healed = load_fftoday_projections(
        [2013], weeks=(1, 2), cache_dir=str(tmp_path), reader=_fake_reader
    )
    assert set(healed.week) == {1, 2}


def test_custom_rosters_do_not_replace_default_joined_cache(tmp_path, monkeypatch):
    from src.analysis import fftoday_loader as mod

    monkeypatch.setattr(mod, "FFTODAY_DEFAULT_WEEKS", (1,))
    monkeypatch.setattr(mod.nfl_source, "rosters", lambda _: _rosters())
    default = mod.load_fftoday_with_gsis_id([2013], str(tmp_path), reader=_fake_reader)
    path = next(tmp_path.glob("*joined*.parquet"))
    original = path.read_bytes()
    custom = _rosters()
    custom["player_id"] = "custom-" + custom["player_id"]
    injected = mod.load_fftoday_with_gsis_id(
        [2013], str(tmp_path), rosters=custom, reader=_fake_reader
    )
    assert injected.player_id.str.startswith("custom-").all()
    assert path.read_bytes() == original
    resumed = mod.load_fftoday_with_gsis_id([2013], str(tmp_path), reader=_fake_reader)
    pd.testing.assert_frame_equal(default, resumed)


# A FFToday WR row: [Chg, Player(anchor), Team, Opp, rAtt, rYd, rTD, Rec, recYd, recTD, FPts]
_WR_PAGE = """
<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td><td>Team</td><td>Opp</td>
<td>Att</td><td>Yard</td><td>TD</td><td>Rec</td><td>Yard</td><td>TD</td><td>FPts</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/2753/Calvin_Johnson?LeagueID=">Calvin Johnson</a></td>
<td>DET</td><td>MIN</td><td>0.0</td><td>0.0</td><td>0.0</td><td>8.0</td><td>110.0</td><td>1.0</td><td>17.0</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/1000/Jimmy_Graham?LeagueID=">Jimmy Graham</a></td>
<td>NO</td><td>ATL</td><td>0.0</td><td>0.0</td><td>0.0</td><td>9.0</td><td>90.0</td><td>1.0</td><td>15.0</td></tr>
</table>
"""

# A FFToday QB row: [Chg, Player, Team, Opp, Comp, Att, pYd, pTD, INT, rAtt, rYd, rTD, FPts]
_QB_PAGE = """
<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td><td>Team</td><td>Opp</td>
<td>Comp</td><td>Att</td><td>Yard</td><td>TD</td><td>INT</td><td>Att</td><td>Yard</td><td>TD</td><td>FPts</td></tr>
<tr><td>&nbsp;</td><td><a href="/stats/players/2515/Aaron_Rodgers?LeagueID=">Aaron Rodgers</a></td>
<td>GB</td><td>SF</td><td>24.0</td><td>37.0</td><td>290.0</td><td>2.0</td><td>0.0</td><td>4.0</td><td>18.0</td><td>0.0</td><td>24.5</td></tr>
</table>
"""

_EMPTY_PAGE = "<table><tr class='tableclmhdr'><td>Chg</td><td>Player</td></tr></table>"


def _fake_reader(url: str) -> str:
    """Return crafted HTML keyed by the PosID in the URL (offense: QB=10, WR=30)."""
    posid = int(url.split("PosID=")[1].split("&")[0])
    if posid == FFTODAY_POS_IDS["WR"]:
        return _WR_PAGE
    if posid == FFTODAY_POS_IDS["QB"]:
        return _QB_PAGE
    return _EMPTY_PAGE  # RB/TE -> parsed empty -> skipped


# ---------- parsing --------------------------------------------------------------


def test_parse_wr_row_maps_receiving_columns():
    df = _parse_projection_html(_WR_PAGE, "WR")
    cj = df[df["player_name"] == "Calvin Johnson"].iloc[0]
    assert cj["team"] == "DET"
    assert cj["opponent"] == "MIN"
    assert cj["receptions"] == 8.0
    assert cj["receiving_yards"] == 110.0
    assert cj["receiving_tds"] == 1.0
    assert cj["fftoday_projected_pts"] == 17.0
    # WR carries no passing stats -> 0-filled, incl. the always-0 fumbles_lost.
    assert cj["passing_yards"] == 0.0
    assert cj["fumbles_lost"] == 0.0
    assert cj["fftoday_player_id"] == "2753"


def test_parse_qb_row_maps_passing_and_rushing_columns():
    df = _parse_projection_html(_QB_PAGE, "QB")
    ar = df[df["player_name"] == "Aaron Rodgers"].iloc[0]
    assert ar["team"] == "GB"  # correct historical team (not FantasyPros' current team)
    assert ar["passing_yards"] == 290.0
    assert ar["passing_tds"] == 2.0
    assert ar["interceptions"] == 0.0
    assert ar["rushing_yards"] == 18.0
    assert ar["rushing_tds"] == 0.0


def test_parse_skips_rows_without_a_player_anchor():
    # Header/spacer rows (no /stats/players anchor) must not become data rows.
    assert len(_parse_projection_html(_EMPTY_PAGE, "WR")) == 0


# ---------- load + cache ---------------------------------------------------------


def test_load_projections_injected_reader(tmp_path):
    df = load_fftoday_projections([2013], weeks=(1,), cache_dir=str(tmp_path), reader=_fake_reader)
    assert set(df["position"]) == {"QB", "WR"}  # RB/TE returned empty pages -> skipped
    assert {"player_name", "team", "season", "week", "position", "receptions"} <= set(df.columns)
    assert (df["season"] == 2013).all()
    # A second call hits the parquet cache (no reader needed).
    cached = load_fftoday_projections(
        [2013],
        weeks=(1,),
        cache_dir=str(tmp_path),
        reader=lambda u: (_ for _ in ()).throw(AssertionError),
    )
    assert len(cached) == len(df)


def test_cache_key_distinguishes_sampled_from_contiguous_seasons(tmp_path):
    # v1 keyed the cache on min/max season only, so a sampled [2013, 2015] pull
    # silently satisfied a later [2013, 2014, 2015] request — wrong data, no error.
    sampled = load_fftoday_projections(
        [2013, 2015], weeks=(1,), cache_dir=str(tmp_path), reader=_fake_reader
    )
    assert set(sampled["season"]) == {2013, 2015}

    fetched: list[str] = []

    def counting_reader(url: str) -> str:
        fetched.append(url)
        return _fake_reader(url)

    full = load_fftoday_projections(
        [2013, 2014, 2015], weeks=(1,), cache_dir=str(tmp_path), reader=counting_reader
    )
    assert fetched, "contiguous request must not be served from the sampled-seasons cache"
    assert set(full["season"]) == {2013, 2014, 2015}


@pytest.mark.parametrize("dimension", ["season", "week"])
def test_sparse_cache_key_preserves_interior_values(tmp_path, dimension):
    first = [2013, 2014, 2016] if dimension == "season" else [1, 2, 4]
    second = [2013, 2015, 2016] if dimension == "season" else [1, 3, 4]

    def load(values, reader=_fake_reader):
        return load_fftoday_projections(
            values if dimension == "season" else [2013],
            weeks=[1] if dimension == "season" else values,
            cache_dir=str(tmp_path),
            reader=reader,
        )

    load(first)
    actual = load(second)
    assert set(actual[dimension]) == set(second)

    def no_fetch(url):
        raise AssertionError("permuted duplicate input should use the canonical cache")

    cached = load([*reversed(second), second[0]], no_fetch)
    pd.testing.assert_frame_equal(actual, cached)


def test_joined_cache_preserves_sparse_season_membership(tmp_path, monkeypatch):
    from src.analysis import fftoday_loader

    monkeypatch.setattr(
        fftoday_loader.nfl_source,
        "rosters",
        lambda seasons: pd.concat(
            [_rosters().assign(season=s) for s in seasons], ignore_index=True
        ),
    )
    for seasons in ([2013, 2014, 2016], [2013, 2015, 2016]):
        actual = load_fftoday_with_gsis_id(seasons, cache_dir=str(tmp_path), reader=_fake_reader)
        assert set(actual["season"]) == set(seasons)


def test_min_season_guard():
    with pytest.raises(ValueError, match="archive starts at"):
        load_fftoday_projections([2009], weeks=(1,), reader=_fake_reader)


# ---------- gsis-id bridge -------------------------------------------------------


def _rosters():
    return pd.DataFrame(
        [
            {
                "player_name": "Calvin Johnson",
                "season": 2013,
                "team": "DET",
                "position": "WR",
                "player_id": "00-0026035",
            },
            {
                "player_name": "Aaron Rodgers",
                "season": 2013,
                "team": "GB",
                "position": "QB",
                "player_id": "00-0023459",
            },
            # Wrong team on purpose -> exercises the (name, season, position) single-id fallback.
            {
                "player_name": "Jimmy Graham",
                "season": 2013,
                "team": "XXX",
                "position": "WR",
                "player_id": "00-0027686",
            },
        ]
    )


def test_bridge_attaches_gsis_primary_and_fallback(tmp_path):
    joined = load_fftoday_with_gsis_id(
        [2013],
        cache_dir=str(tmp_path),
        rosters=_rosters(),
        min_match_rate=0.0,
        reader=_fake_reader,
    )
    by_name = joined.set_index("player_name")["player_id"].to_dict()
    assert by_name["Calvin Johnson"] == "00-0026035"  # primary (name+season+team+pos)
    assert by_name["Aaron Rodgers"] == "00-0023459"
    assert by_name["Jimmy Graham"] == "00-0027686"  # fallback despite wrong roster team
    # Only matched rows survive.
    assert joined["player_id"].notna().all()


def test_bridge_raises_below_min_match_rate(tmp_path):
    empty_rosters = pd.DataFrame(
        columns=["player_name", "season", "team", "position", "player_id"]
    ).astype({"season": int})
    with pytest.raises(RuntimeError, match="match rate"):
        load_fftoday_with_gsis_id(
            [2013],
            cache_dir=str(tmp_path),
            rosters=empty_rosters,
            min_match_rate=0.90,
            reader=_fake_reader,
        )


# ---------- joined-cache identity validity ---------------------------------------


def _identity_rosters(count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": f"00-{i}",
                "player_name": f"Player {i}",
                "position": "QB",
                "team": "KC",
                "season": 2024,
            }
            for i in range(count)
        ]
    )


@pytest.fixture
def joined_provider(monkeypatch, tmp_path):
    """Synthetic complete projections + rosters; counts source loads per join."""
    from src.analysis import fftoday_loader as mod

    state = {"rosters": _identity_rosters(), "count": 2, "loads": 0, "roster_loads": 0}

    def projections(*args, **kwargs):
        state["loads"] += 1
        frame = pd.DataFrame(
            [
                {
                    "player_name": f"Player {i}",
                    "position": "QB",
                    "season": 2024,
                    "week": 1,
                    "team": "KC",
                    "opponent": "BUF",
                }
                for i in range(state["count"])
            ]
        )
        frame.attrs[mod._FETCH_COMPLETE_ATTR] = True
        return frame

    def rosters(seasons):
        state["roster_loads"] += 1
        return state["rosters"].copy()

    monkeypatch.setattr(mod, "load_fftoday_projections", projections)
    monkeypatch.setattr(mod.nfl_source, "rosters", rosters)

    def run(threshold=1.0, **kwargs):
        return mod.load_fftoday_with_gsis_id(
            [2024], cache_dir=str(tmp_path), min_match_rate=threshold, **kwargs
        )

    return run, state, tmp_path


@pytest.mark.parametrize("corruption", ["None", "nan", "missing_column", "actual_null"])
def test_warm_join_rebuilds_invalid_identity_cache(joined_provider, corruption):
    from src.data.identity import valid_player_ids

    run, state, root = joined_provider
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    cached = pd.read_parquet(path)
    if corruption == "missing_column":
        cached = cached.drop(columns="player_id")
    else:
        cached.loc[0, "player_id"] = None if corruption == "actual_null" else corruption
    cached.to_parquet(path)
    actual = run()
    assert state["loads"] == state["roster_loads"] == 2
    pd.testing.assert_frame_equal(actual, expected)
    assert valid_player_ids(actual.player_id).all()


def test_healthy_warm_join_preserves_ids_without_loading_sources(joined_provider):
    run, state, root = joined_provider
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    before = path.read_bytes()
    actual = run()
    assert state["loads"] == state["roster_loads"] == 1
    assert path.read_bytes() == before
    pd.testing.assert_frame_equal(actual, expected)


def test_exact_threshold_uses_original_projection_denominator_on_warm_join(joined_provider):
    run, state, _ = joined_provider
    state["count"] = 10
    state["rosters"] = _identity_rosters(9)
    first = run(0.9)
    assert len(first) == 9  # matched-only rows
    pd.testing.assert_frame_equal(run(0.9), first)
    assert state["loads"] == 1
    # Both thresholds share the rounded mr90 cache filename. The matched-only
    # rows must not turn the original 9/10 coverage into 9/9.
    with pytest.raises(RuntimeError, match="match rate"):
        run(0.9001)
    assert state["loads"] == 2


@pytest.mark.parametrize("metadata", ["missing", None, 1, "2", True])
def test_legacy_or_invalid_denominator_metadata_must_rebuild(joined_provider, metadata):
    from src.analysis import fftoday_loader as mod

    run, state, root = joined_provider
    expected = run()
    path = next(root.glob("*joined*.parquet"))
    cached = pd.read_parquet(path)
    if metadata == "missing":
        cached.attrs.pop(mod._JOIN_SOURCE_ROWS_ATTR, None)
    else:
        cached.attrs[mod._JOIN_SOURCE_ROWS_ATTR] = metadata
    cached.to_parquet(path)
    actual = run()
    assert state["loads"] == 2
    pd.testing.assert_frame_equal(actual, expected)


# ---------- expert registration --------------------------------------------------


def test_fftoday_registered_as_expert():
    experts = {e.name: e for e in aec._build_experts(nflcom_loader=None, sleeper_loader=None)}
    assert "fftoday" in experts
    ff = experts["fftoday"]
    assert ff.label == "FFToday"
    assert ff.skipped == frozenset({"K", "DST"})
    # Comparisons use the common projected components, not full display totals.
    assert ff.project is aec._project_sleeper_comparison

"""Live identities require corroboration and expose omissions to consumers."""

import copy

import pandas as pd
import pytest

from src.serving import espn_live, upcoming_status
from src.serving.roster_identity import current_rosters, resolve_player

pytestmark = pytest.mark.unit


@pytest.fixture
def sources(monkeypatch):
    payload = {
        "season": {"year": 2026},
        "athletes": [
            {
                "position": "offense",
                "items": [
                    {
                        "id": "4431346",
                        "displayName": "Mark Redman",
                        "position": {"abbreviation": "TE"},
                        "dateOfBirth": "2002-01-20T08:00Z",
                    }
                ],
            }
        ],
    }
    reference = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "team": "GB",
                "position": "TE",
                "player_id": "00-0040598",
                "full_name": "Mark Redman",
                "birth_date": "2002-01-20",
                "status": "ACT",
            }
        ]
    )
    monkeypatch.setattr(espn_live, "_espn_to_gsis", {})
    monkeypatch.setattr(espn_live, "_get_json", lambda _: payload)
    return payload, reference


def fetch(reference):
    return espn_live.fetch_active_rosters({"9": "GB"}, season=2026, week=1, rosters_df=reference)


def test_recovers_active_player_and_shares_identity_with_injury_depth_consumers(sources):
    _, reference = sources
    frame = fetch(reference)
    assert frame.player_id.tolist() == ["00-0040598"]
    assert "4431346" not in espn_live.espn_to_gsis_map()
    assert frame.attrs["recovered_ids"] == {"4431346": "00-0040598"}
    records = [{"espn_id": "4431346", "status": "Out", "position": "TE", "team": "GB"}]
    ids = frame.attrs["recovered_ids"]
    injuries = espn_live.fetch_injuries_df(2026, 1, records=records, id_map=ids)
    assert injuries.gsis_id.tolist() == ["00-0040598"]
    assert espn_live.fetch_injury_status_map(2026, 1, records=records, id_map=ids) == {
        "00-0040598": 0.0
    }
    metadata = frame.attrs["source_metadata"]
    assert metadata["status"] == "available"
    assert metadata["parsed_players"] == metadata["mapped_players"] == 1
    assert metadata["recovered_players"] == 1


@pytest.mark.parametrize("status", ["EXE", "RES", "INA"])
def test_repeat_refresh_revalidates_recovered_eligibility(sources, status):
    _, reference = sources
    first = fetch(reference)
    reference.loc[0, "status"] = status
    second = fetch(reference)
    assert first.player_id.tolist() == ["00-0040598"]
    assert second.empty
    assert second.attrs["recovered_ids"] == {}
    assert second.attrs["source_metadata"]["status"] == "partial"


def test_recovery_is_not_reused_when_the_next_weekly_reference_is_unavailable(sources):
    _, reference = sources
    assert not fetch(reference).empty
    second = fetch(None)
    assert second.empty
    assert (
        second.attrs["source_metadata"]["unresolved_players"][0]["reason"]
        == "identity_reference_unavailable"
    )


def test_build_local_recovery_reaches_depth_and_expert_consumers(sources, monkeypatch):
    _, reference = sources
    frame = fetch(reference)
    ids = frame.attrs["recovered_ids"]
    monkeypatch.setattr(espn_live, "_get_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        espn_live, "_parse_depthchart", lambda _: [{"espn_id": "4431346", "order": 2}]
    )
    monkeypatch.setattr(
        espn_live,
        "_parse_fantasy_projections",
        lambda *args: [
            {"espn_id": "4431346", "position": "TE", "ppr_total": 2.5, "receptions": 1.0}
        ],
    )
    assert espn_live.fetch_depth_chart_ranks(2026, {"9": "GB"}, id_map=ids) == {"00-0040598": 2.0}
    points = espn_live.fetch_fantasy_projections(2026, 1, id_map=ids)
    assert points.player_id.tolist() == ["00-0040598"]
    assert points.espn_ppr_total.tolist() == [2.5]
    assert "4431346" not in espn_live.espn_to_gsis_map()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("season", 2025),
        ("week", 2),
        ("team", "LA"),
        ("position", "WR"),
        ("birth_date", "2002-01-21"),
        ("birth_date", None),
        ("player_id", "null"),
        ("player_id", None),
    ],
)
def test_wrong_context_or_missing_identity_cannot_create_player(sources, column, value):
    _, reference = sources
    reference.loc[0, column] = value
    frame = fetch(reference)
    assert frame.empty
    assert frame.attrs["source_metadata"]["status"] == "partial"
    assert len(frame.attrs["source_metadata"]["unresolved_players"]) == 1


def test_ambiguous_same_name_and_birthday_remains_unresolved(sources):
    _, reference = sources
    reference = pd.concat([reference, reference.assign(player_id="00-0099999")])
    frame = fetch(reference)
    assert frame.empty
    assert frame.attrs["source_metadata"]["unresolved_players"][0]["reason"] == "ambiguous_identity"


@pytest.mark.parametrize("status", ["EXE", "INA", "RES", "UNKNOWN"])
def test_fallback_cannot_turn_uncertain_eligibility_into_active_player(sources, status):
    _, reference = sources
    reference.loc[0, "status"] = status
    frame = fetch(reference)
    assert frame.empty
    metadata = frame.attrs["source_metadata"]
    assert metadata["unresolved_players"][0]["reason"] == f"eligibility_unconfirmed:{status}"
    quality = upcoming_status.data_quality({"roster": metadata})
    assert any(issue["source"] == "roster" for issue in quality["issues"])


def test_conflicting_espn_identity_is_not_overridden(sources):
    _, reference = sources
    reference["espn_id"] = "123"
    frame = fetch(reference)
    assert frame.empty
    assert (
        frame.attrs["source_metadata"]["unresolved_players"][0]["reason"]
        == "conflicting_espn_identity"
    )


def test_missing_birthday_in_espn_is_not_a_name_only_match(sources):
    payload, reference = sources
    del payload["athletes"][0]["items"][0]["dateOfBirth"]
    assert fetch(reference).empty


def test_authoritative_crosswalk_still_works_without_weekly_reference(sources, monkeypatch):
    monkeypatch.setattr(espn_live, "_espn_to_gsis", {"4431346": "00-0040598"})
    frame = fetch(None)
    assert frame.player_id.tolist() == ["00-0040598"]
    assert frame.attrs["source_metadata"]["recovered_players"] == 0


def test_wrong_roster_season_fails_instead_of_publishing(sources):
    payload, reference = sources
    payload["season"]["year"] = 2025
    with pytest.raises(ValueError, match="confirm season"):
        fetch(reference)


def test_identity_resolution_does_not_modify_reference(sources):
    payload, reference = sources
    before = reference.copy(deep=True)
    player = espn_live._parse_roster_players(copy.deepcopy(payload), "GB")[0]
    assert resolve_player(player, current_rosters(reference, 2026, 1))[0] == "00-0040598"
    pd.testing.assert_frame_equal(reference, before)

"""Forecast source contracts: event venue, units, horizon and explicit gaps."""

import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from src.serving import espn_live
from src.serving import forecast_weather as weather

pytestmark = pytest.mark.unit


def payload(kickoff, *, units="°F", wind=12.0):
    return {
        "hourly_units": {"temperature_2m": units, "wind_speed_10m": "mp/h"},
        "hourly": {
            "time": [kickoff.isoformat()],
            "temperature_2m": [58.0],
            "wind_speed_10m": [wind],
        },
    }


def test_forecast_nearest_hour_and_units():
    kickoff = pd.Timestamp("2027-09-12T17:20Z")
    p = payload(kickoff.floor("h"))
    assert weather._forecast_at(p, kickoff) == (58.0, 12.0)
    with pytest.raises(ValueError, match="horizon"):
        weather._forecast_at(p, kickoff + pd.Timedelta(days=2))
    with pytest.raises(ValueError, match="units"):
        weather._forecast_at(payload(kickoff, units="°C"), kickoff)
    with pytest.raises(ValueError, match="invalid"):
        weather._forecast_at(payload(kickoff, wind=-1), kickoff)


def test_neutral_venue_overrides_inherited_dome_and_home_coordinates(monkeypatch, tmp_path):
    kickoff = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=3)).floor("h")
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {"LAX01": (33.95, -118.34)})
    requested = []

    def read(url):
        requested.append(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query))
        return json.dumps(payload(kickoff)).encode()

    monkeypatch.setattr(weather, "_read_url", read)
    monkeypatch.setattr(espn_live, "_get_json", lambda _: {"grass": True})
    games = pd.DataFrame(
        [
            {
                "game_id": "neutral",
                "stadium_id": "LAX01",
                "stadium": "SoFi Stadium",
                "venue_name": "Melbourne Cricket Ground",
                "venue_id": "9119",
                "venue_indoor": False,
                "neutral_site": True,
                "roof": "dome",
                "surface": "matrixturf",
                "kickoff": kickoff.isoformat(),
            }
        ]
    )
    result, status = weather.enrich_forecasts(games)
    assert result.iloc[0]["roof"] == "outdoors"
    assert result.iloc[0]["surface"] == "grass"
    assert result.iloc[0]["temp"] == 58
    assert result.iloc[0]["wind"] == 12
    assert float(requested[0]["longitude"][0]) > 140  # Australia, never LA
    assert requested[0]["wind_speed_unit"] == ["mph"]
    assert status[0]["weather"] == "forecast"
    assert status[0]["retrieved_at"]

    # The actual venue surface must survive the schedule cache and reach the
    # feature consumed by saved models, not just the forecast response.
    from src.serving import upcoming_week
    from src.shared import weather_features

    schedule = result.assign(
        season=2026,
        week=1,
        game_type="REG",
        home_team="LA",
        away_team="SF",
        home_score=np.nan,
        away_score=np.nan,
        spread_line=3.0,
        total_line=46.0,
        home_rest=7,
        away_rest=7,
        div_game=1,
    )
    path = tmp_path / "schedule.parquet"
    schedule.assign(surface="matrixturf").to_parquet(path)
    monkeypatch.setattr(upcoming_week, "_schedules_path", lambda: str(path))
    monkeypatch.setattr(weather_features, "_schedule_cache", None)
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: pd.read_parquet(path))
    upcoming_week._augment_schedules_cache(schedule)
    players = pd.DataFrame(
        {
            "player_id": ["LA-QB", "SF-QB"],
            "recent_team": ["LA", "SF"],
            "season": [2026, 2026],
            "week": [1, 1],
        }
    )
    weather_features.merge_schedule_features(players)
    assert players["is_grass"].tolist() == [1, 1]
    assert players["is_dome"].tolist() == [0, 0]


@pytest.mark.parametrize("grass,expected", [(True, "grass"), (False, "artificial"), (None, None)])
def test_neutral_surface_uses_actual_venue_even_when_names_match(monkeypatch, grass, expected):
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {})
    monkeypatch.setattr(espn_live, "_get_json", lambda _: {"grass": grass})
    result, status = weather.enrich_forecasts(
        pd.DataFrame(
            [
                {
                    "game_id": "neutral",
                    "venue_id": "9119",
                    "venue_name": "Unmapped neutral ground",
                    "stadium": "Unmapped neutral ground",
                    "neutral_site": True,
                    "venue_indoor": False,
                    "roof": "dome",
                    "surface": "matrixturf",
                }
            ]
        )
    )
    actual = result.iloc[0]["surface"]
    assert pd.isna(actual) if expected is None else actual == expected
    assert status[0]["surface"] == (expected or "unknown")


def test_neutral_surface_failure_does_not_reuse_home_turf(monkeypatch):
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {})
    monkeypatch.setattr(espn_live, "_get_json", lambda _: (_ for _ in ()).throw(TimeoutError()))
    result, status = weather.enrich_forecasts(
        pd.DataFrame(
            [
                {
                    "game_id": "x",
                    "venue_id": "9119",
                    "neutral_site": True,
                    "venue_indoor": False,
                    "surface": "matrixturf",
                }
            ]
        )
    )
    assert pd.isna(result.iloc[0]["surface"])
    assert status[0]["surface"] == "unknown"


def test_ordinary_surface_keeps_game_metadata_without_venue_fetch(monkeypatch):
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {})
    monkeypatch.setattr(espn_live, "_get_json", lambda _: pytest.fail("unnecessary venue fetch"))
    result, _ = weather.enrich_forecasts(
        pd.DataFrame(
            [
                {
                    "game_id": "x",
                    "venue_id": "3673",
                    "neutral_site": False,
                    "venue_indoor": False,
                    "surface": "fieldturf",
                }
            ]
        )
    )
    assert result.iloc[0]["surface"] == "fieldturf"


@pytest.mark.parametrize(
    "roof,indoor,expected",
    [("dome", True, "indoor"), (None, True, "unknown_roof"), ("outdoors", False, "unavailable")],
)
def test_weather_failure_is_not_calm_weather(monkeypatch, roof, indoor, expected):
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {"TEST": (1.0, 2.0)})
    monkeypatch.setattr(weather, "_read_url", lambda _: (_ for _ in ()).throw(TimeoutError()))
    kickoff = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat()
    result, states = weather.enrich_forecasts(
        pd.DataFrame(
            [
                {
                    "game_id": "x",
                    "stadium_id": "TEST",
                    "roof": roof,
                    "venue_indoor": indoor,
                    "kickoff": kickoff,
                    "neutral_site": False,
                }
            ]
        )
    )
    assert states[0]["weather"] == expected
    if expected == "indoor":
        assert result.iloc[0]["temp"] == 65
        assert result.iloc[0]["wind"] == 0
    else:
        assert pd.isna(result.iloc[0]["temp"])
        assert pd.isna(result.iloc[0]["wind"])


def test_unknown_neutral_venue_does_not_borrow_home_stadium(monkeypatch):
    monkeypatch.setattr(weather, "stadium_coordinates", lambda: {"LAX01": (33.95, -118.34)})
    monkeypatch.setattr(
        weather, "_read_url", lambda _: pytest.fail("must not fetch guessed location")
    )
    result, status = weather.enrich_forecasts(
        pd.DataFrame(
            [
                {
                    "game_id": "x",
                    "stadium_id": "LAX01",
                    "stadium": "SoFi Stadium",
                    "venue_name": "New overseas ground",
                    "venue_indoor": False,
                    "neutral_site": True,
                    "roof": "dome",
                    "kickoff": "2027-09-12T17:00Z",
                }
            ]
        )
    )
    assert status[0]["weather"] == "unknown_venue"
    assert pd.isna(result.iloc[0]["wind"])

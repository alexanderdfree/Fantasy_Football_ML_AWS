"""Live schedule enrichment must reach the actual model feature merge."""

import pandas as pd
import pytest

from src.serving import live_schedule as live
from src.shared import weather_features

pytestmark = pytest.mark.unit


def sample():
    schedules = pd.DataFrame(
        {
            "game_id": ["2026_01_SF_LA"],
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["LA"],
            "away_team": ["SF"],
            "gameday": ["2026-09-10"],
            "gametime": ["20:35"],
            "home_score": [None],
            "away_score": [None],
            "spread_line": [2.5],
            "total_line": [47.5],
            "roof": ["dome"],
            "surface": ["matrixturf"],
            "home_rest": [10],
            "away_rest": [7],
            "div_game": [1],
            "temp": [None],
            "wind": [None],
        }
    )
    slate = pd.DataFrame(
        {
            "game_id": ["401872657"],
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "home_team": ["LA"],
            "away_team": ["SF"],
            "home_score": [None],
            "away_score": [None],
            "spread_line": [3.5],
            "total_line": [48.5],
            "kickoff": ["2026-09-11T00:35Z"],
            "venue": [{"id": "9119", "fullName": "Melbourne Cricket Ground"}],
            "neutral_site": [True],
            "forecast_temp": [55.0],
        }
    )
    return slate, schedules


def test_neutral_venue_live_odds_weather_and_rest_reach_model(monkeypatch):
    slate, schedules = sample()
    monkeypatch.setattr(live.espn_live, "_get_json", lambda url: {"grass": True, "indoor": False})
    monkeypatch.setattr(
        live, "_venue_forecast", lambda venue, kickoff: {"temp": 58.0, "wind": 11.0}
    )
    enriched, meta = live.enrich_schedule_rows(slate, schedules)
    assert len(enriched) == 1  # numeric ESPN ID did not duplicate the nflverse game
    monkeypatch.setattr(weather_features, "_load_schedules", lambda: enriched)
    players = pd.DataFrame(
        {
            "player_id": ["A", "B"],
            "season": [2026, 2026],
            "week": [1, 1],
            "recent_team": ["LA", "SF"],
        }
    )
    result = weather_features.merge_schedule_features(players)
    assert result.implied_team_total.tolist() == [26.0, 22.5]
    assert result.is_dome.tolist() == [0, 0]
    assert result.is_grass.tolist() == [1, 1]
    assert result.is_divisional.tolist() == [1, 1]
    assert result.rest_advantage.tolist() == [3, -3]
    assert result.temp_adjusted.tolist() == [58.0, 58.0]
    assert result.wind_adjusted.tolist() == [11.0, 11.0]
    assert meta["coverage"] == {"forecast": 1}


def test_forecast_outage_is_reported_and_keeps_espn_temperature(monkeypatch):
    slate, schedules = sample()
    monkeypatch.setattr(live.espn_live, "_get_json", lambda url: {"grass": True, "indoor": False})
    monkeypatch.setattr(live, "_venue_forecast", lambda *args: {})
    enriched, meta = live.enrich_schedule_rows(slate, schedules)
    assert enriched.temp.iloc[0] == 55.0
    assert pd.isna(enriched.wind.iloc[0])
    assert meta["coverage"] == {"partial_or_unavailable": 1}


def test_forecast_uses_kickoff_hour_and_explicit_units():
    payload = {
        "hourly_units": {"temperature_2m": "°F", "wind_speed_10m": "mp/h"},
        "hourly": {
            "time": ["2026-09-13T16:00", "2026-09-13T17:00"],
            "temperature_2m": [70.0, 73.0],
            "wind_speed_10m": [3.0, 9.0],
        },
    }
    assert live._forecast_at_kickoff(payload, "2026-09-13T17:15Z") == {"temp": 73.0, "wind": 9.0}
    assert live._forecast_at_kickoff(payload, "2026-10-13T17:15Z") == {}
    payload["hourly_units"]["wind_speed_10m"] = "km/h"
    with pytest.raises(ValueError, match="units"):
        live._forecast_at_kickoff(payload, "2026-09-13T17:00Z")


def test_missing_schedule_context_fails_instead_of_neutralizing_every_feature():
    slate, schedules = sample()
    with pytest.raises(ValueError, match="missing"):
        live.enrich_schedule_rows(slate, schedules.drop(columns=["roof"]))

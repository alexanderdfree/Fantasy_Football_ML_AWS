"""Current game metadata and kickoff forecasts for the CI projection builder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.data import nfl_source
from src.data.source_result import SourceResult, SourceStatus
from src.data import espn_live

WEATHER_SOURCE = "https://open-meteo.com/"
_COUNTRIES = {
    "USA": "US",
    "United States": "US",
    "Australia": "AU",
    "United Kingdom": "GB",
    "Germany": "DE",
    "Spain": "ES",
    "Ireland": "IE",
    "Brazil": "BR",
    "Mexico": "MX",
}
_STATES = dict(
    pair.split(":")
    for pair in [
        "AZ:Arizona",
        "CA:California",
        "CO:Colorado",
        "FL:Florida",
        "GA:Georgia",
        "IL:Illinois",
        "IN:Indiana",
        "LA:Louisiana",
        "MA:Massachusetts",
        "MD:Maryland",
        "MI:Michigan",
        "MN:Minnesota",
        "MO:Missouri",
        "NC:North Carolina",
        "NJ:New Jersey",
        "NV:Nevada",
        "NY:New York",
        "OH:Ohio",
        "PA:Pennsylvania",
        "TN:Tennessee",
        "TX:Texas",
        "WA:Washington",
        "WI:Wisconsin",
    ]
)


def _forecast_at_kickoff(payload: dict, kickoff: str) -> dict:
    """Read the nearest forecast hour, with explicit Fahrenheit/mph units."""
    hourly = payload.get("hourly") or {}
    units = payload.get("hourly_units") or {}
    if units.get("temperature_2m") != "°F" or units.get("wind_speed_10m") not in ("mp/h", "mph"):
        raise ValueError("Unexpected weather forecast units")
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    target = pd.Timestamp(kickoff)
    if target.tzinfo is None:
        raise ValueError("Kickoff requires a timezone")
    if not len(times):
        return {}
    offsets = abs(times - target)
    idx = offsets.argmin()
    if pd.isna(times[idx]) or offsets[idx] > pd.Timedelta(hours=1):
        return {}  # offseason / outside the forecast horizon
    temp = hourly["temperature_2m"][idx]
    wind = hourly["wind_speed_10m"][idx]
    return {
        key: float(value)
        for key, value in (("temp", temp), ("wind", wind))
        if value is not None and np.isfinite(value)
    }


def _venue_forecast(venue: dict, kickoff: str) -> dict:
    """Forecast at the event's city, including international/neutral venues."""
    target = pd.Timestamp(kickoff)
    delta = target - pd.Timestamp.now("UTC")
    if delta < pd.Timedelta(0) or delta > pd.Timedelta(days=15):
        return {}
    address = venue.get("address") or {}
    city = address.get("city")
    country = _COUNTRIES.get(address.get("country"), address.get("country"))
    if not city or not country or len(country) != 2:
        return {}
    query = urlencode({"name": city, "count": 20, "countryCode": country, "language": "en"})
    places = espn_live._get_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}")
    matches = [
        p
        for p in places.get("results", [])
        if p.get("country_code") == country and p.get("name", "").casefold() == city.casefold()
    ]
    state = _STATES.get(address.get("state")) if country == "US" else None
    if state:
        matches = [p for p in matches if p.get("admin1") == state]
    if not matches:
        return {}
    place = max(matches, key=lambda p: p.get("population", 0))
    query = urlencode(
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
            "forecast_days": 16,
        }
    )
    return _forecast_at_kickoff(
        espn_live._get_json(f"https://api.open-meteo.com/v1/forecast?{query}"), kickoff
    )


def enrich_schedule_rows(live: pd.DataFrame, schedules: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep real venue/rest/date context and replace only live lines/weather.

    Schedule identity uses season/week/teams, not the vendors' different game
    IDs. Historical completed games remain in the returned season calendar so
    attention tokens and future rest calculations keep their context.
    """
    keys = ["season", "week", "home_team", "away_team"]
    required = {
        *keys,
        "game_type",
        "gameday",
        "gametime",
        "roof",
        "surface",
        "home_rest",
        "away_rest",
        "div_game",
        "home_score",
        "away_score",
    }
    if not required.issubset(schedules):
        raise ValueError(f"Live schedule context is missing {sorted(required - set(schedules))}")
    base = schedules[schedules["game_type"].eq("REG")].drop_duplicates(keys).copy()
    joined = live.merge(base, on=keys, how="left", suffixes=("_live", ""), validate="one_to_one")
    if joined["gameday"].isna().any():
        raise ValueError("Some upcoming games have no matching current-season schedule")

    def enrich(row):
        row = row.copy()
        for col in ("spread_line", "total_line"):
            if pd.notna(row[f"{col}_live"]):
                row[col] = row[f"{col}_live"]
        # Scores for scheduled games are unknown. Never accept accidental
        # prior-season/backfilled scores as observations of the upcoming game.
        row["home_score"] = np.nan
        row["away_score"] = np.nan
        venue = row.get("venue") or {}
        venue_id = venue.get("id")
        if row.get("neutral_site") and not venue_id:
            raise ValueError("Cannot verify a neutral game's venue without an ID")
        if venue_id:
            try:
                detail = espn_live._get_json(f"{espn_live._CORE_BASE}/venues/{venue_id}")
                if row.get("neutral_site") and not all(
                    isinstance(detail.get(key), bool) for key in ("grass", "indoor")
                ):
                    raise ValueError("Neutral venue details require grass and indoor flags")
                if "grass" in detail:
                    row["surface"] = "grass" if detail["grass"] else "artificial"
                # A neutral game's nflverse roof can still describe the nominal
                # home stadium (2026 Rams at Melbourne). ESPN knows the venue.
                if row.get("neutral_site") and "indoor" in detail:
                    row["roof"] = "dome" if detail["indoor"] else "outdoors"
                elif detail.get("indoor") is False:
                    row["roof"] = "outdoors"
            except Exception as exc:  # network boundary; retain known schedule metadata
                print(f"[live_schedule] venue {venue_id} unavailable: {exc!r}")
                if row.get("neutral_site"):
                    raise ValueError("Cannot verify a neutral game's venue") from exc
        covered = row["roof"] in ("dome", "closed")
        if covered:
            row["temp"], row["wind"] = 65.0, 0.0
            row["_weather_status"] = "covered_venue"
        elif row["roof"] not in ("outdoors", "open"):
            # A retractable stadium's indoor flag does not say whether its
            # roof will be open. Do not feed outdoor forecasts into that gap.
            row["temp"], row["wind"] = np.nan, np.nan
            row["_weather_status"] = "unknown_roof"
        else:
            try:
                forecast = _venue_forecast(venue, row["kickoff"])
            except Exception as exc:  # optional forecast; report missing coverage
                print(f"[live_schedule] forecast for {venue.get('fullName')} unavailable: {exc!r}")
                forecast = {}
            row["temp"] = forecast.get("temp", row.get("forecast_temp", np.nan))
            row["wind"] = forecast.get("wind", np.nan)
            row["_weather_status"] = (
                "forecast" if {"temp", "wind"}.issubset(forecast) else "partial_or_unavailable"
            )
        return row

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(joined)))) as pool:
        enriched = pd.DataFrame(pool.map(enrich, joined.to_dict("records")))
    status = {
        "provider": "Open-Meteo",
        "url": WEATHER_SOURCE,
        "games": len(enriched),
        "coverage": enriched["_weather_status"].value_counts().to_dict(),
        "by_game": [
            {
                "game_id": str(row["game_id"]),
                "venue": (row.get("venue") or {}).get("fullName"),
                "kickoff": row.get("kickoff"),
                "roof": row["roof"] if isinstance(row["roof"], str) else "unknown",
                "weather": {"covered_venue": "indoor", "forecast": "forecast"}.get(
                    row["_weather_status"], "unavailable"
                ),
                "retrieved_at": pd.Timestamp.now("UTC").isoformat(),
            }
            for row in enriched.to_dict("records")
        ],
    }
    categories = status["coverage"]
    assumed = int(categories.get("covered_venue", 0))
    forecast = int(categories.get("forecast", 0))
    incomplete = len(enriched) - assumed - forecast
    partial = int(
        (
            enriched["_weather_status"].eq("partial_or_unavailable")
            & enriched[["temp", "wind"]].notna().any(axis=1)
        ).sum()
    )
    unavailable = incomplete - partial
    availability = (
        SourceStatus.EMPTY
        if enriched.empty
        else SourceStatus.UNAVAILABLE
        if unavailable == len(enriched)
        else SourceStatus.PARTIAL
        if incomplete
        else SourceStatus.AVAILABLE
    )
    period = {}
    if not live.empty and live["season"].nunique() == 1:
        period["season"] = int(live["season"].iloc[0])
    if not live.empty and live["week"].nunique() == 1:
        period["week"] = int(live["week"].iloc[0])
    status["source_result"] = SourceResult.capture(
        enriched[[*keys, "temp", "wind", "_weather_status"]],
        provider="Open-Meteo; ESPN fallback; venue context",
        status=availability,
        effective_period=period,
        coverage={
            "games": len(enriched),
            "forecast_games": forecast,
            "assumed_indoor_games": assumed,
            "partial_games": partial,
            "unavailable_games": unavailable,
        },
        value_kind="forecast_or_venue_assumption_or_provider_fallback",
    ).metadata()
    for game, row in zip(status["by_game"], enriched.to_dict("records"), strict=True):
        game["value_origin"] = {"indoor": "venue_assumption", "forecast": "forecast"}.get(
            game["weather"],
            "provider_fallback_partial"
            if pd.notna(row["temp"]) or pd.notna(row["wind"])
            else "missing_for_imputation",
        )
    columns = list(dict.fromkeys([*base.columns, "temp", "wind"]))
    updates = enriched.reindex(columns=columns)
    result = pd.concat([base, updates], ignore_index=True).drop_duplicates(keys, keep="last")
    return result, status


def fetch_schedule_context(season: int, live: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    return enrich_schedule_rows(live, nfl_source.schedules([season]))

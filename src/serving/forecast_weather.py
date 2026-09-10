"""Kickoff forecasts for the CI builder; no network work in the web process."""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import pandas as pd

# Immutable community stadium gazetteer linked by nflverse-data issue #57.
# Resolve the GAME's stadium_id, never the home team's usual stadium.
_STADIUMS_URL = "https://github.com/user-attachments/files/17464644/stadiums.csv"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# International venues added after the 2024 gazetteer. Key by the actual ESPN
# event name so an inherited home-stadium ID cannot move the forecast abroad.
_INTERNATIONAL_VENUES = {
    "Melbourne Cricket Ground": (-37.8199, 144.9834),
    "Maracana Stadium": (-22.9121, -43.2302),
    "Maracanã": (-22.9121, -43.2302),
    "Stade de France": (48.9244, 2.3601),
    "Santiago Bernabéu": (40.4531, -3.6883),
    "Santiago Bernabeu Stadium": (40.4531, -3.6883),
    "Allianz Arena": (48.2188, 11.6247),
    "FC Bayern Munich Stadium": (48.2188, 11.6247),
}


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


@lru_cache(maxsize=1)
def stadium_coordinates() -> dict[str, tuple[float, float]]:
    table = pd.read_csv(io.BytesIO(_read_url(_STADIUMS_URL)))
    return {
        str(row.stadium_id): (float(row.latitude), float(row.longitude))
        for row in table.itertuples()
        if pd.notna(row.latitude) and pd.notna(row.longitude)
    }


def _forecast_at(payload: dict, kickoff: pd.Timestamp) -> tuple[float, float]:
    units = payload.get("hourly_units", {})
    if units.get("temperature_2m") != "°F" or units.get("wind_speed_10m") != "mp/h":
        raise ValueError("forecast units differ from the model's Fahrenheit/mph encoding")
    hourly = payload["hourly"]
    times = pd.to_datetime(hourly["time"], utc=True)
    # Nearest forecast hour, bounded so an out-of-horizon kickoff never borrows
    # the last forecast from days earlier.
    distances = abs(times - kickoff)
    idx = int(distances.argmin())
    if distances[idx] > pd.Timedelta(minutes=60):
        raise ValueError("kickoff is outside the forecast horizon")
    temp, wind = float(hourly["temperature_2m"][idx]), float(hourly["wind_speed_10m"][idx])
    if not np.isfinite([temp, wind]).all() or wind < 0:
        raise ValueError("forecast contains missing or invalid weather")
    return temp, wind


def enrich_forecasts(schedules: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Enrich upcoming games; unavailable values remain NaN for train imputation.

    ESPN's outdoor flag overrides stale venue defaults (e.g. LA's dome copied
    onto Melbourne). An indoor flag alone does not settle a retractable roof.
    Forecast retrieval time and availability travel with the artifact.
    """
    frame = schedules.copy()
    retrieved = datetime.now(UTC).isoformat()
    try:
        coordinates = stadium_coordinates()
    except Exception as exc:  # noqa: BLE001 - optional network source
        print(f"[forecast_weather] stadium lookup unavailable: {exc!r}")
        coordinates = {}

    def forecast(row):
        status = {
            "game_id": str(row.get("game_id", "")),
            "venue": row.get("venue_name"),
            "kickoff": row.get("kickoff"),
            "retrieved_at": retrieved,
        }
        roof = row.get("roof")
        indoor = row.get("venue_indoor")
        if indoor is False or isinstance(indoor, np.bool_) and not indoor:
            roof = "outdoors"
        elif row.get("neutral_site") and row.get("venue_name") != row.get("stadium"):
            roof = None
        status["roof"] = roof if isinstance(roof, str) else "unknown"
        if roof in ("dome", "closed"):
            status["weather"] = "indoor"
            return roof, 65.0, 0.0, status
        if roof not in ("outdoors", "open"):
            status["weather"] = "unknown_roof"
            return None, np.nan, np.nan, status
        point = _INTERNATIONAL_VENUES.get(row.get("venue_name"))
        # A neutral-site name disagreement makes the schedule's stadium ID
        # untrustworthy too. Unknown new venues degrade instead of guessing.
        if point is None and not (
            row.get("neutral_site") and row.get("venue_name") != row.get("stadium")
        ):
            point = coordinates.get(str(row.get("stadium_id")))
        if point is None:
            status["weather"] = "unknown_venue"
            return roof, np.nan, np.nan, status
        try:
            kickoff = pd.to_datetime(row.get("kickoff"), utc=True)
            if pd.isna(kickoff) or kickoff <= pd.Timestamp.now(tz="UTC"):
                raise ValueError("no future kickoff timestamp")
            query = urllib.parse.urlencode(
                {
                    "latitude": point[0],
                    "longitude": point[1],
                    "hourly": "temperature_2m,wind_speed_10m",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone": "UTC",
                    "forecast_days": 16,
                }
            )
            temp, wind = _forecast_at(json.loads(_read_url(f"{_FORECAST_URL}?{query}")), kickoff)
            status.update(weather="forecast", latitude=point[0], longitude=point[1])
            return roof, temp, wind, status
        except Exception as exc:  # noqa: BLE001 - optional forecast boundary
            status["weather"] = "unavailable"
            print(f"[forecast_weather] {status['game_id']}: {exc!r}")
            return roof, np.nan, np.nan, status

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(forecast, frame.to_dict("records")))
    for col, idx in (("roof", 0), ("temp", 1), ("wind", 2)):
        frame[col] = [v[idx] for v in values]
    return frame, [v[3] for v in values]

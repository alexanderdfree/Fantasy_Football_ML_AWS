"""Live non-ESPN signals for the upcoming-week build.

ESPN ([espn_live.py]) supplies the real-time slate (schedule, lines, rosters,
depth chart, game-day status). This module adds the two signals that are best
sourced from the **same nflverse feeds the model trains on** (so the values stay
on the exact training encoding rather than a new vendor's):

* **practice_status** — nflverse ``load_injuries`` (carries the official
  Did Not Participate / Limited / Full participation field). Falls back to
  Sleeper's free players dump only when nflverse is empty *and* it's plausibly
  in-season (nflverse injuries has gone down before; Sleeper is a free backstop).
* **contract_\\*** — nflverse ``load_contracts`` (OTC), via the same
  ``derive_active_contracts`` the training pipeline uses.

Everything here is **defensive** (never raises into the poller): a failure or an
empty result degrades to ``{}`` / an empty frame, and the caller then falls back
to carry-forward / the healthy default.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pandas as pd

from src.data import nfl_source
from src.data.external_sources import CONTRACT_FEATURE_COLUMNS, derive_active_contracts

# Mirrors the ``practice_map`` in [src/data/loader.py] (the TRAINING encoding);
# keys are the canonical nflverse practice-report strings (verified against a
# live ``load_injuries`` pull). Worst (lowest) value wins per player-week.
_PRACTICE_STATUS_NUM: dict[str, float] = {
    "Full Participation in Practice": 2.0,
    "Limited Participation in Practice": 1.0,
    "Did Not Participate In Practice": 0.0,
}
_PRACTICE_HEALTHY_DEFAULT = 2.0  # mirrors _CONTEXT_DEFAULTS["practice_status"]

# NFL practice reports only exist in-season (Wk 1 ~ early Sept through the
# playoffs ~ early Feb). Outside that window nflverse injuries for the upcoming
# season is legitimately empty (no practices yet), so skip the large Sleeper
# fallback fetch rather than burn a 14MB download every refresh for nothing.
_IN_SEASON_MONTHS = frozenset({9, 10, 11, 12, 1, 2})

_SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
_SLEEPER_TIMEOUT_S = 30


# --------------------------------------------------------------------------
# practice_status
# --------------------------------------------------------------------------
def fetch_practice_status_map(season: int, week: int) -> dict[str, float]:
    """``{player_id: practice_status}`` for the upcoming (season, week).

    Primary: nflverse ``load_injuries`` (exact training encoding). Fallback
    (only when nflverse is empty AND it's plausibly in-season): Sleeper's free
    players dump. Empty → ``{}`` (caller defaults to healthy). Never raises.
    """
    primary = _nflverse_practice_map(season, week)
    if primary:
        return primary
    if datetime.now(UTC).month in _IN_SEASON_MONTHS:
        return _sleeper_practice_map()
    return {}


def _nflverse_practice_map(season: int, week: int) -> dict[str, float]:
    try:
        df = nfl_source.injuries([season])
    except Exception as e:  # noqa: BLE001 - network/data boundary
        print(f"[live_sources] nflverse injuries fetch failed: {e!r}")
        return {}
    if df is None or df.empty or "practice_status" not in df.columns:
        return {}
    df = df[(df["season"] == season) & (df["week"] == week)]
    out: dict[str, float] = {}
    for pid, status in zip(df.get("gsis_id", []), df.get("practice_status", []), strict=False):
        num = _PRACTICE_STATUS_NUM.get(status)
        if pid is None or num is None:
            continue
        key = str(pid)
        out[key] = min(num, out.get(key, _PRACTICE_HEALTHY_DEFAULT))  # worst-per-player
    return out


def _sleeper_practice_to_num(value) -> float | None:
    """Map a Sleeper ``practice_participation`` value → the training encoding.

    Sleeper's vocabulary is undocumented (the field is empty in the offseason),
    so this is best-effort over the plausible spellings; an unknown/empty value
    returns ``None`` so the row keeps the healthy default rather than guessing.
    """
    if not value:
        return None
    s = str(value).strip().lower()
    if s in ("full", "fp", "full participation in practice"):
        return 2.0
    if s in ("limited", "lp", "limited participation in practice"):
        return 1.0
    if s in ("dnp", "did not participate", "did not participate in practice", "out"):
        return 0.0
    return None


def _parse_sleeper_practice(payload: dict) -> dict[str, float]:
    """Pure parser: Sleeper players dump → ``{player_id(gsis): practice_status}``.

    Keyed by each record's ``gsis_id``; records without a gsis id or with an
    unmapped practice value are skipped. Fixture-tested (no network).
    """
    out: dict[str, float] = {}
    for p in (payload or {}).values():
        if not isinstance(p, dict):
            continue
        gsis = p.get("gsis_id")
        num = _sleeper_practice_to_num(p.get("practice_participation"))
        if gsis and num is not None:
            key = str(gsis)
            out[key] = min(num, out.get(key, _PRACTICE_HEALTHY_DEFAULT))
    return out


def _sleeper_practice_map() -> dict[str, float]:
    """Fetch + parse the Sleeper players dump (the practice fallback). ``{}`` on
    failure. The dump is ~14MB and Sleeper asks for ≤1 fetch/day — this only
    fires when nflverse is empty in-season (a nflverse outage), so it's rare."""
    try:
        req = urllib.request.Request(
            _SLEEPER_PLAYERS_URL, headers={"User-Agent": "ff-predictor/1.0"}
        )
        with urllib.request.urlopen(req, timeout=_SLEEPER_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        print(f"[live_sources] Sleeper practice fallback failed: {e!r}")
        return {}
    return _parse_sleeper_practice(payload)


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------
def fetch_contract_features(season: int) -> pd.DataFrame:
    """Current-season ``contract_*`` features indexed by ``player_id``.

    Reuses the training deriver (``derive_active_contracts``) over the live OTC
    feed (``nfl_source.contracts()``), so 2026 rows get the active-as-of-2026
    contract (advancing ``contract_age``/``contract_years_remaining`` vs a stale
    carry-forward). Empty frame (right columns) on failure. Never raises.
    """
    cols = list(CONTRACT_FEATURE_COLUMNS)
    try:
        derived = derive_active_contracts(nfl_source.contracts(), [season])
    except Exception as e:  # noqa: BLE001 - network/data boundary
        print(f"[live_sources] contracts derive failed: {e!r}")
        return pd.DataFrame(columns=cols)
    if derived is None or derived.empty:
        return pd.DataFrame(columns=cols)
    return derived.set_index("player_id")[cols]

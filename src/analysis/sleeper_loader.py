"""Fetch + cache + gsis-join Sleeper (RotoWire) weekly NFL projections.

Source: Sleeper's undocumented projections endpoint
``https://api.sleeper.app/projections/nfl/{season}/{week}`` (free, no auth). Every
record carries ``company: "rotowire"`` — so this is **one** additional expert
(RotoWire), not a consensus. Covers offense (QB/RB/WR/TE, joined via the gsis
crosswalk) and DST (team-keyed); K is totals-only and out of scope.

Lives under ``src/analysis/`` (not ``src/data/``) on purpose: ``src/data/`` is a
global retrain trigger in ``src/scripts/scope_positions.py`` and this loader is
analysis-only. Mirrors the cache + network-defensiveness idiom of
``src/data/nflcom_loader.py``.

Two public entry points (parallel to ``nflcom_loader``):

    load_sleeper_projections(seasons, ...) -> pd.DataFrame
        One row per (sleeper_player_id, position, season, week). Raw stats mapped
        to our internal target names. Cached to
        ``data/raw/sleeper_projections_v1_{min}_{max}_{weeks}.parquet``.

    load_sleeper_with_gsis_id(seasons, ...) -> pd.DataFrame
        Same frame, joined to ``player_id`` (gsis_id) via the nflverse
        ``ff_playerids`` crosswalk (``nfl_source.player_ids()``), the same bridge
        pattern used for ESPN-QBR (``external_sources.py``) and PFR (``loader.py``).

PROVENANCE CAVEAT: Sleeper does not document whether these historical projections
are the as-of-kickoff snapshot or a later backfill. Spot evidence (fractional
expected-value stats that do not match actuals) suggests genuine pre-game
projections, but callers should sanity-check RotoWire's error magnitude against a
known expert (NFL.com) before trusting the comparison — see the comparison
script's provenance gate.


Implementation is shared with src.serving.expert_sources so offline and live
normalization, retries, cache keys, and ID joins stay aligned.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence

import pandas as pd

from src.config import CACHE_DIR
from src.serving import expert_sources as _shared
from src.serving.expert_sources import (
    _MIN_SEASON as _MIN_SEASON,
)
from src.serving.expert_sources import (
    _REQUEST_TIMEOUT_S,
    _RETRY_BACKOFF_S,
)
from src.serving.expert_sources import (
    SLEEPER_DEFAULT_WEEKS as SLEEPER_DEFAULT_WEEKS,
)
from src.serving.expert_sources import (
    SLEEPER_DST_STAT_MAP as SLEEPER_DST_STAT_MAP,
)
from src.serving.expert_sources import (
    SLEEPER_FETCH_POSITIONS as SLEEPER_FETCH_POSITIONS,
)
from src.serving.expert_sources import (
    SLEEPER_OFFENSE_POSITIONS as SLEEPER_OFFENSE_POSITIONS,
)
from src.serving.expert_sources import (
    SLEEPER_STAT_MAP as SLEEPER_STAT_MAP,
)
from src.serving.expert_sources import (
    _projection_url as _projection_url,
)
from src.serving.expert_sources import (
    _weeks_signature as _weeks_signature,
)


def _default_reader(url: str) -> list:
    """Fetch a Sleeper projections URL and return the decoded JSON list."""
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-ml-research"})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
        return json.load(resp)


def _read_one_week(
    season: int,
    week: int,
    positions: Sequence[str],
    *,
    reader=_default_reader,
    max_retries: int = 1,
    backoff_s: float = _RETRY_BACKOFF_S,
) -> list | None:
    return _shared._read_one_sleeper_week(
        season, week, positions, reader=reader, max_retries=max_retries, backoff_s=backoff_s
    )


def load_sleeper_projections(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    positions: Sequence[str] = SLEEPER_FETCH_POSITIONS,
    reader=_default_reader,
) -> pd.DataFrame:
    return _shared.load_sleeper_projections(
        seasons, cache_dir, force_refresh, weeks=weeks, positions=positions, reader=reader
    )


def load_sleeper_with_gsis_id(
    seasons: Sequence[int],
    cache_dir: str = CACHE_DIR,
    force_refresh: bool = False,
    *,
    weeks: Sequence[int] | None = None,
    reader=_default_reader,
    player_ids_loader=None,
) -> pd.DataFrame:
    return _shared.load_sleeper_with_gsis_id(
        seasons,
        cache_dir,
        force_refresh,
        weeks=weeks,
        reader=reader,
        player_ids_loader=player_ids_loader,
    )

"""Unit tests for src/analysis/expert_intervals.py.

The default loaders hit the network — out of scope for unit tests. These inject
stub NFL.com / Sleeper / actuals loaders built so the leak + stationarity season
filter, the quantile fit, monotone band rearrangement, held-out coverage, bounded
examples, and the position skips are all exercised on synthetic multi-season
frames. Also an import smoke so a signature break fails the unit shard.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import expert_intervals as mod

pytestmark = pytest.mark.unit

_QB_TARGETS = ("passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds")
_FIT_SEASONS = (2021, 2022, 2024)  # genuine, stationary
_LEAK_SEASON = 2023  # actuals == projection ⇒ look-ahead
_EVAL_SEASON = 2025
_ALL_SEASONS = (*_FIT_SEASONS, _LEAK_SEASON, _EVAL_SEASON)
_PLAYERS = tuple(f"00-00{i:05d}" for i in range(1, 13))  # 12 players
_WEEKS = tuple(range(1, 10))  # 9 weeks → 108 rows/season


def _base_stats(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Deterministic 'true' QB raw stats for n rows."""
    return {
        "passing_yards": rng.uniform(180, 320, n),
        "passing_tds": rng.uniform(0.5, 2.5, n),
        "interceptions": rng.uniform(0, 1.2, n),
        "rushing_yards": rng.uniform(0, 30, n),
        "rushing_tds": rng.uniform(0, 0.4, n),
    }


def _grid() -> pd.DataFrame:
    rows = [
        {"player_id": p, "player_name": f"QB {p[-2:]}", "season": s, "week": w}
        for s in _ALL_SEASONS
        for p in _PLAYERS
        for w in _WEEKS
    ]
    return pd.DataFrame(rows)


def _nflcom_loader(seasons, force_refresh=False):
    """Raw NFL.com QB frame: projection == the deterministic 'true' stats."""
    g = _grid()
    g = g[g["season"].isin(set(int(s) for s in seasons))].reset_index(drop=True)
    rng = np.random.default_rng(0)
    stats = _base_stats(rng, len(g))
    g["position"] = "QB"
    for c, v in stats.items():
        g[c] = v
    g["fumbles_lost"] = 0.0
    # NFL.com's own points col (unused for QB aggregation but required by the projector).
    g["nflcom_projected_pts"] = g["passing_yards"] * 0.04 + g["passing_tds"] * 4.0
    return g


def _actuals_loader(seasons):
    """nflverse-shaped actuals: projection + noise, except the leak season (noise=0)."""
    proj = _nflcom_loader(seasons)
    rng = np.random.default_rng(1)
    out = proj.copy()
    noise_scale = {"passing_yards": 150.0, "passing_tds": 0.9, "interceptions": 0.6}
    for c, sc in noise_scale.items():
        noise = rng.normal(0, sc, len(out))
        leaked = out["season"].to_numpy() == _LEAK_SEASON
        out[c] = np.where(leaked, out[c], np.clip(out[c] + noise, 0, None))
    out["fantasy_points"] = 0.0  # recomputed by the aggregator; placeholder
    return out


def _dst_actuals_loader(seasons):
    return pd.DataFrame(columns=[*mod._KEYS, "actual_pts"])


def _run(**kw):
    defaults = dict(
        eval_seasons=(_EVAL_SEASON,),
        min_fit_season=2021,
        nflcom_loader=_nflcom_loader,
        sleeper_loader=_nflcom_loader,  # same shape works for the Sleeper projector path
        actuals_loader=_actuals_loader,
        dst_actuals_loader=_dst_actuals_loader,
    )
    defaults.update(kw)
    return mod.build_intervals(**defaults)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_module_imports_cleanly() -> None:
    for attr in (
        "build_intervals",
        "main",
        "lookahead_seasons",
        "_select_fit_seasons",
        "_fit_quantile_params",
        "_apply_bands",
        "_calibrate",
    ):
        assert hasattr(mod, attr)
    assert pytest.approx(0.8) == mod.NOMINAL_COVERAGE
    assert mod.TAUS == (0.1, 0.5, 0.9)


def test_apply_bands_is_monotone() -> None:
    # Deliberately crossing lines (floor slope steepest) — rearrangement must fix it.
    params = {
        0.1: {"intercept": 0.0, "slope": 1.5},
        0.5: {"intercept": 3.0, "slope": 1.0},
        0.9: {"intercept": 6.0, "slope": 0.6},
    }
    x = np.array([0.0, 10.0, 30.0])
    floor, median, ceiling = mod._apply_bands(params, x)
    assert np.all(floor <= median) and np.all(median <= ceiling)


def test_select_fit_seasons_drops_leak_and_nonstationary() -> None:
    rng = np.random.default_rng(2)
    parts = []
    # Genuine stationary: 2022, 2024 (std ~6 → near-exact ~0.07). Fully backfilled:
    # 2023 (actual==projection → near-exact 1.0). Non-stationary tight: 2021 (std ~3,
    # genuine near-exact ~0.13 but < 0.8×the recent season's spread).
    for s, sd in [(2021, 3.0), (2022, 6.0), (2023, 0.0), (2024, 6.0)]:
        proj = rng.uniform(5, 25, 120)
        actual = proj + (rng.normal(0, sd, 120) if sd else 0.0)
        parts.append(pd.DataFrame({"season": s, "projection": proj, "actual": actual}))
    panel = pd.concat(parts, ignore_index=True)
    kept, excluded = mod._select_fit_seasons(panel, {_EVAL_SEASON})
    assert 2023 in excluded and excluded[2023] == "look-ahead"
    assert 2021 in excluded and excluded[2021] == "non-stationary"
    assert set(kept) == {2022, 2024}


def test_select_fit_seasons_catches_partial_backfill() -> None:
    """The bug a residual-std detector misses: a season ~40% backfilled still has a
    normal-looking std (genuine rows dominate the variance) but must be dropped — the
    copied rows pile up at residual ≈ 0, which the near-exact detector catches."""
    rng = np.random.default_rng(3)
    n = 300
    proj = rng.uniform(5, 25, n)
    # 40% exact copies of the actual, 60% genuine (std ~6) → overall std stays ~5.
    is_copy = rng.random(n) < 0.40
    actual = np.where(is_copy, proj, proj + rng.normal(0, 6, n))
    leak = pd.DataFrame({"season": 2022, "projection": proj, "actual": actual})
    genuine = pd.DataFrame(
        {"season": 2024, "projection": rng.uniform(5, 25, n), "actual": rng.uniform(5, 25, n)}
    )
    genuine["actual"] = genuine["projection"] + rng.normal(0, 6, n)
    panel = pd.concat([leak, genuine], ignore_index=True)
    assert panel[panel.season == 2022].eval("actual - projection").std() > 3.0  # std looks normal
    kept, excluded = mod._select_fit_seasons(panel, {_EVAL_SEASON})
    assert excluded.get(2022) == "look-ahead"  # caught despite the normal std
    assert kept == [2024]


# --------------------------------------------------------------------------- #
# build_intervals end-to-end (stubbed loaders)
# --------------------------------------------------------------------------- #


def test_intervals_calibrated_and_leak_excluded() -> None:
    result = _run()
    qb = result["intervals"]["nflcom"]["QB"]
    assert not qb.get("skipped")
    # Leak season excluded; the genuine stationary seasons fit. (In-memory keys are
    # ints — they only become strings once round-tripped through JSON.)
    assert qb["excluded_seasons"].get(_LEAK_SEASON) == "look-ahead"
    assert set(qb["fit_seasons"]) == set(_FIT_SEASONS)
    # Held-out coverage is computed and in a sane band (synthetic noise, not the real
    # ≈0.8 — that is verified on the committed JSON in the serving contract test).
    assert 0.55 <= qb["calibration"]["coverage"] <= 0.95
    assert qb["calibration"]["n_eval"] == len(_PLAYERS) * len(_WEEKS)
    # The look-ahead season bubbles up to the source meta.
    assert _LEAK_SEASON in result["sources_meta"]["nflcom"]["look_ahead_seasons"]


def test_params_and_examples_shape() -> None:
    qb = _run()["intervals"]["nflcom"]["QB"]
    assert set(qb["params"]) == {"floor", "median", "ceiling"}
    for p in qb["params"].values():
        assert set(p) == {"intercept", "slope"}
    examples = qb["examples"]
    assert 0 < len(examples) <= mod._MAX_EXAMPLES
    # One row per distinct player (deduped), each fully formed + self-consistent.
    assert len({e["player_id"] for e in examples}) == len(examples)
    for e in examples:
        assert e["floor"] <= e["median"] <= e["ceiling"]
        assert e["in_band"] == (e["floor"] <= e["actual"] <= e["ceiling"])


def test_coverage_holes_and_totals_only() -> None:
    result = _run()
    # NFL.com has no DST; RotoWire has no K.
    assert result["intervals"]["nflcom"]["DST"] is None
    assert result["intervals"]["rotowire"]["K"] is None
    # NFL.com K is totals-only (flagged), and DST actuals are absent here → K skip is
    # the only nflcom totals path; assert the flag wiring on a present K block instead.
    assert ("nflcom", "K") in mod._TOTALS_ONLY


def test_main_writes_parseable_json(tmp_path) -> None:
    # main() uses the network default loaders; build via stubs and dump the same way
    # (json.dump with mod._json_default) to exercise the serialization contract.
    out = tmp_path / "expert_intervals.json"
    result = _run()
    out.write_text(json.dumps(result, default=mod._json_default))
    payload = json.loads(out.read_text())
    assert payload["nominal_coverage"] == pytest.approx(0.8)
    assert set(payload["intervals"]) == {"nflcom", "rotowire"}
    assert payload["intervals"]["nflcom"]["QB"]["calibration"]["coverage"] is not None

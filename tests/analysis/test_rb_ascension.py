"""Unit tests for src/analysis/rb_ascension.py.

The CLI ``main()`` runs the full RB pipeline (slow, data-dependent) and pulls
nflverse weekly data, so it is out of scope here. These cover the pure helpers
against tiny synthetic frames so a future change to the cohort/lag/metric logic
is caught by CI, and assert that importing the module does not fire the pipeline
(``run`` is imported lazily inside ``main()``) — the drift class CLAUDE.md flags
for operator-only CLIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import rb_ascension as a

pytestmark = pytest.mark.unit


def _row(pid, name, wk, car, tgt, fp, team="T"):
    return {
        "player_id": pid,
        "player_display_name": name,
        "player_name": name,
        "position": "RB",
        "recent_team": team,
        "season": 2023,
        "week": wk,
        "season_type": "REG",
        "carries": car,
        "targets": tgt,
        "receptions": tgt,
        "rushing_yards": car * 4,
        "receiving_yards": tgt * 6,
        "fantasy_points": fp,
        "fantasy_points_ppr": fp + tgt,
    }


def _weekly() -> pd.DataFrame:
    """Lead back L (wk1-3, absent wk4) + backup B who ascends wk4 + a scrub."""
    rows = [
        _row("L", "Lead Back", 1, 15, 2, 12.0),
        _row("L", "Lead Back", 2, 16, 2, 13.0),
        _row("L", "Lead Back", 3, 14, 2, 11.0),  # wk4 absent -> injury-linked
        _row("B", "Backup Asc", 1, 3, 1, 3.0),
        _row("B", "Backup Asc", 2, 2, 1, 2.5),
        _row("B", "Backup Asc", 3, 4, 1, 4.0),
        _row("B", "Backup Asc", 4, 20, 4, 18.0),  # ascension: prior3 opp ~4, opp 24
        _row("S", "Scrub", 1, 1, 0, 0.5, team="U"),
        _row("S", "Scrub", 2, 1, 0, 0.5, team="U"),
        _row("S", "Scrub", 3, 1, 0, 0.5, team="U"),
        _row("S", "Scrub", 4, 2, 0, 1.0, team="U"),
    ]
    return pd.DataFrame(rows)


def _test_df() -> pd.DataFrame:
    """A pipeline-style test_df: one ascension row, two established."""
    return pd.DataFrame(
        {
            "rolling_mean_carries_L3": [3.0, 15.0, 0.0],
            "rolling_mean_targets_L3": [1.0, 3.0, 0.0],
            "carries": [20, 18, 1],
            "targets": [4, 5, 0],
            "fantasy_points": [18.0, 17.0, 0.5],
            "pred_ridge_total": [5.0, 16.0, 1.0],
            "pred_lgbm_total": [4.0, 15.0, 0.8],
            "pred_nn_total": [4.5, 15.5, 0.9],
            "pred_attn_nn_total": [6.0, 16.0, 1.1],
        }
    )


def test_import_is_cheap_and_defers_pipeline():
    # run() is imported lazily inside main(), so a bare import cannot pull
    # torch / the pipeline.
    assert not hasattr(a, "run")
    for fn in (
        "prepare_weekly",
        "find_ascension_events",
        "add_injury_attribution",
        "label_ascension_rows",
        "convergence_table",
        "cohort_model_table",
        "main",
    ):
        assert callable(getattr(a, fn))
    assert a.BACKUP_OPP == 8.0
    assert a.WORKHORSE_OPP == 18.0
    assert a.ACTUAL == "fantasy_points"
    assert a.MODELS["LightGBM"] == "pred_lgbm_total"
    assert {a.ASCENSION, a.ESTABLISHED, a.UNKNOWN} == {"ascension", "established", "unknown"}


def test_find_ascension_events_isolates_the_transition():
    ev = a.find_ascension_events(a.prepare_weekly(_weekly()))
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row["player_id"] == "B"
    assert row["week"] == 4
    assert row["prior3_games"] == 3
    assert row["opp"] == 24
    assert row["prior3_opp"] == pytest.approx(4.0)  # opp=car+tgt: (4+3+5)/3
    # carry_share_L3 = shifted-L3 player carries / team-RB carries = 9 / 54.
    assert row["carry_share_l3"] == pytest.approx(9 / 54)
    # week-W realized share = B's 20 carries / 20 team-RB carries (L absent) = 1.0.
    assert row["game_carry_share"] == pytest.approx(1.0)


def test_thresholds_are_respected():
    prepared = a.prepare_weekly(_weekly())
    # Raise the workhorse bar above B's 24 opportunities -> no events.
    assert a.find_ascension_events(prepared, workhorse_opp=25).empty
    # Drop the backup ceiling below B's prior-3g 4.0 opp/g -> no events.
    assert a.find_ascension_events(prepared, backup_opp=3.0).empty


def test_injury_attribution_flags_absent_lead_back():
    prepared = a.prepare_weekly(_weekly())
    ev = a.find_ascension_events(prepared)
    # Lead back L (45 prior carries) has no week-4 row -> attributed to injury,
    # even with an empty injury-report set.
    linked = a.add_injury_attribution(ev, prepared, inj_out=set())
    assert linked.tolist() == [True]
    # Empty events -> empty Series, no crash.
    assert a.add_injury_attribution(ev.iloc[0:0], prepared, set()).empty


def test_label_ascension_rows_and_missing_column_guard():
    df = _test_df()
    labels = a.label_ascension_rows(df)
    assert labels.tolist() == ["ascension", "established", "established"]
    # A missing required column degrades the whole column to "unknown".
    assert a.label_ascension_rows(df.drop(columns=["carries"])).unique().tolist() == ["unknown"]


def test_cohort_model_table_reports_underprediction():
    df = _test_df()
    df["role_change"] = a.label_ascension_rows(df)
    tbl = a.cohort_model_table(df)
    ridge_asc = tbl[(tbl["model"] == "Ridge") & (tbl["bucket"] == "ascension")].iloc[0]
    # Ridge predicts 5.0 on the lone ascension row (actual 18) -> MAE 13, bias -13.
    assert ridge_asc["n"] == 1
    assert ridge_asc["mae"] == pytest.approx(13.0)
    assert ridge_asc["bias"] == pytest.approx(-13.0)
    # Every model under-predicts the ascension week (negative bias).
    asc = tbl[tbl["bucket"] == "ascension"]
    assert (asc["bias"] < 0).all()
    assert set(tbl["model"]) == set(a.MODELS)


def test_convergence_table_offsets():
    prepared = a.prepare_weekly(_weekly())
    ev = a.find_ascension_events(prepared)
    conv = a.convergence_table(ev, prepared, max_offset=3)
    assert conv["offset"].tolist() == ["W+0", "W+1", "W+2", "W+3"]
    w0 = conv[conv["offset"] == "W+0"].iloc[0]
    assert w0["n"] == 1
    assert w0["realized_fp"] == pytest.approx(18.0)
    assert w0["l3_input_fp"] == pytest.approx(9.5 / 3)  # prior3 fp mean (3+2.5+4)/3
    # B has no week 5+, so later offsets are empty.
    assert conv[conv["offset"] == "W+1"].iloc[0]["n"] == 0
    assert np.isnan(conv[conv["offset"] == "W+1"].iloc[0]["realized_fp"])


def test_offense_depth_ranks_min_over_offense_only():
    # min(depth_team) per player-week over Offense rows; non-Offense ignored;
    # str depth_team coerced to numeric (matches the legacy + ESPN-normalized dtype).
    depth = pd.DataFrame(
        {
            "gsis_id": ["p1", "p1", "p1", "p2"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "formation": ["Offense", "Offense", "Defense", "Offense"],
            "depth_team": ["3", "1", "1", "2"],
        }
    )
    out = a._offense_depth_ranks(depth)
    assert len(out) == 2
    assert out.set_index("gsis_id")["rank"].to_dict() == {"p1": 1, "p2": 2}  # Defense row ignored


def test_espn_schema_flows_to_offense_ranks():
    """Guards the fixed bug: a 2025-style ESPN frame (no ``formation``/``depth_team``)
    must still yield offensive ranks once normalized the loader's way — the raw-shim
    path silently dropped it. Reuses the real ``_normalize_espn_depth`` adapter."""
    from src.data.loader import _normalize_espn_depth

    espn = pd.DataFrame(
        {
            "dt": ["2025-09-05 12:00:00", "2025-09-06 12:00:00"],
            "team": ["KC", "KC"],
            "gsis_id": ["00-0000001", "00-0000001"],
            "pos_grp": ["3WR 1TE", "3WR 1TE"],  # offensive package per _is_espn_offense
            "pos_rank": [2, 1],
        }
    )
    schedules = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2025-09-07"],
            "home_team": ["KC"],
            "away_team": ["DET"],
        }
    )
    canonical = _normalize_espn_depth(espn, schedules, 2025)
    assert not canonical.empty  # the ESPN season is NOT dropped
    ranks = a._offense_depth_ranks(canonical)
    assert len(ranks) == 1
    row = ranks.iloc[0]
    assert row["season"] == 2025
    assert row["rank"] == 1  # latest snapshot before kickoff (Sep 6), min rank

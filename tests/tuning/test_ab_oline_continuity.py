"""Unit pins for the E2 O-line continuity A/B spec (src/tuning/ab_oline_continuity.py).

No training: spec resolution, OL top-5 aggregation + team-code normalization, the
W-1 leakage lags (distinct-starters / prev-overlap / stable-streak), injector
row/order preservation, whitelist mutator branch placement, the rank-metric fn,
and the module-level-import launcher-safety guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.tuning.ab_harness as H
import src.tuning.ab_oline_continuity as spec

pytestmark = pytest.mark.unit


# ---------- spec resolution -------------------------------------------------


def test_spec_resolves_as_dotted():
    resolved = H.resolve_spec("src.tuning.ab_oline_continuity")
    assert resolved.positions == ["QB", "RB", "WR", "TE"]
    assert resolved.baseline == "baseline"
    assert set(resolved.variants) == {"baseline", "oline_continuity"}
    assert resolved.variants["baseline"].is_baseline_shape
    arm = resolved.variants["oline_continuity"]
    assert arm.frame_injector is not None
    assert arm.cfg_mutator is not None
    assert arm.expect_ridge_identical is False


def test_variant_names_are_fleet_only_safe():
    import re

    for name in ("baseline", "oline_continuity"):
        assert re.fullmatch(r"[A-Za-z0-9_]+", name)


def test_module_level_imports_are_launcher_safe():
    """launch_ab imports this spec on the lightweight runner (no training deps)
    to size the grid, so no module-level import may pull src.data / nflreadpy —
    those must be deferred into _build_team_week_table (runs only in-container).
    Regression guard mirroring the #1479 smoke ModuleNotFoundError: nflreadpy."""
    import ast
    import pathlib

    src = pathlib.Path(spec.__file__).read_text()
    banned = ("src.data", "nflreadpy")
    for node in ast.parse(src).body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert not any(name == b or name.startswith(b + ".") for b in banned), (
                f"module-level import {name!r} is launcher-hostile; defer it into a function"
            )


# ---------- OL top-5 aggregation + team normalization -----------------------


def _snap_game(game_id, season, week, team, players):
    """players: list of (pfr_id, position, offense_pct)."""
    return pd.DataFrame(
        {
            "game_id": game_id,
            "season": season,
            "week": week,
            "team": team,
            "pfr_player_id": [p[0] for p in players],
            "position": [p[1] for p in players],
            "offense_pct": [p[2] for p in players],
        }
    )


def test_aggregate_selects_top5_ol_and_normalizes_team():
    # 6 OL (one a combo "G/OT"), plus a WR that must be excluded; STL→LA rename.
    game = _snap_game(
        "g1",
        2015,
        1,
        "STL",
        [
            ("ol1", "T", 1.0),
            ("ol2", "G", 1.0),
            ("ol3", "C", 1.0),
            ("ol4", "G/OT", 0.98),
            ("ol5", "OT", 0.95),
            ("ol6", "G", 0.30),  # 6th OL, below top-5 -> excluded
            ("wr1", "WR", 1.0),  # not OL
        ],
    )
    out = spec._aggregate_oline_games(game)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["recent_team"] == "LA"  # STL normalized
    assert row["top5"] == frozenset({"ol1", "ol2", "ol3", "ol4", "ol5"})  # top-5, WR + 6th out


def test_legacy_team_codes_all_normalize():
    for legacy, modern in (("OAK", "LV"), ("SD", "LAC"), ("STL", "LA")):
        game = _snap_game("g", 2015, 1, legacy, [(f"p{i}", "G", 1.0) for i in range(5)])
        assert spec._aggregate_oline_games(game).iloc[0]["recent_team"] == modern


# ---------- season-to-date continuity leakage lags --------------------------


def _lines_frame(rows):
    """rows: list of (team, season, week, set_of_ids)."""
    return pd.DataFrame(
        [
            {"recent_team": t, "season": s, "week": w, "top5": frozenset(ids)}
            for t, s, w, ids in rows
        ]
    )


def test_continuity_lags_are_leakage_safe():
    L = {"a", "b", "c", "d", "e"}  # base line
    L2 = {"a", "b", "c", "d", "x"}  # one change vs L
    games = _lines_frame(
        [
            ("KC", 2024, 1, L),
            ("KC", 2024, 2, L),  # same as wk1
            ("KC", 2024, 3, L),  # same again
            ("KC", 2024, 4, L2),  # one swap
            ("KC", 2025, 1, L),  # new season resets
        ]
    )
    full = spec._season_to_date_continuity(games)
    out = full[full["season"] == 2024].set_index("week")  # scope: wk1 exists in both seasons

    # Week 1: opener, all NaN.
    assert np.isnan(out.loc[1, "oline_distinct_starters_std"])
    assert np.isnan(out.loc[1, "oline_prev_overlap_std"])
    assert np.isnan(out.loc[1, "oline_stable_streak_std"])
    # Week 2: one prior game -> 5 distinct; overlap needs 2 priors -> NaN; streak=1.
    assert out.loc[2, "oline_distinct_starters_std"] == 5.0
    assert np.isnan(out.loc[2, "oline_prev_overlap_std"])
    assert out.loc[2, "oline_stable_streak_std"] == 1.0
    # Week 3: two priors (both L) -> distinct 5, overlap 5, streak 2.
    assert out.loc[3, "oline_distinct_starters_std"] == 5.0
    assert out.loc[3, "oline_prev_overlap_std"] == 5.0
    assert out.loc[3, "oline_stable_streak_std"] == 2.0
    # Week 4: three priors (L,L,L) -> distinct 5, overlap(wk3,wk2)=5, streak 3.
    assert out.loc[4, "oline_distinct_starters_std"] == 5.0
    assert out.loc[4, "oline_prev_overlap_std"] == 5.0
    assert out.loc[4, "oline_stable_streak_std"] == 3.0

    # 2025 wk1 resets despite 2024 history.
    r = full[(full["season"] == 2025) & (full["week"] == 1)].iloc[0]
    assert np.isnan(r["oline_distinct_starters_std"])


def test_continuity_counts_distinct_and_breaks_streak_on_change():
    L = {"a", "b", "c", "d", "e"}
    L2 = {"a", "b", "c", "d", "x"}  # swap e->x
    games = _lines_frame([("KC", 2024, 1, L), ("KC", 2024, 2, L2), ("KC", 2024, 3, L2)])
    out = spec._season_to_date_continuity(games).set_index("week")
    # Week 3: priors are L (wk1) and L2 (wk2) -> union {a,b,c,d,e,x}=6 distinct;
    # overlap(wk2=L2, wk1=L)=4; streak: wk2==... only 1 prior game at the tail
    # differs from the one before -> streak counts run ending at wk2 = 1.
    assert out.loc[3, "oline_distinct_starters_std"] == 6.0
    assert out.loc[3, "oline_prev_overlap_std"] == 4.0
    assert out.loc[3, "oline_stable_streak_std"] == 1.0


# ---------- injector ---------------------------------------------------------


def test_inject_preserves_rows_and_leaves_missing_nan(monkeypatch):
    tbl = pd.DataFrame(
        {
            "recent_team": ["KC", "KC"],
            "season": [2024, 2024],
            "week": [2, 3],
            "oline_distinct_starters_std": [5.0, 6.0],
            "oline_prev_overlap_std": [np.nan, 5.0],
            "oline_stable_streak_std": [1.0, 2.0],
        }
    )
    monkeypatch.setattr(spec, "_build_team_week_table", lambda seasons: tbl)

    def frame(weeks):
        return pd.DataFrame(
            {
                "player_id": [f"p{w}" for w in weeks],
                "recent_team": ["KC"] * len(weeks),
                "season": [2024] * len(weeks),
                "week": weeks,
                "position": ["RB"] * len(weeks),
            }
        )

    train, val, test = frame([1, 2]), frame([3]), frame([4])
    ot, ov, ote = spec._inject_oline(train, val, test)
    for orig, out in ((train, ot), (val, ov), (test, ote)):
        assert len(out) == len(orig)
        assert list(out["player_id"]) == list(orig["player_id"])
        assert set(spec.FEATURE_COLS) <= set(out.columns)
    assert np.isnan(ot.loc[0, "oline_distinct_starters_std"])  # wk1: no table row
    assert ot.loc[1, "oline_distinct_starters_std"] == 5.0
    assert np.isnan(ote.loc[0, "oline_distinct_starters_std"])  # wk4 unmatched


def test_inject_raises_on_duplicate_table_key(monkeypatch):
    dup = pd.DataFrame(
        {
            "recent_team": ["KC", "KC"],
            "season": [2024, 2024],
            "week": [2, 2],
            "oline_distinct_starters_std": [5.0, 6.0],
            "oline_prev_overlap_std": [4.0, 4.0],
            "oline_stable_streak_std": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(spec, "_build_team_week_table", lambda seasons: dup)
    f = pd.DataFrame({"player_id": ["p"], "recent_team": ["KC"], "season": [2024], "week": [2]})
    with pytest.raises(pd.errors.MergeError):
        spec._inject_oline(f.copy(), f.copy(), f.copy())


# ---------- whitelist mutator ------------------------------------------------


def _fake_cfg():
    return {
        "get_feature_columns_fn": lambda: ["existing_a", "existing_b"],
        "attn_static_features": ["existing_a"],
    }


def test_mutator_extends_both_branches():
    cfg = spec._mut_whitelist(_fake_cfg())
    cols = cfg["get_feature_columns_fn"]()
    assert cols[:2] == ["existing_a", "existing_b"]
    assert set(spec.FEATURE_COLS) <= set(cols)
    assert set(spec.FEATURE_COLS) <= set(cfg["attn_static_features"])
    for col in spec.FEATURE_COLS:  # season-to-date, never windowed -> static-legal
        assert not any(tok in col for tok in ("_L3", "_L5", "_L8", "rolling", "ewma", "trend"))


def test_mutator_without_attn_key_is_safe():
    out = spec._mut_whitelist({"get_feature_columns_fn": lambda: ["x"]})
    assert set(spec.FEATURE_COLS) <= set(out["get_feature_columns_fn"]())
    assert "attn_static_features" not in out


def test_mutator_static_append_is_idempotent():
    cfg = _fake_cfg()
    cfg["attn_static_features"] = ["existing_a", *spec.FEATURE_COLS]
    out = spec._mut_whitelist(cfg)
    assert len(out["attn_static_features"]) == len(set(out["attn_static_features"]))


# ---------- metric_fn --------------------------------------------------------


def _synthetic_result(with_elite_col=True):
    rng = np.random.default_rng(0)
    rows = []
    for w in (1, 2):
        for i in range(14):
            actual = float(5 + i + rng.normal(0, 0.1))
            rows.append(
                {
                    "player_id": f"p{i}",
                    "season": 2025,
                    "week": w,
                    "fantasy_points": actual,
                    "pred_ridge_total": actual + 1.0,
                    "pred_nn_total": actual - 1.0,
                    "pred_attn_nn_total": actual + 0.5,
                    "pred_lgbm_total": actual - 0.5,
                }
            )
    df = pd.DataFrame(rows)
    if with_elite_col:
        df["prior_season_mean_fantasy_points"] = df["player_id"].str[1:].astype(float)
    return {"test_df": df}


def test_metric_fn_emits_ridge_mae_and_rank_metrics():
    out = spec.metric_fn(_synthetic_result(), "RB")
    assert out["Ridge"]["mae"] == pytest.approx(1.0, abs=1e-6)
    for row in out.values():
        assert {"mae", "hit12", "spearman", "regret12", "regret_lineup"} <= set(row)
    assert out["Ridge"]["hit12"] == pytest.approx(1.0)
    assert out["Ridge"]["regret12"] == pytest.approx(0.0, abs=1e-9)


def test_metric_fn_elite_slice_guards_on_column():
    with_elite = spec.metric_fn(_synthetic_result(True), "RB")
    assert with_elite["Ridge"]["elite24_bias"] == pytest.approx(1.0, abs=1e-6)
    without = spec.metric_fn(_synthetic_result(False), "RB")
    assert "elite24_mae" not in without["Ridge"]

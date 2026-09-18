"""Unit tests for the week-1 games-gap A/B spec (src/tuning/ab_games_gap.py).

No training here (that's the fleet run). Coverage: spec resolution + design shape,
the career-gap injector's cross-split semantics (the concat-before-diff trap: the
splits are season-disjoint, so a per-frame diff would turn every val/test week-1
row into a career debut), the itt-empty injector's first-in-season gating, the
mutators' branch placement, and the cohort metric (incl. the #1137 passing-yards
guard's column gating).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tuning import ab_games_gap as G
from src.tuning import ab_harness as H

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# spec resolution + design shape
# --------------------------------------------------------------------------- #
def test_games_gap_spec_resolves_as_dotted():
    """QB/RB/WR/TE grid; baseline is a pure identity; both feature arms carry a
    frame injector + cfg mutator and declare ``expect_ridge_identical=False``;
    the RB re-add arm is cfg-only and report-only (it no-ops where the column is
    already whitelisted, so a hard sentinel expectation would misfire)."""
    spec = H.resolve_spec("src.tuning.ab_games_gap")
    assert spec.dotted == "src.tuning.ab_games_gap"
    assert spec.positions == ["QB", "RB", "WR", "TE"]
    assert spec.baseline == "baseline"
    assert set(spec.variants) == {"baseline", "career_gap", "itt_empty", "rb_returning"}
    assert spec.variants["baseline"].is_baseline_shape
    for name in ("career_gap", "itt_empty"):
        v = spec.variants[name]
        assert v.frame_injector is not None
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is False
    rb = spec.variants["rb_returning"]
    assert rb.frame_injector is None  # cfg-only: stays runnable if pointed at K/DST
    assert rb.cfg_mutator is not None
    assert rb.expect_ridge_identical is None


# --------------------------------------------------------------------------- #
# +career_gap injector — cross-split gap semantics
# --------------------------------------------------------------------------- #
def _row(pid, season, week, **kw):
    return dict(player_id=pid, season=season, week=week, implied_team_total=21.0, **kw)


def _split_frames():
    """Season-disjoint splits mirroring production (train 2023, val 2024, test 2025).

    X plays 2023 wks 15/17/18 (train), 2024 wks 1/4 (val), 2025 wk 1 (test).
    Z has a >2-year gap (2013 wk1 -> 2016 wk1, both train). Y debuts in test.
    Val/test get non-Range indexes + shuffled row order to pin index-alignment.
    """
    train = pd.DataFrame(
        [
            _row("X", 2023, 15),
            _row("X", 2023, 17),
            _row("X", 2023, 18),
            _row("Z", 2013, 1),
            _row("Z", 2016, 1),
        ]
    )
    val = pd.DataFrame([_row("X", 2024, 4), _row("X", 2024, 1)], index=[10, 11])
    test = pd.DataFrame([_row("Y", 2025, 1), _row("X", 2025, 1)], index=[7, 3])
    return train, val, test


def test_inject_career_gap_concat_before_diff():
    """The trap: X's 2024 wk1 (val) gap must be 35 weeks back to 2023 wk18
    (train) — not 1 (per-season diff fillna) and not 104 (per-frame debut)."""
    train, val, test = G._inject_career_gap(*_split_frames())
    v = val.set_index("week")[G.GAP_COL]
    assert v.loc[1] == pytest.approx(35.0)  # (2024*52+1) - (2023*52+18)
    assert v.loc[4] == pytest.approx(3.0)  # 2024 wk1 -> wk4
    t = test.set_index("player_id")[G.GAP_COL]
    assert t.loc["X"] == pytest.approx(49.0)  # (2025*52+1) - (2024*52+4)
    assert t.loc["Y"] == pytest.approx(G.GAP_CAP)  # career debut sentinel
    tr = train.set_index(["player_id", "season", "week"])[G.GAP_COL]
    assert tr.loc[("X", 2023, 15)] == pytest.approx(G.GAP_CAP)  # first data row = debut
    assert tr.loc[("X", 2023, 17)] == pytest.approx(2.0)  # bye week
    assert tr.loc[("X", 2023, 18)] == pytest.approx(1.0)  # consecutive weeks
    assert tr.loc[("Z", 2016, 1)] == pytest.approx(G.GAP_CAP)  # 156 clips to the cap


def test_inject_career_gap_preserves_frames():
    """Original index values, row order, and existing columns survive; the new
    column is float64 everywhere (Ridge/scaler-friendly)."""
    train, val, test = _split_frames()
    orig_val = val.drop(columns=[], inplace=False).copy()
    train2, val2, test2 = G._inject_career_gap(train, val, test)
    assert list(val2.index) == list(orig_val.index)
    pd.testing.assert_frame_equal(val2.drop(columns=[G.GAP_COL]), orig_val)
    for df in (train2, val2, test2):
        assert df[G.GAP_COL].dtype == np.float64
        assert df[G.GAP_COL].between(1.0, G.GAP_CAP).all()


# --------------------------------------------------------------------------- #
# +itt_empty injector — first-in-season gating
# --------------------------------------------------------------------------- #
def test_inject_itt_empty_first_in_season_rows_only():
    df = pd.DataFrame(
        [
            dict(player_id="A", season=2025, week=2, implied_team_total=21.0),  # not first
            dict(player_id="A", season=2025, week=1, implied_team_total=24.5),  # first
            dict(player_id="B", season=2025, week=3, implied_team_total=17.0),  # not first
            dict(player_id="B", season=2025, week=2, implied_team_total=18.0),  # first (wk2!)
            dict(player_id="C", season=2025, week=1, implied_team_total=np.nan),  # first, NaN
        ],
        index=[4, 9, 2, 0, 5],
    )
    out, _, _ = G._inject_itt_empty(df, df.iloc[0:0].copy(), df.iloc[0:0].copy())
    got = out.set_index(["player_id", "week"])[G.ITT_COL]
    assert got.loc[("A", 1)] == pytest.approx(24.5)
    assert got.loc[("A", 2)] == 0.0
    assert got.loc[("B", 2)] == pytest.approx(18.0)  # first in-season game, not week 1
    assert got.loc[("B", 3)] == 0.0
    assert got.loc[("C", 1)] == 0.0  # NaN implied total mirrors the pipeline's 0-fill


# --------------------------------------------------------------------------- #
# config mutators — whitelist + static-branch placement
# --------------------------------------------------------------------------- #
def _fake_cfg() -> dict:
    return {"get_feature_columns_fn": lambda: ["x"], "attn_static_features": ["x"]}


def test_mut_career_gap_and_itt_place_columns():
    for mut, col in ((G._mut_career_gap, G.GAP_COL), (G._mut_itt_empty, G.ITT_COL)):
        cfg = mut(_fake_cfg())
        assert col in cfg["get_feature_columns_fn"]()
        assert col in cfg["attn_static_features"]  # non-windowed scalar: static-eligible
        # A cfg without the attention key (K/DST shapes) must not crash.
        assert col in mut({"get_feature_columns_fn": lambda: ["x"]})["get_feature_columns_fn"]()


def test_mut_rb_returning_add_if_absent():
    col = "is_returning_from_absence"
    cfg = G._mut_rb_returning(_fake_cfg())
    assert cfg["get_feature_columns_fn"]().count(col) == 1
    assert cfg["attn_static_features"].count(col) == 1
    # Already whitelisted (QB/WR/TE shape) -> no duplicate entry appended.
    cfg2 = G._mut_rb_returning(
        {"get_feature_columns_fn": lambda: ["x", col], "attn_static_features": ["x", col]}
    )
    assert cfg2["get_feature_columns_fn"]().count(col) == 1
    assert cfg2["attn_static_features"].count(col) == 1


# --------------------------------------------------------------------------- #
# metric — cohort slices + the #1137 guard
# --------------------------------------------------------------------------- #
def test_metric_fn_cohort_slices_and_guard():
    y = np.array([10.0, 12.0, 8.0, 20.0, 6.0])
    df = pd.DataFrame(
        {
            "player_id": ["A", "A", "A", "B", "B"],
            "season": [2025] * 5,
            "week": [1, 2, 3, 2, 3],
            "fantasy_points": y,
            "is_returning_from_absence": [0, 0, 0, 0, 1],
            "pred_ridge_total": y + 1.0,  # uniform +1 over-prediction
            "pred_lgbm_total": y - 2.0,  # uniform -2 under-prediction
            "passing_yards": [250.0, 300.0, 200.0, 150.0, 100.0],
            "pred_ridge_passing_yards": [260.0, 310.0, 210.0, 160.0, 110.0],
        }
    )
    out = G.metric_fn({"test_df": df}, "QB")

    assert out["Ridge"]["mae"] == pytest.approx(1.0)  # feeds the harness Ridge sentinel
    assert out["Ridge"]["week1_n"] == 1.0  # A wk1 only
    assert out["Ridge"]["week1_bias"] == pytest.approx(1.0)
    assert out["LightGBM"]["week1_bias"] == pytest.approx(-2.0)
    assert out["Ridge"]["week1_actual"] == pytest.approx(10.0)
    assert out["Ridge"]["rest_n"] == 4.0
    assert out["Ridge"]["npg0_n"] == 2.0  # A wk1 + B wk2 (first in-season games)
    assert out["Ridge"]["returning_n"] == 1.0  # B wk3
    assert out["Ridge"]["returning_bias"] == pytest.approx(1.0)
    # #1137 guard: per-target column exists for Ridge only in this toy frame.
    assert out["Ridge"]["passing_yards_mae"] == pytest.approx(10.0)
    assert "passing_yards_mae" not in out["LightGBM"]  # stacked-mode / non-QB shape

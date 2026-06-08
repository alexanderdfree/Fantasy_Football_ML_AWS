"""Unit tests for the WR boom-signal A/B spec (src/tuning/ab_boom_signals_wr.py).

No training here (that's the manual harness run). Coverage: spec resolution + design
shape, the frame injector's column build (leakage-safe shift, WR-scoped team totals,
non-WR rows untouched), the config mutators' branch placement (rolling NEVER reaches the
static branch — the stop-rule), and the boom-subgroup metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tuning import ab_boom_signals_wr as B
from src.tuning import ab_harness as H

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# spec resolution + design shape
# --------------------------------------------------------------------------- #
def test_boom_signals_spec_resolves_as_dotted():
    """Imports + resolves and keeps its shape: WR-only, four arms, every ``+`` arm carries
    a frame injector + cfg mutator and declares ``expect_ridge_identical=False`` (a real
    whitelist feature must move Ridge); ``baseline`` is a pure identity (no injector)."""
    spec = H.resolve_spec("src.tuning.ab_boom_signals_wr")
    assert spec.dotted == "src.tuning.ab_boom_signals_wr"
    assert spec.positions == ["WR"]
    assert spec.baseline == "baseline"
    assert set(spec.variants) == {"baseline", "+rz", "+parity", "+all"}
    assert spec.variants["baseline"].is_baseline_shape  # identity → subgroup slice is native
    for name in ("+rz", "+parity", "+all"):
        v = spec.variants[name]
        assert v.frame_injector is not None
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is False


# --------------------------------------------------------------------------- #
# frame injector
# --------------------------------------------------------------------------- #
def _wr_frame() -> pd.DataFrame:
    """KC: WR A (lead) + WR B over 3 weeks, plus an RB whose targets must NOT count toward
    the team-WR denominator. A: targets 10/8/6, redzone_targets 2/0/1."""
    rows = []
    for wk, (a_t, a_rz, a_rzs, b_t) in enumerate(
        [(10, 2, 0.5, 2), (8, 0, 0.0, 4), (6, 1, 0.25, 6)], start=1
    ):
        rows.append(
            dict(
                player_id="A",
                position="WR",
                recent_team="KC",
                season=2023,
                week=wk,
                targets=a_t,
                carries=0,
                redzone_targets=a_rz,
                redzone_target_share=a_rzs,
                prior_season_mean_receptions=4.0,
                prior_season_mean_targets=6.0,
            )  # fmt: skip
        )
        rows.append(
            dict(
                player_id="B",
                position="WR",
                recent_team="KC",
                season=2023,
                week=wk,
                targets=b_t,
                carries=0,
                redzone_targets=0,
                redzone_target_share=0.0,
                prior_season_mean_receptions=2.0,
                prior_season_mean_targets=4.0,
            )  # fmt: skip
        )
    # An RB on the same team-game: its 5 targets must be excluded from team-WR totals.
    rows.append(
        dict(
            player_id="R",
            position="RB",
            recent_team="KC",
            season=2023,
            week=1,
            targets=5,
            carries=12,
            redzone_targets=3,
            redzone_target_share=0.0,
            prior_season_mean_receptions=1.0,
            prior_season_mean_targets=2.0,
        )  # fmt: skip
    )
    return pd.DataFrame(rows)


def test_inject_builds_columns_and_is_wr_scoped():
    df = _wr_frame()
    out, _, _ = B._inject_wr_signals(df, df.copy(), df.copy())
    g = out.set_index(["player_id", "week"])

    # Team-WR targets wk1 = A(10) + B(2) = 12 — the RB's 5 are excluded (WR-scoped).
    assert g.loc[("A", 1), "game_target_share"] == pytest.approx(10 / 12)
    assert g.loc[("B", 1), "game_target_share"] == pytest.approx(2 / 12)
    # HHI wk1 = (10/12)^2 + (2/12)^2 (RB not in the WR concentration).
    assert g.loc[("A", 1), "game_target_hhi"] == pytest.approx((10 / 12) ** 2 + (2 / 12) ** 2)
    # Non-WR (RB) rows are left at 0.0 — never read by the WR pipeline.
    assert g.loc[("R", 1), "game_target_share"] == 0.0


def test_inject_rolling_is_leakage_safe():
    """``redzone_targets_L3`` must be shift-1 rolling: week 1 sees no prior (NaN), week 3 =
    mean of weeks 1-2 — never the current week's own value."""
    df = _wr_frame()
    out, _, _ = B._inject_wr_signals(df, df.copy(), df.copy())
    a = out[out["player_id"] == "A"].set_index("week")["redzone_targets_L3"]
    assert pd.isna(a.loc[1])  # no prior game → leakage-safe
    assert a.loc[2] == pytest.approx(2.0)  # mean(week1=2)
    assert a.loc[3] == pytest.approx(1.0)  # mean(week1=2, week2=0)


def test_inject_prior_season_catch_rate():
    df = _wr_frame()
    out, _, _ = B._inject_wr_signals(df, df.copy(), df.copy())
    a = out[out["player_id"] == "A"]["prior_season_mean_catch_rate"].iloc[0]
    assert a == pytest.approx(4.0 / 6.0)  # receptions / targets, targets >= 0.5 guard passes


# --------------------------------------------------------------------------- #
# config mutators — branch placement (the static-branch stop-rule)
# --------------------------------------------------------------------------- #
def _fake_cfg() -> dict:
    return {
        "get_feature_columns_fn": lambda: ["x"],
        "attn_static_features": ["x"],
        "attn_history_stats": ["x"],
    }


def test_mut_rz_places_columns_correctly():
    cfg = B._mut_rz(_fake_cfg())
    feats = cfg["get_feature_columns_fn"]()
    # Whitelist gets rolling + prior; static gets ONLY prior (rolling stays out — stop-rule).
    assert set(B._RZ_ROLL) <= set(feats)
    assert set(B._RZ_PRIOR) <= set(feats)
    assert set(B._RZ_PRIOR) <= set(cfg["attn_static_features"])
    assert not (set(B._RZ_ROLL) & set(cfg["attn_static_features"]))  # rolling never static
    assert set(B._RZ_HIST) <= set(cfg["attn_history_stats"])
    # Raw per-game red-zone is history-only, never whitelisted (would be leakage).
    assert not (set(B._RZ_HIST) & set(feats))


def test_mut_parity_and_all_compose():
    cfg = B._mut_all(_fake_cfg())
    feats = cfg["get_feature_columns_fn"]()
    assert set(B._RZ_ROLL) <= set(feats) and set(B._PARITY_ROLL) <= set(feats)
    assert set(B._PARITY_PRIOR) <= set(cfg["attn_static_features"])
    assert set(B._PARITY_HIST) <= set(cfg["attn_history_stats"])
    # opportunity_index_L3 is rolling → whitelist but NOT static.
    assert "opportunity_index_L3" not in cfg["attn_static_features"]


def test_mutator_does_not_need_attn_keys():
    # A position config without the attention keys (rare) must not crash.
    out = B._mut_rz({"get_feature_columns_fn": lambda: ["x"]})
    assert set(B._RZ_ROLL) <= set(out["get_feature_columns_fn"]())


# --------------------------------------------------------------------------- #
# metric — boom subgroup
# --------------------------------------------------------------------------- #
def test_metric_fn_boom_subgroup():
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 22.0, 30.0])  # Q4 (>=q75) = top quartile
    df = pd.DataFrame(
        {
            "fantasy_points": y,
            "receiving_tds": [0, 0, 0, 0, 0, 1, 2, 2],
            "pred_ridge_total": y + 1.0,  # uniform +1 over-prediction
            "pred_lgbm_total": y + 1.0,
        }
    )
    out = B.metric_fn({"test_df": df}, "WR")
    assert "mae" in out["Ridge"]  # overall MAE feeds the harness Ridge sentinel
    assert out["Ridge"]["mae"] == pytest.approx(1.0)
    # Q4 = top quartile by fantasy points; bias is the uniform +1 over-prediction.
    assert out["Ridge"]["q4_n"] == pytest.approx((y >= np.quantile(y, 0.75)).sum())
    assert out["Ridge"]["q4_bias"] == pytest.approx(1.0)
    # rztd cut present (receiving_tds >= 1) and correlation computed on the slice.
    assert out["LightGBM"]["rztd_n"] == pytest.approx(3.0)
    assert out["Ridge"]["q4_corr"] == pytest.approx(1.0)  # pred = y + 1 → perfectly correlated

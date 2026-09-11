"""Unit tests for the cross-position QB-context A/B spec (src/tuning/ab_qb_context_receivers.py).

No training here (that's the manual harness run). Coverage: spec resolution + design shape,
the team-week QB-context build (prior-role leakage safety, out-set is QB-only and rank-aware,
broadcast onto receiver rows, neutral fill for QB-less team-weeks), the config mutator's
branch placement (static branch yes, ``attn_history_stats`` NEVER — the stop-rule), and the
qbout-cohort metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data.nfl_source as nfl_source
from src.tuning import ab_harness as H
from src.tuning import ab_qb_context_receivers as Q

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# spec resolution + design shape
# --------------------------------------------------------------------------- #
def test_spec_resolves_as_dotted():
    """WR/TE/RB, four arms; every ``+`` arm carries the injector + a mutator and declares
    ``expect_ridge_identical=False``; ``baseline`` injects (for cohort slicing) but mutates
    nothing, so the model it trains is production-identical."""
    spec = H.resolve_spec("src.tuning.ab_qb_context_receivers")
    assert spec.positions == ["WR", "TE", "RB"]
    assert spec.baseline == "baseline"
    assert set(spec.variants) == {"baseline", "+qb_out", "+qb_quality", "+both"}
    base = spec.variants["baseline"]
    assert base.frame_injector is not None and base.cfg_mutator is None
    for name in ("+qb_out", "+qb_quality", "+both"):
        v = spec.variants[name]
        assert v.frame_injector is not None
        assert v.cfg_mutator is not None
        assert v.expect_ridge_identical is False


# --------------------------------------------------------------------------- #
# frame injector — synthetic two-team, three-week frame
# --------------------------------------------------------------------------- #
def _row(pid, pos, team, wk, exp=np.nan, season=2023):
    return dict(
        player_id=pid,
        position=pos,
        recent_team=team,
        season=season,
        week=wk,
        total_fantasy_points_exp=exp,
    )


def _frame() -> pd.DataFrame:
    """KC: starter S plays wks 1-2 (exp-FP 20, 40) then sits Out wk3; backup B starts wk3
    (his first game). WR W plays all three weeks. DAL: QB D plays wks 1-3 (exp 10); a
    deep-backup B2 played wk1 (exp 1) and is Out wk3 — an out BACKUP must not flag. WR Z
    plays for NYJ, which has no QB rows at all (neutral fill)."""
    rows = [
        _row("S", "QB", "KC", 1, 20.0),
        _row("S", "QB", "KC", 2, 40.0),
        _row("B", "QB", "KC", 3, 15.0),
        _row("W", "WR", "KC", 1),
        _row("W", "WR", "KC", 2),
        _row("W", "WR", "KC", 3),
        _row("D", "QB", "DAL", 1, 10.0),
        _row("D", "QB", "DAL", 2, 10.0),
        _row("D", "QB", "DAL", 3, 10.0),
        _row("B2", "QB", "DAL", 1, 1.0),
        _row("X", "WR", "DAL", 3),
        _row("Z", "WR", "NYJ", 1),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def _stub_injuries(monkeypatch):
    """Out-set feed: KC's S Out wk3, DAL's B2 Out wk3, plus rows the filter must drop —
    a Questionable QB and an Out WR (QB-only out-set)."""
    inj = pd.DataFrame(
        {
            "position": ["QB", "QB", "QB", "WR"],
            "report_status": ["Out", "Out", "Questionable", "Out"],
            "season": [2023, 2023, 2023, 2023],
            "team": ["KC", "DAL", "DAL", "KC"],
            "week": [3, 3, 2, 3],
            "gsis_id": ["S", "B2", "D", "W"],
        }
    )
    monkeypatch.setattr(nfl_source, "injuries", lambda seasons: inj)


def test_inject_flags_starter_out_week(_stub_injuries):
    df = _frame()
    out, _, _ = Q._inject_qb_context(df, df.copy(), df.copy())
    g = out.set_index(["player_id", "week"])

    # KC wk3: S (prior role mean(20, 40) = 30) is Out and out-ranks B (no prior games → 0).
    assert g.loc[("W", 3), "team_qb_out"] == 1.0
    assert g.loc[("W", 3), "team_qb_vacated_role"] == pytest.approx(30.0)
    assert g.loc[("W", 3), "team_expected_qb_role"] == pytest.approx(0.0)


def test_inject_prior_role_is_leakage_safe(_stub_injuries):
    """Week-2 context must use only week-1's exp-FP (20), never fold in week 2's own 40."""
    df = _frame()
    out, _, _ = Q._inject_qb_context(df, df.copy(), df.copy())
    g = out.set_index(["player_id", "week"])
    assert g.loc[("W", 2), "team_expected_qb_role"] == pytest.approx(20.0)
    assert g.loc[("W", 2), "team_qb_out"] == 0.0
    # Week 1 has no prior games anywhere → all-zero context (documented screen limitation).
    assert g.loc[("W", 1), "team_expected_qb_role"] == 0.0


def test_inject_out_backup_does_not_flag(_stub_injuries):
    """DAL wk3: B2 (prior role 1) is Out but D (prior role 10) plays — no vacancy above the
    present top, so the event columns stay 0 and quality reads D's prior role."""
    df = _frame()
    out, _, _ = Q._inject_qb_context(df, df.copy(), df.copy())
    g = out.set_index(["player_id", "week"])
    assert g.loc[("X", 3), "team_qb_out"] == 0.0
    assert g.loc[("X", 3), "team_qb_vacated_role"] == 0.0
    assert g.loc[("X", 3), "team_expected_qb_role"] == pytest.approx(10.0)


def test_inject_fills_neutral_for_qbless_team_week(_stub_injuries):
    df = _frame()
    out, _, _ = Q._inject_qb_context(df, df.copy(), df.copy())
    z = out[out["player_id"] == "Z"].iloc[0]
    assert (
        z["team_qb_out"] == 0.0
        and z["team_qb_vacated_role"] == 0.0
        and z["team_expected_qb_role"] == 0.0
    )


# --------------------------------------------------------------------------- #
# config mutators — static branch yes, history branch NEVER (stop-rule)
# --------------------------------------------------------------------------- #
def _fake_cfg() -> dict:
    return {
        "get_feature_columns_fn": lambda: ["x"],
        "attn_static_features": ["x"],
        "attn_history_stats": ["x"],
    }


def test_mutators_place_columns_static_only():
    for cols, mut in [
        (Q._EVENT, Q._make_whitelister(Q._EVENT)),
        (Q._QUALITY, Q._make_whitelister(Q._QUALITY)),
        (Q._ALL_COLS, Q._make_whitelister(Q._ALL_COLS)),
    ]:
        cfg = mut(_fake_cfg())
        assert set(cols) <= set(cfg["get_feature_columns_fn"]())
        assert set(cols) <= set(cfg["attn_static_features"])
        assert cfg["attn_history_stats"] == ["x"]  # never touched — stop-rule


def test_mutator_does_not_need_attn_keys():
    out = Q._make_whitelister(Q._EVENT)({"get_feature_columns_fn": lambda: ["x"]})
    assert set(Q._EVENT) <= set(out["get_feature_columns_fn"]())


# --------------------------------------------------------------------------- #
# metric — qbout cohort
# --------------------------------------------------------------------------- #
def test_metric_fn_qbout_cohort():
    y = np.array([2.0, 4.0, 6.0, 8.0, 20.0, 24.0])
    df = pd.DataFrame(
        {
            "fantasy_points": y,
            "pred_ridge_total": y + 1.0,  # uniform +1 over-prediction
            "team_qb_out": [0, 0, 0, 0, 1, 1],
        }
    )
    out = Q.metric_fn({"test_df": df}, "WR")
    assert out["Ridge"]["mae"] == pytest.approx(1.0)  # overall feeds the Ridge sentinel
    assert out["Ridge"]["rmse"] == pytest.approx(1.0)
    assert out["Ridge"]["qbout_n"] == pytest.approx(2.0)
    assert out["Ridge"]["qbout_bias"] == pytest.approx(1.0)


def test_metric_fn_without_cohort_column():
    df = pd.DataFrame({"fantasy_points": [1.0, 2.0], "pred_ridge_total": [1.0, 2.0]})
    out = Q.metric_fn({"test_df": df}, "WR")
    assert "qbout_n" not in out["Ridge"] and out["Ridge"]["mae"] == pytest.approx(0.0)

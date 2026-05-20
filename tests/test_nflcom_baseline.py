"""Unit tests for src.analysis.analysis_nflcom_baseline.

The script no longer trains models — it just pulls NFL.com projections + actuals
from the web and scores both through our PPR aggregator. Tests construct
synthetic NFL.com + actuals frames and inject them via the loader hooks; nothing
hits the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import analysis_nflcom_baseline as ab
from src.config import SCORING_PPR, SCORING_STANDARD

pytestmark = pytest.mark.unit


# ---------- Fixtures --------------------------------------------------------


def _make_qb_actuals(*, seasons=(2025,), n_weeks: int = 4) -> pd.DataFrame:
    """Two QBs × n_weeks weeks × n_seasons. Realistic raw stats + computed FP."""
    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for player_id in ("00-Q1", "00-Q2"):
            for week in range(1, n_weeks + 1):
                py = rng.uniform(180, 320)
                ptd = rng.uniform(1.0, 3.0)
                ints = rng.uniform(0.0, 1.5)
                ry = rng.uniform(0.0, 50.0)
                rtd = rng.uniform(0.0, 0.6)
                fum_lost = rng.uniform(0.0, 0.4)
                fp = (
                    py * SCORING_PPR["passing_yards"]
                    + ptd * SCORING_PPR["passing_tds"]
                    + ints * SCORING_PPR["interceptions"]
                    + ry * SCORING_PPR["rushing_yards"]
                    + rtd * SCORING_PPR["rushing_tds"]
                    + fum_lost * SCORING_PPR["fumbles_lost"]
                )
                rows.append(
                    {
                        "player_id": player_id,
                        "season": int(season),
                        "week": week,
                        "position": "QB",
                        "passing_yards": py,
                        "passing_tds": ptd,
                        "interceptions": ints,
                        "rushing_yards": ry,
                        "rushing_tds": rtd,
                        "fumbles_lost": fum_lost,
                        "fantasy_points": fp,
                    }
                )
    return pd.DataFrame(rows)


def _make_nflcom_for_qb(actuals: pd.DataFrame, *, perfect: bool = False) -> pd.DataFrame:
    """Build an NFL.com-shaped frame keyed by (player_id, season, week, position)."""
    rng = np.random.default_rng(7)
    out_rows = []
    for _, row in actuals.iterrows():
        if perfect:
            scaled = {
                t: row[t]
                for t in (
                    "passing_yards",
                    "passing_tds",
                    "interceptions",
                    "rushing_yards",
                    "rushing_tds",
                    "fumbles_lost",
                )
            }
        else:
            scaled = {
                "passing_yards": row["passing_yards"] + rng.normal(0, 5),
                "passing_tds": row["passing_tds"] + rng.normal(0, 0.1),
                "interceptions": row["interceptions"] + rng.normal(0, 0.1),
                "rushing_yards": row["rushing_yards"] + rng.normal(0, 1),
                "rushing_tds": row["rushing_tds"] + rng.normal(0, 0.05),
                "fumbles_lost": row["fumbles_lost"] + rng.normal(0, 0.05),
            }
        out_rows.append(
            {
                "player_id": row["player_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "position": "QB",
                "nflcom_projected_pts": 18.0,
                **scaled,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
            }
        )
    return pd.DataFrame(out_rows)


# ---------- Aggregation tests ------------------------------------------------


def test_project_nflcom_to_ppr_qb_uses_scoring_dict():
    """NFL.com QB row with known stats should yield exactly SCORING_PPR-weighted total."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-001",
                "season": 2025,
                "week": 1,
                "position": "QB",
                "passing_yards": 300.0,
                "passing_tds": 2.0,
                "interceptions": 1.0,
                "rushing_yards": 50.0,
                "rushing_tds": 0.5,
                "fumbles_lost": 0.2,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "nflcom_projected_pts": 22.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "QB", scoring_format="ppr")
    expected = (
        300.0 * SCORING_PPR["passing_yards"]
        + 2.0 * SCORING_PPR["passing_tds"]
        + 1.0 * SCORING_PPR["interceptions"]
        + 50.0 * SCORING_PPR["rushing_yards"]
        + 0.5 * SCORING_PPR["rushing_tds"]
        + 0.2 * SCORING_PPR["fumbles_lost"]
    )
    assert out["nflcom_pred_total"].iloc[0] == pytest.approx(expected)
    assert out["nflcom_projected_pts"].iloc[0] == pytest.approx(22.0)


def test_project_nflcom_to_ppr_rb_ppr_vs_standard_propagates():
    """Verify scoring_format reaches the aggregator: 5 receptions = +5 PPR, 0 standard."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-RB1",
                "season": 2025,
                "week": 1,
                "position": "RB",
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "receiving_yards": 80.0,
                "receiving_tds": 0.0,
                "receptions": 5.0,
                "fumbles_lost": 0.0,
                "nflcom_projected_pts": 13.0,
            }
        ]
    )
    ppr = ab._project_nflcom_to_ppr(df, "RB", scoring_format="ppr")
    std = ab._project_nflcom_to_ppr(df, "RB", scoring_format="standard")
    assert ppr["nflcom_pred_total"].iloc[0] == pytest.approx(
        80 * SCORING_PPR["receiving_yards"] + 5 * SCORING_PPR["receptions"]
    )
    assert std["nflcom_pred_total"].iloc[0] == pytest.approx(
        80 * SCORING_STANDARD["receiving_yards"] + 5 * SCORING_STANDARD["receptions"]
    )


def test_project_nflcom_to_ppr_drops_unmatched():
    """Rows with player_id NaN are dropped — they can't be joined."""
    df = pd.DataFrame(
        [
            {
                "player_id": np.nan,
                "season": 2025,
                "week": 1,
                "position": "QB",
                "passing_yards": 200.0,
                "passing_tds": 1.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "fumbles_lost": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "nflcom_projected_pts": 12.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "QB")
    assert len(out) == 0


def test_project_nflcom_to_ppr_k_uses_native_pts():
    """K can't decompose into raw stats — uses NFL.com's native projection directly."""
    df = pd.DataFrame(
        [
            {
                "player_id": "00-K1",
                "season": 2025,
                "week": 1,
                "position": "K",
                "nflcom_projected_pts": 8.5,
                "passing_yards": 0.0,
                "passing_tds": 0.0,
                "interceptions": 0.0,
                "rushing_yards": 0.0,
                "rushing_tds": 0.0,
                "receiving_yards": 0.0,
                "receiving_tds": 0.0,
                "receptions": 0.0,
                "fumbles_lost": 0.0,
            }
        ]
    )
    out = ab._project_nflcom_to_ppr(df, "K")
    assert out["nflcom_pred_total"].iloc[0] == pytest.approx(8.5)
    assert not any(c.startswith("nflcom_pred_") and c != "nflcom_pred_total" for c in out.columns)


# ---------- Position-comparison tests ---------------------------------------


def test_compute_position_comparison_perfect_nflcom_zero_mae():
    """When NFL.com's projections == actuals, MAE = 0."""
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=3)
    nflcom = _make_nflcom_for_qb(actuals, perfect=True)

    result = ab._compute_position_comparison(
        "QB", nflcom, actuals, eval_seasons=(2025,), scoring_format="ppr"
    )
    assert result["position"] == "QB"
    assert "2025" in result["per_season"]
    pooled = result["pooled"]
    assert pooled["metrics"]["mae"] == pytest.approx(0.0, abs=1e-9)
    # Per-target breakout populated for QB.
    assert set(pooled["per_target"].keys()) == {
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
    }


def test_compute_position_comparison_noisy_nflcom_positive_mae():
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=4)
    nflcom = _make_nflcom_for_qb(actuals, perfect=False)
    result = ab._compute_position_comparison(
        "QB", nflcom, actuals, eval_seasons=(2025,), scoring_format="ppr"
    )
    assert result["pooled"]["metrics"]["mae"] > 0


def test_compute_position_comparison_skipped_when_no_overlap():
    """Actuals in 2025, NFL.com in 2024 → per-season block reports skipped."""
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=2)
    nflcom = _make_nflcom_for_qb(actuals, perfect=True)
    nflcom["season"] = 2024

    result = ab._compute_position_comparison(
        "QB", nflcom, actuals, eval_seasons=(2025,), scoring_format="ppr"
    )
    block = result["per_season"]["2025"]
    assert block.get("skipped") is True
    assert result["pooled"]["skipped"] is True


def test_compute_position_comparison_multi_season_pools_correctly():
    """Two seasons: pooled n_matched = sum(per_season n_matched), pooled MAE matches
    a freshly-computed MAE on the concatenated error vectors (NOT the average)."""
    actuals_24 = _make_qb_actuals(seasons=(2024,), n_weeks=3)
    actuals_25 = _make_qb_actuals(seasons=(2025,), n_weeks=2)
    actuals = pd.concat([actuals_24, actuals_25], ignore_index=True)
    nflcom = _make_nflcom_for_qb(actuals, perfect=False)

    result = ab._compute_position_comparison(
        "QB", nflcom, actuals, eval_seasons=(2024, 2025), scoring_format="ppr"
    )

    assert "2024" in result["per_season"]
    assert "2025" in result["per_season"]
    per_season_n = sum(result["per_season"][str(s)]["n_matched"] for s in (2024, 2025))
    assert result["pooled"]["n_matched"] == per_season_n

    # Pooled MAE != arithmetic mean of per-season MAEs when sample counts
    # differ, so verify pooled matches a from-scratch recomputation.
    nflcom_pred = ab._project_nflcom_to_ppr(nflcom, "QB", "ppr")
    actuals_q = ab._actuals_for_position(actuals, "QB", [2024, 2025])
    actuals_q["actual_pts"] = ab._aggregate_actuals_to_ppr(actuals_q, "QB", "ppr")
    joined = actuals_q.merge(nflcom_pred, on=["player_id", "season", "week"], how="inner")
    expected_mae = float(np.mean(np.abs(joined["actual_pts"] - joined["nflcom_pred_total"])))
    assert result["pooled"]["metrics"]["mae"] == pytest.approx(expected_mae)


def test_aggregate_actuals_to_ppr_qb_matches_scoring_dict():
    """For QB, _aggregate_actuals_to_ppr should reproduce the same PPR fp the fixture
    computed manually."""
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=1)
    actuals_q = ab._actuals_for_position(actuals, "QB", [2025])
    fp_ours = ab._aggregate_actuals_to_ppr(actuals_q, "QB", "ppr")
    np.testing.assert_array_almost_equal(fp_ours, actuals_q["fantasy_points"].to_numpy())


def test_aggregate_actuals_to_ppr_k_computes_from_raw_stats():
    """K actuals: fg_made_distance * 0.1 + pat_made - fg_missed - pat_missed.
    Matches src.k.targets.compute_targets's formula."""
    df = pd.DataFrame(
        [
            # 3 FGs totaling 100 yards (10 pts), 2 PAT (2 pts), 0 missed = 12.0
            {
                "fg_made_distance": 100.0,
                "pat_made": 2.0,
                "fg_missed": 0.0,
                "pat_missed": 0.0,
            },
            # 1 FG 50 yards (5 pts), 0 PAT, 1 FG miss, 1 XP miss = 5 - 1 - 1 = 3.0
            {
                "fg_made_distance": 50.0,
                "pat_made": 0.0,
                "fg_missed": 1.0,
                "pat_missed": 1.0,
            },
        ]
    )
    out = ab._aggregate_actuals_to_ppr(df, "K", "ppr")
    np.testing.assert_array_almost_equal(out, [12.0, 3.0])


# ---------- main() smoke test -----------------------------------------------


def test_main_writes_json_and_handles_dst_skip(tmp_path):
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=3)
    nflcom = _make_nflcom_for_qb(actuals, perfect=False)

    def fake_nflcom_loader(seasons, force_refresh=False):
        return nflcom

    def fake_actuals_loader(seasons):
        return actuals

    output_dir = tmp_path / "analysis_output"
    result = ab.main(
        eval_seasons=(2025,),
        scoring_format="ppr",
        positions=("QB", "DST"),
        output_dir=str(output_dir),
        nflcom_loader=fake_nflcom_loader,
        actuals_loader=fake_actuals_loader,
    )
    assert "QB" in result["positions"]
    assert "DST" in result["positions"]
    assert result["positions"]["DST"]["skipped"] is True
    assert result["positions"]["QB"]["pooled"]["metrics"]["mae"] > 0

    json_path = output_dir / "nflcom_baseline.json"
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk["eval_seasons"] == [2025]
    assert on_disk["scoring"] == "ppr"


def test_main_passes_force_refresh_through(tmp_path):
    actuals = _make_qb_actuals(seasons=(2025,), n_weeks=2)
    nflcom = _make_nflcom_for_qb(actuals, perfect=False)
    seen = {}

    def fake_nflcom_loader(seasons, force_refresh=False):
        seen["force_refresh"] = force_refresh
        return nflcom

    def fake_actuals_loader(seasons):
        return actuals

    ab.main(
        eval_seasons=(2025,),
        positions=("QB",),
        output_dir=str(tmp_path),
        nflcom_loader=fake_nflcom_loader,
        actuals_loader=fake_actuals_loader,
        force_refresh_nflcom=True,
    )
    assert seen["force_refresh"] is True


def test_main_multi_season_run_writes_both_seasons(tmp_path):
    """main() invoked with two seasons should populate per_season for both."""
    a24 = _make_qb_actuals(seasons=(2024,), n_weeks=2)
    a25 = _make_qb_actuals(seasons=(2025,), n_weeks=2)
    actuals = pd.concat([a24, a25], ignore_index=True)
    nflcom = _make_nflcom_for_qb(actuals, perfect=False)

    result = ab.main(
        eval_seasons=(2024, 2025),
        positions=("QB",),
        output_dir=str(tmp_path),
        nflcom_loader=lambda seasons, force_refresh=False: nflcom,
        actuals_loader=lambda seasons: actuals,
    )
    qb_per_season = result["positions"]["QB"]["per_season"]
    assert set(qb_per_season.keys()) == {"2024", "2025"}
    assert qb_per_season["2024"]["n_matched"] > 0
    assert qb_per_season["2025"]["n_matched"] > 0

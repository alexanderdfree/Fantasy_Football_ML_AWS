"""Unit pins for the B2 PROE/pace A/B spec (src/tuning/ab_proe_pace.py).

No training: exercises spec resolution, the team-week aggregation math, the
W-1 leakage lag, injector row/order preservation, whitelist mutator branch
placement, and the rank-metric fn — the conventions of
test_ab_boom_signals_wr.py / test_ab_games_gap.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.tuning.ab_harness as H
import src.tuning.ab_proe_pace as spec

pytestmark = pytest.mark.unit


# ---------- spec resolution -------------------------------------------------


def test_spec_resolves_as_dotted():
    resolved = H.resolve_spec("src.tuning.ab_proe_pace")
    assert resolved.positions == ["QB", "RB", "WR", "TE"]
    assert resolved.baseline == "baseline"
    assert set(resolved.variants) == {"baseline", "proe_pace"}
    assert resolved.variants["baseline"].is_baseline_shape
    arm = resolved.variants["proe_pace"]
    assert arm.frame_injector is not None
    assert arm.cfg_mutator is not None
    assert arm.expect_ridge_identical is False


def test_variant_names_are_fleet_only_safe():
    import re

    for name in ("baseline", "proe_pace"):
        assert re.fullmatch(r"[A-Za-z0-9_]+", name)


def test_module_level_imports_are_launcher_safe():
    """launch_ab imports this spec on the lightweight runner (no training deps)
    to size the grid, so no module-level import may pull src.data / nflreadpy —
    those must be deferred into _build_team_week_table (runs only in-container).
    Regression guard for the #1479 smoke ModuleNotFoundError: nflreadpy."""
    import ast
    import pathlib

    src = pathlib.Path(spec.__file__).read_text()
    banned = ("src.data", "nflreadpy")
    for node in ast.parse(src).body:  # module body only — deferred imports are nested
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert not any(name == b or name.startswith(b + ".") for b in banned), (
                f"module-level import {name!r} is launcher-hostile; defer it into a function"
            )


# ---------- team-game aggregation -------------------------------------------


def _pbp_fixture() -> pd.DataFrame:
    """One team, one game: 4 plays, 2 neutral, 1 kneel; hand-checkable."""
    return pd.DataFrame(
        {
            "game_id": ["g1"] * 5,
            "season": [2024] * 5,
            "week": [1] * 5,
            "posteam": ["KC"] * 5,
            "drive": [1, 1, 1, 2, 2],
            "pass": [1, 0, 1, 1, 0],
            "pass_oe": [10.0, np.nan, -4.0, 6.0, np.nan],
            "wp": [0.5, 0.5, 0.95, 0.6, 0.5],
            "game_seconds_remaining": [3600.0, 3570.0, 3540.0, 1800.0, 1770.0],
            "qb_kneel": [0, 0, 0, 0, 1],
            "qb_spike": [0, 0, 0, 0, 0],
        }
    )


def test_aggregate_team_games_hand_computed():
    out = spec._aggregate_team_games(_pbp_fixture())
    assert len(out) == 1
    row = out.iloc[0]
    # Kneel dropped -> 4 plays; PROE over non-NaN pass_oe of kept plays.
    assert row["game_plays"] == 4
    assert row["game_proe"] == pytest.approx((10.0 - 4.0 + 6.0) / 3)
    # Neutral (0.2<=wp<=0.8) kept plays: rows 0,1 (drive 1) + row 3 (drive 2).
    assert row["game_neutral_pass_rate"] == pytest.approx(2 / 3)
    # Only one within-drive neutral snap gap: 3600->3570 = 30s (drive 2 has a
    # single neutral play; the wp=0.95 play breaks drive 1's chain anyway).
    assert row["game_neutral_sec_per_play"] == pytest.approx(30.0)


def test_neutral_gap_cap_excludes_breaks():
    pbp = _pbp_fixture()
    pbp.loc[1, "game_seconds_remaining"] = 3600.0 - 100.0  # 100s > cap
    out = spec._aggregate_team_games(pbp)
    assert np.isnan(out.iloc[0]["game_neutral_sec_per_play"])


# ---------- season-to-date W-1 lag -------------------------------------------


def test_season_to_date_shift_is_leakage_safe():
    games = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2024, 2024, 2024, 2025],
            "week": [1, 2, 4, 1],  # week 3 bye: shift is by game, not week
            "posteam": ["KC"] * 4,
            "game_proe": [10.0, 20.0, 30.0, 99.0],
            "game_plays": [60, 70, 80, 90],
            "game_neutral_pass_rate": [0.5, 0.6, 0.7, 0.9],
            "game_neutral_sec_per_play": [28.0, 30.0, 32.0, 40.0],
        }
    )
    tbl = spec._season_to_date_shift(games)
    kc24 = tbl[(tbl["recent_team"] == "KC") & (tbl["season"] == 2024)].set_index("week")
    assert np.isnan(kc24.loc[1, "team_proe_std"])  # opener sees nothing
    assert kc24.loc[2, "team_proe_std"] == pytest.approx(10.0)
    assert kc24.loc[4, "team_proe_std"] == pytest.approx(15.0)  # mean of wk1+wk2
    # New season resets: 2025 opener NaN despite 2024 history.
    kc25 = tbl[(tbl["season"] == 2025)].set_index("week")
    assert np.isnan(kc25.loc[1, "team_proe_std"])
    assert "recent_team" in tbl.columns and "posteam" not in tbl.columns


# ---------- injector ----------------------------------------------------------


def test_inject_preserves_rows_and_leaves_missing_nan(monkeypatch):
    tbl = pd.DataFrame(
        {
            "recent_team": ["KC", "KC"],
            "season": [2024, 2024],
            "week": [2, 3],
            "team_proe_std": [10.0, 15.0],
            "team_neutral_pass_rate_std": [0.5, 0.55],
            "team_neutral_sec_per_play_std": [29.0, 29.5],
            "team_plays_pg_std": [65.0, 66.0],
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
    out_train, out_val, out_test = spec._inject_proe(train, val, test)

    for orig, out in ((train, out_train), (val, out_val), (test, out_test)):
        assert len(out) == len(orig)
        assert list(out["player_id"]) == list(orig["player_id"])  # order preserved
        assert set(spec.FEATURE_COLS) <= set(out.columns)
    assert np.isnan(out_train.loc[0, "team_proe_std"])  # opener: no table row
    assert out_train.loc[1, "team_proe_std"] == pytest.approx(10.0)
    assert out_val.loc[0, "team_proe_std"] == pytest.approx(15.0)
    assert np.isnan(out_test.loc[0, "team_proe_std"])  # unmatched week stays NaN


def test_inject_raises_on_duplicate_table_key(monkeypatch):
    dup = pd.DataFrame(
        {
            "recent_team": ["KC", "KC"],
            "season": [2024, 2024],
            "week": [2, 2],
            "team_proe_std": [1.0, 2.0],
            "team_neutral_pass_rate_std": [0.5, 0.5],
            "team_neutral_sec_per_play_std": [29.0, 29.0],
            "team_plays_pg_std": [65.0, 65.0],
        }
    )
    monkeypatch.setattr(spec, "_build_team_week_table", lambda seasons: dup)
    f = pd.DataFrame({"player_id": ["p"], "recent_team": ["KC"], "season": [2024], "week": [2]})
    with pytest.raises(pd.errors.MergeError):  # merge validate="m:1" rejects the dup key
        spec._inject_proe(f.copy(), f.copy(), f.copy())


# ---------- whitelist mutator -------------------------------------------------


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
    # Season-to-date rates only — the static branch must never see windowed cols.
    for col in spec.FEATURE_COLS:
        assert not any(tok in col for tok in ("_L3", "_L5", "_L8", "rolling", "ewma", "trend"))


def test_mutator_without_attn_key_is_safe():
    cfg = {"get_feature_columns_fn": lambda: ["x"]}
    out = spec._mut_whitelist(cfg)
    assert set(spec.FEATURE_COLS) <= set(out["get_feature_columns_fn"]())
    assert "attn_static_features" not in out


def test_mutator_static_append_is_idempotent():
    cfg = _fake_cfg()
    cfg["attn_static_features"] = ["existing_a", *spec.FEATURE_COLS]
    out = spec._mut_whitelist(cfg)
    assert len(out["attn_static_features"]) == len(set(out["attn_static_features"]))


# ---------- metric_fn ----------------------------------------------------------


def _synthetic_result(with_elite_col=True):
    rng = np.random.default_rng(0)
    n_players, weeks = 14, [1, 2]
    rows = []
    for w in weeks:
        for i in range(n_players):
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
    assert out["Ridge"]["mae"] == pytest.approx(1.0, abs=1e-6)  # sentinel feed
    for model_row in out.values():
        assert {"mae", "hit12", "spearman", "regret12", "regret_lineup"} <= set(model_row)
    # Perfectly rank-preserving offsets -> perfect ordering metrics, zero regret.
    assert out["Ridge"]["hit12"] == pytest.approx(1.0)
    assert out["Ridge"]["spearman"] == pytest.approx(1.0, abs=1e-3)
    assert out["Ridge"]["regret12"] == pytest.approx(0.0, abs=1e-9)


def test_metric_fn_elite_slice_guards_on_column():
    with_elite = spec.metric_fn(_synthetic_result(True), "RB")
    assert "elite24_mae" in with_elite["Ridge"] and "elite24_bias" in with_elite["Ridge"]
    assert with_elite["Ridge"]["elite24_bias"] == pytest.approx(1.0, abs=1e-6)
    without = spec.metric_fn(_synthetic_result(False), "RB")
    assert "elite24_mae" not in without["Ridge"]


def test_regret_lineup_uses_position_size():
    result = _synthetic_result()
    out_rb = spec.metric_fn(result, "RB")  # lineup 24 > 14 players/week -> NaN
    assert np.isnan(out_rb["Ridge"]["regret_lineup"])
    out_qb = spec.metric_fn(result, "QB")  # lineup 12 <= 14 -> computed
    assert out_qb["Ridge"]["regret_lineup"] == pytest.approx(0.0, abs=1e-9)

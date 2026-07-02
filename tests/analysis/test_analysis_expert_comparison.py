"""Unit + smoke tests for src/analysis/analysis_expert_comparison.py.

The default loaders train the pipeline / hit the network — out of scope for unit
tests. These inject stub model + NFL.com + Sleeper loaders so the per-expert join,
same-sample intersection, significance wiring, position skips, and nested JSON
output are all exercised on synthetic frames. Also an import smoke so a signature
break fails the unit shard rather than only at PR-review time.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis import analysis_expert_comparison as mod

pytestmark = pytest.mark.unit

# QB target columns the offense projectors aggregate (POSITION_TARGET_MAP["QB"]).
_QB_TARGETS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
)


def _qb_stats(i: int, wk: int) -> dict:
    """Varied raw QB stats so aggregated totals differ row-to-row (keeps DM /
    Wilcoxon non-degenerate)."""
    return {
        "passing_yards": 220.0 + 5 * i + wk,
        "rushing_yards": 10.0 + i,
        "passing_tds": 1.0 + (i % 3),
        "rushing_tds": float(i % 2),
        "interceptions": float((i + wk) % 2),
        "fumbles_lost": 0.0,
    }


def _model_df_qb() -> pd.DataFrame:
    """Model held-out test_df: players P1..P6 over weeks 1-2, season 2025."""
    rows = []
    for wk in (1, 2):
        for i in range(1, 7):
            actual = 10.0 + i + wk
            rows.append(
                {
                    "player_id": f"P{i}",
                    "season": 2025,
                    "week": wk,
                    "fantasy_points": actual,
                    "pred_attn_nn_total": actual + (0.5 if i % 2 else -0.7),
                }
            )
    return pd.DataFrame(rows)


def _nflcom_qb() -> pd.DataFrame:
    """Raw NFL.com frame for players P4..P9 over weeks 1-2 (overlap w/ model = P4,P5,P6)."""
    rows = []
    for wk in (1, 2):
        for i in range(4, 10):
            row = {
                "position": "QB",
                "player_id": f"P{i}",
                "season": 2025,
                "week": wk,
                "nflcom_projected_pts": 12.0 + i,
            }
            row.update(_qb_stats(i, wk))
            rows.append(row)
    return pd.DataFrame(rows)


def _sleeper_qb() -> pd.DataFrame:
    """Gsis-joined Sleeper frame for players P3..P8 over weeks 1-2 (overlap w/ model
    = P3,P4,P5,P6). Shape mirrors ``load_sleeper_with_gsis_id`` output: player_id is
    the gsis-bridged id, plus position + target columns."""
    rows = []
    for wk in (1, 2):
        for i in range(3, 9):
            row = {"position": "QB", "player_id": f"P{i}", "season": 2025, "week": wk}
            row.update(_qb_stats(i + 1, wk))  # offset so Sleeper ≠ NFL.com projection
            rows.append(row)
    return pd.DataFrame(rows)


_DST_TARGETS = (
    "def_sacks",
    "def_ints",
    "def_fumble_rec",
    "def_fumbles_forced",
    "def_safeties",
    "def_tds",
    "def_blocked_kicks",
    "special_teams_tds",
    "points_allowed",
    "yards_allowed",
)


def _model_df_dst() -> pd.DataFrame:
    """Model DST test_df: teams T1..T6 over weeks 1-2 (player_id = team abbrev)."""
    rows = []
    for wk in (1, 2):
        for i in range(1, 7):
            actual = 6.0 + i + wk
            rows.append(
                {
                    "player_id": f"T{i}",
                    "season": 2025,
                    "week": wk,
                    "fantasy_points": actual,
                    "pred_attn_nn_total": actual + (0.5 if i % 2 else -0.7),
                }
            )
    return pd.DataFrame(rows)


def _sleeper_dst() -> pd.DataFrame:
    """Sleeper DST frame: teams T3..T8 (overlap w/ model = T3,T4,T5,T6). Team-keyed
    player_id + the 10 DST target columns (varied so totals differ row-to-row)."""
    rows = []
    for wk in (1, 2):
        for i in range(3, 9):
            row = {"position": "DST", "player_id": f"T{i}", "season": 2025, "week": wk}
            for j, t in enumerate(_DST_TARGETS):
                row[t] = float((i + j + wk) % 3)
            row["points_allowed"] = 14.0 + i
            row["yards_allowed"] = 300.0 + 10.0 * i
            rows.append(row)
    return pd.DataFrame(rows)


def _stub_model_loader(pos, eval_seasons, scoring_format):
    return _model_df_dst() if pos == "DST" else _model_df_qb()


def _stub_nflcom_loader(seasons):
    return _nflcom_qb()


def _stub_sleeper_loader(seasons):
    # Both offense (QB) and DST rows, mirroring the real loader's combined frame.
    return pd.concat([_sleeper_qb(), _sleeper_dst()], ignore_index=True)


def _stub_fftoday_loader(seasons):
    # FFToday is offense-only (no K/DST); reuse the offense QB frame.
    return _sleeper_qb()


def _run(tmp_path, **kw):
    """main() with all loaders stubbed (no network/training)."""
    defaults = dict(
        eval_seasons=(2025,),
        output_dir=str(tmp_path),
        n_boot=50,
        seed=0,
        model_preds_loader=_stub_model_loader,
        nflcom_loader=_stub_nflcom_loader,
        sleeper_loader=_stub_sleeper_loader,
        fftoday_loader=_stub_fftoday_loader,
    )
    defaults.update(kw)
    return mod.main(**defaults)


def test_module_imports_cleanly() -> None:
    for attr in (
        "main",
        "_compare_one_position",
        "_default_model_preds",
        "_build_experts",
        "_project_sleeper_to_ppr",
        "ExpertSource",
    ):
        assert hasattr(mod, attr)
    assert mod._MODEL_PRED_COLS[0] == "pred_attn_nn_total"


def test_experts_each_same_sample_and_significance(tmp_path) -> None:
    result = _run(tmp_path, positions=("QB", "DST"))

    # Nested per-expert output (NFL.com + Sleeper + FFToday).
    assert set(result["experts"]) == {"nflcom", "sleeper", "fftoday"}
    nflcom_qb = result["experts"]["nflcom"]["positions"]["QB"]
    sleeper_qb = result["experts"]["sleeper"]["positions"]["QB"]
    # FFToday is offense-only: QB scored (same offense frame as Sleeper), DST skipped.
    assert result["experts"]["fftoday"]["positions"]["QB"]["n_matched"] == 8
    assert result["experts"]["fftoday"]["positions"]["DST"]["skipped"] is True

    # Per-expert same-sample intersection differs by source coverage:
    #   NFL.com P4..P9 ∩ model P1..P6 = {P4,P5,P6} × 2 weeks = 6
    #   Sleeper P3..P8 ∩ model P1..P6 = {P3,P4,P5,P6} × 2 weeks = 8
    assert nflcom_qb["n_matched"] == 6
    assert sleeper_qb["n_matched"] == 8

    for block in (nflcom_qb, sleeper_qb):
        assert block["model_col"] == "pred_attn_nn_total"
        for side in ("model", "expert"):
            assert set(block[side]) >= {"mae", "rmse", "r2", "top_k_hit_rate", "spearman"}
        assert set(block["delta_mae"]) == {"value", "ci_lo", "ci_hi"}
        assert set(block["dm_mae"]) >= {"dm_stat", "p_value", "favored"}
        assert "bootstrap_rmse" in block and "wilcoxon_abs_error" in block
        # delta = model - expert; sign agrees with the raw MAE difference.
        assert (block["delta_mae"]["value"] < 0) == (block["model"]["mae"] < block["expert"]["mae"])


def test_sleeper_covers_dst_nflcom_skips(tmp_path) -> None:
    result = _run(tmp_path, positions=("DST",), n_boot=10)
    # NFL.com has no DST projections -> skipped.
    assert result["experts"]["nflcom"]["positions"]["DST"]["skipped"] is True
    # Sleeper now covers DST (team-keyed): model teams T1..T6 ∩ Sleeper T3..T8 =
    # T3..T6 × 2 weeks = 8 matched.
    sl_dst = result["experts"]["sleeper"]["positions"]["DST"]
    assert not sl_dst.get("skipped")
    assert sl_dst["n_matched"] == 8
    assert set(sl_dst["model"]) >= {"mae", "rmse"}


def test_sleeper_skips_kicker(tmp_path) -> None:
    result = _run(tmp_path, positions=("K",), n_boot=10)
    # NFL.com covers K (totals-only); Sleeper covers offense + DST but not K.
    assert result["experts"]["sleeper"]["positions"]["K"]["skipped"] is True


def test_writes_parseable_nested_json(tmp_path) -> None:
    _run(tmp_path, positions=("QB",), n_boot=20)
    out = tmp_path / "expert_comparison.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert set(payload["experts"]) == {"nflcom", "sleeper", "fftoday"}
    assert payload["experts"]["nflcom"]["positions"]["QB"]["n_matched"] == 6
    assert payload["experts"]["sleeper"]["positions"]["QB"]["n_matched"] == 8
    # The Sleeper provenance caveat rides along in the output.
    assert "provenance" in payload["experts"]["sleeper"]["note"].lower()


def test_missing_model_column_is_skipped(tmp_path) -> None:
    def bad_loader(pos, eval_seasons, scoring_format):
        return pd.DataFrame(
            {"player_id": ["P4"], "season": [2025], "week": [1], "fantasy_points": [10.0]}
        )

    # No pred_* column ⇒ each expert's QB block short-circuits to a skip.
    result = _run(tmp_path, positions=("QB",), n_boot=10, model_preds_loader=bad_loader)
    assert result["experts"]["nflcom"]["positions"]["QB"]["skipped"] is True
    assert result["experts"]["sleeper"]["positions"]["QB"]["skipped"] is True


def test_failed_expert_load_is_skipped_not_fatal(tmp_path) -> None:
    """A loader that raises (e.g. Sleeper network/RuntimeError) skips that expert
    but the other expert still produces a comparison."""

    def boom(seasons):
        raise RuntimeError("sleeper unavailable")

    result = _run(tmp_path, positions=("QB",), n_boot=10, sleeper_loader=boom)
    assert result["experts"]["nflcom"]["positions"]["QB"]["n_matched"] == 6
    assert result["experts"]["sleeper"]["positions"]["QB"]["skipped"] is True


def test_no_scoring_warning_with_injected_loader(tmp_path, capsys) -> None:
    """Injected loaders control their own scoring, so the non-PPR mismatch warning
    must NOT fire for them (it is scoped to the default model loader)."""
    _run(tmp_path, scoring_format="half_ppr", positions=("QB",), n_boot=10)
    assert "WARNING: --scoring-format" not in capsys.readouterr().out

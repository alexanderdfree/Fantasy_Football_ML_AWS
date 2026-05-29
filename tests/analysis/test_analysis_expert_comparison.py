"""Unit + smoke tests for src/analysis/analysis_expert_comparison.py.

The default loaders train the pipeline / hit the network — out of scope for unit
tests. These inject stub model + NFL.com loaders so the join, same-sample
intersection, significance wiring, DST skip, and JSON output are all exercised on
synthetic frames. Also an import smoke so a signature break fails the unit shard
rather than only at PR-review time.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis import analysis_expert_comparison as mod

pytestmark = pytest.mark.unit

# QB target columns NFL.com's projector aggregates (POSITION_TARGET_MAP["QB"]).
_QB_TARGETS = (
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
)


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
                    # Model is close to actual but not exact, with per-row variation.
                    "pred_attn_nn_total": actual + (0.5 if i % 2 else -0.7),
                }
            )
    return pd.DataFrame(rows)


def _nflcom_qb() -> pd.DataFrame:
    """Raw NFL.com frame for players P4..P9 over weeks 1-2 (overlap = P4,P5,P6)."""
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
            # Varied raw stats so the aggregated total differs from actual and
            # row-to-row (keeps DM / Wilcoxon non-degenerate).
            row.update(
                {
                    "passing_yards": 220.0 + 5 * i + wk,
                    "rushing_yards": 10.0 + i,
                    "passing_tds": 1.0 + (i % 3),
                    "rushing_tds": float(i % 2),
                    "interceptions": float((i + wk) % 2),
                    "fumbles_lost": 0.0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _stub_model_loader(pos, eval_seasons, scoring_format):
    return _model_df_qb()


def _stub_nflcom_loader(seasons):
    return _nflcom_qb()


def test_module_imports_cleanly() -> None:
    assert hasattr(mod, "main")
    assert hasattr(mod, "_compare_one_position")
    assert hasattr(mod, "_default_model_preds")
    assert mod._MODEL_PRED_COLS[0] == "pred_attn_nn_total"


def test_head_to_head_same_sample_and_significance(tmp_path) -> None:
    result = mod.main(
        eval_seasons=(2025,),
        positions=("QB", "DST"),
        output_dir=str(tmp_path),
        n_boot=50,
        seed=0,
        model_preds_loader=_stub_model_loader,
        nflcom_loader=_stub_nflcom_loader,
    )
    qb = result["positions"]["QB"]

    # Same-sample: only P4,P5,P6 overlap, across 2 weeks ⇒ 6 matched rows
    # (P1..P3 are model-only, P7..P9 expert-only and must be excluded).
    assert qb["n_matched"] == 6
    assert qb["model_col"] == "pred_attn_nn_total"

    # Both sides scored; head-to-head + significance blocks present.
    for side in ("model", "expert"):
        assert set(qb[side]) >= {"mae", "rmse", "r2", "top_k_hit_rate", "spearman"}
    assert set(qb["delta_mae"]) == {"value", "ci_lo", "ci_hi"}
    assert set(qb["dm_mae"]) >= {"dm_stat", "p_value", "favored"}
    assert "bootstrap_rmse" in qb and "wilcoxon_abs_error" in qb

    # delta = model - expert; sign must agree between the bootstrap point estimate
    # and the raw MAE difference.
    assert (qb["delta_mae"]["value"] < 0) == (qb["model"]["mae"] < qb["expert"]["mae"])


def test_dst_is_skipped(tmp_path) -> None:
    result = mod.main(
        eval_seasons=(2025,),
        positions=("DST",),
        output_dir=str(tmp_path),
        n_boot=10,
        model_preds_loader=_stub_model_loader,
        nflcom_loader=_stub_nflcom_loader,
    )
    assert result["positions"]["DST"]["skipped"] is True


def test_writes_parseable_json(tmp_path) -> None:
    mod.main(
        eval_seasons=(2025,),
        positions=("QB",),
        output_dir=str(tmp_path),
        n_boot=20,
        model_preds_loader=_stub_model_loader,
        nflcom_loader=_stub_nflcom_loader,
    )
    out = tmp_path / "expert_comparison.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["expert_source"] == "nflcom"
    assert payload["positions"]["QB"]["n_matched"] == 6


def test_missing_model_column_is_skipped(tmp_path) -> None:
    def bad_loader(pos, eval_seasons, scoring_format):
        return pd.DataFrame(
            {"player_id": ["P4"], "season": [2025], "week": [1], "fantasy_points": [10.0]}
        )

    # No pred_* column ⇒ _compare_one_position's model-column check short-circuits
    # to a skip (rather than raising).
    result = mod.main(
        eval_seasons=(2025,),
        positions=("QB",),
        output_dir=str(tmp_path),
        n_boot=10,
        model_preds_loader=bad_loader,
        nflcom_loader=_stub_nflcom_loader,
    )
    assert result["positions"]["QB"]["skipped"] is True


def test_no_scoring_warning_with_injected_loader(tmp_path, capsys) -> None:
    """Injected loaders control their own scoring, so the non-PPR model/expert
    mismatch warning must NOT fire for them (it is scoped to the default loader)."""
    mod.main(
        eval_seasons=(2025,),
        scoring_format="half_ppr",
        positions=("QB",),
        output_dir=str(tmp_path),
        n_boot=10,
        model_preds_loader=_stub_model_loader,
        nflcom_loader=_stub_nflcom_loader,
    )
    assert "WARNING: --scoring-format" not in capsys.readouterr().out

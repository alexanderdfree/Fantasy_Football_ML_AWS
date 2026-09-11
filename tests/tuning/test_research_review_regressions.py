"""Research integration: explicit inputs, fair populations, and eager execution."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.context import RunContext, use_context
from src.tuning import ab_harness as harness
from src.tuning import ab_oline_confirm as confirm
from src.tuning import ab_qb_context_receivers as receivers

pytestmark = pytest.mark.unit


def sample():
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(14)],
            "season": 2025,
            "week": 1,
            "season_type": "REG",
            "position": "TE",
            "fantasy_points": np.arange(14, dtype=float),
            "pred_ridge_total": np.arange(14, dtype=float),
            "pred_attn_nn_total": np.arange(14, 0, -1, dtype=float),
            "pred_lgbm_total": np.arange(14, 0, -1, dtype=float),
        }
    )


def test_confirm_uses_supplied_splits(monkeypatch):
    from src.benchmarking import benchmark

    def forbidden():
        pytest.fail("must not reload ambient splits")

    monkeypatch.setattr(benchmark, "_load_full_featured_frame", forbidden)
    frame = sample()
    output = confirm._make_injector(2025, False)(
        frame.assign(season=2023), frame.assign(season=2024), frame
    )
    assert [set(part.season) for part in output] == [{2023}, {2024}, {2025}]
    assert output[2].player_id.tolist() == frame.player_id.tolist()


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_confirm_has_one_finite_population_and_uses_context_root(tmp_path, monkeypatch, missing):
    frame = sample()
    slate = frame[["player_id", "season", "week", "position"]].assign(
        rotowire_pred=frame.pred_lgbm_total
    )
    frame.loc[13, "pred_attn_nn_total"] = missing
    context = RunContext(tmp_path / "out", tmp_path / "data")
    seen = []

    def read(path):
        seen.append(Path(path))
        return slate

    monkeypatch.setattr(pd, "read_parquet", read)
    with use_context(context):
        result = confirm.metric_fn({"test_df": frame}, "TE")
    assert seen == [context.raw_root / confirm._ROTOWIRE_SLATE]
    for source in ("attn_nn", "lgbm", "rotowire"):
        assert result[source]["slate_n"] == 13
        assert result[source]["regret"] == 12


def test_confirm_rejects_duplicate_archive_keys(monkeypatch):
    frame = sample()
    slate = frame[["player_id", "season", "week", "position"]].assign(rotowire_pred=0.0)
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.concat([slate, slate.iloc[:1]]))
    with pytest.raises(pd.errors.MergeError):
        confirm.metric_fn({"test_df": frame}, "TE")


def test_confirm_reports_empty_intersection_for_every_source(monkeypatch):
    frame = sample()
    slate = frame[["player_id", "season", "week", "position"]].assign(rotowire_pred=np.nan)
    monkeypatch.setattr(pd, "read_parquet", lambda _: slate)
    output = confirm.metric_fn({"test_df": frame}, "TE")
    for source in ("attn_nn", "lgbm", "rotowire"):
        assert output[source]["slate_n"] == output[source]["slate_available"] == 0
        assert np.isnan(output[source]["regret"])


@pytest.mark.parametrize("team,canonical", [("OAK", "LV"), ("STL", "LA"), ("SD", "LAC")])
def test_qb_absence_joins_normalized_team_keys(monkeypatch, team, canonical):
    rows = pd.DataFrame(
        {
            "player_id": ["starter", "backup", "receiver"],
            "position": ["QB", "QB", "WR"],
            "season": 2016,
            "week": [1, 2, 2],
            "recent_team": canonical,
            "total_fantasy_points_exp": [20.0, 1.0, 0.0],
        }
    )
    injuries = pd.DataFrame(
        {
            "position": ["QB"],
            "season": [2016],
            "team": [team],
            "week": [2],
            "gsis_id": ["starter"],
            "report_status": ["Out"],
        }
    )
    monkeypatch.setattr("src.data.nfl_source.injuries", lambda _: injuries)
    result, _, _ = receivers._inject_qb_context(rows, rows.iloc[:0], rows.iloc[:0])
    row = result.loc[result.player_id.eq("receiver")].iloc[0]
    assert row.team_qb_out == 1 and row.team_qb_vacated_role == 20


@pytest.mark.parametrize(
    "name", ["games_gap", "oline_confirm", "oline_continuity", "proe_pace", "qb_context_receivers"]
)
def test_research_explicit_stacking_is_rejected_before_compute(name):
    spec = harness.resolve_spec(f"src.tuning.ab_{name}")
    assert not spec.supports_stacked
    with pytest.raises(ValueError, match="does not support stacked"):
        harness.build_stacked_units(spec)
    with pytest.raises(ValueError, match="does not support stacked"):
        harness.run_ab(f"src.tuning.ab_{name}", stacked_seeds=True)


def test_cuda_auto_selection_routes_research_to_eager(monkeypatch, capsys):
    monkeypatch.setattr("src.shared.utils.cuda_enabled", lambda: True)
    monkeypatch.setattr(harness, "build_stacked_units", lambda _: pytest.fail("stacked path used"))
    assert harness.main(["--spec", "src.tuning.ab_oline_confirm", "--list"]) == 0
    assert "24 cells" in capsys.readouterr().out


def test_workflow_accepts_historical_plus_variants_and_rejects_shell_metacharacters():
    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/ab-batch.yml").read_text()
    assert "'^[A-Za-z0-9_+]+$'" in workflow
    for variant in receivers.VARIANTS:
        assert re.fullmatch(r"[A-Za-z0-9_+]+", variant.name)
    for value in ("x;y", "$(whoami)", "`id`", "a b", "x|y"):
        assert re.fullmatch(r"[A-Za-z0-9_+]+", value) is None

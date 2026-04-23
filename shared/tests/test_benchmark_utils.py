"""Coverage tests for ``shared/benchmark_utils.py``.

The file wires together summary-row construction, comparison-table
rendering, git-hash capture, and history append. These tests exercise
every branch (elasticnet/attention/lgbm/cv variants present or not) on
synthetic pipeline result dicts — no real training required.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from shared.benchmark_utils import (
    _best_model_mae,
    _json_default,
    _per_target,
    append_to_history,
    get_git_hash,
    print_comparison_table,
    print_history_comparison,
    summarize_pipeline_result,
)


# --------------------------------------------------------------------------
# get_git_hash
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_get_git_hash_returns_string_in_repo():
    """In a git repo, ``get_git_hash`` returns a non-empty short-hash string."""
    h = get_git_hash()
    assert isinstance(h, str)
    # Either a real 7-char hex hash or the fallback
    assert h == "unknown" or (len(h) >= 6 and all(c in "0123456789abcdef" for c in h))


@pytest.mark.unit
def test_get_git_hash_returns_unknown_when_git_missing(monkeypatch):
    """If subprocess raises FileNotFoundError (git not installed), returns
    ``unknown``."""

    def _boom(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert get_git_hash() == "unknown"


@pytest.mark.unit
def test_get_git_hash_returns_unknown_on_called_process_error(monkeypatch):
    """``CalledProcessError`` (non-zero exit) also falls through to unknown."""

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(subprocess, "check_output", _fail)
    assert get_git_hash() == "unknown"


# --------------------------------------------------------------------------
# append_to_history
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_append_to_history_creates_new_file(tmp_path, capsys):
    path = tmp_path / "history.json"
    append_to_history(str(path), {"run_id": "r1"})
    assert path.exists()
    assert json.loads(path.read_text()) == [{"run_id": "r1"}]
    assert "appended" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_append_to_history_appends_to_existing(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([{"run_id": "r1"}]))
    append_to_history(str(path), {"run_id": "r2"})
    entries = json.loads(path.read_text())
    assert [e["run_id"] for e in entries] == ["r1", "r2"]


@pytest.mark.unit
def test_append_to_history_quarantines_corrupt_json(tmp_path, capsys):
    """Malformed JSON file → quarantined with .corrupt-{ts} suffix and fresh
    history is started with just the new entry."""
    path = tmp_path / "history.json"
    path.write_text("{not valid json")
    append_to_history(str(path), {"run_id": "rN"})
    out = capsys.readouterr().out
    assert "corrupt" in out
    # Fresh file has just the new entry
    assert json.loads(path.read_text()) == [{"run_id": "rN"}]
    # Quarantine file exists
    quarantined = list(tmp_path.glob("history.json.corrupt-*"))
    assert len(quarantined) == 1


@pytest.mark.unit
def test_append_to_history_serializes_sets_via_json_default(tmp_path):
    """``_json_default`` is wired in; sets become sorted lists in the JSON."""
    path = tmp_path / "history.json"
    append_to_history(str(path), {"positions": {"QB", "RB", "WR"}})
    payload = json.loads(path.read_text())
    assert payload[0]["positions"] == ["QB", "RB", "WR"]


@pytest.mark.unit
def test_json_default_raises_on_unknown_type():
    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())


# --------------------------------------------------------------------------
# _per_target + _best_model_mae
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_per_target_excludes_total_and_rounds():
    metrics = {
        "passing_yards": {"mae": 24.56789, "r2": 0.7123},
        "passing_tds": {"mae": 0.66666, "r2": 0.3456},
        "total": {"mae": 99.9, "r2": 0.1},
    }
    out = _per_target(metrics)
    assert "total" not in out
    assert out["passing_yards"] == {"mae": 24.568, "r2": 0.712}


@pytest.mark.unit
def test_best_model_mae_picks_lowest_mae():
    """Includes every model variant when present; picks the min."""
    s = {
        "ridge_mae": 5.0,
        "nn_mae": 4.5,
        "elasticnet_mae": 4.8,
        "attn_nn_mae": 4.3,
        "lgbm_mae": 4.6,
    }
    best, mae = _best_model_mae(s)
    assert best == "Attn"
    assert mae == 4.3


@pytest.mark.unit
def test_best_model_mae_subset_optional_models():
    """Only ridge + nn present → picks min of those two."""
    best, mae = _best_model_mae({"ridge_mae": 5.0, "nn_mae": 6.0})
    assert best == "Ridge"
    assert mae == 5.0


# --------------------------------------------------------------------------
# summarize_pipeline_result
# --------------------------------------------------------------------------


def _basic_result() -> dict:
    """Minimal pipeline-result dict with only Ridge + NN + rankings."""
    return {
        "ridge_metrics": {
            "total": {"mae": 5.0, "r2": 0.3},
            "yards": {"mae": 20.0, "r2": 0.25},
        },
        "nn_metrics": {
            "total": {"mae": 4.5, "r2": 0.35},
            "yards": {"mae": 18.0, "r2": 0.3},
        },
        "ridge_ranking": {"season_avg_hit_rate": 0.42},
        "nn_ranking": {"season_avg_hit_rate": 0.48},
    }


@pytest.mark.unit
def test_summarize_minimal_ridge_plus_nn():
    r = _basic_result()
    s = summarize_pipeline_result("QB", r)
    assert s["position"] == "QB"
    assert s["ridge_mae"] == 5.0
    assert s["nn_mae"] == 4.5
    assert s["nn_wins_mae"] is True
    assert "yards" in s["nn_per_target"]
    assert s["nn_top12"] == 0.48


@pytest.mark.unit
def test_summarize_includes_elasticnet_when_present():
    r = _basic_result()
    r["elasticnet_metrics"] = {
        "total": {"mae": 4.8, "r2": 0.32},
        "yards": {"mae": 19.0, "r2": 0.28},
    }
    r["elasticnet_ranking"] = {"season_avg_hit_rate": 0.45}
    s = summarize_pipeline_result("RB", r)
    assert s["elasticnet_mae"] == 4.8
    assert "yards" in s["elasticnet_per_target"]
    assert s["elasticnet_top12"] == 0.45


@pytest.mark.unit
def test_summarize_includes_attention_and_lgbm():
    r = _basic_result()
    r["attn_nn_metrics"] = {
        "total": {"mae": 4.3, "r2": 0.4},
        "yards": {"mae": 17.0, "r2": 0.33},
    }
    r["attn_nn_ranking"] = {"season_avg_hit_rate": 0.5}
    r["lgbm_metrics"] = {
        "total": {"mae": 4.6, "r2": 0.34},
        "yards": {"mae": 18.5, "r2": 0.29},
    }
    r["lgbm_ranking"] = {"season_avg_hit_rate": 0.47}
    s = summarize_pipeline_result("WR", r)
    assert s["attn_nn_mae"] == 4.3
    assert s["lgbm_mae"] == 4.6
    assert "yards" in s["attn_nn_per_target"]
    assert "yards" in s["lgbm_per_target"]


@pytest.mark.unit
def test_summarize_includes_cv_block():
    r = _basic_result()
    r["cv_metrics"] = {
        "ridge": {"total": {"mae_mean": 5.1, "mae_std": 0.25}},
        "nn": {"total": {"mae_mean": 4.6, "mae_std": 0.3}},
    }
    r["best_cv_alpha"] = 10.0
    s = summarize_pipeline_result("TE", r)
    assert s["cv_ridge_mae_mean"] == 5.1
    assert s["cv_ridge_mae_std"] == 0.25
    assert s["cv_nn_mae_mean"] == 4.6
    assert s["best_cv_alpha"] == 10.0


@pytest.mark.unit
def test_summarize_passes_through_elapsed_and_phases():
    r = _basic_result()
    r["elapsed_sec"] = 123
    r["phase_seconds"] = {"train": 60, "eval": 10}
    s = summarize_pipeline_result("K", r)
    assert s["elapsed_sec"] == 123
    assert s["phase_seconds"] == {"train": 60, "eval": 10}


# --------------------------------------------------------------------------
# print_comparison_table
# --------------------------------------------------------------------------


def _canned_summary(pos: str, **extras) -> dict:
    base = {
        "position": pos,
        "ridge_mae": 5.0,
        "ridge_r2": 0.3,
        "nn_mae": 4.5,
        "nn_r2": 0.35,
        "nn_wins_mae": True,
        "ridge_top12": 0.42,
        "nn_top12": 0.48,
        "ridge_per_target": {"yards": {"mae": 20.0, "r2": 0.25}},
        "nn_per_target": {"yards": {"mae": 18.0, "r2": 0.3}},
    }
    base.update(extras)
    return base


@pytest.mark.unit
def test_print_comparison_ridge_only(capsys):
    """Basic Ridge+NN table (no elasticnet/attn/lgbm/cv)."""
    s = _canned_summary("QB")
    s["elapsed_sec"] = 42
    print_comparison_table([s], header="Smoke", show_time=True)
    out = capsys.readouterr().out
    assert "Smoke" in out
    assert "QB" in out
    assert "R-squared" in out
    assert "Top-12" in out
    assert "Per-Target MAE" in out
    assert "Per-Target R" in out  # unicode R²


@pytest.mark.unit
def test_print_comparison_with_all_variants(capsys):
    """With elasticnet + attention + lgbm + CV, the extra columns + CV block fire."""
    s = _canned_summary(
        "WR",
        elasticnet_mae=4.8,
        elasticnet_r2=0.33,
        elasticnet_top12=0.45,
        elasticnet_per_target={"yards": {"mae": 19.0, "r2": 0.28}},
        attn_nn_mae=4.3,
        attn_nn_r2=0.4,
        attn_nn_top12=0.5,
        attn_nn_per_target={"yards": {"mae": 17.0, "r2": 0.33}},
        lgbm_mae=4.6,
        lgbm_r2=0.34,
        lgbm_top12=0.47,
        lgbm_per_target={"yards": {"mae": 18.5, "r2": 0.29}},
        cv_ridge_mae_mean=5.1,
        cv_ridge_mae_std=0.25,
        cv_nn_mae_mean=4.6,
        cv_nn_mae_std=0.3,
        best_cv_alpha=10.0,
    )
    print_comparison_table([s], header="AllModels", show_time=False)
    out = capsys.readouterr().out
    assert "ENet" in out
    assert "Attn" in out
    assert "LGBM" in out
    assert "Cross-Validation Metrics" in out


# --------------------------------------------------------------------------
# print_history_comparison
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_history_comparison_missing_file_noop(tmp_path, capsys):
    print_history_comparison(str(tmp_path / "nope.json"), [_canned_summary("QB")])
    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_history_comparison_corrupt_file_prints_warning(tmp_path, capsys):
    path = tmp_path / "history.json"
    path.write_text("{not valid json")
    print_history_comparison(str(path), [_canned_summary("QB")])
    assert "could not read" in capsys.readouterr().out


@pytest.mark.unit
def test_history_comparison_prints_per_position_tables(tmp_path, capsys):
    path = tmp_path / "history.json"
    history = [
        {
            "timestamp": "2026-03-01T12:00:00",
            "git_hash": "abc1234",
            "note": "prior run",
            "results": [{"position": "QB", "ridge_mae": 6.0, "nn_mae": 5.5, "nn_top12": 0.4}],
        },
        {  # new run (excluded from the 'history' rows)
            "timestamp": "2026-04-01T12:00:00",
            "git_hash": "def5678",
            "note": "new",
            "results": [{"position": "QB", "ridge_mae": 5.0}],
        },
    ]
    path.write_text(json.dumps(history))

    new = _canned_summary("QB", attn_nn_mae=4.3, lgbm_mae=4.6)
    print_history_comparison(str(path), [new], last_n=5)
    out = capsys.readouterr().out
    assert "QB history" in out
    assert "prior run" in out
    assert "> NEW" in out

"""Unit tests for the pure helpers in src.analysis.artifact_eval.

The full ``build_test_df_from_artifacts`` path needs saved model artifacts +
data splits (exercised by the CLI smoke / diagnostics, not here); these tests
cover the drift-free pure helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import artifact_eval as ae

pytestmark = pytest.mark.unit


def test_attach_predictions_writes_total_and_per_target_columns():
    pos_test = pd.DataFrame({"player_id": ["a", "b"], "fantasy_points": [10.0, 20.0]})
    preds = {"yards": np.array([5.0, 7.0]), "tds": np.array([1.0, 2.0])}
    targets = ["yards", "tds"]
    total_fn = lambda p: p["yards"] + 6.0 * p["tds"]  # noqa: E731 - tiny test stub
    ae.attach_predictions(pos_test, "ridge", preds, targets, total_fn)
    assert list(pos_test["pred_ridge_yards"]) == [5.0, 7.0]
    assert list(pos_test["pred_ridge_tds"]) == [1.0, 2.0]
    # total = yards + 6*tds
    assert list(pos_test["pred_ridge_total"]) == [11.0, 19.0]


def test_make_total_fn_uses_target_signs_for_K_like_position():
    reg = {"target_signs": {"fg_points": 1.0, "misses": -1.0}}
    total_fn = ae._make_total_fn("K", ["fg_points", "misses"], reg, "ppr")
    preds = {"fg_points": np.array([9.0, 3.0]), "misses": np.array([1.0, 0.0])}
    # sign-vectored sum: fg_points - misses
    assert list(total_fn(preds)) == [8.0, 3.0]


def test_attn_supported_for_flat_incl_opp_history_but_not_nested():
    assert ae._attn_is_supported({}) is True
    assert ae._attn_is_supported({"attn_history_structure": "flat"}) is True
    # Opponent-history side branch (skill positions) IS supported in v1.
    assert ae._attn_is_supported({"opp_attn_history_stats": ["def_sacks"]}) is True
    # Nested per-kick variant (K) is not yet handled.
    assert ae._attn_is_supported({"attn_history_structure": "nested"}) is False


def _populate(d):
    """Create dir ``d`` (Path) with one sentinel file so it counts as populated."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text("{}")


def test_resolve_model_dir_prefers_served_then_falls_back_to_producer(tmp_path, monkeypatch):
    """Served path wins when populated; else the local producer path; else a LOUD raise.

    Guards the served-vs-producer mismatch that silently produced "no models found":
    reg["model_dir"] = src/{pos}/outputs/models (served, what serving + --sync read)
    vs the local producer path {pos}/outputs/models (what a local run / Batch writes).
    """
    monkeypatch.chdir(tmp_path)
    served = tmp_path / "src" / "wr" / "outputs" / "models"
    producer = tmp_path / "wr" / "outputs" / "models"
    reg = {"model_dir": str(served)}

    # Neither exists -> raise loudly, naming BOTH paths (no silent empty result).
    with pytest.raises(FileNotFoundError) as exc_info:
        ae.resolve_model_dir("WR", reg)
    msg = str(exc_info.value)
    assert str(served) in msg and "wr/outputs/models" in msg

    # Producer-only -> fall back to it (the local-run case this fix unblocks).
    _populate(producer)
    assert ae.resolve_model_dir("WR", reg) == ae._producer_model_dir("WR") == "wr/outputs/models"

    # Served populated -> preferred over the producer path.
    _populate(served)
    assert ae.resolve_model_dir("WR", reg) == str(served)


def test_resolve_model_dir_treats_empty_served_dir_as_absent(tmp_path, monkeypatch):
    """An EMPTY served dir (failed/partial sync) must not shadow a populated producer dir."""
    monkeypatch.chdir(tmp_path)
    served = tmp_path / "src" / "wr" / "outputs" / "models"
    producer = tmp_path / "wr" / "outputs" / "models"
    reg = {"model_dir": str(served)}

    served.mkdir(parents=True)  # exists but EMPTY (e.g. model_sync mkdir'd, extract failed)
    _populate(producer)  # the fresh local run
    assert ae.resolve_model_dir("WR", reg) == "wr/outputs/models"  # falls through to producer

    # Both empty -> still a loud raise (not a masked "no models found").
    import shutil

    shutil.rmtree(producer)
    producer.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        ae.resolve_model_dir("WR", reg)


def test_resolve_model_dir_override_takes_precedence(tmp_path):
    reg = {"model_dir": "src/wr/outputs/models"}  # would not exist; override must win first
    assert ae.resolve_model_dir("WR", reg, override="/custom/models") == "/custom/models"


def test_warn_if_sync_noop_is_loud_when_bucket_unset_or_blank(monkeypatch, capsys):
    """--sync silently no-ops when FF_MODEL_S3_BUCKET is unset OR whitespace-only.

    The whitespace case mirrors model_sync's ``.strip()`` gate: a stray-space export
    would no-op the sync, so warn_if_sync_noop must flag it too (bare truthiness wouldn't).
    """
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert ae.warn_if_sync_noop() is False
    assert "FF_MODEL_S3_BUCKET is unset" in capsys.readouterr().out

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "   ")  # whitespace-only -> model_sync .strip() no-ops
    assert ae.warn_if_sync_noop() is False
    assert "FF_MODEL_S3_BUCKET is unset" in capsys.readouterr().out

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "some-bucket")
    assert ae.warn_if_sync_noop() is True
    assert capsys.readouterr().out == ""  # bucket set -> no warning


# --- Stale-artifact (Ridge-tell) drift check ---------------------------------- #
def _write_metrics(model_dir, ridge_mae):
    import json

    (model_dir / "benchmark_metrics.json").write_text(
        json.dumps(
            {
                "git_sha": "abc123",
                "split_run_id": "sid",
                "ridge_metrics": {"total": {"mae": ridge_mae}},
            }
        )
    )


def test_reconstruction_verdict_thresholds():
    assert ae._reconstruction_verdict(0.03, 0.10, 0.30) == "ok"
    assert ae._reconstruction_verdict(0.10, 0.10, 0.30) == "ok"  # boundary inclusive
    assert ae._reconstruction_verdict(0.15, 0.10, 0.30) == "warn"
    assert ae._reconstruction_verdict(0.86, 0.10, 0.30) == "fail"


def _recon_df():
    # |10-11| + |20-19| = 2, mean = 1.0 reconstructed Ridge MAE
    return pd.DataFrame({"pred_ridge_total": [10.0, 20.0], "fantasy_points": [11.0, 19.0]})


def test_validate_reconstruction_ok_when_recorded_matches(tmp_path, capsys):
    _write_metrics(tmp_path, 1.0)  # recorded == reconstructed -> Δ=0
    res = ae.validate_reconstruction("QB", _recon_df(), model_dir=str(tmp_path))
    assert res["status"] == "ok"
    assert res["reconstructed_ridge_mae"] == 1.0
    assert "STALE" not in capsys.readouterr().out  # quiet when healthy


def test_validate_reconstruction_flags_stale_and_strict_raises(tmp_path, capsys):
    _write_metrics(tmp_path, 2.0)  # recorded 2.0 vs reconstructed 1.0 -> Δ=1.0 -> fail
    res = ae.validate_reconstruction("QB", _recon_df(), model_dir=str(tmp_path))
    assert res["status"] == "fail"
    assert res["delta"] == 1.0
    assert "STALE ARTIFACT" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        ae.validate_reconstruction("QB", _recon_df(), model_dir=str(tmp_path), strict=True)


def test_validate_reconstruction_warn_band(tmp_path):
    _write_metrics(tmp_path, 1.15)  # Δ=0.15 -> warn band (warn_tol 0.10 < Δ <= fail_tol 0.30)
    assert (
        ae.validate_reconstruction("QB", _recon_df(), model_dir=str(tmp_path))["status"] == "warn"
    )


def test_validate_reconstruction_skips_gracefully(tmp_path):
    # no benchmark_metrics.json -> skipped (no reference)
    assert (
        ae.validate_reconstruction("QB", _recon_df(), model_dir=str(tmp_path))["status"]
        == "skipped"
    )
    _write_metrics(tmp_path, 1.0)
    # non-skill position (no FP total in splits) -> skipped
    assert (
        ae.validate_reconstruction("K", _recon_df(), model_dir=str(tmp_path))["status"] == "skipped"
    )
    # non-PPR scoring (recorded MAE is PPR) -> skipped
    assert (
        ae.validate_reconstruction(
            "QB", _recon_df(), model_dir=str(tmp_path), scoring_format="standard"
        )["status"]
        == "skipped"
    )
    # missing pred_ridge_total (Ridge load failed upstream) -> skipped
    assert (
        ae.validate_reconstruction(
            "QB", pd.DataFrame({"fantasy_points": [1.0]}), model_dir=str(tmp_path)
        )["status"]
        == "skipped"
    )

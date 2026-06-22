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


def test_resolve_model_dir_prefers_served_then_falls_back_to_producer(tmp_path, monkeypatch):
    """Served path wins when present; else the local producer path; else a LOUD raise.

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
    producer.mkdir(parents=True)
    assert ae.resolve_model_dir("WR", reg) == ae._producer_model_dir("WR") == "wr/outputs/models"

    # Served present -> preferred over the producer path.
    served.mkdir(parents=True)
    assert ae.resolve_model_dir("WR", reg) == str(served)


def test_resolve_model_dir_override_takes_precedence(tmp_path):
    reg = {"model_dir": "src/wr/outputs/models"}  # would not exist; override must win first
    assert ae.resolve_model_dir("WR", reg, override="/custom/models") == "/custom/models"


def test_warn_if_sync_noop_is_loud_only_when_bucket_unset(monkeypatch, capsys):
    """--sync silently no-ops when FF_MODEL_S3_BUCKET is unset; that must be loud."""
    monkeypatch.delenv("FF_MODEL_S3_BUCKET", raising=False)
    assert ae.warn_if_sync_noop() is False
    assert "FF_MODEL_S3_BUCKET is unset" in capsys.readouterr().out

    monkeypatch.setenv("FF_MODEL_S3_BUCKET", "some-bucket")
    assert ae.warn_if_sync_noop() is True
    assert capsys.readouterr().out == ""  # bucket set -> no warning

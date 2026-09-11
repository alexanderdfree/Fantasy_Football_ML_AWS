"""Evidence boundaries must fail before a misleading Batch result is accepted."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.shared.comparison_scoring import scoring_components
from src.tuning import ab_merge_readiness as spec

pytestmark = pytest.mark.unit


def _pin_replay(tmp_path, monkeypatch, source):
    path = tmp_path / "replay.parquet"
    source.to_parquet(path, index=False)
    monkeypatch.setenv("FF_MERGE_READINESS_QB_REPLAY", str(path))
    monkeypatch.setenv("FF_MERGE_READINESS_QB_REPLAY_SHA256", spec._sha256(path.read_bytes()))


def _test_frame():
    return pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "season": [2025] * 3,
            "week": [1] * 3,
            "position": ["QB", "QB", "RB"],
            "is_top_available": [0.0, 1.0, 1.0],
            "inherited_opportunity": [0.0, 9.0, 3.0],
            "passing_yards": [100, 200, 0],
        }
    )


def test_qb_replay_joins_keys_and_changes_only_declared_features(tmp_path, monkeypatch):
    test = _test_frame()
    source = test.iloc[[1, 0]].copy()
    source["is_top_available"] = [0.0, 1.0]
    source["inherited_opportunity"] = [0.0, 5.0]
    source["passing_yards"] = -999  # Extra input columns cannot replace truth.
    _pin_replay(tmp_path, monkeypatch, source)
    replay, provenance = spec._qb_replay(test)
    expected = test.copy()
    expected.loc[:1, "is_top_available"] = [1.0, 0.0]
    expected.loc[:1, "inherited_opportunity"] = [5.0, 0.0]
    pd.testing.assert_frame_equal(replay, expected)
    pd.testing.assert_frame_equal(test, _test_frame())
    assert provenance["matched_rows"] == 2


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "checksum"])
def test_qb_replay_rejects_incomplete_or_unverified_inputs(tmp_path, monkeypatch, defect):
    source = _test_frame().iloc[:2].copy()
    if defect == "missing":
        source = source.iloc[:1]
    elif defect == "duplicate":
        source = pd.concat([source, source.iloc[:1]])
    elif defect == "nonfinite":
        source.loc[0, "inherited_opportunity"] = np.nan
    _pin_replay(tmp_path, monkeypatch, source)
    if defect == "checksum":
        monkeypatch.setenv("FF_MERGE_READINESS_QB_REPLAY_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        spec._qb_replay(_test_frame())


def test_rows_preserve_certified_missing_truth_and_raw_predictions(monkeypatch):
    frame = _test_frame().iloc[:2].copy()
    targets = scoring_components("QB")
    for target in targets:
        frame[target] = 0.0
        for family in spec.FAMILIES:
            frame[f"pred_{family}_{target}"] = 0.0
    for family in spec.FAMILIES:
        frame[f"pred_{family}_total"] = [3.0, 4.0]
    frame["fantasy_points"] = [999.0, 888.0]
    frame["actual_projected_total"] = [np.nan, 1.0]
    frame.attrs["actual_projected_total_metadata"] = {
        "basis": "configured_target_aggregation_v1",
        "targets": list(targets),
        "scoring_format": "ppr",
    }
    monkeypatch.setattr(
        spec,
        "reference_selection",
        lambda pos, rows, ref, n: (pd.Series(True, index=rows.index), {"status": "available"}),
    )
    rows, _ = spec._rows(frame, "QB", targets, pd.DataFrame())
    assert rows.comparison_available.tolist() == [False, True]
    assert pd.isna(rows.comparison_actual.iloc[0])
    assert rows.comparison_actual.iloc[1] == 1.0
    assert "pred_attn_nn_passing_yards" in rows
    assert spec._row_metrics(rows, "native")["native:ridge"] == {
        "n": 1,
        "unavailable": 1,
        "mae": 3.0,
        "rmse": 3.0,
        "bias": 3.0,
    }


def test_capture_observer_preserves_call_and_records_actual_engagement(monkeypatch):
    from src.shared.training import MultiHeadHistoryTrainer, MultiHeadTrainer

    calls = []

    def original(trainer, loader):
        calls.append((trainer, loader))
        trainer._graphed_step = object()
        return True

    monkeypatch.setattr(MultiHeadTrainer, "_maybe_graph_full_step", original)
    spec._CAPTURE_EVENTS.clear()
    spec._observe_capture()
    spec._observe_capture()  # Installing twice must not duplicate observations.
    loader = object()
    for cls in (MultiHeadTrainer, MultiHeadHistoryTrainer):
        trainer = object.__new__(cls)
        trainer.model = object()
        trainer.criterion = SimpleNamespace(compute_combined_capturable=lambda: None)
        trainer.device = "cuda:0"
        trainer._use_amp = False
        trainer._amp_dtype = None
        trainer._graphed_step = None
        assert trainer._maybe_graph_full_step(loader) is True
    assert len(calls) == 2
    assert all(item[1] is loader for item in calls)
    evidence = spec._capture_evidence({"execution": {"device": "cuda:0"}})
    assert evidence["required"] is True
    assert len(evidence["events"]) == 2
    assert all(event["graph_present"] for event in evidence["events"])


def test_capture_gate_rejects_capability_without_actual_graph(monkeypatch):
    monkeypatch.delenv("FF_DEVICE", raising=False)
    spec._CAPTURE_EVENTS.clear()
    with pytest.raises(ValueError, match="full-step capture did not engage"):
        spec._capture_evidence({"execution": {"device": "cuda:0"}})
    assert spec._capture_evidence({"execution": {"device": "cpu"}})["required"] is False


def test_output_sink_refuses_production_prefix_and_uses_cell_key(monkeypatch):
    import boto3

    calls = []
    monkeypatch.setattr(
        boto3,
        "client",
        lambda name: SimpleNamespace(put_object=lambda **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setenv("FF_AB_RUN_ID", "test-run")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("FF_AB_S3_PREFIX", "models")
    with pytest.raises(ValueError, match="explicit ab_runs"):
        spec._evidence_sink("QB-corrected-42")
    monkeypatch.setenv("FF_AB_S3_PREFIX", "ab_runs")
    write = spec._evidence_sink("QB-corrected-42")
    assert write("manifest-abc.json", b"{}") == (
        "s3://test-bucket/ab_runs/test-run/readiness/QB-corrected-42/manifest-abc.json"
    )
    assert calls[0]["IfNoneMatch"] == "*"
    assert calls[0]["Body"] == b"{}"


def test_spec_keeps_existing_grid_and_rejects_stacking(monkeypatch):
    from src.tuning.ab_harness import build_cells, resolve_spec

    assert len(build_cells(resolve_spec(spec))) == 30
    monkeypatch.setenv("FF_AB_STACKED", "1")
    with pytest.raises(ValueError, match="nonstacked"):
        spec.corrected({})


def test_output_sink_accepts_authorized_experiment_but_rejects_path_escape(monkeypatch):
    import boto3

    calls = []
    monkeypatch.setattr(
        boto3,
        "client",
        lambda name: SimpleNamespace(put_object=lambda **kwargs: calls.append(kwargs)),
    )
    monkeypatch.setenv("FF_AB_RUN_ID", "test-run")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    prefix = "experiments/merge-readiness/20260911T181252Z/ab_runs"
    monkeypatch.setenv("FF_AB_S3_PREFIX", prefix)
    spec._evidence_sink("WR-corrected-42")("manifest-abc.json", b"{}")
    assert calls[0]["Key"] == f"{prefix}/test-run/readiness/WR-corrected-42/manifest-abc.json"
    for invalid in (f"{prefix}/../models", "ab_runs/../models", "ab_runs//models", "models"):
        monkeypatch.setenv("FF_AB_S3_PREFIX", invalid)
        with pytest.raises(ValueError, match="explicit ab_runs"):
            spec._evidence_sink("WR-corrected-42")


def test_complete_callback_writes_replay_manifest_and_numeric_aggregate(tmp_path, monkeypatch):
    from src.analysis import artifact_eval
    from src.qb.run_pipeline import CONFIG
    from src.training.context import RunContext, use_context
    from src.training.contracts import resolve_recipe
    from src.tuning.ab_harness import aggregate, resolve_spec

    class Result(dict):
        recipe = resolve_recipe("QB", CONFIG)

    test = _test_frame()
    native = test.iloc[:2].copy()
    targets = Result.recipe["targets"]
    for target in targets:
        native[target] = 0.0
        for family in spec.FAMILIES:
            native[f"pred_{family}_{target}"] = 0.0
    native["fantasy_points"] = 0.0
    for family in spec.FAMILIES:
        native[f"pred_{family}_total"] = 0.0
    source = test.iloc[:2].copy()
    source["inherited_opportunity"] = [5.0, 0.0]
    _pin_replay(tmp_path, monkeypatch, source)
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    (data / "raw").mkdir()
    for name in ("train", "val", "test"):
        test.to_parquet(data / "splits" / f"{name}.parquet", index=False)
    reference_file = data / "raw" / "weekly_evaluation_reference_v1.parquet"
    reference_file.write_bytes(b"pinned reference")
    monkeypatch.setattr(spec, "load_reference", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(
        spec,
        "reference_selection",
        lambda pos, rows, ref, n: (pd.Series(True, index=rows.index), {"status": "available"}),
    )
    monkeypatch.setattr(spec.ab_nn_correctness, "metric_fn", lambda *args: {})

    def predict(position, train, val, replay, **kwargs):
        pd.testing.assert_frame_equal(train, test)
        pd.testing.assert_frame_equal(val, test)
        out = native.copy()
        out[spec.QB_FIELDS] = replay.loc[replay.position.eq("QB"), spec.QB_FIELDS]
        return out

    monkeypatch.setattr(artifact_eval, "build_test_df_from_artifacts", predict)
    monkeypatch.setenv("FF_MERGE_READINESS_OUTPUT", str(tmp_path / "evidence"))
    monkeypatch.delenv("FF_AB_RUN_ID", raising=False)
    monkeypatch.setenv("FF_DEVICE", "cpu")
    monkeypatch.setattr(spec, "_ARM", "baseline")
    spec._CAPTURE_EVENTS.clear()
    result = Result(
        test_df=native,
        per_target_preds={
            family: {target: np.zeros(2) for target in targets} for family in spec.FAMILIES
        },
        cohorts={name: {"status": "available"} for name in spec.REQUIRED_COHORTS},
        execution={"device": "cpu"},
    )
    with use_context(RunContext(tmp_path / "outputs", data, seed=42)):
        metrics = spec.metric_fn(result, "QB")
    manifest_path = next((tmp_path / "evidence").rglob("manifest-*.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.name == f"manifest-{spec._sha256(manifest_path.read_bytes())}.json"
    assert manifest["evaluations"]["native"]["n_rows"] == 2
    assert manifest["evaluations"]["pregame_replay"]["n_rows"] == 2
    assert metrics["readiness"]["required_cohorts_available"] == 4
    summarized = aggregate(
        resolve_spec(spec),
        [{"position": "QB", "variant": "baseline", "seed": 42, "ok": True, "metrics": metrics}],
    )
    assert summarized["n_ok"] == 1

"""Replay of synthetic cohorts against fake checkpoints built on the real QB spec."""

import hashlib
import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.analysis.synthetic_history import HistoryRecipe, generate_cohort, write_cohort
from src.analysis.synthetic_replay import (
    load_cohort,
    main,
    prediction_inputs,
    replay_cohort,
    requested_families,
)
from src.prediction.bundle import read_bundle, write_bundle
from src.prediction.predictor import Predictor, legacy_schema
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import (
    compute_feature_cols_hash,
    wrap_state_dict,
    write_scaler_meta,
)
from src.shared.models import RidgeMultiTarget
from src.shared.neural_net import MultiHeadNet, MultiHeadNetWithHistory
from src.shared.registry import get_inference_spec
from tests.analysis.conftest import fake_artifacts

pytestmark = pytest.mark.unit


def _qb_artifacts(
    directory, families=("attn_nn",), *, seed=33, extra_static=(), data_ids=None, opp_stats=()
):
    """Fake QB checkpoints whose input schemas are the real production whitelists."""
    return fake_artifacts(
        directory,
        "QB",
        families,
        seed=seed,
        extra_static=extra_static,
        data_ids=data_ids,
        opp_stats=opp_stats,
    )


def _cohort(tmp_path, source, name="cohort", **kwargs):
    kwargs.setdefault("history_games", 3)
    kwargs.setdefault("window", "exact")
    recipe = HistoryRecipe(name=name, cases=5, block_games=2, **kwargs)
    return write_cohort(generate_cohort(source, recipe), tmp_path / name)


def test_replay_records_attention_responses_from_context_and_history(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models)
    cohort = load_cohort(_cohort(tmp_path, qb_source))
    predictions, manifest = replay_cohort(cohort, str(models), ["attn_nn"])
    targets = list(get_inference_spec("QB")["targets"])
    assert len(predictions) == 5
    assert set(manifest["prediction_columns"]) == {
        *(f"pred_attn_nn_{t}" for t in targets),
        "pred_attn_nn_total",
        "pred_attn_nn_total_half_ppr",
        "pred_attn_nn_total_standard",
    }
    assert predictions["exact_window"].all()
    predictor = Predictor.from_bundle(models, "attn_nn", position="QB")
    direct = predictor.predict_raw(prediction_inputs(predictor, cohort))
    for target in targets:
        np.testing.assert_array_equal(predictions[f"pred_attn_nn_{target}"], direct[target])
    np.testing.assert_array_equal(
        predictions["pred_attn_nn_total"], predictions_to_fantasy_points("QB", direct, "ppr")
    )
    np.testing.assert_array_equal(
        predictions["pred_attn_nn_total_standard"],
        predictions_to_fantasy_points("QB", direct, "standard"),
    )
    assert manifest["families_excluded"] == {}
    assert manifest["families_requested"] == ["attn_nn"]
    assert manifest["identity_control"]["status"] == "skipped"
    assert manifest["families"]["attn_nn"]["provenance"]["data_id"] == "fake"
    assert not any(column.startswith("actual") for column in predictions.columns)


def test_transformed_cohort_replays_attention_only_with_context_control(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"))
    transforms = [{"op": "scale", "stats": ["passing_yards"], "factor": 1.5}]
    cohort = load_cohort(
        _cohort(
            tmp_path,
            qb_source,
            name="fixture",
            transforms=transforms,
            opaque_signal_policy="keep_donor",
        )
    )
    assert cohort.manifest["history_kind"] == "transformed"
    predictions, manifest = replay_cohort(
        cohort, str(models), ["attn_nn", "ridge"], source=qb_source
    )
    assert list(manifest["families"]) == ["attn_nn"]
    assert "transformed" in manifest["families_excluded"]["ridge"]
    assert manifest["history_kind"] == "transformed" and manifest["fixture"] is True
    assert manifest["cohort_name"] == "fixture"
    assert manifest["sampling_identity_sha256"] == cohort.manifest["sampling_identity_sha256"]
    control = manifest["identity_control"]
    assert control["status"] == "context_only"
    assert control["families"]["attn_nn"]["checks"] == ["static_values"]
    baseline = load_cohort(_cohort(tmp_path, qb_source, name="donor"))
    plain, _ = replay_cohort(baseline, str(models), ["attn_nn"])
    assert list(plain["case_id"]) != list(predictions["case_id"])
    assert list(plain["donor_player_id"]) == list(predictions["donor_player_id"])
    field = cohort.manifest["history_columns"].index("passing_yards")
    np.testing.assert_allclose(
        cohort.arrays["history"][:, :3, field], 1.5 * baseline.arrays["history"][:, :3, field]
    )


def test_identity_control_matches_production_inputs_and_predictions(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge", "nn"))
    cohort = load_cohort(_cohort(tmp_path, qb_source))
    _, manifest = replay_cohort(cohort, str(models), ["attn_nn", "ridge", "nn"], source=qb_source)
    control = manifest["identity_control"]
    assert control["status"] == "passed"
    assert set(control["families"]) == {"attn_nn", "ridge", "nn"}
    attention = control["families"]["attn_nn"]
    assert attention["cases"] == attention["exact_window_cases"] == 5
    assert attention["compared_predictions"] == 5
    assert attention["checks"] == [
        "static_values",
        "history_prefix",
        "mask",
        "predictions_on_exact_windows",
    ]
    for family, entry in control["families"].items():
        assert entry["max_abs_prediction_delta"] <= 1e-5, family


def test_truncated_windows_are_counted_not_verified(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"))
    # The exact three-game windows (weeks 1, 2, 4) score below 17 points in the
    # fixture; the bound keeps only forecasts with more real history than the window.
    cohort = load_cohort(_cohort(tmp_path, qb_source, window="any", min_history_ppg=17.0))
    assert not cohort.cases["exact_window"].any()
    predictions, manifest = replay_cohort(
        cohort, str(models), ["attn_nn", "ridge"], source=qb_source
    )
    control = manifest["identity_control"]
    assert control["status"] == "inputs_only"
    assert control["families"]["attn_nn"]["compared_predictions"] == 0
    assert control["families"]["attn_nn"]["checks"] == ["static_values", "history_prefix", "mask"]
    assert "truncate" in manifest["families_excluded"]["ridge"]
    assert not predictions["exact_window"].any()


def test_bootstrap_cohort_control_verifies_context_only(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"))
    cohort = load_cohort(_cohort(tmp_path, qb_source, name="boot", mode="block_bootstrap"))
    _, manifest = replay_cohort(cohort, str(models), ["attn_nn", "ridge"], source=qb_source)
    control = manifest["identity_control"]
    assert control["status"] == "context_only"
    assert control["families"]["attn_nn"]["checks"] == ["static_values"]
    assert "resampled" in manifest["families_excluded"]["ridge"]
    with pytest.raises(ValueError, match="no requested family is replayable"):
        replay_cohort(cohort, str(models), ["ridge"])


def test_identity_control_detects_static_drift_and_foreign_source(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models)
    cohort = load_cohort(_cohort(tmp_path, qb_source))
    drifted = qb_source.copy()
    drifted["depth_chart_rank"] = 2.0
    with pytest.raises(ValueError, match="source values differ"):
        replay_cohort(cohort, str(models), ["attn_nn"], source=drifted)
    foreign = qb_source.copy()
    foreign["qbr_total"] = foreign["qbr_total"] + 1
    with pytest.raises(ValueError, match="source values differ"):
        replay_cohort(cohort, str(models), ["attn_nn"], source=foreign)


def test_mixed_training_generations_are_refused(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"), data_ids={"ridge": "older-release"})
    cohort = load_cohort(_cohort(tmp_path, qb_source))
    with pytest.raises(ValueError, match="different training generations"):
        replay_cohort(cohort, str(models), ["attn_nn", "ridge"])
    _, manifest = replay_cohort(cohort, str(models), ["ridge"])
    assert manifest["families"]["ridge"]["provenance"]["data_id"] == "older-release"


def test_schema_mismatches_fail_loud(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, extra_static=["not_a_feature"])
    cohort = load_cohort(_cohort(tmp_path, qb_source))
    with pytest.raises(ValueError, match="missing features required by attn_nn"):
        replay_cohort(cohort, str(models), ["attn_nn"])
    good = tmp_path / "good"
    _qb_artifacts(good)
    predictor = Predictor.from_bundle(good, "attn_nn", position="QB")
    drifted = {**cohort.manifest, "history_columns": cohort.manifest["history_columns"][:-1]}
    with pytest.raises(ValueError, match="history columns differ"):
        prediction_inputs(predictor, replace(cohort, manifest=drifted))
    opponent = tmp_path / "opponent"
    _qb_artifacts(opponent, opp_stats=["opp_a", "opp_b"])
    predictor = Predictor.from_bundle(opponent, "attn_nn", position="QB")
    with pytest.raises(ValueError, match="opponent stream"):
        prediction_inputs(predictor, cohort)


def test_cohort_manifest_shape_and_files_are_verified(tmp_path, qb_source):
    tampered = _cohort(tmp_path, qb_source, name="tampered")
    (tampered / "context.parquet").write_bytes(b"not a parquet")
    with pytest.raises(ValueError, match="cohort artifact mismatch: context.parquet"):
        load_cohort(tampered)
    stale = _cohort(tmp_path, qb_source, name="stale")
    manifest_path = stale / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "schema_version": 1}))
    with pytest.raises(ValueError, match="schema_version 3 required"):
        load_cohort(stale)
    manifest_path.write_text(json.dumps({k: v for k, v in manifest.items() if k != "recipe"}))
    with pytest.raises(ValueError, match="manifest is missing"):
        load_cohort(stale)
    files = {k: v for k, v in manifest["files"].items() if k != "context.parquet"}
    manifest_path.write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="does not list"):
        load_cohort(stale)
    bootstrap = _cohort(tmp_path, qb_source, name="unlocked", mode="block_bootstrap")
    unlocked = json.loads((bootstrap / "manifest.json").read_text())
    unlocked["model_input_readiness"]["ridge"]["ready"] = True
    (bootstrap / "manifest.json").write_text(json.dumps(unlocked))
    with pytest.raises(ValueError, match="readiness disagrees"):
        load_cohort(bootstrap)


def test_requested_families_resolves_all_and_rejects_unknown(tmp_path, monkeypatch):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "nn"))
    assert requested_families(["all"], str(models)) == (["nn", "attn_nn"], {})
    assert requested_families(["attn_nn", "ridge", "attn_nn"], str(models))[0] == [
        "ridge",
        "attn_nn",
    ]
    with pytest.raises(ValueError, match="unknown model families"):
        requested_families(["tabpfn"], str(models))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with pytest.raises(ValueError, match="legacy artifact directory"):
        requested_families(["all"], str(legacy))
    monkeypatch.setattr(
        "src.analysis.synthetic_replay.bundled_families", lambda _: ("ridge", "lgbm")
    )
    monkeypatch.setattr("src.analysis.synthetic_replay.platform.system", lambda: "Darwin")
    families, excluded = requested_families(["all"], str(models))
    assert families == ["ridge"] and "libomp" in excluded["lgbm"]
    monkeypatch.setattr("src.analysis.synthetic_replay.platform.system", lambda: "Linux")
    assert requested_families(["all"], str(models)) == (["ridge", "lgbm"], {})


def test_cli_json_no_overwrite_and_hashes_round_trip(tmp_path, qb_source, capsys):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "nn"))
    cohort_dir = _cohort(tmp_path, qb_source)
    source = tmp_path / "source.parquet"
    qb_source.to_parquet(source, index=False)
    output = tmp_path / "replay"
    argv = [
        "--cohort",
        str(cohort_dir),
        "--output",
        str(output),
        "--families",
        "attn_nn",
        "nn",
        "--model-dir",
        str(models),
        "--source",
        str(source),
    ]
    assert main(argv) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["families"] == ["attn_nn", "nn"]
    assert stdout["identity_control"] == "passed"
    manifest = json.loads((output / "replay_manifest.json").read_text())
    assert manifest["files"] == {
        "predictions.parquet": hashlib.sha256(
            (output / "predictions.parquet").read_bytes()
        ).hexdigest()
    }
    assert (
        manifest["cohort_manifest_sha256"]
        == hashlib.sha256((cohort_dir / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        manifest["identity_control"]["source_file_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert manifest["sync"] is None
    bundle = read_bundle(models, "attn_nn")
    assert manifest["bundle_ids"]["attn_nn"] == bundle.bundle_id
    assert manifest["families"]["attn_nn"]["feature_cols_hash"] == compute_feature_cols_hash(
        bundle.inputs.features
    )
    frame = pd.read_parquet(output / "predictions.parquet")
    assert len(frame) == 5 and "pred_nn_total" in frame
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "output already exists" in capsys.readouterr().err
    assert not list(tmp_path.glob(".replay-*"))

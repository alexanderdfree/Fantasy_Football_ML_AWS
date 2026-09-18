"""Replay of synthetic cohorts against fake checkpoints built on the real QB spec."""

import hashlib
import json

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
from src.features.engineer import get_attn_static_columns
from src.prediction.bundle import read_bundle, write_bundle
from src.prediction.predictor import Predictor
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import (
    compute_feature_cols_hash,
    wrap_state_dict,
    write_scaler_meta,
)
from src.shared.models import RidgeMultiTarget
from src.shared.neural_net import MultiHeadNet, MultiHeadNetWithHistory
from src.shared.registry import get_inference_spec

pytestmark = pytest.mark.unit


def _qb_artifacts(directory, families=("attn_nn",), *, seed=33, extra_static=()):
    """Fake QB checkpoints whose input schemas are the real production whitelists."""
    directory.mkdir(parents=True, exist_ok=True)
    cfg = get_inference_spec("QB")
    features = list(cfg["get_feature_columns_fn"]())
    static = get_attn_static_columns(features, cfg["attn_static_features"]) + list(extra_static)
    targets = list(cfg["targets"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for family in families:
        if family == "ridge":
            model = RidgeMultiTarget(targets, alpha=1.0)
            X = rng.normal(size=(24, len(features))).astype(np.float32)
            model.fit(X, {t: rng.normal(size=24).astype(np.float32) for t in targets})
            model.save(str(directory))
            write_bundle(directory, "QB", family, cfg, features, model, data_id="fake")
            continue
        columns = static if family == "attn_nn" else features
        if family == "attn_nn":
            model = MultiHeadNetWithHistory(
                len(columns),
                len(cfg["attn_history_stats"]),
                targets,
                **cfg["attn_nn_kwargs_static"],
            )
        else:
            model = MultiHeadNet(len(columns), targets, **cfg["nn_kwargs"])
        scaler = StandardScaler().fit(rng.normal(size=(8, len(columns))).astype(np.float32))
        stem = "attention_nn" if family == "attn_nn" else "nn"
        weight_stem = "attention_nn" if family == "attn_nn" else "multihead_nn"
        joblib.dump(scaler, directory / f"{stem}_scaler.pkl")
        write_scaler_meta(directory / f"{stem}_scaler_meta.json", columns, targets)
        torch.save(
            wrap_state_dict(model.state_dict(), columns, targets),
            directory / f"qb_{weight_stem}.pt",
        )
        write_bundle(
            directory,
            "QB",
            family,
            cfg,
            columns,
            model,
            preprocessing={"clip": [-4.0, 4.0]},
            data_id="fake",
        )
    return cfg


def _cohort(tmp_path, source, name="cohort", **kwargs):
    recipe = HistoryRecipe(name=name, cases=5, history_games=5, block_games=3, **kwargs)
    return write_cohort(generate_cohort(source, recipe), tmp_path / name)


def test_replay_records_attention_responses_from_context_and_history(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models)
    cohort_dir = _cohort(tmp_path, qb_source)
    cohort = load_cohort(cohort_dir)
    predictions, manifest = replay_cohort(cohort, str(models), ["attn_nn"])
    targets = list(get_inference_spec("QB")["targets"])
    assert len(predictions) == 5
    assert set(manifest["prediction_columns"]) == {
        *(f"pred_attn_nn_{t}" for t in targets),
        "pred_attn_nn_total",
        "pred_attn_nn_total_half_ppr",
        "pred_attn_nn_total_standard",
    }
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
    assert manifest["identity_control"]["status"] == "skipped"
    assert "accuracy" not in json.dumps(manifest["response_semantics"]).replace(
        "forecast accuracy", ""
    )
    assert not any(column.startswith("actual") for column in predictions.columns)


@pytest.mark.parametrize("history_games,all_exact", [(5, True), (3, False)])
def test_identity_control_matches_production_inputs_and_predictions(
    tmp_path, qb_source, history_games, all_exact
):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"))
    recipe = HistoryRecipe(name="identity", cases=5, history_games=history_games, block_games=2)
    cohort = load_cohort(write_cohort(generate_cohort(qb_source, recipe), tmp_path / "identity"))
    _, manifest = replay_cohort(cohort, str(models), ["attn_nn", "ridge"], source=qb_source)
    control = manifest["identity_control"]
    assert control["status"] == "passed"
    assert set(control["families"]) == {"attn_nn", "ridge"}
    attention = control["families"]["attn_nn"]
    assert attention["cases"] == 5
    assert (attention["exact_window_cases"] == 5) is all_exact
    assert attention["checks"] == [
        "static_values",
        "history_prefix",
        "mask",
        "predictions_on_exact_windows",
    ]
    assert control["families"]["ridge"]["exact_window_cases"] == 5


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
    bootstrap = load_cohort(_cohort(tmp_path, qb_source, name="boot", mode="block_bootstrap"))
    with pytest.raises(ValueError, match="identity control requires a replay cohort"):
        replay_cohort(bootstrap, str(models), ["attn_nn"], source=qb_source)


def test_flat_families_are_excluded_for_resampled_histories(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "ridge"))
    cohort = load_cohort(_cohort(tmp_path, qb_source, mode="block_bootstrap"))
    predictions, manifest = replay_cohort(cohort, str(models), ["attn_nn", "ridge"])
    assert list(manifest["families"]) == ["attn_nn"]
    assert "windowed" in manifest["families_excluded"]["ridge"]
    assert not any(column.startswith("pred_ridge") for column in predictions.columns)
    with pytest.raises(ValueError, match="no requested family is replayable"):
        replay_cohort(cohort, str(models), ["ridge"])


def test_schema_mismatches_fail_loud(tmp_path, qb_source):
    models = tmp_path / "models"
    _qb_artifacts(models, extra_static=["not_a_feature"])
    cohort_dir = _cohort(tmp_path, qb_source)
    cohort = load_cohort(cohort_dir)
    with pytest.raises(ValueError, match="missing features required by attn_nn"):
        replay_cohort(cohort, str(models), ["attn_nn"])
    good = tmp_path / "good"
    _qb_artifacts(good)
    predictor = Predictor.from_bundle(good, "attn_nn", position="QB")
    drifted = json.loads(json.dumps(cohort.manifest))
    drifted["history_columns"] = drifted["history_columns"][:-1]
    with pytest.raises(ValueError, match="history columns differ"):
        prediction_inputs(predictor, cohort.__class__(**{**cohort.__dict__, "manifest": drifted}))
    tampered = _cohort(tmp_path, qb_source, name="tampered")
    (tampered / "context.parquet").write_bytes(b"not a parquet")
    with pytest.raises(ValueError, match="cohort artifact mismatch: context.parquet"):
        load_cohort(tampered)
    stale = _cohort(tmp_path, qb_source, name="stale")
    manifest = json.loads((stale / "manifest.json").read_text())
    manifest["schema_version"] = 1
    (stale / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version 2 required"):
        load_cohort(stale)


def test_requested_families_resolves_all_and_rejects_unknown(tmp_path):
    models = tmp_path / "models"
    _qb_artifacts(models, families=("attn_nn", "nn"))
    assert requested_families(["all"], str(models)) == ["nn", "attn_nn"]
    assert requested_families(["attn_nn", "ridge", "attn_nn"], str(models)) == ["ridge", "attn_nn"]
    with pytest.raises(ValueError, match="unknown model families"):
        requested_families(["tabpfn"], str(models))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    with pytest.raises(ValueError, match="legacy artifact directory"):
        requested_families(["all"], str(legacy))


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
    assert not list(tmp_path.glob(".replay-*"))

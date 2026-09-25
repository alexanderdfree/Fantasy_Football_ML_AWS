"""Synthetic evidence only: never fit a model or scaler or access AWS."""

import hashlib
import importlib
import io
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis import audit_development_report as report
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.comparison_scoring import comparison_actuals, score_actual_components
from src.shared.comparison_truth import attach_comparison_actuals
from src.shared.evaluation_cohorts import _identity

pytestmark = pytest.mark.unit


def plan(purpose="smoke"):
    positions = ["WR"] if purpose == "smoke" else ["RB", "WR", "TE"]
    origins = [2022] if purpose == "smoke" else [2022, 2023]
    seeds = [42] if purpose == "smoke" else [42, 123, 7]
    return {
        "schema": "audit-development-plan/v1",
        "purpose": purpose,
        "candidate": "count_precision",
        "source_sha": "1" * 40,
        "baseline_sha": "2" * 40,
        "candidate_sha": "3" * 40,
        "data_release": "4" * 64,
        "image_digest": "sha256:" + "5" * 64,
        "campaign_prefix": "ab_runs/test-campaign",
        "positions": positions,
        "origins": origins,
        "seeds": seeds,
        "runs": [
            dict(
                prefix=f"ab_runs/test-campaign/{purpose}-{year}",
                origin=year,
                positions=positions,
                seeds=seeds,
            )
            for year in origins
        ],
    }


def fixture_record(position="WR", *, missing=False, neural_fp32=False):
    targets = importlib.import_module(f"src.{position.lower()}.config").POSITION_CONFIG.targets
    truth = pd.DataFrame(
        [
            {"player_id": player, "season": year, "week": 1, **{target: 2.0 for target in targets}}
            for year in [2021, 2022]
            for player in ["a", "b"]
        ]
    )
    available = pd.Series(True, index=truth.index)
    if missing:
        available.iloc[-1] = False
    truth = attach_comparison_actuals(truth, position, available)
    rows = truth[truth.season.eq(2022)].copy().reset_index(drop=True)
    for family in report.FAMILIES:
        for index, target in enumerate(targets):
            values = rows[target] + ((index + 1) / 7 if neural_fp32 else 0.2)
            if neural_fp32 and family in ("nn", "attn_nn"):
                values = values.astype(np.float32)
            rows[f"pred_{family}_{target}"] = values
        rows[f"pred_{family}_total"] = (
            predictions_to_fantasy_points(
                position, {t: rows[f"pred_{family}_{t}"].to_numpy() for t in targets}, "ppr"
            )
            if position == "K"
            else score_actual_components(rows, position, prefix=f"pred_{family}_")
        )
    rows["comparison_actual"] = comparison_actuals(rows, position)
    valid = rows[rows.comparison_actual.notna()].copy()
    valid["fantasy_points"] = valid.comparison_actual
    metrics = {}
    models = {}
    for family in report.FAMILIES:
        value = report._numbers(valid[f"pred_{family}_total"] - valid.comparison_actual)
        metrics[f"all:{family}"] = value
        metrics[f"elite_top24:{family}"] = value
        models[family] = value
    data = {"rows": rows.to_parquet(index=False)}
    pred = {t: (rows[t] + 0.2).to_numpy() for t in targets}
    actual = {t: rows[t].to_numpy() for t in targets}
    error = predictions_to_fantasy_points(position, pred, "ppr") - predictions_to_fantasy_points(
        position, actual, "ppr"
    )
    scores = report._numbers(error)
    npz = io.BytesIO()
    np.savez_compressed(
        npz,
        **{f"prediction__{t}": v for t, v in pred.items()},
        **{f"truth__{t}": v for t, v in actual.items()},
    )
    data["npz"] = npz.getvalue()
    state = io.BytesIO()
    torch.save({"weight": torch.tensor([1.0])}, state)
    data["state"] = state.getvalue()
    record = {
        "schema": "isolated-audit-development/v1",
        "promotion_eligible": False,
        "position": position,
        "origin": 2022,
        "seed": 42,
        "variant": "baseline",
        "source_sha": "1" * 40,
        "baseline_sha": "2" * 40,
        "data_release": "4" * 64,
        "candidate_pins": {"count_precision": "3" * 40},
        "batch_job_id": "job",
        "gpu": "NVIDIA L4",
        "tf32_matmul": True,
        "tf32_cudnn": True,
        "execution": {
            "regime": "eager",
            "device": "cuda:0",
            "seed": 42,
            "overrides": {"FF_AMP_DTYPE": "fp32"},
        },
        "prepared_hashes": {
            key: "a" * 64
            for key in [
                "train",
                "val",
                "test",
                "X_train",
                "X_val",
                "X_test",
                *[f"y_{s}/{t}" for s in ("train", "val", "test") for t in targets],
            ]
        },
        "row_hash": hashlib.sha256(
            pd.util.hash_pandas_object(rows[report.KEYS], index=False).values.tobytes()
        ).hexdigest(),
        "truth_hash": hashlib.sha256(
            pd.util.hash_pandas_object(
                rows[[*report.KEYS, "comparison_actual"]], index=False
            ).values.tobytes()
        ).hexdigest(),
        "rows": "rows",
        "metrics": metrics,
        "cohorts": {
            "elite_top24": {
                "status": "available",
                "n": len(valid),
                "cohort_hash": _identity(valid),
                "models": models,
            },
            "weekly_reference_top24": {
                "status": "unavailable",
                "n": None,
                "models": {},
                "reason": "pre2024",
            },
        },
        "inference_parity_passed": True,
        "inference_parity_max_absolute_error": {
            f"pred_{f}_{t}": 0.0 for f in report.FAMILIES for t in (*targets, "total")
        },
        "trainers": [
            {
                "family": family,
                "device": "cuda:0",
                "amp": False,
                "graph": not (position == "K" and family == "attn_nn"),
                "stop_reason": "budget",
                "epochs_executed": 1,
                "validation_raw": "npz",
                "checkpoint": "state",
                "restored_scores": {k: scores[k] for k in ("mae", "rmse")},
                "checkpoint_selection": {
                    "epoch": 1,
                    "score": 1.0,
                    "validation_curve": [1.0],
                    "fixed_epochs": False,
                    "validation_metrics": {
                        f"val_fantasy_{k}_ppr": scores[k] for k in ("mae", "rmse")
                    },
                },
            }
            for family in ("nn", "attn_nn")
        ],
    }
    return record, truth, data


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_full_rescoring_all_positions_without_fitting(position):
    record, truth, data = fixture_record(position, missing=position == "K")
    verified = report.verify_record(record, truth, data.__getitem__)
    assert verified.scores["all:nn"]["n"] == (1 if position == "K" else 2)


def test_k_neural_rescore_preserves_native_fp32_sign_vector():
    record, truth, data = fixture_record("K", neural_fp32=True)
    rows = pd.read_parquet(io.BytesIO(data["rows"]))
    promoted = score_actual_components(rows, "K", prefix="pred_nn_")
    assert not np.array_equal(promoted.to_numpy(), rows.pred_nn_total.to_numpy())
    report.verify_record(record, truth, data.__getitem__)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["metrics"]["all:nn"].update(n=1), "Incomplete samples"),
        (lambda r: r["metrics"]["all:nn"].update(mae=99), "Rescore mismatch"),
        (lambda r: r["cohorts"]["elite_top24"].update(status="unavailable"), "protected cohort"),
        (
            lambda r: r["inference_parity_max_absolute_error"].pop("pred_nn_total"),
            "Incomplete saved-inference",
        ),
        (lambda r: r["trainers"][0]["checkpoint_selection"].update(epoch=2), "Invalid selected"),
        (lambda r: r["trainers"][0]["restored_scores"].update(mae=99), "Rescore mismatch"),
        (lambda r: r["prepared_hashes"].pop("X_train"), "fingerprints"),
        (lambda r: r["trainers"][0].update(graph=False), "graph policy"),
    ],
)
def test_bad_receipts_fail_closed(mutation, match):
    record, truth, data = fixture_record()
    mutation(record)
    with pytest.raises(ValueError, match=match):
        report.verify_record(record, truth, data.__getitem__)


def test_missing_actuals_cannot_be_silently_restored():
    record, truth, data = fixture_record("K", missing=True)
    truth["actual_projected_total"] = truth["actual_projected_total"].fillna(0)
    with pytest.raises(AssertionError):
        report.verify_record(record, truth, data.__getitem__)


def test_smoke_is_not_a_development_seed_and_runs_cannot_overlap():
    development = plan("development")
    report.validate_plan(development)
    development["runs"].append(
        {
            "prefix": "ab_runs/test-campaign/smoke",
            "origin": 2022,
            "positions": ["WR"],
            "seeds": [42],
        }
    )
    with pytest.raises(ValueError, match="overlapping"):
        report.validate_plan(development)
    development = plan("development")
    development["origins"] = [2022]
    with pytest.raises(ValueError, match="Both development years"):
        report.validate_plan(development)


def test_observer_source_bridge_is_explicit_and_replaces_the_complete_position():
    development = report.validate_plan(plan("development"))
    alternate = "6" * 40
    development["sources"][alternate] = {
        "image_digest": "sha256:" + "7" * 64,
        "bridge": {"path": "todo/audit-source-bridge-20260923.json", "sha256": "8" * 64},
    }
    for run in development["runs"]:
        run["include_positions"] = ["RB", "WR"]
        run["superseded_positions"] = {"TE": "observer_position_identity"}
    development["runs"] += [
        dict(
            prefix=f"ab_runs/test-campaign/te-replacement-{origin}",
            origin=origin,
            positions=["TE"],
            seeds=[42, 123, 7],
            source_sha=alternate,
        )
        for origin in [2022, 2023]
    ]
    result = report.validate_plan(development)
    assert set(result["sources"]) == {"1" * 40, alternate}
    assert "source_sha" not in result
    incomplete = deepcopy(development)
    incomplete["runs"][-1]["seeds"] = [42]
    with pytest.raises(ValueError, match="overlapping run plan"):
        report.validate_plan(incomplete)
    undeclared = deepcopy(development)
    del undeclared["sources"][alternate]["bridge"]
    with pytest.raises(ValueError, match="explicit observer-only bridge"):
        report.validate_plan(undeclared)
    cherry_pick = deepcopy(development)
    cherry_pick["runs"][0]["superseded_positions"] = {"TE": "poor metrics"}
    with pytest.raises(ValueError, match="supersession reason"):
        report.validate_plan(cherry_pick)


def _verified_arms():
    record, truth, data = fixture_record()
    baseline = report.verify_record(record, truth, data.__getitem__)
    repeated = deepcopy(baseline)
    repeated.record["variant"] = "baseline_rep"
    candidate = deepcopy(baseline)
    candidate.record["variant"] = "count_precision"
    return baseline, repeated, candidate


def test_controls_hardware_and_complete_grid_are_mandatory():
    baseline, repeated, candidate = _verified_arms()
    candidate.rows["pred_nn_total"] += 1
    result = report.summarize([baseline, repeated, candidate], plan())
    assert not result["verification_passed"] and not result["development_qualified"]
    baseline, repeated, candidate = _verified_arms()
    candidate.record["gpu"] = "NVIDIA A10G"
    result = report.summarize([baseline, repeated, candidate], plan())
    assert any("Unpaired gpu" in x for x in result["errors"])
    result = report.summarize([baseline, repeated], plan())
    assert not result["verification_passed"]
    with pytest.raises(ValueError, match="Duplicate evidence"):
        report.summarize([baseline, baseline, repeated, candidate], plan())


def test_smoke_never_qualifies_and_tradeoffs_fail():
    baseline, repeated, candidate = _verified_arms()
    for key in ("all:attn_nn", "elite_top24:attn_nn"):
        candidate.scores[key]["mae"] -= 0.01
        candidate.scores[key]["rmse"] -= 0.01
    result = report.summarize([baseline, repeated, candidate], plan())
    assert result["verification_passed"] and result["comparisons"][0]["passes"]
    assert not result["development_qualified"] and not result["merge_authorized"]
    candidate.scores["elite_top24:attn_nn"]["rmse"] += 1
    assert not report.summarize([baseline, repeated, candidate], plan())["comparisons"][0]["passes"]


def test_archive_rejects_corruption_and_foreign_receipts(tmp_path):
    archive = report.Archive(tmp_path, "bucket", offline=True)
    prefix = "ab_runs/campaign/run/"
    body = b"verified bytes"
    digest = hashlib.sha256(body).hexdigest()
    key = prefix + digest + "-object.json"
    path = tmp_path / "objects" / key
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    receipt = {"uri": "s3://bucket/" + key, "sha256": digest, "bytes": len(body)}
    assert archive.receipt(receipt, prefix) == body
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        archive.receipt(receipt, prefix)
    receipt["uri"] = receipt["uri"].replace("campaign", "different")
    with pytest.raises(ValueError, match="outside declared"):
        archive.receipt(receipt, prefix)


def test_report_contains_three_seed_uncertainty_only_for_complete_development():
    base, repeat, candidate = _verified_arms()
    cells = []
    for position in ["RB", "WR", "TE"]:
        for origin in [2022, 2023]:
            for seed, delta in [(42, -0.01), (123, -0.02), (7, -0.03)]:
                for template in [base, repeat, candidate]:
                    value = deepcopy(template)
                    value.record.update(position=position, origin=origin, seed=seed)
                    value.record["execution"]["seed"] = seed
                    if value.record["variant"] == "count_precision":
                        for key in ["all:attn_nn", "elite_top24:attn_nn"]:
                            value.scores[key] = {
                                **value.scores[key],
                                "mae": value.scores[key]["mae"] + delta,
                                "rmse": value.scores[key]["rmse"] + delta,
                            }
                    cells.append(value)
    result = report.summarize(cells, plan("development"))
    assert result["verification_passed"] and result["development_qualified"]
    assert len(result["comparisons"]) == 6
    assert result["comparisons"][0]["deltas"]["all:mae"]["std"] == pytest.approx(0.01)
    assert len(result["comparisons"][0]["deltas"]["all:mae"]["ci95_seed_mean"]) == 2
    json.dumps(result, allow_nan=False)


def test_analyze_downloaded_run_and_reject_mixed_pins(tmp_path, monkeypatch):
    manifest_plan = plan()
    prefix = manifest_plan["runs"][0]["prefix"] + "/"
    archive = report.Archive(tmp_path, "bucket", offline=True)
    record, truth, data = fixture_record()

    def put(key, payload):
        path = tmp_path / "objects" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def receipt(payload, name):
        digest = hashlib.sha256(payload).hexdigest()
        key = prefix + "evidence/" + digest + "-" + name
        put(key, payload)
        return {"uri": "s3://bucket/" + key, "sha256": digest, "bytes": len(payload)}

    run = {
        "spec": "src.tuning.ab_audit_count_development",
        "image_sha": manifest_plan["source_sha"],
        "image_digest": manifest_plan["image_digest"],
        "variants": ["baseline", "baseline_rep", "count_precision"],
        "positions": ["WR"],
        "seeds": [42],
        "jobs": {"WR": "job"},
        "extra_env": {"FF_AUDIT_ORIGIN": "2022", "FF_AMP_DTYPE": "fp32"},
        "cuda_graph": "auto",
    }
    put(prefix + "run.json", json.dumps(run).encode())
    for arm in run["variants"]:
        cell = deepcopy(record)
        cell["variant"] = arm
        cell["rows"] = receipt(data["rows"], "rows.parquet")
        for trainer in cell["trainers"]:
            trainer["validation_raw"] = receipt(data["npz"], "validation.npz")
            trainer["checkpoint"] = receipt(data["state"], "checkpoint.pt")
        receipt(json.dumps(cell).encode(), "manifest.json")
        small = {
            "ok": True,
            "metrics": {
                **cell["metrics"],
                "repair": {"origin": 2022, "trainer_count": 2, "weekly_reference_available": 0},
            },
        }
        put(prefix + f"cells/WR-{arm}-42.json", json.dumps(small).encode())
    monkeypatch.setattr(
        archive,
        "keys",
        lambda prefix: [
            str(p.relative_to(tmp_path / "objects"))
            for p in (tmp_path / "objects" / prefix).rglob("*")
            if p.is_file()
        ],
    )
    monkeypatch.setattr(report, "hydrate_truth", lambda archive, plan: {"WR": truth})
    result = report.analyze(manifest_plan, archive)
    assert result["verification_passed"] and result["records"] == 3
    run["image_digest"] = "sha256:" + "9" * 64
    put(prefix + "run.json", json.dumps(run).encode())
    result = report.analyze(manifest_plan, archive)
    assert not result["verification_passed"]
    assert any("Mixed source/image" in error for error in result["errors"])

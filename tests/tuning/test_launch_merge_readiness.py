"""Submission and evidence gates without AWS calls or local model fitting."""

import hashlib
import io
import json
import subprocess
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from src.tuning import launch_merge_readiness as controller

pytestmark = pytest.mark.unit


@pytest.fixture
def campaign():
    return controller.Campaign(
        nn_image_sha="a" * 40,
        combined_image_sha="b" * 40,
        nn_data_prefix=f"{controller.CAMPAIGN_PREFIX}/data/nn",
        combined_data_prefix=f"{controller.CAMPAIGN_PREFIX}/data/qb",
        nn_data_release="c" * 64,
        combined_data_release="d" * 64,
        replay_sha256="e" * 64,
    )


@pytest.fixture
def aws():
    objects = {}
    s3, ecr = MagicMock(), MagicMock()
    s3.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(objects[kw["Key"]])}
    s3.list_objects_v2.side_effect = lambda **kw: {
        "Contents": [{"Key": key} for key in objects if key.startswith(kw["Prefix"])]
    }
    s3.get_paginator.return_value.paginate.side_effect = lambda **kw: [
        {"Contents": [{"Key": key} for key in objects if key.startswith(kw["Prefix"])]}
    ]

    def put(**kw):
        assert kw["IfNoneMatch"] == "*"
        assert kw["Key"] not in objects
        objects[kw["Key"]] = kw["Body"]

    s3.put_object.side_effect = put
    ecr.describe_images.return_value = {"imageDetails": [{"imageDigest": f"sha256:{'f' * 64}"}]}
    return s3, ecr, objects


def add_cell(objects, campaign, run, cell_key, *, mutate=None):
    pos, variant, seed = cell_key.split("-")
    identity = {"position": pos, "variant": variant, "seed": int(seed)}
    prefix = f"{controller.RESULTS_PREFIX}/{run.run_id}/readiness/{cell_key}/"
    evaluations = {}
    for regime in ("native", "pregame_replay") if pos == "QB" else ("native",):
        raw = f"{cell_key}/{regime}".encode()
        digest = hashlib.sha256(raw).hexdigest()
        key = f"{prefix}{regime}-{digest}.parquet"
        objects[key] = raw
        evaluations[regime] = {"n_rows": 3, "sha256": digest, "location": f"s3://bucket/{key}"}
    manifest = {
        **identity,
        "schema_version": 1,
        "image_sha": run.image_sha,
        "data_release": run.data_release,
        "evaluations": evaluations,
        "replay_input": {"sha256": campaign.replay_sha256},
        "cuda_capture": {
            "required": True,
            "enabled_gate": True,
            "full_step_enabled_gate": True,
            "events": [
                {
                    "family": family,
                    "device": "cuda:0",
                    "use_amp": False,
                    "returned_true": True,
                    "graph_present": True,
                    "capturable_loss": True,
                }
                for family in ("nn", "attn_nn")
            ],
        },
    }
    cell = {
        **identity,
        "ok": True,
        "provenance": {"git_sha": run.image_sha},
        "metrics": {
            "readiness": {
                "native_rows": 3,
                "pregame_rows": 3 if pos == "QB" else 0,
                "nn_full_step_capture": 1,
                "attn_nn_full_step_capture": 1,
                "required_cohorts_available": 4,
            }
        },
    }
    if mutate:
        mutate(cell, manifest)
    raw = json.dumps(manifest).encode()
    objects[f"{prefix}manifest-{hashlib.sha256(raw).hexdigest()}.json"] = raw
    objects[f"{controller.RESULTS_PREFIX}/{run.run_id}/cells/{cell_key}.json"] = json.dumps(
        cell
    ).encode()


def test_bounded_grid_and_launch_contract(campaign):
    smoke = controller.phase_runs(campaign, "run-1", "smoke")
    full = controller.phase_runs(campaign, "run-1", "full")
    assert [len(run.cell_keys) for run in smoke] == [2, 2]
    assert [len(run.cell_keys) for run in full] == [30, 6]
    assert len({run.run_id for run in [*smoke, *full]}) == 4
    assert full[0].positions == ("QB", "RB", "WR", "TE", "DST")
    assert full[1].positions == ("QB",)
    for run in [*smoke, *full]:
        cmd = controller.launch_command(campaign, run, wait_timeout=18000, attempt_timeout=10800)
        assert cmd[1:3] == ["-m", "src.tuning.launch_ab"]
        assert "--stacked-seeds" not in cmd
        assert cmd[cmd.index("--wait") + 1] == "true"
        assert cmd[cmd.index("--cuda-graph") + 1] == "auto"
        assert "FF_AMP_DTYPE=fp32" in cmd and "FF_AB_STACKED=0" in cmd
        assert run.image_sha in cmd and run.data_prefix in cmd


def test_dry_run_does_not_touch_aws_or_git(campaign, monkeypatch, capsys):
    monkeypatch.setattr(controller, "verify_sources", MagicMock(side_effect=AssertionError))
    controller.main(
        [
            "--config-json",
            json.dumps(asdict(campaign)),
            "--run-id",
            "example",
            "--phase",
            "full",
            "--dry-run",
        ]
    )
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(len(row["cells"]) for row in rows) == 36


@pytest.mark.parametrize(
    "field,value",
    [
        ("nn_image_sha", "main"),
        ("nn_data_release", "legacy"),
        ("replay_sha256", ""),
        ("combined_data_prefix", "data"),
        ("nn_data_prefix", f"{controller.CAMPAIGN_PREFIX}/data/../production"),
    ],
)
def test_rejects_unpinned_or_production_inputs(campaign, field, value):
    with pytest.raises(ValueError):
        controller.Campaign.from_json(json.dumps({**asdict(campaign), field: value}))


def test_binds_selected_release_in_launcher_environment(campaign, monkeypatch):
    monkeypatch.setenv("FF_DATA_RELEASE", "old")
    monkeypatch.setenv("FF_DATASET_ID", "old")
    monkeypatch.setenv("FF_DATA_FORMAT", "old")
    invoke = MagicMock()
    monkeypatch.setattr(controller.subprocess, "run", invoke)
    run = controller.phase_runs(campaign, "run", "smoke")[0]
    controller.launch_run(campaign, run, wait_timeout=18000, attempt_timeout=10800)
    env = invoke.call_args.kwargs["env"]
    assert env["FF_DATA_RELEASE"] == campaign.nn_data_release
    assert "FF_DATASET_ID" not in env and "FF_DATA_FORMAT" not in env
    assert invoke.call_args.kwargs["check"] is True


@pytest.mark.parametrize("kind", [0, 1])
def test_accepts_observed_capture_and_hashed_artifacts(campaign, aws, kind):
    s3, _, objects = aws
    run = controller.phase_runs(campaign, "run", "smoke")[kind]
    add_cell(objects, campaign, run, run.cell_keys[0])
    controller.verify_cell(s3, "bucket", campaign, run, run.cell_keys[0])


@pytest.mark.parametrize(
    "failure", ["cell", "image", "release", "capture", "amp", "replay", "cohort"]
)
def test_smoke_gate_rejects_invalid_evidence(campaign, aws, failure):
    s3, _, objects = aws
    run = controller.phase_runs(campaign, "run", "smoke")[1]

    def mutate(cell, manifest):
        if failure == "cell":
            cell["ok"] = False
        elif failure == "image":
            cell["provenance"]["git_sha"] = "wrong"
        elif failure == "release":
            manifest["data_release"] = "wrong"
        elif failure == "capture":
            manifest["cuda_capture"]["events"][0]["returned_true"] = False
        elif failure == "amp":
            manifest["cuda_capture"]["events"][0]["use_amp"] = True
        elif failure == "replay":
            manifest["replay_input"]["sha256"] = "wrong"
        elif failure == "cohort":
            cell["metrics"]["readiness"]["required_cohorts_available"] = 3

    add_cell(objects, campaign, run, run.cell_keys[0], mutate=mutate)
    with pytest.raises(ValueError):
        controller.verify_cell(s3, "bucket", campaign, run, run.cell_keys[0])


def test_frame_corruption_fails_gate(campaign, aws):
    s3, _, objects = aws
    run = controller.phase_runs(campaign, "run", "smoke")[0]
    add_cell(objects, campaign, run, run.cell_keys[0])
    key = next(key for key in objects if key.endswith(".parquet"))
    objects[key] = b"changed"
    with pytest.raises(ValueError, match="content hash mismatch"):
        controller.verify_cell(s3, "bucket", campaign, run, run.cell_keys[0])


def test_full_cannot_launch_without_completed_smoke(campaign, aws, monkeypatch):
    s3, ecr, _ = aws
    launch = MagicMock()
    monkeypatch.setattr(controller, "launch_run", launch)
    with pytest.raises(KeyError):
        controller.run_phase(
            campaign,
            phase="full",
            run_id="run",
            s3=s3,
            ecr=ecr,
            bucket="bucket",
            spec_sha256="f" * 64,
            wait_timeout=10,
            attempt_timeout=60,
        )
    launch.assert_not_called()


def test_failure_or_timeout_never_writes_success_receipt(campaign, aws, monkeypatch):
    s3, ecr, objects = aws
    monkeypatch.setattr(
        controller,
        "launch_run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, "launch_ab")),
    )
    verify = MagicMock()
    monkeypatch.setattr(controller, "verify_phase", verify)
    with pytest.raises(subprocess.CalledProcessError):
        controller.run_phase(
            campaign,
            phase="smoke",
            run_id="run",
            s3=s3,
            ecr=ecr,
            bucket="bucket",
            spec_sha256="f" * 64,
            wait_timeout=10,
            attempt_timeout=60,
        )
    verify.assert_not_called()
    assert not any(key.endswith("smoke-complete.json") for key in objects)


def test_full_rejects_changed_ecr_digest(campaign, aws, monkeypatch):
    s3, ecr, objects = aws
    request = {
        "campaign": asdict(campaign),
        "spec_sha256": "f" * 64,
        "image_digests": {"nn": "sha256:" + "0" * 64, "combined": "sha256:" + "0" * 64},
    }
    prefix = f"{controller.RESULTS_PREFIX}/controller/run/"
    objects[prefix + "request.json"] = json.dumps(request).encode()
    objects[prefix + "smoke-complete.json"] = json.dumps({"verified_cells": 4, **request}).encode()
    launch = MagicMock()
    monkeypatch.setattr(controller, "launch_run", launch)
    with pytest.raises(ValueError, match="digest"):
        controller.run_phase(
            campaign,
            phase="full",
            run_id="run",
            s3=s3,
            ecr=ecr,
            bucket="bucket",
            spec_sha256="f" * 64,
            wait_timeout=10,
            attempt_timeout=60,
        )
    launch.assert_not_called()


def test_smoke_then_full_verifies_and_records_all_40_cells(campaign, aws, monkeypatch):
    s3, ecr, objects = aws
    launched = []

    def launch(campaign, run, **kwargs):
        launched.append(run)
        for key in run.cell_keys:
            add_cell(objects, campaign, run, key)

    monkeypatch.setattr(controller, "launch_run", launch)
    for phase in ("smoke", "full"):
        controller.run_phase(
            campaign,
            phase=phase,
            run_id="run",
            s3=s3,
            ecr=ecr,
            bucket="bucket",
            spec_sha256="f" * 64,
            wait_timeout=10,
            attempt_timeout=60,
        )
    assert sorted(len(run.cell_keys) for run in launched) == [2, 2, 6, 30]
    prefix = f"{controller.RESULTS_PREFIX}/controller/run/"
    assert json.loads(objects[prefix + "smoke-complete.json"])["verified_cells"] == 4
    assert json.loads(objects[prefix + "full-complete.json"])["verified_cells"] == 36


def test_existing_namespace_is_not_reused(campaign, aws, monkeypatch):
    s3, ecr, objects = aws
    run = controller.phase_runs(campaign, "run", "smoke")[1]
    objects[f"{controller.RESULTS_PREFIX}/{run.run_id}/run.json"] = b"{}"
    launch = MagicMock()
    monkeypatch.setattr(controller, "launch_run", launch)
    with pytest.raises(ValueError, match="namespace already exists"):
        controller.run_phase(
            campaign,
            phase="smoke",
            run_id="run",
            s3=s3,
            ecr=ecr,
            bucket="bucket",
            spec_sha256="f" * 64,
            wait_timeout=10,
            attempt_timeout=60,
        )
    launch.assert_not_called()


def test_mismatched_spec_source_fails(campaign, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / controller.SPEC_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"current spec")
    monkeypatch.setattr(
        controller.subprocess, "run", MagicMock(return_value=MagicMock(stdout=b"old spec"))
    )
    with pytest.raises(ValueError, match="differs"):
        controller.verify_sources(campaign)

"""Execute EC2 wrapper rendering/dispatch with local command fakes only."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.scripts.ec2_wrapper import render_update

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _workflow():
    return yaml.safe_load((ROOT / ".github/workflows/train-ec2.yml").read_text())


def _step(name):
    return next(step for step in _workflow()["jobs"]["train"]["steps"] if step["name"] == name)


def _executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


def test_ec2_selects_and_registers_exact_image_before_instance_compute():
    doc = _workflow()
    events = doc.get("on", doc.get(True))
    assert events["workflow_dispatch"]["inputs"]["image_sha"]["required"] is True
    steps = doc["jobs"]["train"]["steps"]
    image = _step("Resolve immutable training image")
    assert steps.index(image) < steps.index(_step("Start instance (no-op if already running)"))
    assert _step("Checkout repo")["with"]["fetch-depth"] == 0
    assert "src.scripts.resolve_training_image ec2" in image["run"]
    registration = _step("Register selected training source")
    data = _step("Wait for compatible published training data")
    start = _step("Start instance (no-op if already running)")
    assert steps.index(image) < steps.index(data) < steps.index(start)
    assert steps.index(image) < steps.index(registration) < steps.index(start)
    assert "register_training_source" in registration["run"]
    for name in (
        "Run training for all positions (sequential)",
        "Verify exact EC2 run artifacts",
        "Append EC2 run to benchmark_history/",
        "Build and publish serving cache before rollout",
    ):
        env = _step(name)["env"]
        assert env["FF_TRAIN_GIT_SHA"] == "${{ steps.image.outputs.image_sha }}"
        assert env["FF_TRAIN_IMAGE_ID"] == "${{ steps.image.outputs.image_uri }}"
        assert env["FF_LEGACY_RUN_ID"] == "ec2:${{ github.run_id }}:${{ github.run_attempt }}"
    assert (
        _step("Build and publish serving cache before rollout")["env"]["FF_BUILD_POSITIONS"]
        == "${{ needs.detect.outputs.positions }}"
    )


def test_all_ec2_workflow_shell_blocks_parse_without_execution():
    for step in _workflow()["jobs"]["train"]["steps"]:
        if "run" in step:
            script = re.sub(r"\$\{\{.*?\}\}", "expression", step["run"])
            result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
            assert result.returncode == 0, (step["name"], result.stderr)


@pytest.mark.parametrize("requested", ["a" * 40, "", "abc", "$(touch injected)"])
def test_image_resolution_delegates_literal_source_to_shared_resolver(tmp_path, requested):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    trace = tmp_path / "trace.jsonl"
    _executable(
        binaries / "python",
        "import json,os,sys\n"
        "with open(os.environ['TRACE'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n",
    )
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "TRACE": str(trace),
        "REQUESTED_IMAGE_SHA": requested,
    }
    result = subprocess.run(
        ["bash", "-c", _step("Resolve immutable training image")["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in trace.read_text().splitlines()] == [
        ["-m", "src.scripts.resolve_training_image", "ec2", "--sha", requested]
    ]
    assert not (tmp_path / "injected").exists()


def _render(tmp_path):
    target = tmp_path / "ff-train"
    update = render_update(
        "us-east-1",
        "ff-predictor-training",
        target=target,
        cache_dir=tmp_path / "cache",
        cache_owner=None,
    )
    result = subprocess.run(["sh"], input=update, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return target, update


def test_remote_dispatch_preserves_selected_identity_for_every_position(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    _executable(binaries / "aws", "print('ssm-test-command')\n")
    trace = tmp_path / "runs.jsonl"
    train = tmp_path / "ff-train"
    names = (
        "FF_TRAIN_GIT_SHA",
        "FF_TRAIN_IMAGE",
        "FF_TRAIN_IMAGE_ID",
        "FF_LEGACY_RUN_ID",
        "FF_DATA_RELEASE",
        "FF_DATASET_ID",
        "FF_DATA_FORMAT",
        "FF_MODEL_S3_PREFIX",
    )
    _executable(
        train,
        "import json,os,sys\n"
        + f"record={{'args':sys.argv[1:],'env':{{key:os.environ[key] for key in {names!r}}}}}\n"
        + "with open(os.environ['TRACE'],'a') as stream: stream.write(json.dumps(record)+'\\n')\n",
    )
    image = "registry/ff-training@sha256:" + "b" * 64
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "TRACE": str(trace),
        "AWS_REGION": "us-east-1",
        "POSITIONS": "QB RB",
        "SEED": "42",
        "FF_TRAIN_GIT_SHA": "a" * 40,
        "FF_TRAIN_IMAGE": image,
        "FF_TRAIN_IMAGE_ID": image,
        "FF_LEGACY_RUN_ID": "ec2:77:3",
        "FF_DATA_RELEASE": "c" * 64,
        "FF_DATASET_ID": "c" * 64,
        "FF_DATA_FORMAT": "data-release-v1",
        "FF_MODEL_S3_PREFIX": "sandbox prefix",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    script = _step("Run training for all positions (sequential)")["run"]
    script = re.sub(r"\$\{\{.*?\}\}", "instance", script)
    script = script.replace("/tmp/ssm-params.json", str(tmp_path / "params.json"))
    result = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    params = json.loads((tmp_path / "params.json").read_text())
    assert params["executionTimeout"] == ["7800"]
    command = params["commands"][0].replace("/usr/local/bin/ff-train", str(train))
    result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [record["args"] for record in records] == [["QB", "42"], ["RB", "42"]]
    assert all(record["env"] == {name: env[name] for name in names} for record in records)


def test_warm_host_migration_is_atomic_idempotent_and_shared_with_bootstrap(tmp_path):
    target, update = _render(tmp_path)
    assert update.startswith("set -eu\n")
    content, inode = target.read_bytes(), target.stat().st_ino
    assert subprocess.run(["bash", "-n", str(target)], capture_output=True).returncode == 0
    assert b"FF_TRAIN_IMAGE_ID" in content and b"FF_LEGACY_RUN_ID" in content
    assert b"imageTag=latest" not in content and b'IMAGE="' + b":latest" not in content
    result = subprocess.run(["sh"], input=update, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == content and target.stat().st_ino == inode
    assert "already current" in result.stdout
    assert not list(tmp_path.glob(".ff-train.*"))


@pytest.mark.parametrize(
    "image,sha", [("repository:latest", "a" * 40), ("repository@sha256:" + "b" * 64, "abc")]
)
def test_wrapper_rejects_mutable_or_unidentified_images_before_host_actions(tmp_path, image, sha):
    target, _ = _render(tmp_path)
    env = {
        **os.environ,
        "FF_TRAIN_IMAGE": image,
        "FF_TRAIN_GIT_SHA": sha,
        "FF_LEGACY_RUN_ID": "ec2:77:3",
    }
    result = subprocess.run(["bash", str(target), "QB"], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "requires a full source SHA and digest-pinned image" in result.stderr


@pytest.mark.parametrize("preflight_fails", [False, True])
def test_wrapper_passes_exact_image_and_intent_and_checks_source_before_gpu(
    tmp_path, preflight_fails
):
    target, _ = _render(tmp_path)
    # Only filesystem locations change for this local execution; no root/AWS/GPU access.
    body = target.read_text().replace("/opt/ff", str(tmp_path / "host"))
    body = body.replace("/var/lock/ff-train.lock", str(tmp_path / "run.lock"))
    body = body.replace("/var/log/ff-train", str(tmp_path / "logs"))
    target.write_text(body)
    (tmp_path / "logs").mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    trace = tmp_path / "docker.jsonl"
    _executable(
        binaries / "docker",
        "import json,os,sys\nargs=sys.argv[1:]\nwith open(os.environ['TRACE'],'a') as f: f.write(json.dumps(args)+'\\n')\nif '--entrypoint' in args and os.environ['FAIL_PREFLIGHT']=='1': raise SystemExit(1)\n",
    )
    _executable(binaries / "flock", "")
    _executable(binaries / "date", "print('2026-09-10T12:00:00Z')\n")
    image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ff-training@sha256:" + "b" * 64
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "TRACE": str(trace),
        "FF_TRAIN_IMAGE": image,
        "FF_TRAIN_GIT_SHA": "a" * 40,
        "FF_LEGACY_RUN_ID": "ec2:77:3",
        "FF_MODEL_S3_PREFIX": "sandbox-models",
        "FF_DATA_RELEASE": "c" * 64,
        "FF_DATASET_ID": "c" * 64,
        "FF_DATA_FORMAT": "data-release-v1",
        "FAIL_PREFLIGHT": "1" if preflight_fails else "0",
    }
    result = subprocess.run(
        ["bash", str(target), "QB", "42"], env=env, capture_output=True, text=True
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    preflight = next(call for call in calls if "--entrypoint" in call)
    assert image in preflight and "FF_TRAIN_GIT_SHA=" + "a" * 40 in preflight
    gpu = [call for call in calls if "--gpus" in call]
    if preflight_fails:
        assert result.returncode != 0 and gpu == []
    else:
        assert result.returncode == 0, result.stderr
        assert len(gpu) == 1 and calls.index(preflight) < calls.index(gpu[0])
        for value in (
            image,
            "FF_TRAIN_IMAGE_ID=" + image,
            "FF_LEGACY_RUN_ID=ec2:77:3",
            "FF_DATA_RELEASE=" + "c" * 64,
            "FF_DATASET_ID=" + "c" * 64,
            "FF_DATA_FORMAT=data-release-v1",
            "FF_TRAIN_GIT_SHA=" + "a" * 40,
            "FF_MODEL_S3_PREFIX=sandbox-models",
        ):
            assert value in gpu[0]
        assert gpu[0][-4:] == ["--position", "QB", "--seed", "42"]

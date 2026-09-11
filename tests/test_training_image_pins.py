"""Exercise generated/warm-host launch scripts and workflow image forwarding."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ff-training@sha256:" + "1" * 64
pytestmark = pytest.mark.unit


def steps(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())["jobs"]["train"]["steps"]


def generated_script(tmp_path):
    source = (ROOT / "infra/ec2/user-data.sh").read_text()
    source = source[source.index("cat > /usr/local/bin/ff-train <<EOF") :]
    source = source[: source.index("\nEOF") + len("\nEOF")]
    target = tmp_path / "ff-train"
    source = source.replace("/usr/local/bin/ff-train", str(target))
    subprocess.run(
        ["bash"],
        input=source,
        text=True,
        check=True,
        env={**os.environ, "REGION": "us-east-1", "BUCKET": "bucket"},
    )
    body = target.read_text().replace("/opt/ff", str(tmp_path / "opt"))
    body = body.replace("/var/lock", str(tmp_path / "locks"))
    body = body.replace("/var/log/ff-train", str(tmp_path / "logs"))
    (tmp_path / "locks").mkdir()
    (tmp_path / "logs").mkdir()
    target.write_text(body)
    return target


def fake_docker(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir()
    docker = bins / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['DOCKER_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(1 if sys.argv[1:3]==['image','inspect'] else 0)\n"
    )
    docker.chmod(0o755)
    lock = bins / "flock"
    lock.write_text("#!/bin/sh\nexit 0\n")
    lock.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bins}:{os.environ['PATH']}",
        "DOCKER_LOG": str(tmp_path / "docker.log"),
    }


def test_generated_ec2_trainer_runs_resolved_digest_if_latest_moves(tmp_path):
    script = generated_script(tmp_path)
    env = fake_docker(tmp_path)
    env.update(
        FF_TRAIN_IMAGE=IMAGE,
        FF_TRAIN_GIT_SHA=SHA,
        FF_DATA_RELEASE="c" * 64,
        FF_DATASET_ID="c" * 64,
        FF_DATA_FORMAT="data-release-v1",
        FF_LEGACY_RUN_ID="ec2:42:1",
    )
    subprocess.run(["bash", str(script), "DST", "42"], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in (tmp_path / "docker.log").read_text().splitlines()]
    assert ["pull", IMAGE] in calls
    run = next(call for call in calls if "--gpus" in call)
    assert IMAGE in run and not any(":latest" in arg for arg in run)
    assert f"FF_TRAIN_GIT_SHA={SHA}" in run
    assert "FF_DATA_RELEASE=" + "c" * 64 in run
    assert "FF_DATASET_ID=" + "c" * 64 in run
    assert "FF_DATA_FORMAT=data-release-v1" in run
    assert run[-4:] == ["--position", "DST", "--seed", "42"]


@pytest.mark.parametrize("image", ["", "repository:latest"])
def test_ec2_trainer_without_digest_pin_fails_before_docker(tmp_path, image):
    script = generated_script(tmp_path)
    env = fake_docker(tmp_path)
    env.update(FF_TRAIN_IMAGE=image, FF_TRAIN_GIT_SHA=SHA)
    result = subprocess.run(["bash", str(script), "DST"], env=env, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / "docker.log").exists()


@pytest.mark.parametrize("old_source_override", [False, True])
def test_warm_host_upgrade_honors_pin_and_is_idempotent(tmp_path, old_source_override):
    from src.scripts.ec2_wrapper import render_update

    script = tmp_path / "ff-train"
    old_retag = (
        'if [ -n "${FF_TRAIN_GIT_SHA:-}" ]; then\n  IMAGE="${IMAGE%:*}:$FF_TRAIN_GIT_SHA"\nfi\n'
        if old_source_override
        else ""
    )
    script.write_text(
        '#!/bin/bash\nIMAGE="repository:latest"\n' + old_retag + 'printf "%s" "$IMAGE"\n'
    )
    patch = render_update(
        "us-east-1", "bucket", target=script, cache_dir=tmp_path / "cache", cache_owner=None
    )
    subprocess.run(["sh"], input=patch, text=True, check=True)
    first = script.read_bytes()
    subprocess.run(["sh"], input=patch, text=True, check=True)
    assert script.read_bytes() == first
    assert b"IMAGE%:*" not in first
    assert b"FF_DATASET_ID" in first and b"FF_TRAIN_IMAGE_ID" in first
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "workflow,resolver_id", [("train-ec2.yml", "image"), ("train-batch.yml", "revision")]
)
def test_workflow_gates_actual_resolved_image_and_forwards_same_sha(workflow, resolver_id):
    values = steps(workflow)
    resolver = next(s for s in values if s.get("id") == resolver_id)
    gate_id = "data-release" if workflow == "train-ec2.yml" else "build_plan"
    gate = next(s for s in values if s.get("id") == gate_id)
    train = next(s for s in values if s.get("id") == "train")
    assert values.index(resolver) < values.index(gate) < values.index(train)
    if workflow == "train-ec2.yml":
        expected = "${{ steps.image.outputs.image_sha }}"
        assert gate["env"]["HEAD_SHA"] == train["env"]["FF_TRAIN_GIT_SHA"] == expected
        assert train["env"]["FF_TRAIN_IMAGE"] == "${{ steps.image.outputs.image_uri }}"
        assert "shlex.quote(os.environ[name])" in train["run"]
    else:
        assert gate["env"]["CODE_SHA"] == "${{ steps.revision.outputs.image_sha }}"
        assert train["env"]["FF_TRAIN_GIT_SHA"] == "${{ steps.build_plan.outputs.git_sha }}"
        assert train["env"]["FF_BUILD_PLAN_ID"] == "${{ steps.build_plan.outputs.build_plan_id }}"
        assert "using bare CPU job definition" not in train["run"]

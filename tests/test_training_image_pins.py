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
    env.update(FF_TRAIN_IMAGE=IMAGE, FF_TRAIN_GIT_SHA=SHA, FF_DATA_RELEASE="release-a")
    subprocess.run(["bash", str(script), "DST", "42"], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in (tmp_path / "docker.log").read_text().splitlines()]
    assert ["pull", IMAGE] in calls
    run = next(call for call in calls if call[0] == "run")
    assert IMAGE in run and not any(":latest" in arg for arg in run)
    assert f"FF_TRAIN_GIT_SHA={SHA}" in run
    assert "FF_DATA_RELEASE=release-a" in run
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
    patch_step = next(
        s for s in steps("train-ec2.yml") if s["name"] == "Ensure feature-cache mount in ff-train"
    )
    patch = patch_step["run"][patch_step["run"].index("if ! grep -q '^IMAGE=") :].split(
        "\nBASH", 1
    )[0]
    script = tmp_path / "ff-train"
    old_retag = (
        'if [ -n "${FF_TRAIN_GIT_SHA:-}" ]; then\n  IMAGE="${IMAGE%:*}:$FF_TRAIN_GIT_SHA"\nfi\n'
        if old_source_override
        else ""
    )
    script.write_text(
        '#!/bin/bash\nIMAGE="repository:latest"\n' + old_retag + 'printf "%s" "$IMAGE"\n'
    )
    patch = patch.replace("/usr/local/bin/ff-train", str(script))
    subprocess.run(["bash"], input=patch, text=True, check=True)
    first = script.read_text()
    subprocess.run(["bash"], input=patch, text=True, check=True)
    assert script.read_text() == first
    assert "IMAGE%:*" not in first
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "FF_TRAIN_IMAGE": IMAGE, "FF_TRAIN_GIT_SHA": SHA},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == IMAGE


@pytest.mark.parametrize(
    "workflow,resolver_id", [("train-ec2.yml", "image"), ("train-batch.yml", "revision")]
)
def test_workflow_gates_actual_resolved_image_and_forwards_same_sha(workflow, resolver_id):
    values = steps(workflow)
    resolver = next(s for s in values if s.get("id") == resolver_id)
    gate = next(s for s in values if s.get("id") == "data-release")
    train = next(s for s in values if s.get("id") == "train")
    assert values.index(resolver) < values.index(gate) < values.index(train)
    expected = "${{ steps." + resolver_id + ".outputs.image_sha }}"
    assert gate["env"]["HEAD_SHA"] == train["env"]["FF_TRAIN_GIT_SHA"] == expected
    if workflow == "train-ec2.yml":
        assert train["env"]["FF_TRAIN_IMAGE"] == "${{ steps.image.outputs.image_uri }}"
        assert "FF_TRAIN_IMAGE='$FF_TRAIN_IMAGE' /usr/local/bin/ff-train" in train["run"]
    else:
        assert (
            train["env"]["FF_JOB_DEFINITION_REVISION"] == "${{ steps.revision.outputs.revision }}"
        )
        assert "using bare CPU job definition" not in train["run"]

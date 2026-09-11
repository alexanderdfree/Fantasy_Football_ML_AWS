"""Exercise the real local image entrypoint with isolated CLI transports."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_repo(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "src/batch"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "src/batch/build_and_push.sh", scripts / "build_and_push.sh")
    (repo / "src/model.py").write_text("MODEL = 1\n")
    for args in (
        ["init", "--initial-branch=main"],
        ["add", "src"],
        ["-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-m", "source"],
    ):
        subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text('#!/bin/sh\nif [ "$1" = sts ]; then echo 123; else echo test-password; fi\n')
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPTURE_DOCKER"\n'
        'if [ "$1" = login ]; then cat > "$CAPTURE_DOCKER_PASSWORD"; fi\n'
    )
    aws.chmod(0o755)
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "CAPTURE_DOCKER": str(tmp_path / "docker-args"),
        "CAPTURE_DOCKER_PASSWORD": str(tmp_path / "docker-password"),
        "AWS_REGION": "us-east-1",
        "ECR_REPO": "ff-training",
        "USE_PULL_THROUGH": "0",
        "PULL_THROUGH_PREFIX": "",
    }
    env.pop("IMAGE_TAG", None)
    return repo, sha, env


def test_image_entrypoint_records_actual_source_and_uses_identifiable_default_tag(build_repo):
    repo, sha, env = build_repo
    result = subprocess.run(
        ["bash", "src/batch/build_and_push.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    calls = Path(env["CAPTURE_DOCKER"]).read_text().splitlines()
    assert f"TRAIN_GIT_SHA={sha}" in calls
    assert f"123.dkr.ecr.us-east-1.amazonaws.com/ff-training:{sha}" in calls
    assert not any(":latest" in argument for argument in calls)
    assert Path(env["CAPTURE_DOCKER_PASSWORD"]).read_text() == "test-password\n"


def test_dirty_source_fails_before_aws_or_docker(build_repo):
    repo, _, env = build_repo
    (repo / "src/model.py").write_text("MODEL = 2\n")
    result = subprocess.run(
        ["bash", "src/batch/build_and_push.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "commit source changes" in result.stderr
    assert not Path(env["CAPTURE_DOCKER"]).exists()


def test_docker_bakes_and_validates_the_source_identity(tmp_path):
    run = next(
        line.removeprefix("RUN ")
        for line in (ROOT / "src/batch/Dockerfile.train").read_text().splitlines()
        if line.startswith("RUN ") and "Path('.training-source-sha')" in line
    )
    env = {**os.environ, "TRAIN_GIT_SHA": "a" * 40}
    result = subprocess.run(["sh", "-c", run], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / ".training-source-sha").read_text() == "a" * 40 + "\n"
    (tmp_path / ".training-source-sha").unlink()
    env["TRAIN_GIT_SHA"] = "latest"
    result = subprocess.run(["sh", "-c", run], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / ".training-source-sha").exists()

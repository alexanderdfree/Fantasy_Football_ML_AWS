"""CI image metadata preserves provenance and branch publication boundaries."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _steps(workflow, job):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    return document["jobs"][job]["steps"]


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/codex/test-build"])
@pytest.mark.parametrize("pull_through", [True, False])
@pytest.mark.parametrize("cache_hit", ["true", "false", ""])
def test_training_build_pins_checkout_and_keeps_branch_off_latest(
    tmp_path, ref, pull_through, cache_hit
):
    steps = _steps("batch-image.yml", "build-and-push")
    build = next(step for step in steps if step.get("id") == "build")
    register = next(
        step for step in steps if step.get("name") == "Register new job definition revision"
    )
    assert steps.index(build) < steps.index(register)
    assert register["env"]["IMAGE_URI"] == "${{ steps.build.outputs.image_uri }}"
    assert "github.ref == 'refs/heads/main'" in register["if"]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "source",
    )
    sha = git("rev-parse", "HEAD")
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    (bin_path / "aws").write_text("#!/bin/sh\necho 1\n" if pull_through else "#!/bin/sh\nexit 1\n")
    (bin_path / "docker").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'with open(os.environ["BUILD_ARGS"], "w") as stream: json.dump(sys.argv[1:], stream)\n'
    )
    for path in bin_path.iterdir():
        path.chmod(0o755)
    output = tmp_path / "outputs"
    captured = tmp_path / "build-args.json"
    env = dict(
        os.environ,
        PATH=f"{bin_path}:{os.environ['PATH']}",
        ECR_REGISTRY="registry.example",
        ECR_REPOSITORY="training",
        IMAGE_TAG="b" * 40,
        TRAIN_GIT_SHA="not-the-checkout",
        GITHUB_REF=ref,
        GITHUB_OUTPUT=str(output),
        BUILD_ARGS=str(captured),
        AWS_REGION="us-east-1",
        CACHE_HIT=cache_hit,
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", build["run"]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(captured.read_text())
    assert args[:2] == ["buildx", "build"]
    assert f"TRAIN_GIT_SHA={sha}" in args
    assert "not-the-checkout" not in str(args)
    assert ("--build-arg=PULL_THROUGH_PREFIX=registry.example/dockerhub/" in args) is pull_through
    tags = [args[i + 1] for i, value in enumerate(args) if value == "--tag"]
    expected = ["registry.example/training:" + "b" * 40]
    if ref == "refs/heads/main":
        expected.append("registry.example/training:latest")
    assert tags == expected
    assert output.read_text().strip() == "image_uri=" + expected[0]
    assert args.count("--cache-from") == 1
    assert args[args.index("--cache-from") + 1] == "type=local,src=/tmp/.buildx-cache"
    assert args.count("--cache-to") == (0 if cache_hit == "true" else 1)
    if cache_hit != "true":
        assert (
            args[args.index("--cache-to") + 1] == "type=local,dest=/tmp/.buildx-cache-new,mode=max"
        )


def test_image_archives_have_separate_keys_and_no_cross_image_restore():
    prefixes = set()
    for workflow, job, image in [
        ("batch-image.yml", "build-and-push", "training"),
        ("deploy.yml", "deploy", "serving"),
    ]:
        steps = _steps(workflow, job)
        caches = [step for step in steps if step.get("uses", "").startswith("actions/cache")]
        assert len(caches) == 1
        prefix = "buildx-" + image + "-${{ runner.os }}-${{ runner.arch }}-"
        assert caches[0]["with"]["key"].startswith(prefix)
        assert caches[0]["with"]["restore-keys"].strip() == prefix
        assert caches[0]["id"] == "buildx-cache"
        prefixes.add(prefix)
        build = next(step for step in steps if "docker buildx build" in step.get("run", ""))
        assert build["run"].count("--cache-from") == build["run"].count("--cache-to") == 1
        assert "type=gha" not in build["run"]
        assert build["env"]["CACHE_HIT"] == "${{ steps.buildx-cache.outputs.cache-hit }}"
        move = next(step for step in steps if step.get("name") == "Move BuildKit cache")
        assert move["if"] == "steps.buildx-cache.outputs.cache-hit != 'true'"
    assert len(prefixes) == 2


@pytest.mark.parametrize("cache_hit", ["true", "false", ""])
def test_serving_exact_hit_skips_only_discarded_cache_export(tmp_path, cache_hit):
    build = next(
        step
        for step in _steps("deploy.yml", "deploy")
        if step.get("name") == "Build, tag, push image"
    )
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    (bin_path / "docker").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'with open(os.environ["BUILD_ARGS"], "w") as stream: json.dump(sys.argv[1:], stream)\n'
    )
    (bin_path / "docker").chmod(0o755)
    captured = tmp_path / "args.json"
    env = dict(
        os.environ,
        PATH=f"{bin_path}:{os.environ['PATH']}",
        BUILD_ARGS=str(captured),
        ECR_REGISTRY="registry.example",
        ECR_REPOSITORY="serving",
        IMAGE_TAG="a" * 40,
        CACHE_HIT=cache_hit,
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", build["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(captured.read_text())
    assert args[args.index("--platform") + 1] == "linux/arm64"
    assert "--push" in args
    assert args[args.index("--cache-from") + 1] == "type=local,src=/tmp/.buildx-cache"
    assert args.count("--cache-to") == (0 if cache_hit == "true" else 1)
    assert [args[i + 1] for i, value in enumerate(args) if value == "-t"] == [
        "registry.example/serving:" + "a" * 40,
        "registry.example/serving:latest",
    ]

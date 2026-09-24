"""Serving cache transport preserves the existing build/publication arguments."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def serving_steps():
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    return workflow["jobs"]["deploy"]["steps"]


def test_serving_archive_key_and_restore_prefix_are_image_and_architecture_scoped():
    caches = [step for step in serving_steps() if step.get("uses", "").startswith("actions/cache")]
    assert len(caches) == 1
    options = caches[0]["with"]
    prefix = "buildx-serving-${{ runner.os }}-${{ runner.arch }}-"
    assert options["key"] == prefix + "${{ hashFiles('requirements-serving.txt', 'Dockerfile') }}"
    assert options["restore-keys"].strip() == prefix
    assert options["path"] == "/tmp/.buildx-cache"


def test_serving_build_keeps_platform_tags_context_and_push_with_one_cache_backend(tmp_path):
    build = next(step for step in serving_steps() if step.get("name") == "Build, tag, push image")
    assert build["env"]["IMAGE_TAG"] == "${{ github.sha }}"
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'with open(os.environ["CAPTURED_BUILD_ARGS"], "w") as stream:\n'
        "    json.dump(sys.argv[1:], stream)\n"
    )
    docker.chmod(0o755)
    captured = tmp_path / "build-args.json"
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "ECR_REGISTRY": "registry.example",
        "ECR_REPOSITORY": "serving",
        "IMAGE_TAG": "a" * 40,
        "CAPTURED_BUILD_ARGS": str(captured),
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", build["run"]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(captured.read_text()) == [
        "buildx",
        "build",
        "--platform",
        "linux/arm64",
        "--provenance=false",
        "--cache-from",
        "type=local,src=/tmp/.buildx-cache",
        "--cache-to",
        "type=local,dest=/tmp/.buildx-cache-new,mode=max",
        "-t",
        "registry.example/serving:" + "a" * 40,
        "-t",
        "registry.example/serving:latest",
        "--push",
        ".",
    ]

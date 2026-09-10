"""The local build entrypoint must stamp the actual source it copies."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dirty", [False, True])
def test_local_build_stamps_full_source_sha_before_aws_calls(tmp_path, dirty):
    repo = tmp_path / "repo"
    source = repo / "src/batch"
    source.mkdir(parents=True)
    script = Path(__file__).resolve().parents[2] / "src/batch/build_and_push.sh"
    shutil.copy2(script, source / script.name)
    (source / "Dockerfile.train").write_text("FROM scratch\n")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("add", "src")
    git("commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD")
    if dirty:
        (source / "uncommitted.py").write_text("uncommitted source\n")

    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    (bin_path / "aws").write_text(
        '#!/bin/sh\ncase "$1" in sts) echo 123456789012;; ecr) echo placeholder;; esac\n'
    )
    (bin_path / "docker").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'if sys.argv[1] == "login": sys.stdin.read()\n'
        'with open(os.environ["BUILD_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
    )
    for file in bin_path.iterdir():
        file.chmod(0o755)
    log = tmp_path / "docker.jsonl"
    env = dict(
        os.environ,
        PATH=f"{bin_path}:{os.environ['PATH']}",
        BUILD_LOG=str(log),
        TRAIN_GIT_SHA="not-the-source",
        IMAGE_TAG="preview",
    )
    result = subprocess.run(
        ["bash", str(source / script.name)], cwd=tmp_path, env=env, text=True, capture_output=True
    )
    if dirty:
        assert result.returncode == 1
        assert "commit source changes" in result.stderr
        assert not log.exists()
    else:
        assert result.returncode == 0, result.stderr
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        build = next(call for call in calls if call[0] == "build")
        assert f"TRAIN_GIT_SHA={sha}" in build
        assert any(arg.endswith(":preview") for arg in build)
        assert "not-the-source" not in str(calls)

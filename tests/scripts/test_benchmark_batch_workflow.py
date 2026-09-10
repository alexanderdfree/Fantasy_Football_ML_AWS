"""Execute the isolated benchmark's input boundary with local command stubs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/benchmark-batch.yml"


def _step(name):
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["benchmark"]["steps"]
    return next(step for step in steps if step["name"] == name)


def _run(step, tmp_path, **env):
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_SHA": "a" * 40,
            "S3_BUCKET": "unused-bucket",
            "AWS_REGION": "us-east-1",
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
            **env,
        },
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("sha", ["", "abc1234", "ABC1234", "a" * 40])
def test_resolve_image_accepts_documented_sha_tags(tmp_path, sha):
    result = _run(_step("Resolve image SHA + experiment prefix"), tmp_path, INPUT_SHA=sha)
    expected = sha or "a" * 40
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs").read_text().splitlines() == [
        f"sha={expected}",
        f"prefix=experiments/benchmark/{expected}/models",
    ]


@pytest.mark.parametrize(
    "sha",
    [
        "abcdef",
        "a" * 41,
        "main",
        "abcdefg",
        "abc1234\nsha=override",
        "abc1234\n",
        "../abc1234",
        "x', None)); print('injected');#",
        "$(touch shell-marker)",
        '"; touch shell-marker; #',
    ],
)
def test_resolve_image_rejects_invalid_input_before_publishing_outputs(tmp_path, sha):
    result = _run(_step("Resolve image SHA + experiment prefix"), tmp_path, INPUT_SHA=sha)
    assert result.returncode != 0
    assert "Invalid image_sha" in result.stderr
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "shell-marker").exists()


def _command(tmp_path, name, source):
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return str(path.parent) + os.pathsep + os.environ["PATH"]


def test_job_revision_resolver_passes_sha_as_data(tmp_path):
    """Even a bypassed input check cannot turn the resolver's SHA into Python code."""
    _command(tmp_path, "aws", "#!/usr/bin/env bash\nexit 1\n")
    path = _command(
        tmp_path,
        "python",
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, types\n"
        "boto3 = types.ModuleType('boto3')\n"
        "boto3.client = lambda *args, **kwargs: None\n"
        "resolver = types.ModuleType('src.tuning.launch_ab')\n"
        "def resolve(sha, client):\n"
        "    pathlib.Path(os.environ['SEEN_SHA']).write_text(json.dumps(sha))\n"
        "    return 'ff-ab-job:8'\n"
        "resolver.resolve_job_definition = resolve\n"
        "sys.modules.update({'boto3': boto3, 'src': types.ModuleType('src'), "
        "'src.tuning': types.ModuleType('src.tuning'), 'src.tuning.launch_ab': resolver})\n"
        "assert sys.argv[1] == '-c'\n"
        "exec(sys.argv[2])\n",
    )
    sha = "x', None)); print('injected');#"
    result = _run(
        _step("Resolve Batch job-definition revision for this image"),
        tmp_path,
        PATH=path,
        HEAD_SHA=sha,
        SEEN_SHA=str(tmp_path / "seen-sha.json"),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "seen-sha.json").read_text()) == sha
    assert (tmp_path / "outputs").read_text().splitlines() == [
        "job_def=ff-ab-job",
        "revision=8",
    ]


def test_collection_passes_sha_as_one_inert_argument(tmp_path):
    path = _command(
        tmp_path,
        "python",
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['SEEN_ARGS']).write_text(json.dumps(sys.argv[1:]))\n",
    )
    step = _step("Collect benchmark metrics from the experiment prefix")
    assert step["env"]["HEAD_SHA"] == "${{ steps.ctx.outputs.sha }}"
    sha = "$(touch shell-marker)"
    result = _run(
        step,
        tmp_path,
        PATH=path,
        HEAD_SHA=sha,
        POSITIONS="QB RB",
        NOTE="local test",
        SEEN_ARGS=str(tmp_path / "args.json"),
    )
    assert result.returncode == 0, result.stderr
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[args.index("--git-hash") + 1] == sha
    assert not (tmp_path / "shell-marker").exists()

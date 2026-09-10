"""Run workflow input boundaries without submitting remote jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.tuning.ab_harness import resolve_spec
from src.tuning.launch_ab import _build_parser

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]


def _step(workflow, job, name):
    doc = yaml.safe_load((_ROOT / ".github/workflows" / workflow).read_text())
    return next(step for step in doc["jobs"][job]["steps"] if step.get("name") == name)


def _command(tmp_path, name, source):
    path = tmp_path / "bin" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return str(path.parent) + os.pathsep + os.environ["PATH"]


@pytest.mark.parametrize(
    "spec,only,expected",
    [
        ("src.tuning.ab_example", "baseline", ["baseline"]),
        (
            "src.tuning.ab_example",
            "+season_recency nn_dropout=0",
            ["baseline", "+season_recency", "nn_dropout=0"],
        ),
        ("src.tuning.ab_opp_def", "-opp_def", ["baseline", "-opp_def"]),
        ("src.tuning.ab_example", "$(touch${IFS}injected)", None),
    ],
)
def test_ab_workflow_preserves_canonical_variant_names(tmp_path, spec, only, expected):
    args_file = tmp_path / "args.json"
    path = _command(
        tmp_path,
        "python",
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n",
    )
    _command(tmp_path, "tee", "#!/bin/sh\ncat\n")
    step = _step("ab-batch.yml", "ab", "Launch A/B + wait + aggregate")
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": path,
            "ARGS_FILE": str(args_file),
            "SPEC": spec,
            "POSITIONS": "RB",
            "SEEDS": "42",
            "ONLY": only,
            "IMAGE_SHA": "a" * 40,
            "CUDA_GRAPH": "auto",
            "ATTEMPT_TIMEOUT_SECONDS": "10800",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads(args_file.read_text())
    assert argv[:2] == ["-m", "src.tuning.launch_ab"]
    args = _build_parser().parse_args(argv[2:])
    if expected is None:
        with pytest.raises(ValueError, match="names not in spec"):
            resolve_spec(args.spec, only=args.only)
        assert not (tmp_path / "injected").exists()
    else:
        assert list(resolve_spec(args.spec, only=args.only).variants) == expected


def _run_rb_assembly(tmp_path, seed):
    params = tmp_path / "params.json"
    calls = tmp_path / "aws-calls"
    path = _command(
        tmp_path,
        "aws",
        f"#!{sys.executable}\n"
        "import os, pathlib\n"
        "pathlib.Path(os.environ['AWS_CALLS']).write_text('called')\n"
        "print('local-command-id')\n",
    )
    step = _step("ablate-rb-gate.yml", "ablate", "Run ablation via docker")
    body = (
        step["run"]
        .replace("${{ steps.img.outputs.uri }}", "registry.invalid/ff-training:test")
        .replace("${{ steps.inst.outputs.id }}", "i-local-test")
        .replace("${{ github.sha }}", "a" * 40)
        .replace("/tmp/ssm-params.json", str(params))
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", body],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": path,
            "AWS_CALLS": str(calls),
            "INPUT_SEED": seed,
            "AWS_REGION": "us-east-1",
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, params, calls


@pytest.mark.parametrize("seed,expected", [("", "42"), ("0", "0"), ("123", "123")])
def test_rb_ablation_assembles_valid_seed(tmp_path, seed, expected):
    result, params, calls = _run_rb_assembly(tmp_path, seed)
    assert result.returncode == 0, result.stderr
    assert calls.exists()
    command = json.loads(params.read_text())["commands"][0]
    assert f"--seed '{expected}'" in command


@pytest.mark.parametrize("seed", ["-1", "1.5", "42; touch injected", "$(touch injected)", "42\n0"])
def test_rb_ablation_rejects_seed_before_ssm_assembly(tmp_path, seed):
    result, params, calls = _run_rb_assembly(tmp_path, seed)
    assert result.returncode != 0
    assert "Invalid seed" in result.stderr
    assert not params.exists()
    assert not calls.exists()
    assert not (tmp_path / "injected").exists()

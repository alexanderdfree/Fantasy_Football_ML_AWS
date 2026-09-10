"""Collect real launcher namespaces through the checked-in workflow command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.tuning import launch_tune

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("graph", ["true", "false", "TRUE", "1", "off"])
def test_retune_workflow_collects_every_submitted_namespace(tmp_path, monkeypatch, graph):
    batch = MagicMock()
    batch.submit_job.return_value = {"jobId": "local-job"}
    monkeypatch.setattr(launch_tune.boto3, "client", lambda *args, **kwargs: batch)
    monkeypatch.setattr(sys, "argv", ["launch_tune", "--cuda-graph", graph, "--wait", "false"])
    launch_tune.main()
    published = {}
    expected_versions = {}
    for call in batch.submit_job.call_args_list:
        override = call.kwargs["containerOverrides"]
        pos = override["command"][1]
        env = {item["name"]: item["value"] for item in override["environment"]}
        version = env["TUNE_NN_STORAGE_VERSION"]
        expected_versions[pos] = version
        published[f"tune_nn/{version}/{pos.lower()}/results.json"] = {
            pos: {"best_val_loss": 1.0, "storage_version": version}
        }
    remote = tmp_path / "remote.json"
    remote.write_text(json.dumps(published))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"sys.path.insert(0, {str(_ROOT)!r})\n"
        "import boto3\n"
        "from botocore.exceptions import ClientError\n"
        "objects = json.loads(pathlib.Path(os.environ['REMOTE_OBJECTS']).read_text())\n"
        "class S3:\n"
        "    def download_file(self, bucket, key, dest):\n"
        "        if key not in objects:\n"
        "            raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')\n"
        "        pathlib.Path(dest).write_text(json.dumps(objects[key]))\n"
        "boto3.client = lambda *args, **kwargs: S3()\n"
        "assert sys.argv[1:3] == ['-m', 'src.tuning.aggregate_results']\n"
        "sys.argv = ['aggregate_results', *sys.argv[3:]]\n"
        "from src.tuning.aggregate_results import main\n"
        "main()\n"
    )
    python.chmod(0o755)
    workflow = yaml.safe_load((_ROOT / ".github/workflows/retune-nn-batch.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["aggregate"]["steps"]
        if step.get("name") == "Aggregate per-position results from S3"
    )
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "REMOTE_OBJECTS": str(remote),
            "POSITIONS": " ".join(launch_tune.SUPPORTED_POSITIONS),
            "CUDA_GRAPH": graph,
            "S3_BUCKET": "unused",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    collected = json.loads((tmp_path / "tune_nn_results.json").read_text())
    assert {pos: entry["storage_version"] for pos, entry in collected.items()} == expected_versions

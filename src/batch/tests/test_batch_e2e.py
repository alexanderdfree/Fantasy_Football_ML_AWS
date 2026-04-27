"""End-to-end smoke for the batch training CLI (src/batch/train.py).

Invokes the CLI with `--dry-run`, which stubs out the heavy position
pipeline and S3 calls while still exercising:
  - argparse and --position validation
  - env-var resolution (MODEL_OUTPUT_DIR, S3_BUCKET, REQUIRE_GPU)
  - artifact layout + benchmark_metrics.json serialization
  - the skip-S3 code path

Covers reviewer concern P2.4: "CLI entry point has no end-to-end smoke."

Budget: < 5s per parametrized position on CPU.
"""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_CLI = Path(PROJECT_ROOT) / "src" / "batch" / "train.py"


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.parametrize("position", ["QB", "DST"])
def test_dry_run_exits_zero_and_writes_artifacts(position, tmp_path, monkeypatch):
    """Invoke `src/batch/train.py --position <POS> --dry-run` via subprocess.

    Subprocess (rather than importing main()) to exercise the real argparse
    path, the `if __name__ == "__main__"` entry, and PYTHONPATH resolution
    as AWS Batch sees it.
    """
    import subprocess

    model_dir = tmp_path / "model"
    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(model_dir))
    monkeypatch.setenv("S3_BUCKET", "")
    monkeypatch.setenv("REQUIRE_GPU", "0")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(TRAIN_CLI), "--position", position, "--seed", "42", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"Dry-run exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    stub_path = model_dir / f"{position.lower()}_model.stub"
    assert stub_path.exists(), f"Stub artifact missing at {stub_path}"
    metrics_path = model_dir / "benchmark_metrics.json"
    assert metrics_path.exists(), f"Metrics missing at {metrics_path}"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["position"] == position
    assert metrics.get("dry_run") is True
    assert metrics["seed"] == 42


@pytest.mark.e2e
@pytest.mark.integration
def test_dry_run_never_calls_boto3(tmp_path, monkeypatch):
    """Dry-run must not invoke boto3.client. In-process (not subprocess) so
    we can patch boto3 from the parent and prove `.client` is never touched.
    """
    model_dir = tmp_path / "model"
    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(model_dir))
    monkeypatch.setenv("REQUIRE_GPU", "0")

    from src.batch import train as train_mod

    def _boom(*args, **kwargs):
        raise AssertionError("boto3.client was called during --dry-run — S3 isolation broken")

    with (
        mock.patch("sys.argv", ["train.py", "--position", "QB", "--dry-run"]),
        mock.patch.object(train_mod, "boto3") as mock_boto3,
    ):
        mock_boto3.client.side_effect = _boom
        train_mod.main()
        mock_boto3.client.assert_not_called()

    assert (model_dir / "benchmark_metrics.json").exists()


@pytest.mark.e2e
@pytest.mark.integration
def test_dry_run_skips_position_pipeline_import(tmp_path, monkeypatch):
    """Dry-run should not import the heavy QB pipeline module.

    If dry-run accidentally called the real run_qb_pipeline, any of the
    heavy src.shared.pipeline imports (matplotlib, lightgbm, torch training
    loops) would fire. The position-runner-module import check is a fast,
    deterministic proxy.
    """
    model_dir = tmp_path / "model"
    monkeypatch.setenv("MODEL_OUTPUT_DIR", str(model_dir))
    monkeypatch.setenv("REQUIRE_GPU", "0")

    before = {k for k in sys.modules if k.startswith("src.QB.run_qb_pipeline")}

    from src.batch import train as train_mod

    with mock.patch("sys.argv", ["train.py", "--position", "QB", "--dry-run"]):
        train_mod.main()

    after = {k for k in sys.modules if k.startswith("src.QB.run_qb_pipeline")}
    newly_imported = after - before
    assert not newly_imported, f"--dry-run imported heavy pipeline modules: {newly_imported}"

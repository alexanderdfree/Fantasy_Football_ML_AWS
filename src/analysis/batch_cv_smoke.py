"""Disposable, AWS-Batch-only validation of #1612/#1615 CV and #1599 replay fixtures.

This diagnostic image never invokes a training/benchmark publisher. Model
artifacts stay in pytest's temporary directories; only logs and receipts upload.
It is not a model-accuracy experiment or a production image.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("batch_cv_smoke_source.json")
TESTS = {
    "WR": [
        "tests/wr/test_config_tiny.py",
        "tests/wr/test_pipeline_e2e.py",
        "tests/wr/test_run_cv_pipeline.py",
    ],
    "RB": ["tests/rb/test_pipeline_e2e.py", "tests/rb/test_run_cv_pipeline.py"],
    "DST": [
        "tests/analysis/test_synthetic_history_dst.py::test_replay_streams_the_opponent_and_the_control_rebuilds_it",
        "tests/analysis/test_synthetic_history_dst.py::test_replay_refuses_mismatched_streams",
        "tests/analysis/test_synthetic_history_dst.py::test_cli_round_trip_with_the_opponent_frame",
    ],
    "UNIT": ["-m", "unit"],
}
EXPECTED_TESTS = {"WR": 14, "RB": 11, "DST": 3}


def without_tiny_hash(source: str) -> str:
    """Runtime-local AST digest; ast.dump serialization differs across Python versions."""
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CONFIG_TINY" for t in node.targets)
        )
    ]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def validate_source(root: Path, source: dict) -> None:
    for name, expected in source["test_source_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Test source checksum mismatch: {name}")
    for name in source.get("absent_source_files", []):
        if (root / name).exists():
            raise ValueError(f"Unexpected removed source file: {name}")
    baseline = source["baseline_wr_source"].encode("utf-8")
    if (
        hashlib.sha256(baseline).hexdigest()
        != source["baseline_data_producers"]["src/wr/config.py"]
    ):
        raise ValueError("Baseline WR source checksum mismatch")
    # Parse BOTH pinned source texts with this interpreter. Persisted AST dumps
    # are not portable between the host and the Python 3.12 training image.
    current = without_tiny_hash((root / "src/wr/config.py").read_text())
    if current != without_tiny_hash(baseline.decode("utf-8")):
        raise ValueError("WR production config changed beyond CONFIG_TINY")


def validate_data_producers(manifest: dict, source: dict) -> None:
    mismatch = [
        name
        for name, expected in source["baseline_data_producers"].items()
        if manifest.get("producer", {}).get(name) != expected
    ]
    if mismatch:
        raise ValueError(f"Frozen dataset is incompatible with baseline: {mismatch}")


def junit_counts(path: Path) -> dict[str, int]:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }


def suite_passed(target: str, exit_code: int, counts: dict[str, int]) -> bool:
    if exit_code or counts.get("errors", 1) or counts.get("failures", 1):
        return False
    if target == "UNIT":
        # Legitimate platform skips remain visible; empty/all-skipped does not pass.
        return counts.get("tests", 0) > counts.get("skipped", 0)
    return counts.get("tests") == EXPECTED_TESTS[target] and counts.get("skipped") == 0


def isolated_test_environment(
    parent: dict[str, str], output: Path, prefix: str, *, target: str
) -> dict[str, str]:
    """Keep AWS credentials exclusively in the parent hydration/evidence process."""
    env = {name: value for name, value in parent.items() if not name.startswith("AWS_")}
    for name in (
        "ALLOW_SKIP_E2E",
        "FF_CACHE_DIR",
        "PYTEST_ADDOPTS",
        "FF_S3_BUCKET",
        "S3_BUCKET",
        "FF_MODEL_S3_PREFIX",
    ):
        env.pop(name, None)
    empty_config = output / "empty-aws-config"
    empty_config.write_text("")
    env.update(
        {
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_SESSION_TOKEN": "testing",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_SHARED_CREDENTIALS_FILE": str(empty_config),
            "AWS_CONFIG_FILE": str(empty_config),
            "BOTO_CONFIG": str(empty_config),
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_REGION": "us-east-1",
            "FF_MODEL_S3_BUCKET": "",
            "FF_BENCHMARK_SYNC_INTERVAL_S": "0",
            "MODEL_OUTPUT_DIR": str(output / "models"),
            "TMPDIR": str(output),
            "PYTHONPATH": str(ROOT),
        }
    )
    if target != "UNIT":
        env["FF_MODEL_S3_PREFIX"] = f"{prefix}/unpublished-models"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = "1"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", choices=TESTS, required=True)
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--result-prefix", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.data_release):
        parser.error("--data-release must be an immutable release SHA-256")
    if not re.fullmatch(r"diagnostics/cv-smoke/[A-Za-z0-9_-]+", args.result_prefix):
        parser.error("--result-prefix must be diagnostics/cv-smoke/<run-id>")
    source = json.loads(SOURCE.read_text())
    validate_source(ROOT, source)
    command = [
        "bash",
        "scripts/pytest-fair.sh",
        *TESTS[args.position],
        "-n",
        "2",
        "--dist=loadgroup",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if args.dry_run:
        print(json.dumps({"position": args.position, "command": command, "origins": source["prs"]}))
        return 0
    if not os.environ.get("AWS_BATCH_JOB_ID"):
        raise RuntimeError("Fitting test fixtures require AWS Batch")
    if os.environ.get("FF_DEVICE") != "cpu" or os.environ.get("FF_AMP_DTYPE") != "fp32":
        raise RuntimeError("This fixture diagnostic requires explicit CPU FP32")
    built_sha = (ROOT / ".training-source-sha").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", built_sha):
        raise RuntimeError("Missing immutable diagnostic image source identity")

    import boto3

    from src.data.release import download_release, resolve_release

    s3 = boto3.client("s3")
    _, manifest = resolve_release(s3, args.bucket, release_id=args.data_release)
    validate_data_producers(manifest, source)
    download_release(
        s3,
        args.bucket,
        release_id=args.data_release,
        raw_dir=ROOT / "data/raw",
        splits_dir=ROOT / "data/splits",
    )
    job = os.environ["AWS_BATCH_JOB_ID"]
    attempt = os.environ.get("AWS_BATCH_JOB_ATTEMPT", "1")
    prefix = f"{args.result_prefix}/{args.position.lower()}/{job}/attempt-{attempt}"
    with tempfile.TemporaryDirectory(prefix="cv-smoke-") as tmp:
        output = Path(tmp)
        log, junit = output / "pytest.log", output / "junit.xml"
        env = isolated_test_environment(dict(os.environ), output, prefix, target=args.position)
        env["FF_DATA_RELEASE"] = args.data_release
        with log.open("w") as stream:
            result = subprocess.run(
                [*command, f"--basetemp={output / 'pytest'}", f"--junitxml={junit}"],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        counts = junit_counts(junit) if junit.exists() else {}
        passed = suite_passed(args.position, result.returncode, counts)
        receipt = {
            "schema": "batch-cv-fixture-smoke/v1",
            "image_source_sha": built_sha,
            "data_release": args.data_release,
            "source": source,
            "position": args.position,
            "job_id": job,
            "attempt": attempt,
            "execution": {
                "device": "cpu",
                "dtype": "fp32",
                "purpose": "full-unit-gate" if args.position == "UNIT" else "fixture-contract-only",
                "test_aws_credentials": "dummy",
                "model_history_publishing": "disabled",
            },
            "dependency_versions": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            },
            "pytest_exit_code": result.returncode,
            "counts": counts,
            "passed": passed,
        }
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for file in (log, junit, output / "receipt.json", Path("/opt/ml/test-environment.txt")):
            if file.is_file():
                s3.upload_file(str(file), args.bucket, f"{prefix}/{file.name}")
        print(
            json.dumps(
                {"passed": passed, "counts": counts, "evidence": f"s3://{args.bucket}/{prefix}/"}
            )
        )
        return 0 if passed else (result.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())

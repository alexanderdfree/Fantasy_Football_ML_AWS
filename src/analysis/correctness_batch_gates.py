"""Disposable Spot Batch gates; data preparation stages without promotion.

No target runs locally. The image also accepts the existing A/B launcher's
``--position POS --mode tune`` command, restricted to an explicitly selected
A/B spec; it cannot dispatch normal production training.
"""

from __future__ import annotations

import argparse
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
NUMERICAL_TESTS = (
    "tests/shared/test_ztnb_reception_expectation.py",
    "tests/shared/test_count_likelihood_precision.py",
    "tests/analysis/test_verify_count_likelihoods.py",
)


def test_command(target: str) -> list[str]:
    selection = ["-m", "unit"] if target == "unit" else list(NUMERICAL_TESTS)
    if target == "numerical_cpu":
        selection += ["-k", "not native_mps and not native_cuda"]
    elif target == "numerical_gpu":
        selection += ["-k", "not native_mps"]
    return [
        "bash",
        "scripts/pytest-fair.sh",
        *selection,
        "-n",
        "2",
        "--dist=loadgroup",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def test_environment(parent: dict, output: Path, device: str) -> dict:
    """Retain real AWS credentials only in the hydration/receipt parent."""
    env = {key: value for key, value in parent.items() if not key.startswith("AWS_")}
    for key in (
        "ALLOW_SKIP_E2E",
        "FF_CACHE_DIR",
        "PYTEST_ADDOPTS",
        "FF_S3_BUCKET",
        "S3_BUCKET",
        "FF_MODEL_S3_PREFIX",
    ):
        env.pop(key, None)
    empty = output / "empty-aws-config"
    empty.write_text("")
    env.update(
        AWS_ACCESS_KEY_ID="testing",
        AWS_SECRET_ACCESS_KEY="testing",
        AWS_SESSION_TOKEN="testing",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_SHARED_CREDENTIALS_FILE=str(empty),
        AWS_CONFIG_FILE=str(empty),
        BOTO_CONFIG=str(empty),
        AWS_DEFAULT_REGION="us-east-1",
        AWS_REGION="us-east-1",
        FF_MODEL_S3_BUCKET="",
        FF_BENCHMARK_SYNC_INTERVAL_S="0",
        MODEL_OUTPUT_DIR=str(output / "models"),
        TMPDIR=str(output),
        PYTHONPATH=str(ROOT),
        FF_DEVICE=device,
        FF_AMP_DTYPE="fp32",
        REQUIRE_GPU="1" if device == "cuda" else "0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return env


def junit_counts(path: Path) -> dict[str, int]:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    return {
        "tests": len(cases),
        **{
            name: sum(case.find(name) is not None for case in cases)
            for name in ("failure", "error", "skipped")
        },
    }


def suite_passed(target: str, exit_code: int, counts: dict) -> bool:
    return bool(
        exit_code == 0
        and counts.get("tests", 0) > counts.get("skipped", 0)
        and not counts.get("failure", 0)
        and not counts.get("error", 0)
        and (target == "unit" or counts.get("skipped", 0) == 0)
    )


def prepare_data(
    root: Path, source_sha: str, s3, bucket: str, log: Path, *, run=subprocess.run, publish=None
) -> dict:
    """Invoke the actual clean producer, then stage its verified immutable bytes."""
    if any(
        (root / "data" / name).exists() or (root / "data" / name).is_symlink()
        for name in ("raw", "splits")
    ):
        raise ValueError(
            "preparedata requires clean raw/splits; existing releases cannot be relabelled"
        )
    env = dict(os.environ)
    for key in (
        "FF_DATA_RELEASE",
        "FF_DATASET_ID",
        "FF_DATA_FORMAT",
        "FF_CACHE_DIR",
        "FF_BUILD_PLAN_ID",
        "FF_REQUIRE_BUILD_PLAN",
    ):
        env.pop(key, None)
    env.update(
        GITHUB_SHA=source_sha,
        FF_TRAIN_GIT_SHA=source_sha,
        NFLREADPY_CACHE="off",
        NFLREADPY_TIMEOUT="120",
        FF_CAPTURE_PROVIDER_SOURCES="data/raw/provider_sources",
        FF_MODEL_S3_BUCKET="",
        FF_BENCHMARK_SYNC_INTERVAL_S="0",
    )
    with log.open("w") as stream:
        run(
            [sys.executable, "-m", "src.data.maintenance_build"],
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if publish is None:
        from src.data.release import publish_release

        publish = publish_release
    manifest = json.loads((root / "data/splits/release-inputs.json").read_text())
    if manifest.get("git_sha") != source_sha:
        raise ValueError("Data producer source differs from the built image")
    release_id = publish(
        s3,
        bucket,
        raw_dir=root / "data/raw",
        splits_dir=root / "data/splits",
        repo_root=root,
        promote=False,
    )
    return {
        "data_release": release_id,
        "producer_sha": manifest["data_producer_sha256"],
        "input_source": "fresh live provider capture by maintenance_build",
        "promoted": False,
        "files": len(manifest["files"]),
    }


def validate_release(manifest: dict, root: Path) -> None:
    from src.data.release import data_producer_hashes

    mismatch = [
        name
        for name, digest in data_producer_hashes(root).items()
        if manifest.get("producer", {}).get(name) != digest
    ]
    if mismatch:
        raise ValueError(f"Immutable release does not match this image's producers: {mismatch}")


def source_identity(root: Path) -> str:
    source = (root / ".training-source-sha").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Image must contain its immutable source identity")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != source:
        raise ValueError("Exported checkout and built image identity differ")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root, check=False).returncode:
        raise ValueError("Image tracked source differs from its recorded commit")
    return source


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("preparedata", "numerical_cpu", "numerical_gpu", "unit")
    )
    parser.add_argument("--bucket", default="ff-predictor-training")
    parser.add_argument("--data-release")
    parser.add_argument("--result-prefix", default="diagnostics/correctness/20260924")
    parser.add_argument("--dry-run", action="store_true")
    if "--target" not in argv:
        if (
            not os.environ.get("AWS_BATCH_JOB_ID")
            or not os.environ.get("FF_TUNE_AB_SPEC")
            or "--mode" not in argv
            or argv[argv.index("--mode") + 1 :] == []
            or argv[argv.index("--mode") + 1] != "tune"
        ):
            parser.error("Only explicit diagnostic targets or Batch A/B tune dispatch are allowed")
        source_identity(ROOT)
        os.execv(sys.executable, [sys.executable, "-m", "src.batch.train", *argv])
    args = parser.parse_args(argv)
    if not re.fullmatch(
        r"diagnostics/correctness/[A-Za-z0-9_/-]+", args.result_prefix
    ) or ".." in args.result_prefix.split("/"):
        parser.error("Use an isolated diagnostics/correctness/<run> result prefix")
    if args.data_release and not re.fullmatch(r"[0-9a-f]{64}", args.data_release):
        parser.error("--data-release must be an immutable SHA256")
    if args.target == "preparedata" and args.data_release:
        parser.error("preparedata never consumes an old release")
    if args.target == "unit" and not args.data_release:
        parser.error("The full unit gate requires an exact compatible data release")
    command = None if args.target == "preparedata" else test_command(args.target)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "target": args.target,
                    "command": command,
                    "data_release": args.data_release,
                    "promote": False,
                    "aws_calls": 0,
                }
            )
        )
        return 0
    if not os.environ.get("AWS_BATCH_JOB_ID"):
        raise RuntimeError("All correctness gates and data preparation require AWS Batch")
    source = source_identity(ROOT)
    import boto3

    s3 = boto3.client("s3")
    prefix = f"{args.result_prefix}/{args.target}/{os.environ['AWS_BATCH_JOB_ID']}/attempt-{os.environ.get('AWS_BATCH_JOB_ATTEMPT', '1')}"
    receipt = {
        "schema": "correctness-batch-gates/v1",
        "target": args.target,
        "image_source_sha": source,
        "data_release": args.data_release,
        "job_id": os.environ["AWS_BATCH_JOB_ID"],
        "passed": False,
    }
    with tempfile.TemporaryDirectory(prefix="correctness-") as directory:
        output = Path(directory)
        log, junit = output / "run.log", output / "junit.xml"
        code = 1
        try:
            if args.target == "preparedata":
                if os.environ.get("FF_DEVICE") != "cpu" or os.environ.get("FF_AMP_DTYPE") != "fp32":
                    raise RuntimeError("preparedata requires explicit CPU FP32 execution")
                receipt.update(prepare_data(ROOT, source, s3, args.bucket, log))
                code = 0
                receipt["passed"] = True
            else:
                device = "cuda" if args.target == "numerical_gpu" else "cpu"
                if device == "cuda":
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError("Numerical GPU gate requires a real CUDA device")
                    receipt["gpu"] = torch.cuda.get_device_name()
                if args.target != "unit":
                    missing = [name for name in NUMERICAL_TESTS if not (ROOT / name).is_file()]
                    if missing:
                        raise ValueError(f"Candidate numerical tests are absent: {missing}")
                if args.data_release:
                    from src.data.release import download_release, resolve_release

                    _, manifest = resolve_release(s3, args.bucket, release_id=args.data_release)
                    validate_release(manifest, ROOT)
                    download_release(
                        s3,
                        args.bucket,
                        release_id=args.data_release,
                        raw_dir=ROOT / "data/raw",
                        splits_dir=ROOT / "data/splits",
                    )
                env = test_environment(dict(os.environ), output, device)
                if args.data_release:
                    env["FF_DATA_RELEASE"] = args.data_release
                with log.open("w") as stream:
                    if device == "cuda":
                        subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                "import json; from src.analysis.verify_count_likelihoods import verify_count_likelihoods; print(json.dumps(verify_count_likelihoods(), sort_keys=True))",
                            ],
                            cwd=ROOT,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    result = subprocess.run(
                        [*command, f"--basetemp={output / 'pytest'}", f"--junitxml={junit}"],
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                code = result.returncode
                counts = junit_counts(junit) if junit.exists() else {}
                receipt.update(
                    device=device,
                    dtype="fp32",
                    counts=counts,
                    pytest_exit_code=code,
                    test_credentials="dummy",
                    model_history_publishing="disabled",
                    passed=suite_passed(args.target, code, counts),
                )
                code = 0 if receipt["passed"] else (code or 1)
        except Exception as error:
            code = 1
            receipt["error"] = f"{type(error).__name__}: {error}"
            print(receipt["error"], file=sys.stderr)
        finally:
            receipt["dependency_versions"] = {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            }
            receipt["log_sha256"] = (
                hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None
            )
            path = output / "receipt.json"
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            for file in (log, junit, path, Path("/opt/ml/test-environment.txt")):
                if file.is_file():
                    s3.upload_file(str(file), args.bucket, f"{prefix}/{file.name}")
        print(
            json.dumps(
                {
                    "passed": receipt["passed"],
                    "evidence": f"s3://{args.bucket}/{prefix}/",
                    "data_release": receipt.get("data_release"),
                }
            )
        )
        return code


if __name__ == "__main__":
    sys.exit(main())

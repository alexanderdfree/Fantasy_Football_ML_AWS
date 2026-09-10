"""Gate deployments/training on a published snapshot matching its data producers.

Uses the runner's AWS CLI, so deployment need not install the ML environment.
Missing pending markers never allow a rollout: the immutable manifest itself is
checked. A matching failure marker or timeout fails before service mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import tempfile
import time
from pathlib import Path

from src.data.release import (
    DATA_PRODUCER_PATHS,
    DataReleaseError,
    data_producer_hashes,
    producer_fingerprint,
    resolve_compatible_release,
)


class AwsS3:
    """Small read-only adapter for the release manifest reader."""

    def get_object(self, *, Bucket, Key):
        with tempfile.NamedTemporaryFile() as output:
            result = subprocess.run(
                [
                    "aws",
                    "s3api",
                    "get-object",
                    "--bucket",
                    Bucket,
                    "--key",
                    Key,
                    output.name,
                    "--no-cli-pager",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                if any(code in result.stderr for code in ("(NoSuchKey)", "(404)", "(NotFound)")):
                    raise FileNotFoundError(Key)
                raise RuntimeError(f"Unable to read s3://{Bucket}/{Key}: {result.stderr.strip()}")
            return {"Body": io.BytesIO(Path(output.name).read_bytes())}


def producer_hashes_at_revision(revision: str) -> dict[str, str]:
    """Compare to the image's commit even when the CI checkout advanced on main."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        text=True,
    ).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], text=True
    ).splitlines()
    selected = [
        name
        for name in names
        if any(
            name == path or (name.startswith(path + "/") and name.endswith(".py"))
            for path in DATA_PRODUCER_PATHS
        )
    ]
    return {
        name: hashlib.sha256(
            subprocess.check_output(["git", "show", f"{commit}:{name}"])
        ).hexdigest()
        for name in selected
    }


def _marker(s3, bucket, kind, revision):
    try:
        return (
            s3.get_object(Bucket=bucket, Key=f"splits-rebuild-markers/{kind}/{revision}.txt")[
                "Body"
            ]
            .read()
            .decode()
            .strip()
        )
    except FileNotFoundError:
        return None


def wait_for_release(
    s3,
    bucket,
    *,
    revision,
    repo_root=".",
    timeout=3600,
    interval=30,
    clock=time.monotonic,
    sleep=time.sleep,
    expected_hashes=None,
):
    expected = (
        expected_hashes if expected_hashes is not None else data_producer_hashes(Path(repo_root))
    )
    if not expected:
        raise RuntimeError("No data-producing source files found; refusing unchecked rollout")
    deadline = clock() + timeout
    while True:
        try:
            release_id, _ = resolve_compatible_release(s3, bucket, expected)
            print(f"Verified compatible data release {release_id} for {revision}", flush=True)
            return release_id
        except DataReleaseError as error:
            reason = str(error)
        except FileNotFoundError as error:
            reason = f"published snapshot unavailable: {error}"
        pending = _marker(s3, bucket, "pending", revision)
        failed = _marker(s3, bucket, "failed", revision)
        # A retry writes a new pending run ID, so an earlier failed attempt
        # cannot cancel that retry. A ready/compatible release always wins.
        if failed and failed == pending:
            raise RuntimeError(
                f"Data rebuild {failed} failed for {revision}; refusing rollout/training ({reason})"
            )
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError(
                f"No compatible training data after {timeout}s for {revision}: {reason}. Existing service was not changed; inspect refresh-splits before retrying."
            )
        print(f"Waiting for compatible data ({int(remaining)}s remaining): {reason}", flush=True)
        sleep(min(interval, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--pin-training", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    # Read current S3 state, not an inherited shell pin from an earlier run.
    os.environ.pop("FF_DATA_RELEASE", None)
    try:
        expected = producer_hashes_at_revision(args.revision)
        selected = wait_for_release(
            AwsS3(),
            args.bucket,
            revision=args.revision,
            timeout=args.timeout_seconds,
            expected_hashes=expected,
        )
    except (RuntimeError, ValueError, TimeoutError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    if args.pin_training:
        with open(os.environ["GITHUB_ENV"], "a") as stream:
            stream.write(f"FF_DATA_RELEASE={selected}\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"release_id={selected}\n")
            stream.write(f"producer_sha256={producer_fingerprint(expected)}\n")


if __name__ == "__main__":
    main()

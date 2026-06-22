"""Launch parallel AWS Batch training jobs for all positions.

Usage:
    python src/batch/launch.py                         # all positions
    python src/batch/launch.py --positions RB WR       # subset
    python src/batch/launch.py --wait false            # fire and forget
    python src/batch/launch.py --dry-run               # print plan, touch nothing
    python src/batch/launch.py --wait-timeout 1800     # override 3h default
    python src/batch/launch.py --force-upload          # skip ETag dedup
    python src/batch/launch.py --skip-upload           # assume S3 current (CI)

Config (environment variables, all optional):
    FF_S3_BUCKET        (default: ff-predictor-training)
    FF_JOB_QUEUE        (default: ff-training-queue)
    FF_JOB_QUEUE_CPU    (optional)                          CPU split queue
    FF_JOB_DEFINITION   (default: ff-training-job)          GPU job definition
    FF_JOB_DEFINITION_CPU  (optional)                       CPU split job definition
    FF_JOB_DEFINITION_REVISION       (optional)             GPU job-def revision pin
    FF_JOB_DEFINITION_CPU_REVISION   (optional)             CPU job-def revision pin
    FF_WAIT_TIMEOUT     (default: 10800, i.e. 3h)
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError

# --- Configuration (env-var overridable) ---------------------------------
S3_BUCKET = os.environ.get("FF_S3_BUCKET", "ff-predictor-training")
JOB_QUEUE = os.environ.get("FF_JOB_QUEUE", "ff-training-queue")
JOB_QUEUE_CPU = os.environ.get("FF_JOB_QUEUE_CPU", "") or None
JOB_DEFINITION = os.environ.get("FF_JOB_DEFINITION", "ff-training-job")
# CPU job definition used by split cpu/merge branches. Legacy full-mode K/DST
# routing still honors it when explicitly set, but production leaves it unset
# unless --split is active so the full path remains GPU-backed.
JOB_DEFINITION_CPU = os.environ.get("FF_JOB_DEFINITION_CPU", "") or None
JOB_DEFINITION_CPU_REVISION = os.environ.get("FF_JOB_DEFINITION_CPU_REVISION", "") or None
# Pin job submissions to a specific job-definition revision. batch-image.yml
# registers a new revision each time it pushes an image and stashes the
# revision number at s3://ff-predictor-training/job-def-revisions/{sha}.txt;
# train-batch.yml resolves that file and exports the value here. Without this
# pinning, AWS Batch resolves bare job-definition names to the latest active
# revision at submission time, so two concurrent train-batch runs (image A
# and image B, both registered in the same window) end up submitting jobs
# against the same revision — the latest one — which means a workflow run
# triggered by image A can silently train image B's code.
JOB_DEFINITION_REVISION = os.environ.get("FF_JOB_DEFINITION_REVISION", "") or None
# Thread the image's commit SHA through to the containers so train.py can
# stamp it into benchmark_metrics.json. benchmark.py uses it to verify all
# six positions in a run reflect the same image (catches the lingering
# manifest-write race when two train-batch runs land in quick succession,
# even after Layer A pins the job-def revision). Empty -> not passed.
TRAIN_GIT_SHA = os.environ.get("FF_TRAIN_GIT_SHA", "") or None
# Optional breadcrumb file recording the submitted Batch job ids. Set by
# train-batch.yml so its post-timeout recovery step can re-check the SAME jobs
# via `aws batch describe-jobs` after wait_for_jobs gives up — on 2026-06-08
# (run 27161472255) Spot capacity starvation held all six jobs RUNNABLE past
# the 3h wait; they SUCCEEDED 0.5-3.5 min after the workflow stopped looking,
# and the benchmark-history append was silently skipped. Unset (the default,
# workstation runs) writes nothing.
JOB_IDS_FILE = os.environ.get("FF_BATCH_JOB_IDS_FILE", "") or None
# Optional CUDA-graph OVERRIDE, forwarded to the container only when
# FF_CUDA_GRAPH is set in this launcher's environment. The container's
# cuda_graph_enabled() (src/shared/utils.py) AUTODETECTS graphs ON for sm_80+
# (both g6/L4 and g5/A10G qualify), so the production fan-out is graphed by
# default with no value here. train-batch.yml threads the FF_BATCH_CUDA_GRAPH
# repo variable as a fleet override: leave it unset for the autodetect default,
# or set it to 0 to force the whole fan-out back to the eager path (e.g. a
# bit-comparable A/B).
# Graphs are ~1.5-1.8x on the launch-bound GPU branch (both the base/control NN
# and the attention NN) but NOT bit-identical to eager (see
# todo/gpu_launch_bound_levers.md, Lever A).
FF_CUDA_GRAPH = os.environ.get("FF_CUDA_GRAPH", "") or None
# Optional FULL-STEP CUDA-graph OVERRIDE: the wider gather+forward+loss+backward
# capture (cuda_graph_full_enabled() in src/shared/utils.py) is ALSO autodetect-ON
# for sm_80+ (and requires the base FF_CUDA_GRAPH gate). train-batch.yml threads
# the FF_BATCH_CUDA_GRAPH_FULL repo variable as a SEPARATE fleet override: leave it
# unset for the autodetect default, or set it to 0 to degrade the fan-out from the
# full-step graph to the model-only graph (keeps the ~1.84x model-only win) without
# forcing the whole path back to eager (which FF_BATCH_CUDA_GRAPH=0 would do).
FF_CUDA_GRAPH_FULL = os.environ.get("FF_CUDA_GRAPH_FULL", "") or None

from src.shared.registry import ALL_POSITIONS, CPU_ONLY_POSITIONS  # noqa: E402

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
BATCH_LOG_GROUP = "/aws/batch/job"

POLL_INTERVAL_SECONDS = 10
# Hard cap so a stuck RUNNABLE job (Spot capacity, bad IAM, etc.) can't pin
# this script forever. ~3h is generous for a 6-position GPU sweep.
WAIT_TIMEOUT_SECONDS = int(os.environ.get("FF_WAIT_TIMEOUT", 3 * 60 * 60))
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}

# Retry Spot reclaims and transient ECR pull errors; don't retry deterministic
# app failures. Mirrored as the job-definition fallback in
# infra/batch/setup.sh + .github/workflows/batch-image.yml so submitters that
# bypass this launcher (manual `aws batch submit-job`, future tools) get the
# same protection — keep all three in sync.
RETRY_STRATEGY = {
    "attempts": 3,
    "evaluateOnExit": [
        # Spot interruption: host terminated by EC2 -> retry on a fresh pool
        # (SPOT_PRICE_CAPACITY_OPTIMIZED picks one with low current reclaim risk).
        {"onStatusReason": "Host EC2*", "action": "RETRY"},
        # ECR pull blip / pull-through-cache miss: transient, retry.
        {"onReason": "CannotPullContainerError*", "action": "RETRY"},
        # Anything else: exit immediately so we see the real error.
        {"onReason": "*", "action": "EXIT"},
    ],
}


def _job_label(key) -> str:
    if isinstance(key, tuple):
        return "/".join(str(part) for part in key)
    return str(key)


def _console_encode(s: str) -> str:
    # AWS console fragment path components are double-URL-encoded: `/` ->
    # `%2F` -> `$252F` (the `$` is how the console escapes `%`). Percent-
    # encode once, then replace each literal `%` with `$25`; replacing with
    # a bare `$` yields `$2F`, which the console fails to parse.
    return urllib.parse.quote(s, safe="").replace("%", "$25")


def _cloudwatch_url(log_stream_name: str) -> str:
    """Build a console URL that opens the CloudWatch stream for a failed job."""
    group = _console_encode(BATCH_LOG_GROUP)
    stream = _console_encode(log_stream_name)
    return (
        f"https://{AWS_REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={AWS_REGION}#logsV2:log-groups/log-group/{group}/log-events/{stream}"
    )


def _file_md5(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file; returns hex digest (matches S3 ETag for single-part)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _s3_object_etag(s3_client, bucket: str, key: str):
    """Return the S3 object's ETag (minus quotes) or None if the object doesn't exist."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return resp.get("ETag", "").strip('"')


def upload_data(s3_bucket, s3_client=None, force: bool = False):
    """Upload local data splits to S3, skipping files whose ETag matches the local MD5.

    Set force=True to re-upload regardless.
    """
    s3 = s3_client or boto3.client("s3", region_name=AWS_REGION)
    data_dir = "data/splits"
    uploaded = 0
    skipped = 0
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        local_path = os.path.join(data_dir, name)
        s3_key = f"data/{name}"
        if not force:
            remote_etag = _s3_object_etag(s3, s3_bucket, s3_key)
            if remote_etag is not None:
                local_md5 = _file_md5(local_path)
                if remote_etag == local_md5:
                    print(
                        f"  [skip] s3://{s3_bucket}/{s3_key} already up to date ({local_md5[:8]}...)"
                    )
                    skipped += 1
                    continue
        print(f"  [upload] {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
        uploaded += 1
    print(f"Data upload complete: {uploaded} uploaded, {skipped} skipped.\n")


def _job_definition_for(position: str, branch: str = "full") -> str:
    """Pick the right job definition for a position.

    Full mode keeps the legacy default. Split ``nn`` always uses the GPU
    definition; split ``cpu``/``merge`` require the CPU definition and pin it
    with FF_JOB_DEFINITION_CPU_REVISION, never the GPU revision.
    """
    if branch in {"cpu", "merge"}:
        if not JOB_DEFINITION_CPU:
            raise RuntimeError("FF_JOB_DEFINITION_CPU is required for split CPU/merge jobs")
        return (
            f"{JOB_DEFINITION_CPU}:{JOB_DEFINITION_CPU_REVISION}"
            if JOB_DEFINITION_CPU_REVISION
            else JOB_DEFINITION_CPU
        )
    if branch == "nn":
        return (
            f"{JOB_DEFINITION}:{JOB_DEFINITION_REVISION}"
            if JOB_DEFINITION_REVISION
            else JOB_DEFINITION
        )

    # Full-mode CPU routing requires BOTH the CPU job-def AND the CPU queue, so
    # it stays symmetric with _job_queue_for (line ~231) and the --split guard.
    # A def/queue pair must agree for a valid Batch submission; routing a CPU
    # job-def to the default GPU queue is invalid/wasteful, so a partial config
    # (only one of the two vars set) falls back to the GPU def+queue pair.
    use_cpu = position in CPU_ONLY_POSITIONS and JOB_DEFINITION_CPU and JOB_QUEUE_CPU
    base = JOB_DEFINITION_CPU if use_cpu else JOB_DEFINITION
    if use_cpu and JOB_DEFINITION_CPU_REVISION:
        return f"{base}:{JOB_DEFINITION_CPU_REVISION}"
    if JOB_DEFINITION_REVISION and not use_cpu:
        return f"{base}:{JOB_DEFINITION_REVISION}"
    return base


def _job_queue_for(position: str, branch: str = "full") -> str:
    """Pick the right Batch queue for a position.

    Split CPU/merge branches require the CPU queue. Full mode preserves the
    legacy CPU-only route only when both CPU env vars are explicitly configured.
    """
    if branch in {"cpu", "merge"}:
        if not JOB_QUEUE_CPU:
            raise RuntimeError("FF_JOB_QUEUE_CPU is required for split CPU/merge jobs")
        return JOB_QUEUE_CPU
    if branch == "nn":
        return JOB_QUEUE
    if position in CPU_ONLY_POSITIONS and JOB_DEFINITION_CPU and JOB_QUEUE_CPU:
        return JOB_QUEUE_CPU
    return JOB_QUEUE


def submit_job(
    position,
    seed=42,
    batch_client=None,
    *,
    branch: str = "full",
    split_run_id: str | None = None,
    depends_on: list[dict] | None = None,
):
    """Submit a single Batch job. Returns (position-or-branch-key, job_id)."""
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    # int-seconds timestamp collides if two launches happen in the same second;
    # a short uuid suffix makes the name unique without sacrificing readability.
    timestamp = int(time.time())
    suffix = uuid.uuid4().hex[:6]
    job_definition = _job_definition_for(position, branch=branch)
    job_queue = _job_queue_for(position, branch=branch)
    environment = [
        {"name": "S3_BUCKET", "value": S3_BUCKET},
        {"name": "S3_DATA_PREFIX", "value": "data"},
        {"name": "LOG_EVERY", "value": "1"},
    ]
    if TRAIN_GIT_SHA:
        # Stamped into benchmark_metrics.json by train.py; benchmark.py uses
        # it to surface per-position SHA divergence across a single run.
        environment.append({"name": "FF_TRAIN_GIT_SHA", "value": TRAIN_GIT_SHA})
    if FF_CUDA_GRAPH:
        # Override only — graphs autodetect ON for sm_80+ in the container, so a
        # value is needed only to force the eager path (forward "0"). K's nested
        # trainer no-ops capture regardless; cuda_graph_enabled() re-checks
        # compute capability so a value on an ineligible GPU is inert.
        environment.append({"name": "FF_CUDA_GRAPH", "value": FF_CUDA_GRAPH})
    if FF_CUDA_GRAPH_FULL:
        # Full-step capture override only — cuda_graph_full_enabled() autodetects
        # ON for sm_80+, so a value is needed only to force "0" (degrade to the
        # model-only graph; FF_CUDA_GRAPH=0 above is what kills graphs entirely).
        environment.append({"name": "FF_CUDA_GRAPH_FULL", "value": FF_CUDA_GRAPH_FULL})
    # Forward the model-artifact prefix into the container so an isolated
    # benchmark run (FF_MODEL_S3_PREFIX=experiments/...) writes its tarball +
    # manifest under that prefix instead of prod ``models/`` — the serving site
    # only polls ``models/``, so an experiment can never hot-swap into prod.
    model_prefix = os.environ.get("FF_MODEL_S3_PREFIX", "").strip()
    if model_prefix:
        environment.append({"name": "FF_MODEL_S3_PREFIX", "value": model_prefix})
    if branch != "full":
        if not split_run_id:
            raise RuntimeError("split_run_id is required for split branch jobs")
        environment.append({"name": "FF_SPLIT_RUN_ID", "value": split_run_id})
    if branch == "cpu":
        environment.extend(
            [
                {"name": "FF_CPU_BRANCH_CORES", "value": "4"},
                {"name": "FF_DEVICE", "value": "cpu"},
                {"name": "LGBM_N_JOBS", "value": "1"},
                {"name": "LOKY_MAX_CPU_COUNT", "value": "4"},
                {"name": "OPENBLAS_NUM_THREADS", "value": "1"},
                {"name": "OMP_NUM_THREADS", "value": "1"},
                {"name": "MKL_NUM_THREADS", "value": "1"},
                {"name": "NUMEXPR_NUM_THREADS", "value": "1"},
            ]
        )
    command = ["--position", position, "--seed", str(seed)]
    if branch != "full":
        command.extend(["--branch", branch, "--split-run-id", split_run_id])

    submit_kwargs = dict(
        jobName=(
            f"ff-{position.lower()}-{branch}-{timestamp}-{suffix}"
            if branch != "full"
            else f"ff-{position.lower()}-{timestamp}-{suffix}"
        ),
        jobQueue=job_queue,
        jobDefinition=job_definition,
        retryStrategy=RETRY_STRATEGY,
        containerOverrides={
            "command": command,
            "environment": environment,
        },
    )
    if depends_on:
        submit_kwargs["dependsOn"] = depends_on
    response = batch.submit_job(**submit_kwargs)
    job_id = response["jobId"]
    key = (position, branch) if branch != "full" else position
    label = f"{position}/{branch}" if branch != "full" else position
    print(f"[{label}] Submitted job {job_id} (queue: {job_queue}, definition: {job_definition})")
    return key, job_id


def _submit_split_for_position(position: str, seed: int, split_run_id: str, batch_client=None):
    """Submit NN, CPU, and merge jobs for one split position."""
    _, nn_job_id = submit_job(
        position,
        seed,
        batch_client,
        branch="nn",
        split_run_id=split_run_id,
    )
    _, cpu_job_id = submit_job(
        position,
        seed,
        batch_client,
        branch="cpu",
        split_run_id=split_run_id,
    )
    merge_key, merge_job_id = submit_job(
        position,
        seed,
        batch_client,
        branch="merge",
        split_run_id=split_run_id,
        depends_on=[{"jobId": nn_job_id}, {"jobId": cpu_job_id}],
    )
    return {
        (position, "nn"): nn_job_id,
        (position, "cpu"): cpu_job_id,
        merge_key: merge_job_id,
    }


def wait_for_jobs(job_ids, timeout_seconds=None, batch_client=None):
    """Poll Batch until all jobs reach a terminal state (or timeout).

    Args:
        job_ids: dict mapping position -> job_id
        timeout_seconds: wall-clock cap; remaining jobs reported as TIMED_OUT.
            Defaults to the module-level WAIT_TIMEOUT_SECONDS.
        batch_client: optional shared boto3 Batch client.

    Returns:
        dict mapping position -> (status, stopped_at_ms)
        status is "SUCCEEDED", "FAILED", or "TIMED_OUT".
        stopped_at_ms is the Batch `stoppedAt` epoch-ms, or None.

    On FAILED, prints a CloudWatch console URL and an `aws logs` command so
    the log stream is one click or one paste away.
    """
    if timeout_seconds is None:
        timeout_seconds = WAIT_TIMEOUT_SECONDS
    batch = batch_client or boto3.client("batch", region_name=AWS_REGION)
    remaining = dict(job_ids)  # position -> job_id
    results = {}  # position -> (status, stopped_at_ms)
    last_status = {}  # job_id -> last printed status
    deadline = time.monotonic() + timeout_seconds

    while remaining:
        if time.monotonic() > deadline:
            print(f"\nTimeout after {timeout_seconds}s; giving up on {list(remaining.keys())}")
            for pos in remaining:
                results[pos] = ("TIMED_OUT", None)
            break

        ids = list(remaining.values())
        response = batch.describe_jobs(jobs=ids)

        for job in response["jobs"]:
            job_id = job["jobId"]
            status = job["status"]
            # Find position for this job_id
            pos = next(p for p, jid in remaining.items() if jid == job_id)

            label = _job_label(pos)
            if last_status.get(job_id) != status:
                print(f"[{label}] {status}")
                last_status[job_id] = status

            if status in TERMINAL_STATES:
                stopped_at = job.get("stoppedAt")  # ms since epoch or None
                results[pos] = (status, stopped_at)
                if status == "FAILED":
                    reason = job.get("statusReason") or ""
                    container = job.get("container") or {}
                    stream = container.get("logStreamName")
                    print(f"[{label}] FAILED reason: {reason}")
                    if stream:
                        print(f"[{label}] log stream: {stream}")
                        print(f"[{label}] console:    {_cloudwatch_url(stream)}")
                        print(
                            f"[{label}] cli:        aws logs get-log-events "
                            f"--log-group-name {BATCH_LOG_GROUP} "
                            f"--log-stream-name '{stream}' --region {AWS_REGION}"
                        )
                    else:
                        print(f"[{label}] (no log stream — job never started a container)")
                del remaining[pos]

        if remaining:
            time.sleep(POLL_INTERVAL_SECONDS)

    return results


def download_artifacts(positions, stopped_at_by_pos=None, s3_client=None):
    """Download model artifacts from S3 back to local position dirs.

    Resolves the per-position artifact via ``models/{POS}/manifest.json`` rather
    than the legacy ``models/{POS}/model.tar.gz`` mirror, which was removed in
    the parallel-train-batch race fix (two concurrent runs writing the same
    legacy key were last-write-wins). Walks ``stable → current → previous``,
    matching the ``src/batch/benchmark.py::download_metrics`` chain so a
    post-train ``launch.py`` invocation pulls the same artifact CI just
    benchmarked.

    If stopped_at_by_pos is provided (position -> ms-epoch stoppedAt), warn
    loudly when the resolved entry's S3 LastModified is older than the job
    finish — that means we're pulling a stale artifact from a prior run.

    Missing manifest and "all entries failed" both surface a per-position
    skip rather than raising, preserving the original "one bad position
    shouldn't kill a six-position download" behaviour.
    """
    from src.shared.model_sync import load_manifest

    s3 = s3_client or boto3.client("s3", region_name=AWS_REGION)
    stopped_at_by_pos = stopped_at_by_pos or {}
    s3_prefix = os.environ.get("FF_MODEL_S3_PREFIX", "models").strip("/")

    for pos in positions:
        local_model_dir = os.path.join(pos.lower(), "outputs", "models")
        os.makedirs(local_model_dir, exist_ok=True)

        try:
            manifest = load_manifest(s3, S3_BUCKET, s3_prefix, pos)
        except ClientError as e:
            print(f"[{pos}] manifest fetch failed: {e!r} — skipping")
            continue
        if manifest is None:
            print(
                f"[{pos}] No manifest at s3://{S3_BUCKET}/{s3_prefix}/{pos}/manifest.json, skipping"
            )
            continue

        # Resolve in priority order — same chain as benchmark.download_metrics
        # so the artifact a post-train launch.py pulls matches what CI just
        # benchmarked. Skip entries with no resolvable key.
        tried: list[tuple[str, str, str]] = []
        extracted = False
        for label in ("stable", "current", "previous"):
            entry = manifest.get(label)
            if not entry or not entry.get("key"):
                continue
            s3_key = entry["key"]

            # Stale-artifact guard: compare remote LastModified to job stoppedAt.
            try:
                head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey", "NotFound"):
                    tried.append((label, s3_key, "missing"))
                    print(
                        f"[{pos}] {label} key s3://{S3_BUCKET}/{s3_key} missing — falling through"
                    )
                    continue
                raise
            remote_modified = head.get("LastModified")  # datetime, tz-aware
            stopped_at_ms = stopped_at_by_pos.get(pos)
            if remote_modified is not None and stopped_at_ms:
                stopped_at_s = stopped_at_ms / 1000.0
                # +5s fudge for clock skew. LastModified is in UTC.
                if remote_modified.timestamp() + 5 < stopped_at_s:
                    print(
                        f"[{pos}] WARNING: s3://{S3_BUCKET}/{s3_key} LastModified "
                        f"({remote_modified.isoformat()}) is older than job "
                        f"stoppedAt ({stopped_at_s}). Artifact may be stale."
                    )

            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
                print(f"[{pos}] Downloading s3://{S3_BUCKET}/{s3_key} (source={label}) ...")
                try:
                    s3.download_file(S3_BUCKET, s3_key, tmp.name)
                    with tarfile.open(tmp.name, "r:gz") as tar:
                        tar.extractall(local_model_dir, filter="data")
                except (ClientError, tarfile.TarError, OSError) as e:
                    tried.append((label, s3_key, repr(e)))
                    print(f"[{pos}] {label} download/extract failed: {e!r} — falling through")
                    continue
            print(f"[{pos}] Extracted to {local_model_dir}/ (source={label})")
            extracted = True
            break

        if not extracted:
            print(f"[{pos}] WARNING: all manifest entries failed; tried={tried!r}")


def _print_plan(positions, seed, *, split: bool = False, split_run_id: str | None = None):
    """--dry-run: print what would be submitted, touch nothing."""
    print("DRY RUN — no AWS calls will be made.")
    print(f"  region:       {AWS_REGION}")
    print(f"  bucket:       {S3_BUCKET}")
    print(f"  queue:        {JOB_QUEUE}")
    print(f"  definition:   {JOB_DEFINITION}")
    if JOB_DEFINITION_REVISION:
        print(f"  definition rev: {JOB_DEFINITION_REVISION}")
    if JOB_DEFINITION_CPU:
        # Render the actual cpu-only set (the same one driving _job_*_for at
        # L192/206) rather than a hardcoded "(K, DST)" that silently goes stale
        # if a position's cpu_only flag changes (#351 F21).
        _cpu_route = "split cpu/merge branches" if split else ", ".join(sorted(CPU_ONLY_POSITIONS))
        _cpu_route = _cpu_route or "none"
        if JOB_QUEUE_CPU:
            print(f"  cpu queue:    {JOB_QUEUE_CPU} ({_cpu_route} route here)")
        print(f"  cpu def:      {JOB_DEFINITION_CPU} ({_cpu_route} route here)")
        if JOB_DEFINITION_CPU_REVISION:
            print(f"  cpu def rev:  {JOB_DEFINITION_CPU_REVISION}")
    print(f"  wait timeout: {WAIT_TIMEOUT_SECONDS}s")
    print(f"  seed:         {seed}")
    if split:
        print("  split:        true")
        print(f"  split run id: {split_run_id}")
    print("  jobs:")
    for pos in positions:
        if split:
            print(
                f"    - {pos:<4} nn    -> queue {_job_queue_for(pos, branch='nn')}, "
                f"definition {_job_definition_for(pos, branch='nn')}, "
                f"command --position {pos} --seed {seed} --branch nn "
                f"--split-run-id {split_run_id}"
            )
            print(
                f"    - {pos:<4} cpu   -> queue {_job_queue_for(pos, branch='cpu')}, "
                f"definition {_job_definition_for(pos, branch='cpu')}, "
                f"command --position {pos} --seed {seed} --branch cpu "
                f"--split-run-id {split_run_id}"
            )
            print(
                f"    - {pos:<4} merge -> queue {_job_queue_for(pos, branch='merge')}, "
                f"definition {_job_definition_for(pos, branch='merge')}, "
                "dependsOn nn+cpu, "
                f"command --position {pos} --seed {seed} --branch merge "
                f"--split-run-id {split_run_id}"
            )
        else:
            print(f"    - {pos:<4} -> definition {_job_definition_for(pos)}")


def _write_job_ids_file(path, expected_positions, job_ids):
    """Record submitted job ids + the expected position set as JSON.

    Best-effort breadcrumb for train-batch.yml's recovery step (and operator
    forensics after a crash): the workflow re-checks these exact job ids when
    wait_for_jobs exits non-zero, so a wait that expired minutes before the
    jobs reached SUCCEEDED no longer skips the benchmark append. Split-mode
    tuple keys serialize as "POS/branch" labels. Never raises — losing the
    breadcrumb must not fail a launch whose jobs are already submitted.
    """
    payload = {
        "expected_positions": list(expected_positions),
        "jobs": {_job_label(key): job_id for key, job_id in job_ids.items()},
    }
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote submitted job ids to {path}")
    except OSError as e:
        print(f"WARNING: could not write job-ids file {path}: {e!r}")


def _append_benchmark_history(positions, *, note):
    """Best-effort: roll succeeded positions into a benchmark_history row + S3
    mirror so a standalone (non-CI) Batch run shows up in the serving app's
    History tab after the container's next boot. Reuses benchmark.py's
    aggregation so there is exactly one code path. Never affects the launch
    exit code — a flaky S3 mirror shouldn't fail an otherwise-successful run.
    """
    if not positions:
        return
    try:
        # Local import: src/batch/benchmark.py imports from this module, so a
        # module-level import here would be circular.
        from src.batch.benchmark import record_benchmark_run

        record_benchmark_run(positions, backend="batch", note=note)
    except Exception as e:  # noqa: BLE001 — append is a convenience, not a gate
        print(f"[benchmark_history] auto-append skipped: {e!r}")


def main():
    parser = argparse.ArgumentParser(description="Launch AWS Batch training jobs")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=ALL_POSITIONS,
        choices=ALL_POSITIONS,
        help="Positions to train",
    )
    parser.add_argument(
        "--wait",
        default="true",
        help="Wait for jobs to complete (true/false)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned submissions and exit without touching AWS",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=None,
        help=f"Override wait timeout in seconds (default: {WAIT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Upload data splits even if S3 ETag matches the local file",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help=(
            "Skip uploading data/splits/*.parquet (assume S3 already current). "
            "Used by CI (train-batch.yml) where the runner has no local data."
        ),
    )
    parser.add_argument(
        "--append-history",
        default="true",
        help=(
            "After a successful wait, roll the succeeded positions into a "
            "benchmark_history row + S3 mirror so the run appears in the serving "
            "History tab (true/false, default true). CI passes false — "
            "train-batch.yml runs benchmark.py --download-only separately with "
            "the image SHA + PR number."
        ),
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Submit split NN/GPU + Ridge/LGBM CPU branch jobs plus a merge job per position.",
    )
    parser.add_argument(
        "--split-run-id",
        default=os.environ.get("FF_SPLIT_RUN_ID"),
        help="Namespace for staged split artifacts. Auto-generated when --split is set.",
    )
    args = parser.parse_args()
    wait = args.wait.lower() == "true"
    append_history = args.append_history.lower() == "true"
    wait_timeout = args.wait_timeout if args.wait_timeout is not None else WAIT_TIMEOUT_SECONDS
    split_run_id = args.split_run_id
    if args.split:
        if not split_run_id:
            split_run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        if not JOB_DEFINITION_CPU or not JOB_QUEUE_CPU:
            parser.error("--split requires FF_JOB_DEFINITION_CPU and FF_JOB_QUEUE_CPU")

    if args.dry_run:
        _print_plan(args.positions, args.seed, split=args.split, split_run_id=split_run_id)
        return

    # Shared boto3 clients — boto3 clients are thread-safe, no need per-thread.
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    batch_client = boto3.client("batch", region_name=AWS_REGION)

    if args.skip_upload:
        print("Skipping data upload (--skip-upload); assuming S3 splits are current.\n")
    else:
        print("Uploading data splits to S3...")
        upload_data(S3_BUCKET, s3_client=s3_client, force=args.force_upload)

    # Submit all positions in parallel
    if args.split:
        print(
            f"Submitting split Batch jobs for {len(args.positions)} positions: "
            f"{args.positions} (split_run_id={split_run_id})"
        )
    else:
        print(f"Submitting {len(args.positions)} Batch jobs: {args.positions}")
    job_ids = {}  # position -> job_id
    submit_failures = []  # positions whose submit_job raised — never enter job_ids
    with ThreadPoolExecutor(max_workers=len(args.positions)) as pool:
        if args.split:
            futures = {
                pool.submit(
                    _submit_split_for_position,
                    pos,
                    args.seed,
                    split_run_id,
                    batch_client,
                ): pos
                for pos in args.positions
            }
        else:
            futures = {
                pool.submit(submit_job, pos, args.seed, batch_client): pos for pos in args.positions
            }
        for future in as_completed(futures):
            pos = futures[future]
            try:
                submitted = future.result()
                if args.split:
                    job_ids.update(submitted)
                else:
                    pos, job_id = submitted
                    job_ids[pos] = job_id
            except Exception as e:
                print(f"[{pos}] FAILED to submit: {e}")
                submit_failures.append(pos)

    if JOB_IDS_FILE:
        _write_job_ids_file(JOB_IDS_FILE, args.positions, job_ids)

    if not wait:
        print("\nJobs submitted. Use 'aws batch describe-jobs' to check status.")
        # A submit failure leaves the position out of job_ids (and out of the
        # wait-path results below), so it must fail loudly here too.
        if submit_failures:
            print(f"ERROR: {len(submit_failures)} positions failed to submit: {submit_failures}")
            sys.exit(1)
        return

    # Wait for all jobs to complete
    print(
        f"\nWaiting for {len(job_ids)} jobs to complete (polling every {POLL_INTERVAL_SECONDS}s, timeout {wait_timeout}s)..."
    )
    results = wait_for_jobs(job_ids, timeout_seconds=wait_timeout, batch_client=batch_client)

    if args.split:
        succeeded = [
            key[0]
            for key, (status, _) in results.items()
            if isinstance(key, tuple) and key[1] == "merge" and status == "SUCCEEDED"
        ]
    else:
        succeeded = [pos for pos, (status, _) in results.items() if status == "SUCCEEDED"]
    failed = [key for key, (status, _) in results.items() if status == "FAILED"]
    timed_out = [key for key, (status, _) in results.items() if status == "TIMED_OUT"]
    if args.split:
        stopped_at_by_pos = {
            key[0]: stopped_at
            for key, (_, stopped_at) in results.items()
            if isinstance(key, tuple) and key[1] == "merge"
        }
    else:
        stopped_at_by_pos = {pos: stopped_at for pos, (_, stopped_at) in results.items()}

    if failed:
        print(f"\nFailed positions: {failed}")
    if timed_out:
        # TIMED_OUT lands here when `wait_for_jobs` hits the wall-clock cap
        # before Batch reports a terminal state — the job may still be
        # running; failing it from this view prevents downstream artifact
        # downloads but lets the operator decide whether to wait, cancel,
        # or rerun.
        print(f"\nTimed-out positions: {timed_out}")
    if succeeded:
        print(f"\nSucceeded: {succeeded}")
        print("Downloading model artifacts...")
        download_artifacts(succeeded, stopped_at_by_pos=stopped_at_by_pos, s3_client=s3_client)
        if append_history:
            _append_benchmark_history(succeeded, note="Standalone Batch run")

    print("\nAll done.")

    # train-batch.yml's step comment claims "launch.py blocks on
    # wait_for_jobs(); returns 0 only if all positions reach SUCCEEDED" —
    # but the previous code always exited 0, so a position that FAILED or
    # TIMED_OUT was reported on stdout and then silently lost. The "Verify
    # model artifact freshness" step that follows catches MISSING
    # manifests but not the case where a stale prior-run artifact happens
    # to satisfy the freshness window. Exit non-zero so the workflow's
    # post-step actually surfaces failed positions; succeeded artifacts
    # have already been downloaded above for forensics. ``submit_failures``
    # covers positions that never reached the wait loop (submit_job raised),
    # which are otherwise absent from ``results`` and silently ignored.
    if failed or timed_out or submit_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Bounded campaign allocations; each step executes in a fresh child process."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.tuning.campaign_contracts import EXECUTION_ENV, ROOT, identity, source_fingerprint
from src.tuning.campaign_io import CellCheckpoint, Journal, atomic_json, clean, verify_snapshot


def verify_runtime(manifest, data_dir):
    if (
        identity({k: v for k, v in manifest.items() if k != "manifest_id"})
        != manifest["manifest_id"]
    ):
        raise RuntimeError("Campaign manifest checksum mismatch")
    if manifest["backend"] == "local":
        if source_fingerprint() != manifest["source_fingerprint"]:
            raise RuntimeError("Campaign source changed; create a new campaign")
    else:
        from src.artifacts.source import image_source_sha

        stamp = image_source_sha()
        if stamp != manifest["code_sha"]:
            raise RuntimeError("Container source differs from the frozen campaign image")
    data_dir = Path(data_dir)
    seal = json.loads((data_dir / "splits/release-inputs.json").read_text())
    if identity(seal) != manifest["dataset_id"]:
        raise RuntimeError("Campaign input release changed")
    verify_snapshot(data_dir, seal)


def child_environment(manifest, step, unit, output, data_dir):
    env = dict(os.environ)
    for key in tuple(env):
        if (key.startswith("FF_") and not key.startswith("FF_RESULT_")) or (
            manifest["backend"] == "local" and key in EXECUTION_ENV
        ):
            env.pop(key, None)
    # Batch's immutable job definition/image supply native-library defaults
    # such as OPENBLAS_NUM_THREADS=1. Preserve them unless the frozen manifest
    # explicitly overrides them; dropping the caps oversubscribes Ridge CV.
    env.update(manifest["execution_environment"])
    env.update(step["env"])
    device = step["options"].get("device", step["env"].get("FF_DEVICE"))
    if device == "auto":
        device = manifest["execution_environment"].get("FF_DEVICE", "auto")
    if manifest["backend"] == "batch":
        device = "cpu" if unit["resource"] == "cpu" else "cuda"
        env.update(S3_BUCKET=manifest["bucket"], FF_CAMPAIGN_BUCKET=manifest["bucket"])
    else:
        env.pop("S3_BUCKET", None)
    if device:
        env["FF_DEVICE"] = device
    prefix = f"campaign_runs/{manifest['id']}/units/{unit['id']}/steps/{step['id']}"
    env.update(
        {
            "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
            "FF_CAMPAIGN_ID": manifest["id"],
            "FF_CAMPAIGN_MANIFEST_ID": manifest["manifest_id"],
            "FF_CAMPAIGN_STEP": step["id"],
            "FF_CAMPAIGN_UNIT": unit["id"],
            "FF_CAMPAIGN_DATA_DIR": str(data_dir),
            "FF_CACHE_DIR": str(Path(data_dir) / "raw"),
            "FF_CAMPAIGN_STUDY_DIR": str(output / "studies"),
            "FF_CAMPAIGN_STEP_PREFIX": prefix,
            "FF_DATA_RELEASE": manifest["dataset_id"],
            "FF_DATASET_ID": manifest["dataset_id"],
            "FF_DATA_FORMAT": "data-release-v1",
            "S3_DATA_PREFIX": "data",
            "FF_MODEL_S3_PREFIX": f"{prefix}/models",
            "FF_MODEL_S3_BUCKET": "",
            "FF_TRAIN_GIT_SHA": manifest["code_sha"],
            "FF_FRESH": "1" if manifest["fresh"] or step["kind"] == "benchmark" else "0",
        }
    )
    if manifest["backend"] == "batch":
        env["FF_CAMPAIGN_STUDY_PREFIX"] = f"{prefix}/studies"
        # Preserve the Batch NN tuning profile, while the existing tuner
        # still resolves hardware support and K/DST stacking fallbacks.
        if step["kind"] == "nn_tune" and unit["resource"] == "gpu":
            env.setdefault("FF_CUDA_GRAPH_FULL", "1")
    return env


def _invoke_main(module, argv):
    import importlib

    previous = sys.argv
    sys.argv = [module, *argv]
    try:
        importlib.import_module(module).main()
    finally:
        sys.argv = previous


def _flags(options, *, skip=()):
    result = []
    for key, value in options.items():
        if key in skip:
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(option)
        elif isinstance(value, list):
            result.extend([option, *map(str, value)])
        else:
            result.extend([option, str(value)])
    return result


def _benchmark_cell(task):
    from src.benchmarking.benchmark import (
        _cohorts_block,
        _significance_block,
        run_one,
        score_one_origin,
    )
    from src.shared.benchmark_utils import summarize_pipeline_result
    from src.training.context import RunContext, use_context

    pos, origin = task["position"], task["origin"]
    context = RunContext(
        Path(task["output"]), Path(task["data"]), seed=task["seed"], reuse_results=False
    )
    start = time.monotonic()
    with use_context(context):
        if origin is not None:
            _, summary = score_one_origin(pos, origin, seed=context.seed)
        else:
            result = run_one(pos, context=context)
            summary = summarize_pipeline_result(pos, result)
            summary["cohorts"] = _cohorts_block(pos, result)
            if task.get("significance"):
                summary["significance"] = _significance_block(pos, result)
    summary["elapsed_sec"] = round(time.monotonic() - start, 3)
    row = {"position": pos, "origin": origin, "summary": clean(summary)}
    atomic_json(Path(task["output"]) / "cell_result.json", row)
    return row


def _benchmark_failure(task, exc):
    return {"position": task["position"], "origin": task["origin"], "error": str(exc)[:1000]}


def benchmark_step(step, data_dir, output, checkpoint):
    from src.benchmarking.benchmark import finalize_rolling_origin
    from src.benchmarking.parallel_train import _default_jobs, physical_cores
    from src.config import ROLLING_ORIGIN_TEST_SEASONS
    from src.shared.core_pool import start_coordinator
    from src.tuning._execution import run_tasks
    from src.tuning.ab_harness import _init_spec_worker

    origins = ROLLING_ORIGIN_TEST_SEASONS if step["options"].get("rolling_origin") else [None]
    tasks = [
        {
            "position": pos,
            "origin": origin,
            "seed": step["options"].get("seed", 42),
            "data": str(data_dir),
            "output": str(output / f"{pos}-{origin or 'holdout'}"),
            "key": f"{pos}-{origin or 'holdout'}",
            "significance": step["options"].get("significance", False),
        }
        for pos in step["positions"]
        for origin in origins
    ]
    rows, pending = [], []
    for task in tasks:
        previous = checkpoint.load(task["key"], Path(task["output"]))
        if previous is None:
            pending.append(task)
        else:
            rows.append(previous)
    jobs = max(1, min(step["options"].get("jobs", _default_jobs(min(len(tasks), 6))), len(pending)))
    cores = physical_cores()
    with tempfile.TemporaryDirectory(prefix="ff-campaign-pool-") as directory:
        address, active, stop = start_coordinator(cores, directory)
        try:
            active(jobs)

            def save(index, row):
                if "error" not in row:
                    task = pending[index]
                    checkpoint.save(task["key"], row, Path(task["output"]))

            rows.extend(
                run_tasks(
                    pending,
                    _benchmark_cell,
                    max_workers=jobs,
                    on_error=_benchmark_failure,
                    on_result=save,
                    initializer=_init_spec_worker,
                    initargs=(cores, 5, address),
                )
            )
        finally:
            stop()
    errors = [row for row in rows if "error" in row]
    if errors:
        atomic_json(output / "benchmark_errors.json", errors)
        raise RuntimeError(f"{len(errors)} benchmark cells failed")
    rows.sort(key=lambda row: (step["positions"].index(row["position"]), row["origin"] or 0))
    if not step["options"].get("rolling_origin"):
        return {"results": [row["summary"] for row in rows], "fresh_training": True}
    return {
        "results": [
            finalize_rolling_origin(
                pos, [(row["origin"], row["summary"]) for row in rows if row["position"] == pos]
            )
            for pos in step["positions"]
        ],
        "fresh_training": True,
    }


def execute_step(manifest, step, data_dir, output):
    from src.training.context import RunContext, use_context

    verify_runtime(manifest, data_dir)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch

    device = os.environ.get("FF_DEVICE", "auto")
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(f"Requested campaign device is unavailable: {device}")
    link = output / "data"
    if not link.exists():
        try:
            link.symlink_to(data_dir, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            shutil.copytree(data_dir, link)
    os.chdir(output)  # This function runs only in a dedicated child.
    options = step["options"]
    s3 = None
    if manifest["backend"] == "batch":
        import boto3

        s3 = boto3.client("s3", region_name=manifest["region"])
    journal = Journal(
        output / ".checkpoints", s3=s3, bucket=manifest["bucket"], campaign_id=manifest["id"]
    )
    checkpoint = CellCheckpoint(
        journal,
        f"units/{os.environ['FF_CAMPAIGN_UNIT']}/steps/{step['id']}/cells",
        manifest["manifest_id"],
    )
    with use_context(RunContext(output, Path(data_dir), seed=options.get("seed", 42))):
        if step["kind"] == "ab":
            from src.tuning.ab_harness import aggregate, resolve_spec, run_ab

            if manifest["backend"] == "batch":
                from src.tuning.ab_batch import run_batch_entry
                from src.tuning.launch_ab import collect_results

                pos = step["positions"][0]
                run_id = f"{manifest['id']}-{step['id']}-{pos}"
                prefix = f"campaign_runs/{manifest['id']}/ab"
                os.environ.update(
                    FF_TUNE_AB_SPEC=step["spec"], FF_AB_RUN_ID=run_id, FF_AB_S3_PREFIX=prefix
                )
                if "seeds" in options:
                    os.environ["FF_AB_SEEDS"] = ",".join(map(str, options["seeds"]))
                if options.get("only"):
                    os.environ["FF_AB_ONLY"] = ",".join(options["only"])
                if options.get("stacked_seeds"):
                    os.environ["FF_AB_STACKED"] = "1"
                    os.environ["FF_AB_STACKED_EPOCHS"] = str(options.get("stacked_epochs", 30))
                os.environ["FF_FEATURE_CACHE_DISABLE"] = (
                    "0" if options.get("feature_cache") else "1"
                )
                run_batch_entry(pos)
                spec = resolve_spec(
                    step["spec"],
                    positions=[pos],
                    seeds=options.get("seeds"),
                    only=options.get("only"),
                )
                result = aggregate(
                    spec,
                    collect_results(
                        spec,
                        bucket=manifest["bucket"],
                        s3_prefix=prefix,
                        run_id=run_id,
                        s3_client=s3,
                    ),
                )
            else:
                result = run_ab(
                    step["spec"],
                    positions=step["positions"],
                    fresh=manifest["fresh"],
                    checkpoint=checkpoint,
                    **{k: v for k, v in options.items() if k not in {"device", "max_cells"}},
                )
            if result.get("failed") or any(
                row.get("status") != "ok" for row in result.get("sentinel", [])
            ):
                raise RuntimeError("A/B campaign step contains failed cells or sentinel violations")
        elif step["kind"] == "benchmark":
            result = benchmark_step(step, data_dir, output, checkpoint)
        else:
            opts = dict(options)
            # A successful exit must produce this attempt's result; leave study
            # checkpoints in place while removing stale report files.
            for stale in output.glob("tune_*results*.json"):
                stale.unlink()
            if step["kind"] == "nn_tune":
                if manifest["backend"] == "batch":
                    opts.setdefault("parallel_backend", "auto")
                    opts.setdefault("n_jobs", "auto")
                argv = [*step["positions"], *_flags(opts, skip=("device",))]
                if manifest["backend"] == "batch":
                    argv.append("--checkpoint-s3")
                _invoke_main("src.tuning.tune_nn", argv)
                paths = list(output.glob("tune_nn_results*.json"))
            else:
                argv = [*step["positions"], *_flags(opts, skip=("seeds",))]
                if "seeds" in opts:
                    argv += ["--seeds", ",".join(map(str, opts["seeds"]))]
                _invoke_main("src.tuning.tune_lgbm", argv)
                paths = [output / "tune_lgbm_results.json"]
            if not paths or not all(path.is_file() for path in paths):
                raise RuntimeError("Tuner completed without its results artifact")
            result = {path.name: json.loads(path.read_text()) for path in paths}
    verify_runtime(manifest, data_dir)
    atomic_json(output / "result.json", clean(result))


def run_unit(manifest, unit, journal, *, data_dir, directory, attempt=1):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    verify_runtime(manifest, data_dir)
    name = f"units/{unit['id']}/progress.json"
    progress, etag = journal.read(name)
    if progress is None:
        progress = {"manifest_id": manifest["manifest_id"], "unit": unit["id"], "steps": {}}
    if progress["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Campaign progress belongs to a different manifest")
    if progress.get("attempt", 0) > attempt:
        raise RuntimeError("A newer worker owns this campaign unit")
    progress["attempt"] = attempt
    manifest_file = directory / "manifest.json"
    atomic_json(manifest_file, manifest)
    failed = False
    for original in manifest.get("execution_steps", manifest["spec"]["steps"]):
        if original["id"] not in unit["steps"]:
            continue
        step = {
            **original,
            "positions": [unit["position"]] if unit["position"] else original["positions"],
        }
        previous = progress["steps"].get(step["id"], {})
        output = directory / "steps" / step["id"]
        prefix = f"units/{unit['id']}/steps/{step['id']}"
        if previous.get("state") == "SUCCEEDED" and journal.outputs_valid(
            output, prefix, previous.get("outputs", {})
        ):
            continue
        verify_runtime(manifest, data_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").unlink(missing_ok=True)
        progress["steps"][step["id"]] = {"state": "RUNNING", "started_at": time.time()}
        etag = journal.write(name, progress, etag)
        argv = [
            sys.executable,
            "-m",
            "src.tuning.campaign_worker",
            "--execute",
            str(manifest_file),
            "--step",
            step["id"],
            "--output",
            str(output),
            "--positions",
            *step["positions"],
        ]
        start = time.monotonic()
        with (output / "worker.log").open("ab") as log:
            child = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=child_environment(manifest, step, unit, output, data_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            old_handler = signal.getsignal(signal.SIGTERM)

            def terminate(signum, frame, child=child):
                _stop_child(child)
                raise SystemExit(143)

            signal.signal(signal.SIGTERM, terminate)
            try:
                code = child.wait()
            except BaseException:
                _stop_child(child)
                raise
            finally:
                signal.signal(signal.SIGTERM, old_handler)
        verify_runtime(manifest, data_dir)
        files = journal.upload_outputs(output, f"units/{unit['id']}/steps/{step['id']}")
        success = code == 0 and (output / "result.json").is_file()
        progress["steps"][step["id"]] = {
            "state": "SUCCEEDED" if success else "FAILED",
            "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "outputs": files,
        }
        etag = journal.write(name, progress, etag)
        failed |= not success
        print(
            f"[campaign] {unit['id']}/{step['id']}: {progress['steps'][step['id']]['state']}",
            flush=True,
        )
    return 1 if failed else 0


def _stop_child(child):
    if child.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False, capture_output=True
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=55)
    if child.poll() is None:
        if os.name == "nt":
            child.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def resolve_step(step, backend):
    """Resolve a spec in a fresh interpreter, including env-built variant lists."""
    options = dict(step["options"])
    os.environ.update(step["env"])
    if options.get("device"):
        os.environ["FF_DEVICE"] = options["device"]
    if step["kind"] == "ab":
        from src.shared.utils import cuda_enabled
        from src.tuning.ab_ensemble_seeds import stacked_default_seed_list
        from src.tuning.ab_harness import build_cells, resolve_spec

        stacked = options.get("stacked_seeds", backend == "local" and cuda_enabled())
        spec = resolve_spec(
            step["spec"],
            positions=step["positions"],
            seeds=options.get("seeds"),
            only=options.get("only"),
            default_seeds=stacked_default_seed_list() if stacked else None,
        )
        if stacked and not spec.supports_stacked:
            if options.get("stacked_seeds"):
                raise ValueError("This A/B spec does not support stacked execution")
            stacked = False
            spec = resolve_spec(
                step["spec"],
                positions=step["positions"],
                seeds=options.get("seeds"),
                only=options.get("only"),
            )
        if backend == "batch" and len(build_cells(spec)) > options.get("max_cells", 120):
            raise ValueError("A/B grid exceeds max_cells; raise the explicit budget if intended")
        options.update(seeds=list(spec.seeds), only=list(spec.variants), stacked_seeds=stacked)
        step = {**step, "positions": list(spec.positions)}
    return {**step, "options": options}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute")
    mode.add_argument("--resolve")
    parser.add_argument("--backend", choices=("local", "batch"), default="local")
    parser.add_argument("--step")
    parser.add_argument("--output", required=True)
    parser.add_argument("--positions", nargs="+")
    args = parser.parse_args()
    if args.resolve:
        atomic_json(
            args.output, resolve_step(json.loads(Path(args.resolve).read_text()), args.backend)
        )
        return
    if not args.step or not args.positions:
        parser.error("--execute requires --step and --positions")
    manifest = json.loads(Path(args.execute).read_text())
    step = next(
        step
        for step in manifest.get("execution_steps", manifest["spec"]["steps"])
        if step["id"] == args.step
    )
    if not set(args.positions) <= set(step["positions"]):
        raise ValueError("Worker positions differ from campaign")
    step = {**step, "positions": args.positions}
    execute_step(manifest, step, Path(os.environ["FF_CAMPAIGN_DATA_DIR"]), Path(args.output))


if __name__ == "__main__":
    main()

# ADR-0020: Batch GPU Execution Path For The Shared A/B Harness

**Status:** Accepted
**Date:** 2026-06-11
**Supersedes:** none
**Related:** ADR-0013, ADR-0015, ADR-0017

## Context

The shared A/B harness ([src/tuning/ab_harness.py](../../src/tuning/ab_harness.py))
resolved jobs for local environments only — 5080 CUDA `-j6`, 9950X3D 16-core
pool, Mac sequential. A metric-path A/B with no local GPU available ran
sequentially on a Mac CPU (~80 min, FP32/eager), which is not the path
production trains on: the Batch fleet is g6/L4 (sm_89, g5/A10G fallback) with
FP16 AMP and CUDA graphs autodetect-ON — and graphs are deliberately **not**
numerically inert (ADR-0017), so an eager-CPU A/B can mis-rank variants the
graphed GPU path would order differently.

Constraints: production images build only from `main`
([batch-image.yml](../../.github/workflows/batch-image.yml)); a branch A/B
needs a branch image that must not trigger `train-batch.yml` (its
`workflow_run` trigger is filtered to `branches: [main]` — safe) nor leak
into production resolution paths; new A/B files must stay in `src/tuning/`
(`src/batch/` and `src/shared/` edits fire a 6-position retrain via
`scope_positions`); a grid is ~30–60 GPU-cells of a few minutes each, so the
shape must be cost-proportional (~$1–2/run), not a standing fleet.

## Decision

One Spot GPU job **per position**, riding the existing `--mode=tune` env
dispatch; cells run **sequentially in-container with per-cell S3
checkpointing**; the launcher collects and aggregates.

- **Launcher** [src/tuning/launch_ab.py](../../src/tuning/launch_ab.py)
  (mirrors [launch_tune.py](../../src/tuning/launch_tune.py)): submits one job
  per spec position with command `--position {POS} --mode tune` and env
  `FF_TUNE_AB_SPEC=<dotted spec>` + `FF_AB_RUN_ID` (+ optional
  `FF_AB_SEEDS`/`FF_AB_ONLY` overrides); reuses `wait_for_jobs` /
  `RETRY_STRATEGY` from `src/batch/launch.py`; then downloads the per-cell
  JSONs and feeds them through the harness's own `aggregate()` /
  `print_report()` — same tables and Ridge-invariance sentinel as a local run.
- **Container entry** [src/tuning/ab_batch.py](../../src/tuning/ab_batch.py):
  `tune_nn.main()` dispatches on `FF_TUNE_AB_SPEC` (the PR #1142
  `FF_TUNE_ENSEMBLE_AB` pattern — the image ENTRYPOINT is pinned to
  `src.batch.train`, and editing that file fires a retrain). The entry pulls
  data via tune_nn's `_ensure_data_from_s3`, resolves the spec, and runs this
  position's variant × seed cells through the harness's `run_cell` (chdir +
  tmp-dir isolated — served `{pos}/outputs` are never written), uploading each
  result to `s3://{bucket}/ab_runs/{run_id}/cells/{POS}-{variant}-{seed}.json`
  on completion. A Spot reclaim resumes: Batch's retry re-enters, lists the
  prefix, and skips completed cells.
- **Branch images**: `batch-image.yml` dispatched on a non-`main` ref now
  pushes only the SHA tag (no `:latest` — the EC2 rollback path and the
  ablate-rb-gate / retune-lgbm workflows resolve `:latest`) and skips
  job-definition registration (bare-name submissions resolve to the latest
  active revision). Because `containerOverrides` cannot swap an image, the
  launcher instead clones the production GPU definition into a separate
  **`ff-ab-job`** definition with the image re-pointed at
  `ff-training:{--image-sha}` (idempotent: reuses the latest revision when the
  image already matches). Production job-definition names never point at
  branch images.
- **Workflow** [ab-batch.yml](../../.github/workflows/ab-batch.yml)
  (`workflow_dispatch`, mirrors retune-nn-batch.yml): inputs for
  spec/positions/seeds/only/image SHA/cuda_graph; drives `launch_ab` and
  prints the report to the step summary — so an A/B can be kicked from the
  Actions UI by an agent or operator with no local AWS credentials.

No model artifact upload, no `benchmark_history/` append, no ECS touch — cell
results and the aggregated `summary.json` live only under `ab_runs/{run_id}/`.
`--cuda-graph auto` (default) leaves the container's sm_80+ autodetect ON, i.e.
the production graphed metric path; `false` forwards the force-off override
for a bit-comparable eager A/B (ADR-0017).

## Options considered

| Option | 36-cell grid (2var×3seed×6pos) | Per-job overhead (pull+data+imports ~1.5–3 min) | Spot-reclaim cost | Moving parts |
|---|---|---|---|---|
| **Per-position jobs, sequential cells + per-cell S3 checkpoint (chosen)** | ~20–35 min, ~$1 | ×6 | remaining cells of one position | launcher + entry, mirrors launch_tune |
| Array job, one cell per child | ~15–25 min, ~$1.50 | ×36–60 | one cell (free via retry) | + index→cell contract between submitter & image; 36–60 containers of CloudWatch noise |
| Single job, whole grid on one host | ~2.5–4 h | ×1 | hours, unless the same checkpointing is added anyway | fewest, but wastes the fleet |

## Chosen rationale

Per-position fan-out is the twice-proven shape (ADR-0015's tune fleet, D13's
training fan-out) and the env dispatch is the once-proven retrain-safe routing
(PR #1142). Cells stay sequential in-container because each A/B cell is a
*full* pipeline run (Ridge + LightGBM + both NNs) and the 4-vCPU job shape has
no headroom for the concurrent-cell tricks the 16-core local boxes use; the
position axis supplies the parallelism (6 of the 16 hosts the 64-vCPU Spot
quota fits, so an A/B coexists with a concurrent train fan-out). Per-cell
checkpointing gives the resilience array jobs get for free, without the
index→cell contract or the 10× container-overhead multiplier.

## Rejected

- **Array job per cell** — the right scale-up if grids grow past ~100 cells;
  at 30–60 cells the extra wall-clock win is minutes while the overhead,
  log noise, and submitter↔image grid contract are permanent.
- **Registering branch images as `ff-training-job` revisions** — revision
  pinning protects the CI path, but bare-name submitters (manual
  `aws batch submit-job`, local `launch.py` without
  `FF_JOB_DEFINITION_REVISION`) resolve to the latest active revision; a
  separate `ff-ab-job` name removes the hazard class.
- **In-container cell concurrency (e.g. the tuner's 4-worker MPS pattern)** —
  tune trials are attention-NN-only; an A/B cell's LightGBM/Ridge stages would
  contend for the 4 vCPUs. Revisit with measurements if per-position slices
  grow past ~10 cells.
- **A standing GPU instance for A/Bs** — cost-disproportionate for a few
  $1–2 runs per week; Spot submit-per-run matches the existing train/tune
  economics.

## Consequence

Metric-path A/Bs (loss/arch/graph-sensitive changes) can now be validated on
the exact production path before merge: dispatch `batch-image.yml` on the
branch, then `ab-batch.yml` (or `python -m src.tuning.launch_ab --spec … 
--image-sha <branch head>`). The harness's spec contract is unchanged — any
existing `ab_*` spec runs on Batch unmodified. The launcher enforces a
120-cell cost guard (`--max-cells`), the analog of retune-nn-batch's
100-trial ceiling.

## References

[src/tuning/launch_ab.py](../../src/tuning/launch_ab.py) (submit / job-def
clone / collect / aggregate), [src/tuning/ab_batch.py](../../src/tuning/ab_batch.py)
(container entry, per-cell checkpoint + resume), [src/tuning/tune_nn.py](../../src/tuning/tune_nn.py)
(`FF_TUNE_AB_SPEC` dispatch), [.github/workflows/ab-batch.yml](../../.github/workflows/ab-batch.yml),
[.github/workflows/batch-image.yml](../../.github/workflows/batch-image.yml)
(non-main guards), [todo/ab_harness_priority.md](../../todo/ab_harness_priority.md),
tests: [tests/tuning/test_ab_batch.py](../../tests/tuning/test_ab_batch.py),
[tests/tuning/test_launch_ab.py](../../tests/tuning/test_launch_ab.py).

## Changelog

- **2026-06-11** — Initial decision: per-position Spot jobs via the
  `--mode=tune` env dispatch, per-cell S3 checkpoint/resume, `ff-ab-job`
  cloned definition for arbitrary image SHAs, branch-image guards in
  batch-image.yml, ab-batch.yml dispatch workflow.
- **2026-06-15** — Two new `ab_*` consumers of this path: the stackable
  legacy NN ablations `attn_arch` and `scheduler_type` were ported to
  `ab_harness` specs ([src/tuning/ab_attn_arch.py](../../src/tuning/ab_attn_arch.py),
  [src/tuning/ab_scheduler_type.py](../../src/tuning/ab_scheduler_type.py)) so
  they run on the Spot fleet at the GPU-default N=24 stacked width via
  `launch_ab --spec …`. `rb_gate`/`batch_lr` were evaluated and left eager.

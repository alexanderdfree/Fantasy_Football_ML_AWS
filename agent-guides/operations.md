# CI, serving and infrastructure operations

Read only the sections relevant to the task. [AGENTS.md](../AGENTS.md) supplies the shared entrypoint; current code/config and linked decisions supply operational state. Dated measurements describe their recorded regime, not a promise about today.

## CI & training

- [tests.yml](../.github/workflows/tests.yml) runs ruff and pytest with `uv`,
  position/serving/shared shards and per-shard coverage. The
  [Codecov policy](../codecov.yml) owns the 80% component targets and diagnostic
  CLI exclusions. If `Run Tests` silently stops firing after rapid force-push,
  run `pytest` locally before an otherwise-authorized squash merge. Follow
  [delivery gates](delivery.md#pr-and-merge-gates); this exception does not
  authorize `--admin` or bypass another failing/pending check.
- [batch-image.yml](../.github/workflows/batch-image.yml) builds the image;
  `BATCH_ACTIVE` selects [Batch](../.github/workflows/train-batch.yml) versus
  [EC2 rollback](../.github/workflows/train-ec2.yml). `BATCH_SPLIT_ACTIVE`
  selects NN/CPU/merge branches versus monolithic Batch jobs. An unset split
  flag selects monolithic mode; `workflow_dispatch` is the explicit break-glass
  path around the normal backend-selection gate. Verify the live variables
  and artifact state before reporting which route ran.
- [scope_positions.py](../src/scripts/scope_positions.py) owns retrain scope for
  both backends. Update [its contract tests](../tests/scripts/test_scope_positions.py)
  when changing the global-trigger list. [ADR-0019](../docs/adr/0019-split-batch-training-gpu-nn-cpu-ridge-lgbm.md#decision)
  preserves the split-job and merge-only publication contract.
- The GPU fleet uses **one diversified CE**, with GPU instance types and capacity limits configured
  by [infra/batch/setup.sh](../infra/batch/setup.sh). Queue-order fallback is
  not a remedy for Spot-capacity starvation. [ADR-0013](../docs/adr/0013-spot-fan-out-via-aws-batch.md#changelog)
  retains the two-CE reversal, dated quota changes and timing measurements;
  [the Batch runbook](../docs/batch_design.md) and [EC2 runbook](../docs/ec2_design.md)
  hold the operational details. Verify current cloud quotas before sizing a run.
- Preserve the post-train `ecs_rollout`: weight-only polling cannot reload an
  architecture-changing checkpoint safely. The
  [serving-staleness incident](../todo/fixed-archive/fixed-attention-nn-mae-rmse-missing-for-qb-rb-wr-te-in-the-comparison-model-5a0bb8e1.md)
  retains the removal/reinstatement history and NaN failure evidence.
- `deploy.yml` — ECS Flask deploy.
- Optional scheduled maintenance: [ADR-0028](../docs/adr/0028-scheduled-data-maintenance.md) and the [runbook](../infra/maintenance/README.md) define two small Standard workflows for daily/intraday inference and a Thursday correction pass. The stack defaults to disabled schedules and shadow output; verify live stack mode and repository cutover variables before assuming AWS owns publication. `maintenance-image.yml` updates an installed runtime only after its matching serving source deploys.

Use current workflow/configuration for behavior and live state for deployment facts. Historical measurements belong in the linked decisions and incidents.

### Project facts & infrastructure
- Device/dtype constraints, including T4 BF16 exclusion, belong in [platform policy](platform.md#device-and-dtype-policy); an accepted `autocast` argument does not establish kernel support.
- **GPU arch is per-CUDA-wheel, not per-version:** sm_xx comes from the cuXXX wheel build (inspect the selected cuXXX wheel and current requirements) — verify the wheel index, don't trust a version→GPU claim.
- **AWS Batch UserData must be MIME multipart, not raw bash**, else the compute environment flips INVALID (recovery: republish + disable→update→enable).
- **`src/serving/release_changelog.json` is the owner-curated model-release log** rendered by the dashboard's Changelog & Timeline tab (`/api/timeline`). When a notable model release lands (architecture change, tuning wave, full-fleet retrain milestone), append an entry `{version, date, family, model, title, summary, mae, r2, prev_mae, pr}` with metrics from the corresponding `benchmark_history/` run — schema is pinned by [tests/test_app_timeline.py](../tests/test_app_timeline.py); the weekly head-to-head log on that tab is computed live and needs no upkeep.
- **Local benchmark runs must mirror to S3** (`benchmark.py::_maybe_upload_to_s3`, env-gated, `--no-sync` opt-out) to reach the website History tab; don't lift that helper into `src/shared/` (fires a 6-position retrain).
- **Diagnostics live in `src/analysis/`** (import shared helpers); editing `src/shared/` or `src/{pos}/` to wire in read-only tooling fires a 6-position retrain — expose a predicate fn for later drop-in.
- **Split `fantasy_points` is skill-only** (≈0 for K, absent for DST); per-position label analysis from splits is QB/RB/WR/TE-only — K/DST totals exist only post-pipeline on `result["test_df"]`.
- **`/health` distinguishes cold start (200) from affirmative failure (503):** shared initialization or position errors with no loaded positions return 503; useful partial serving remains 200. Public errors are sanitized. Base-data retry or valid hydration clears the shared failure marker.
- **Serving caches publish as one generation:** `models/predictions_cache/current.json` selects an immutable manifest and all four verified cache files. CI builds the complete compatible cache generation before rollout; deployment verifies that source before mutating ECS. Loose files and the predecessor `cache.tar.gz` cannot substitute for it. Runtime/local mode persists revocations across workers; production artifact-only mode does not load models. See [ADR-0027](../docs/adr/0027-versioned-prediction-and-execution-contracts.md).
- **Injury/return:** models over-predict 2+ wk returners (bias, not MAE); attention PE is slot-indexed (`arange(seq_len)`) so gap-blind — attention ties plain NN on returners, LGBM not actually robust (#623).

# Scheduled data maintenance

This stack adds two AWS Step Functions **Standard** workflows. EventBridge
Scheduler launches isolated ARM64 Fargate jobs; the web service continues to
download artifacts from S3. All model/feature implementations and evaluation
season boundaries remain owned by the existing pipelines.

| Workflow | Stages |
|---|---|
| Inference | Pin data/models → source check and inference → validate/publish → verify the public artifact |
| Weekly correction | Pin models → build/stage data and serving cache → activate compatible data → verify rollout → inference/public verification |

The source check and ML work run inside one worker process per job. The workflow
separates only durable job boundaries. A worker receipt is required before
publication, including when an ECS launch itself succeeded. Retries reuse a
completed run's staged outputs. Model manifest changes and a changed serving
revision invalidate a prepared run instead of relabeling older inputs.

## Deployment defaults

**The template defaults to `Mode=shadow` and `EnableSchedules=false`.** Creating
the stack does not enable production maintenance. Shadow jobs write immutable
data candidates and `maintenance/runs/` outputs, and do not change the current
data pointers, serving cache, live forecasts, or ECS service. Their source and
verification receipts live under `maintenance/shadow/`.

The active schedule is:

- Daily source check and inference: 06:15 `America/New_York`.
- Historical correction pass: Thursday 07:15 `America/New_York`.
- Intraday inference: hourly at minute 45 UTC (configurable).
- Independent freshness verifier: every 15 minutes, active mode only.

The daily baseline does not replace intraday forecasts. The public forecast's
current four-hour input-age limit remains in force. NFL/nflverse publication
times guide the morning/Thursday windows; source availability and the producer's
coverage checks determine whether a run can publish. A provider's missing season
is not silently converted into an observed zero.

The daily report records a publication expectation for each source, validates
the upstream inventory/CSV schema, and inspects the pinned raw files actually
consumed for required columns, row/season coverage, and completed team-games.
It separately checks live roster/injury team coverage and age, practice/weather
coverage, completed-game snaps/QBR/opportunity, and ESPN/RotoWire/NFL.com expert
schemas and projected-row coverage. Historical-only expert feeds and ordinary
unprojected bench players remain distinct from source failures. Required gaps
block publication; optional upstream/reporting gaps stay explicit. Only a daily
report that includes the actual live/expert inputs updates daily-check health.
The registry and checks live in `src/maintenance/readiness.py`.

## Install and rehearse

1. Build the worker and matching control package with **Build maintenance
   runtime** (`maintenance-image.yml`) on the intended commit. This workflow
   also runs after a successful matching serving deployment. It publishes
   `ff-maintenance:<full-sha>` to ECR and
   `maintenance/images/<full-sha>/control.zip` to the artifact bucket. Image
   tags are immutable. Runtime updates preserve the installed mode, schedule,
   and network parameters and reject an overtaken serving deployment.
2. Create `ff-maintenance` with `template.yaml`, `CAPABILITY_IAM`, and the
   parameters below. `aws cloudformation deploy` should receive `--s3-bucket`
   and `--s3-prefix maintenance/templates` to retain the deployment template
   alongside its assets. Do not enable schedules yet.
3. Start the inference workflow manually with `{"check_sources":true}`. Shadow
   runs may additionally specify an explicit `data_release` for a replay.
   Start the weekly workflow with `{}`. Inspect the execution, worker logs,
   `result.json`, staged artifacts, and verification receipts. The weekly
   workflow uses its newly staged release for its shadow inference.
4. Confirm all six positions/three scoring formats, source coverage, peak memory,
   runtime, and network access from the actual Fargate subnets. Initial workers
   use 4 vCPU / 16 GiB and CPU inference; resize only after measuring. The
   inference task has a one-hour workflow deadline; preparation has 90 minutes.

Required installation parameters:

| Parameter | Value |
|---|---|
| `WorkerImage` | Full ECR URI with the immutable source SHA tag |
| `LambdaCodeKey` | Matching `maintenance/images/<sha>/control.zip` |
| `ArtifactBucket` | Existing artifact bucket (default `ff-predictor-training`) |
| `ClusterName`, `ServiceName`, `ServiceUrl` | Existing serving deployment |
| `Subnets`, `SecurityGroups` | Existing outbound-capable worker network; no load balancer or inbound listener |
| `ServingTaskRoleArn`, `ServingExecutionRoleArn` | Roles already present in the serving task definition; only active control may pass them |

Worker IAM roles can stage their outputs but cannot advance production pointers
or update ECS. Production publication/activation permissions are conditional on
active mode and belong to the control function. The control Lambda is packaged
without the scientific/ML dependency tree. Its package and the worker carry the
same baked source and data-producer hashes.

Control and worker roles also have `s3:ListBucket` on this artifact bucket:
S3 needs that bucket-level permission to distinguish a missing receipt/manifest
(404) from denied access (403). Object reads and writes retain their prefix scopes.

## Production cutover

After the shadow rehearsals pass:

1. Set the repository variable `FF_MAINTENANCE_LOCK_TABLE` to the stack's
   `LockTable` output. Existing code deployment, historical publication, and
   Batch/EC2 rollover workflows participate in that same lease. Let pre-cutover
   jobs finish before proceeding; older checked-out code does not know the lock.
2. Set `MAINTENANCE_INFERENCE_ARN` to the stack's `InferenceWorkflowArn` output.
   Confirm the maintenance image matches the current serving source SHA.
3. Set `AWS_MAINTENANCE_ACTIVE=true`, then let any already-running legacy
   forecast build drain. This suppresses the GitHub forecast cron and routes
   manual/post-training refresh requests to AWS. AWS execution history and the
   watchdog own completion/freshness after dispatch.
4. Update the stack to `Mode=active`, still with schedules disabled. Run and
   verify inference and one weekly correction manually. If a request arrives
   during this brief transition, it remains a shadow run until the stack is
   active; complete the manual active run before enabling timers.
5. Connect the `AlertsTopicArn` to the owner's notification destination, then
   enable schedules. Confirm an intentional stale fixture/failed execution
   alerts and a healthy public artifact clears the incident.

No notification subscription or production cutover is performed by the code
change alone. New runtime image/package versions follow subsequent successful
serving deployments. Infrastructure/template changes require an explicit stack
update; the image workflow intentionally preserves the installed template.

## Publication and recovery

Each execution pins one historical release and six immutable stable model keys.
The existing live-overlay CLI fetches current-season inputs in its separate
mutable cache. Feature construction is recomputed through the existing pipeline;
this is not an incremental feature store. Training and evaluation partitions are
not advanced, and neither scheduled workflow retrains weights.

The historical producer is shared with `refresh-splits.yml` through
`src.data.maintenance_build`. Preparation runs in a clean task directory, seals
all raw/split dependencies (including captured providers), verifies uploads, and
stages the canonical builder's four cache files and build token before any
current pointer moves. Stable models come from `models/releases/v3/<POS>/manifest.json`.
Activation publishes through `src.artifacts.serving_snapshot` and its conditional
`models/predictions_cache/current.json` pointer. A compatible activation preserves
the serving image and pins both data and snapshot generation through the existing
ECS/ALB readiness transaction. An incompatible/newer code deployment
requires a fresh run against that runtime.

An activation backup retains the prior snapshot pointer, data pointers, task
definition and durable rollout state. Immutable generation files are retained.
Failed activation/rollout restores only this execution's writes; a later code or
data deployment is never rolled back by an older execution. A lost final receipt
after verified readiness is reconciled without redeploying. A failed ECS update
response follows the canonical transaction's rollback path before a new attempt.

For a failed workflow, start a **new execution of the same workflow** with
`{"resume_run_id":"<run_id from the Begin result>"}`. This explicit recovery
path reuses the saved request and completed worker output, including a sealed
weekly candidate, while rechecking publication ownership. After rollback it
reactivates that candidate and verifies it again. Native Step Functions redrive
is not the operator recovery path for the caught `Alert -> Failed` endings.
Saved inference requests expire after four hours and weekly requests after one
day; changed runtime/model/data pointers also require a fresh `{}` execution.
Expired staged forecasts cannot be published by resuming them.

The expiring DynamoDB lease serializes publication/rollout mutations across
AWS and opted-in CI writers. Worker leases prevent overlapping expensive jobs;
the worker checks lease ownership before producing its completion receipt. An
expired owner cannot delete its successor's lease. Do not force-remove a worker
lease while its task can still run; stop/confirm termination first.

The control Lambda and its publication lease allow 900 seconds. Activation waits
at most 600 seconds for exact task/container readiness, with 180 seconds reserved
from the remaining invocation budget for recovery; activation/rollback states
allow 930 seconds. The code-deploy CI lease allows 5,400 seconds for source
verification plus rollout. These settings preserve the existing readiness gates.

Live forecast publication uses conditional S3 writes and rejects an artifact
older than the current input timestamp. Serving downloads the forecast on its
existing one-minute poll; the workflow then checks the exact public run identity
and freshness. Missing/partial staged files and unavailable required models do
not replace the last good forecast. Optional source gaps remain disclosed.

The watchdog monitors public input age/availability plus daily source-check and
weekly correction receipts. Initial thresholds are four hours for in-season
forecast inputs, 26 hours for source checks/verified offseason updates, and eight
days for the correction pass. CloudWatch alarms notify on state changes. A
scheduler delivery DLQ and separate alarm cover invocations that never reached
their target; workflow failures report separately.

For an operational rollback, disable the AWS schedules first, drain/stop active
executions, restore the intended compatible artifacts, and verify the public
endpoint. Set `AWS_MAINTENANCE_ACTIVE=false` only when the legacy forecast
workflow should resume. Keep the shared lease configured while any AWS writer
can still execute.

## Validation

`tests/maintenance/` covers corrupt staging, omitted positions, scoring-format
disagreement, stale inputs, duplicate deliveries, changed model manifests,
shadow isolation, expired owners, activation rollback, lost responses, and
standalone Lambda packaging. Existing data-release/cache/rollout tests remain
part of the focused suite. Use the standard sharded pytest harness.

CloudFormation syntax can be validated using a compact temporary JSON copy.
Validate each rendered state-machine definition with
`aws stepfunctions validate-state-machine-definition --type STANDARD`.
These checks do not replace the actual isolated-prefix Fargate rehearsal before
production cutover.

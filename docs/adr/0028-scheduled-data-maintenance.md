# ADR-0028: Scheduled data maintenance and inference

**Status:** Accepted; deployment is opt-in.

## Context

Historical data maintenance depended on relevant code merges or manual dispatch.
The September 10, 2026 review found a 67-day rebuild gap, while actual live
forecast publications repeatedly exceeded their four-hour freshness target.
Successful deployment and successful schedule delivery do not establish that a
new, coherent forecast reached the public service.

## Decision

Use EventBridge Scheduler and two small Step Functions Standard workflows with
isolated on-demand CPU Fargate workers. The daily/intraday workflow pins data and
models, runs the existing source/inference logic, publishes validated output,
and verifies the public artifact. The weekly workflow separates durable
data/cache preparation from compatible ECS activation and subsequent inference.
Python owns data and model logic; state machines own job boundaries and recovery.

Daily source checks compare upstream revisions and record failures separately
from unchanged data. The Thursday correction pass runs after the midweek NFL
correction window, using the same producer as CI. Readiness/coverage validation
governs publication. Content-addressed releases may remain identical when a
correction pass finds no changed inputs; maintenance receipts record that work.

Preserve ADR-0026's immutable historical inputs and isolated live overlay. An
optional staging mode uploads a candidate without advancing its current/index
pointers. Build the complete serving cache before activation. Keep fitting and
evaluation years unchanged; scheduled data/inference maintenance does not train
or promote model weights.

Use ADR-0027's canonical model manifests and immutable serving generations.
Preparation exports the canonical builder's four verified cache files and build
token under the maintenance run. Activation reuses conditional snapshot
publication; it does not restore the predecessor tarball publication protocol.
The serving task pins both the data release and snapshot generation. A changed
generation requires a verified rollout even when the data release is unchanged.

The existing ECS/ALB deployment transaction remains responsible for readiness
and rollback. Its state is persisted to the maintenance backup before each
mutation, allowing recovery after a lost completion receipt. An unacknowledged
ECS update is rolled back by that transaction; it is not accepted as a completed
rollout without readiness evidence. The control invocation and publication
lease allow 900 seconds; the readiness wait is at most 600 seconds and reserves
180 seconds of remaining invocation time for recovery. State-machine activation
and rollback timeouts exceed the Lambda limit.

Publishers use a shared expiring lease and forecast S3 conditional writes.
Existing CI publication/rollover jobs participate when the coordination variable
is configured. Retain previous data/cache/task-definition state for bounded
activation recovery, and refuse to undo a later deployment. Every run binds its
image source, data release, model keys, and timestamps. An independent verifier
checks the public forecast and maintenance receipts.

Operator recovery starts a new execution with the saved `resume_run_id`, reusing
completed preparation and rechecking publication ownership. Requests expire
after four hours for inference or one day for weekly preparation. Recovery after
rollback reactivates the candidate; expired or superseded runs require fresh
preparation. Caught failures end with an alert, so native redrive is not used.

Direct Scheduler → Fargate is appropriate for an inference-only command. The
combined weekly publication/activation process benefits from visible durable
stages and retries without repeating successful preparation. Step Functions
does not improve worker speed or establish semantic freshness; application
validation and the independent verifier remain necessary.

The CloudFormation stack defaults to disabled schedules and shadow output.
Production cutover follows successful isolated-prefix AWS rehearsals, configured
CI coordination, drained legacy writers, and verified notification delivery.
The approved worker image and control package follow matching serving source
deployments; installing/upgrading infrastructure is explicit.

## References

- [Deployment and recovery runbook](../../infra/maintenance/README.md)
- [Infrastructure](../../infra/maintenance/template.yaml)
- [Workers](../../src/maintenance/worker.py)
- [Control and verification](../../src/maintenance/control.py)
- [Historical producer](../../src/data/maintenance_build.py)

## Changelog

- **2026-09-11** — Consolidate with ADR-0027: canonical staged snapshots, v3 model
  manifests, durable deployment recovery, generation-aware rollout and bounded
  control execution. Local integration validation does not replace the AWS
  shadow rehearsals required before cutover.
- **2026-09-10** — Add opt-in daily/intraday inference, daily source checks, and
  a weekly correction workflow with staged activation and public verification.

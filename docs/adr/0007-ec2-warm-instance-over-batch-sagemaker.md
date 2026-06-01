# ADR-0007: EC2 warm instance over Batch/SageMaker

**Status:** Accepted

> **Status (2026-05-20):** superseded by [D13](0013-spot-fan-out-via-aws-batch.md) as the default. The warm-EC2 implementation in D9 remains the rollback path (`BATCH_ACTIVE=false`); this entry is kept verbatim as the historical decision so D13's trade-off discussion is readable. D13 explicitly addresses the "warm EC2 always wins" framing below — the gap closes once you count parallelism across six positions, not per-position cold-start in isolation.

**Decision.** Train on a single warm EC2 g4dn.xlarge driven by CI. The six per-position training containers run sequentially on the instance via one SSM command, invoked by [.github/workflows/train-ec2.yml](../../.github/workflows/train-ec2.yml). AWS Batch with Spot is kept as a standby path ([docs/batch_design.md](../batch_design.md)), reactivated by setting `BATCH_ACTIVE=true`.

**Context.** Per-position training takes ~2 minutes on a GPU. We went through three iterations: SageMaker first (commit `eedacfc`), then Batch + Spot (`57d52f9` → `ffb3119`), then the warm-EC2 design ([docs/ec2_design.md](../ec2_design.md), landed 2026-04-19). Each pivot was driven by the same realization: a 2-minute training job amplifies cold-start overhead, so eliminating it is worth more than the per-hour savings. The follow-up (D13) measured that parallelism dominates cold-start once each position runs on its own host, which inverted the conclusion.

**Options considered.**

| Option | Cold-start | Cost pattern | Operational overhead |
|---|---|---|---|
| Train locally | 0 s | $0 | Blocks laptop ~12 min per full run; no audit trail |
| SageMaker Training Jobs | 3–5 min | $0.53/hr × 6 | Managed, but full cold-start every run |
| AWS Batch + Spot | 30–90 s (with pull-through + SOCI) | $0.16/hr × 6 ≈ $0.03/run | Scales to zero; own the IAM/ECR/queue |
| **EC2 warm instance (chosen)** | ~0 s (already running) | ~$0.53/hr while active, $0 while stopped via idle auto-shutdown | Single host to babysit; SSM is the only control plane |

**Chosen: EC2 warm instance.** The container is pre-pulled; the CUDA drivers are already loaded. `train-ec2.yml` just `aws ec2 start-instances` (no-op if already running) then sends one SSM command that loops through the positions on the host. Per-run cost is effectively the sequential position loop plus the Actions runtime. An idle auto-shutdown timer ([infra/ec2/auto-shutdown.timer](../../infra/ec2/auto-shutdown.timer)) stops the instance after 4 h quiet, bringing the idle cost to zero; the next push pays the start-up tax once and reuses the warm host for the rest of the day.

The commit↔model relationship is now one-to-one: every merge to `main` produces a measured, logged training run. Under Batch, cold-starts dominated observability — the Actions log was mostly "waiting for compute environment."

**Why Batch remains the standby path.** Batch is strictly better when training *dominates* wall time (long jobs) or when we genuinely want $0-idle with no manual stop semantics. For constant fine-tuning on a 2-minute job, the always-on-but-auto-stopped EC2 pattern dominates *sequential* training. We keep the Batch image pipeline live ([.github/workflows/batch-image.yml](../../.github/workflows/batch-image.yml)) so switching back is one `BATCH_ACTIVE=true` away.

**Superseded by D13 when BATCH_ACTIVE=true.** D13 rebuts the "warm EC2 always wins" framing of this entry by observing that the warm-EC2 path is *sequential* across positions (one T4 can't fit six concurrent NN runs). Fanning out across six Spot g6.xlarge instances — one per position — replaces sum(per-position) with max(per-position) and amortizes the per-host cold-start across the parallelism. The choice between D7's warm-OD path and D13's Spot fan-out is now a one-variable flip; both paths remain runnable.

**Rejected.** SageMaker (`eedacfc` → `57d52f9`): managed overhead without training-time dominance. Kubernetes (GKE/EKS): too much machinery for a single GPU job. Long-lived instance without auto-shutdown: leaves an expensive GPU running unused.

**References.** Active path: [docs/ec2_design.md](../ec2_design.md), [infra/ec2/README.md](../../infra/ec2/README.md), [.github/workflows/train-ec2.yml](../../.github/workflows/train-ec2.yml), [src/batch/train.py](../../src/batch/train.py) (reused as the in-container entrypoint). Standby path: [docs/batch_design.md](../batch_design.md), [src/batch/launch.py](../../src/batch/launch.py). Commit arc: `eedacfc` (SageMaker) → `57d52f9` (pivot to Batch) → `ffb3119` (final Batch) → `4b96c41` / `deb3cc7` (EC2 wiring) → `ec5ab17` (SSM polling fix).

## Changelog

- **2026-06-01** — Corrected the historical EC2 rollback implementation text: current `train-ec2.yml` runs one sequential SSM command on the single T4 host, not six parallel commands. Documentation-only.
- **2026-04-19** — D7/D9 and §2 diagram reconciled with the EC2 training switch; the Batch path is preserved as standby (see [docs/batch_design.md](../batch_design.md)).

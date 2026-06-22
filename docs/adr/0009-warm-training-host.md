# ADR-0009: Warm training host

**Status:** Accepted

> **Status (2026-05-20):** implementation for the **rollback path** under D7. The warm host stays provisioned and is reactivated by `gh variable set BATCH_ACTIVE --body "false"` on the next push. Idle cost is ~$8/mo (EBS only) while stopped. The active push-driven trainer is D13's Batch fan-out.

**Decision.** Keep a single g6.xlarge EC2 instance warm with the training image already pulled and CUDA drivers loaded. Trigger per-push training from CI via SSM `RunCommand`, stream CloudWatch logs back to Actions, and stop the instance after 4 h of inactivity via a systemd timer. The Batch cold-start stack (ECR pull-through + SOCI + aggressive `.dockerignore`) is independently used by the active D13 path; it also helps the first EC2 image pull during user-data.

**Context.** D7 picked the warm-host pattern; this decision is the implementation. The old Batch design had to fight cold-start (image pull, instance provisioning, Docker startup) because Batch intentionally scales to zero. For a 2-minute training job, every second spent warming up is overhead we pay on every run. Leaving a GPU idle at ~$0.80/hr (g6 on-demand; was ~$0.53/hr on the retired g4dn) is also unacceptable — so the design has to stop the instance when it's genuinely unused.

**Chosen (composed).**

| Component | Mechanism | Why it matters |
|---|---|---|
| Deep Learning AMI (Ubuntu 22.04, PyTorch) | NVIDIA drivers, Docker, SSM agent, ECR credsStore pre-installed | First boot is ~90 s; subsequent starts are ~25 s |
| `ff-training:latest` pre-pulled, cached on root EBS | `docker pull` at user-data time, then again on every `systemctl start` | Container-start from the CI command is ~2 s (image is already present) |
| SSM `RunCommand` as the only control plane | No SSH, no open ingress, IAM-scoped per command | Security: instance has egress-only SG; auditability: every run is a logged SSM invocation |
| Per-position `ff-train` helper on PATH | `train-ec2.yml` fires one SSM command that loops through the six positions sequentially | One host, one SSM command to poll — the g6/L4 has memory headroom, so sequential is a predictability choice, not a hardware limit |
| [`auto-shutdown.timer`](../../infra/ec2/auto-shutdown.timer) | systemd timer fires every 15 min, stops instance if idle > 4 h | Brings idle cost to zero; next push pays one start-up (~25 s), subsequent pushes are warm |
| [`cloudwatch-agent.json`](../../infra/ec2/cloudwatch-agent.json) | Ships `/var/log/ff-train/*.log` to `/ff/training` | Logs survive the instance stop/start cycle |

Net effect on a typical rollback push: if the instance is already warm, training starts within seconds; if it was idle and stopped, the first push eats ~25 s of start-up and every subsequent push that day is warm. The total wall-clock time from `git push` to six position artifacts in S3 is roughly 120 minutes (the original g4dn/T4 figure; the L4 rollback host should be faster but the wall-clock is unmeasured) because the single host runs the positions sequentially.

**Batch cold-start stack (now on the active path).** The Batch design planned three optimizations to minimize cold-start: ECR pull-through cache for the PyTorch base image (helps GHA build time, not runtime pulls from ECR to the Spot host), SOCI lazy loading (container starts before image fully pulled), and aggressive `.dockerignore` + explicit `COPY` in the training Dockerfile (~8 GB → ~5–6 GB image). **As of 2026-05-20, only the `.dockerignore` win and the SOCI index publish are realized** — the snapshotter that consumes the index is not running on the default Batch AMI (`ami-03dd7084ddd63d5d0`), so measured cold-start is ~258 s instead of the targeted ~60–90 s. Full breakdown and snapshotter-activation paths in [docs/batch_design.md](../batch_design.md). The `.dockerignore` win is independent of the EC2 choice and also helps the first EC2 image pull during user-data when the rollback path runs.

**Rejected.** Dedicated-instance reserved-pricing: commits to 24/7 usage we don't need. Spot on EC2 with no auto-shutdown: interrupts mid-training. On-demand with no auto-shutdown: burns $0.53/hr through idle weekends. Lambda-backed GPU (not generally available at this size): no GPU, and would add a cold-start problem back.

**References.** [docs/ec2_design.md](../ec2_design.md), [infra/ec2/launch-instance.sh](../../infra/ec2/launch-instance.sh), [infra/ec2/user-data.sh](../../infra/ec2/user-data.sh), [infra/ec2/auto-shutdown.sh](../../infra/ec2/auto-shutdown.sh), [.github/workflows/train-ec2.yml](../../.github/workflows/train-ec2.yml). Standby assets: [src/batch/build_and_push.sh](../../src/batch/build_and_push.sh), [src/batch/Dockerfile.train](../../src/batch/Dockerfile.train). Arc: `4145257` → `8a50eec` → `ffb3119` (Batch cold-start stack) → `4b96c41` → `deb3cc7` (EC2 warm-host implementation) → `ec5ab17` (SSM polling fix).

## Changelog

- **2026-06-22** — EC2 rollback host swapped g4dn.xlarge/T4 → g6.xlarge/L4 (T4 retired from training; the rollback now matches the Batch L4 fleet and the cu130 image). Config/docs only — the path stays disabled (`BATCH_ACTIVE=true`); the live warm instance + `EC2_TRAINER_INSTANCE_ID` are re-provisioned out-of-band when the rollback is next exercised. Note the g6/L4 is sm_89 (sm_80+), so the rollback now runs CUDA-graphed like the Batch fleet (the old T4 stayed eager).
- **2026-06-01** — Corrected the rollback runbook text to match `train-ec2.yml`: one sequential SSM command on the single T4 host, about 120 minutes wall-clock. Documentation-only.

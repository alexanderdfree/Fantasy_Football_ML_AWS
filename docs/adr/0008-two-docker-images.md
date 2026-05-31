# ADR-0008: Two Docker images

**Status:** Accepted

**Decision.** Build and deploy two separate Docker images: a slim `python:3.12-slim` image for the Flask inference service (~150 MB) and a `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` image for GPU training (~5–6 GB). The heavy image is consumed by AWS Batch on the active path (D13) and by the EC2 warm host on the D7 rollback path; both pull from the same ECR tag.

**Context.** Inference runs CPU-only on ECS and does not need CUDA, `torch.cuda.*`, or the pytorch wheel's CUDA libs. Training needs all of them plus `nflreadpy`, `lightgbm`, and the training scripts. A single image would either bloat inference (slow ECS deploys, higher cold-start) or strip training capability.

**Options considered.**

| Option | Inference image size | Training setup | Ops |
|---|---|---|---|
| One shared image | ~5–6 GB | Easy | Slow ECS deploys |
| Two images (chosen) | 150 MB + 5–6 GB | Explicit split | Two pipelines |
| Multi-stage build | Smaller, but fragile | Complex | Debug-hostile |

**Chosen: two images.** They have different requirements, different deploy cadences, and different failure modes. Keeping them separate means the Flask app can deploy without rebuilding torch, and a training dep bump doesn't ship to prod inference.

The training Dockerfile ([src/batch/Dockerfile.train](../../src/batch/Dockerfile.train)) uses *explicit* COPYs rather than `COPY . .` to drop the Flask UI, scratch scripts, and analysis notebooks out of the image — see the comments on lines 25–38 of that file.

**Rejected.** Multi-stage builds that share a base were considered but rejected as debug-hostile: when a training run fails on Batch, the fastest debug path is `docker run` the exact training image locally. A multi-stage build obscures that.

**References.** [Dockerfile](../../Dockerfile) (Flask), [src/batch/Dockerfile.train](../../src/batch/Dockerfile.train) (Batch), [.dockerignore](../../.dockerignore). Landed in commit `0e814a1`.

---

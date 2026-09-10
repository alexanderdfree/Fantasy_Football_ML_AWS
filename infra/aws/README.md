# AWS serving infrastructure

Stands up the ECS Fargate + ALB + ACM stack that serves `alexfree.me`.

Training infrastructure lives in `infra/batch/` and `infra/ec2/`; this
directory manages serving resources.

## One-time setup

Prereqs: AWS CLI v2 configured, Docker with buildx (for ARM64), jq, and the
project Python environment. Set `PYTHON=/path/to/venv/bin/python` when needed.
Run from an up-to-date `main` checkout with `origin/main` fetched; a new seed
registers its source revision in the same release lineage used by training.
Model files are gitignored: populate `src/{pos}/outputs/models/` from a
completed training run before seeding a fresh bucket. Already provisioned
buckets need no seeding.

```bash
# 1. Initialize missing manifests from local artifacts that pass load/predict.
#    Existing manifest-backed models are verified and never overwritten.
bash infra/aws/seed_s3_models.sh

# 2. Push one ARM64 seed image to ECR (bootstrap refuses to run without it).
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
      "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"
# (bootstrap creates the repo, but we need the image tag "bootstrap" to exist
#  before the ECS service can start — create the repo first if needed)
aws ecr describe-repositories --repository-names fantasy-predictor --region us-east-1 \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name fantasy-predictor --region us-east-1
docker buildx build --platform linux/arm64 \
  -t "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fantasy-predictor:bootstrap" \
  --push .

# 3. Bootstrap everything else. Pauses for Namecheap CNAME validation records.
bash infra/aws/bootstrap.sh
```

`bootstrap.sh` writes resource IDs to `infra/aws/.env.out` and prints the ALB
DNS name at the end. Use that for the final Namecheap ALIAS/CNAME records:

```
ALIAS  @    -> <ALB DNS>
CNAME  www  -> <ALB DNS>
```

(Also delete the existing A record pointing at `192.64.119.87` — that's
Namecheap parking.)

Before changing AWS resources, bootstrap downloads each position through the
actual serving manifest consumer and runs the CPU artifact smoke test. A
legacy `models/{POS}/model.tar.gz` alone is insufficient. Run this same
read-only check separately with `bash infra/aws/seed_s3_models.sh --verify-only`.
It validates model loading with the local checkout; it does not replace a
new image's deployment checks or validate the availability of inference data.

Seeding preflights every missing position before its first S3 write and uses
conditional initialization to preserve concurrent training publications.
If interrupted, rerun it to verify completed positions and initialize those
still missing. It does not overwrite a broken existing manifest; repair that
artifact with the training or explicit rollback workflow.

Bootstrap reconciles the task role's `fantasy-s3-read` inline policy from
`task-role-policy.json`: model/data reads, bucket listing, and writes limited
to `models/predictions_cache/*`. Prior-version recovery additionally permits
`GetObjectVersion` only for `models/predictions_cache/upcoming_week.json` and
`ListBucketVersions` only with that exact prefix. S3 versioning must be enabled.
Deploy does not reconcile IAM: apply these two statements to the task role before
rolling out recovery (or use bootstrap for a full infrastructure reconciliation).
Reruns preserve unrelated inline policies.
Bootstrap also registers a fresh task definition and redeploys the service.

## Ongoing

- **Code pushes** → `.github/workflows/deploy.yml` rebuilds the ARM64 image,
  pushes to ECR, re-registers the task def, and force-redeploys the service.
  No manual action.
- **Fresh models from training** → `src/batch/train.py` uploads immutable
  artifacts under `s3://ff-predictor-training/models/{POS}/releases/history/`
  and conditionally promotes `models/{POS}/releases/manifest.json` after
  source-revision ordering and validation checks. The consumer accepts the
  legacy manifest until the first release publication migrates it. The
  flat `models/{POS}/model.tar.gz` mirror is unused. The running Fargate
  task picks them up automatically via the in-flight manifest poller (`src.shared.model_sync.start_refresh_poller`, started in `gunicorn.conf.py`) — no restart or force-deploy required.

## Cost control

`bash infra/aws/teardown.sh` stops the meter: deletes the service, ALB, TG,
and SGs. Keeps cluster, IAM roles, ECR repo, ACM cert, and log group so
re-running `bootstrap.sh` is fast (no ACM revalidation).

Baseline with stack up: ~$54/month. After teardown: ~$0.10/month (ECR storage).

## Files

| File | Purpose |
|---|---|
| `bootstrap.sh` | Verify model readiness, reconcile serving resources, and deploy |
| `teardown.sh` | Cost-control delete of ALB + service + SGs |
| `seed_s3_models.sh` | Initialize missing manifests from validated local artifacts; `--verify-only` is read-only |
| `enable_bucket_versioning.sh` | Enable S3 bucket versioning on the artifacts bucket (operator-run once, idempotent) so a buggy GC prune or console misclick can be recovered. |
| `task-definition.json` | ECS task def template (ARM64 Fargate, 2 vCPU / 8 GB, `/health` check). Placeholders `__ACCOUNT_ID__`, `__REGION__`, `__ECR_URI__`, `__IMAGE_TAG__`, `__FF_MODEL_S3_BUCKET__` are substituted by `bootstrap.sh`. |
| `task-role-policy.json` | Model/data reads, bucket listing, and prediction-cache writes for the task role |
| `task-trust-policy.json` | Trust policy letting ECS tasks assume both the execution and task roles |
| `.env.out` | Resource IDs from the last bootstrap run (gitignored) |

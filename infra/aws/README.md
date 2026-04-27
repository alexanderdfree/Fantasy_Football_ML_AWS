# AWS serving infrastructure

_Last verified: 2026-04-21._

Stands up the ECS Fargate + ALB + ACM stack that serves `alexfree.me`.

Training infra lives in `infra/ec2/` — this directory is serving-only.

## One-time setup

Prereqs: AWS CLI v2 configured, Docker with buildx (for ARM64), jq.

```bash
# 1. Seed S3 with model tarballs from local git-committed models.
#    Harmless no-op if you later overwrite with real EC2 training output.
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

## Ongoing

- **Code pushes** → `.github/workflows/deploy.yml` rebuilds the ARM64 image,
  pushes to ECR, re-registers the task def, and force-redeploys the service.
  No manual action.
- **Fresh models from EC2 training** → `src/batch/train.py` uploads to
  `s3://ff-predictor-training/models/{POS}/model.tar.gz`. The running Fargate
  task will pick them up on next restart (trigger via
  `aws ecs update-service --force-new-deployment`).

## Cost control

`bash infra/aws/teardown.sh` stops the meter: deletes the service, ALB, TG,
and SGs. Keeps cluster, IAM roles, ECR repo, ACM cert, and log group so
re-running `bootstrap.sh` is fast (no ACM revalidation).

Baseline with stack up: ~$54/month. After teardown: ~$0.10/month (ECR storage).

## Files

| File | Purpose |
|---|---|
| `bootstrap.sh` | Idempotent full stand-up |
| `teardown.sh` | Cost-control delete of ALB + service + SGs |
| `seed_s3_models.sh` | Upload git-committed models to S3 so Gap 1 sync has data |
| `task-definition.json` | ECS task def template (ARM64 Fargate, 1 vCPU / 2 GB, `/health` check). Placeholders `__ACCOUNT_ID__`, `__ECR_URI__`, `__IMAGE_TAG__`, `__FF_MODEL_S3_BUCKET__` are substituted by `bootstrap.sh`. |
| `task-role-policy.json` | `s3:GetObject` on `ff-predictor-training/models/*` for the task role |
| `task-trust-policy.json` | Trust policy letting ECS tasks assume both the execution and task roles |
| `.env.out` | Resource IDs from the last bootstrap run (gitignored) |

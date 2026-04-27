# ff-training EC2 trainer

_Last verified: 2026-04-21._

A 24/7 `g4dn.xlarge` that runs `src/batch/train.py` on GPU with zero cold start, driven by `.github/workflows/train-ec2.yml`.

## First-time setup

Prereqs: AWS CLI v2, `gh` CLI, credentials with rights to create IAM + EC2 resources.

```
bash infra/ec2/launch-instance.sh
```

This script is idempotent — re-running it only creates what's missing. It performs a quota preflight (G-instance vCPU quota ≥ 4) and refuses to proceed if the quota request hasn't been approved.

On success it prints the instance ID and the `gh variable set` commands you need:

```
gh variable set EC2_TRAINER_INSTANCE_ID --body "i-xxxxxxxxxxxxxxxxx"
gh variable set BATCH_ACTIVE --body "false"
```

## Verification

1. **SSM reachability** (~2-3 min after launch):
   ```
   aws ssm describe-instance-information \
     --filters "Key=InstanceIds,Values=<id>" --region us-east-1 \
     --query 'InstanceInformationList[0].PingStatus'
   ```
   Expect `Online`.

2. **Bootstrap breadcrumb**:
   ```
   aws ssm start-session --target <id>
   # inside:
   test -f /opt/ff/config/bootstrap-complete && cat /opt/ff/config/bootstrap-complete
   nvidia-smi | head -3
   ```

3. **Smoke test** (K is CPU-only and fast, ~30s):
   ```
   # inside SSM session:
   /usr/local/bin/ff-train K 42
   aws s3 ls s3://ff-predictor-training/models/K/
   ```

4. **CI end-to-end**: push any change matching the `train-ec2.yml` path filter; observe the workflow run, model freshness in S3, and logs in `/ff/training` CloudWatch log group.

## Enable auto-shutdown (recommended after observing usage)

The idle-check timer is **installed but disabled**. Turn it on once you've watched traffic for a day:

```
aws ssm send-command --targets "Key=InstanceIds,Values=<id>" \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["systemctl enable --now ff-auto-shutdown.timer"]' \
  --region us-east-1
```

Default threshold: 4h idle → `shutdown -h`. Override with `IDLE_HOURS`:
```
systemctl edit ff-auto-shutdown.service
# add: [Service]
#      Environment=IDLE_HOURS=8
```

CI auto-starts the instance on the next push (`aws ec2 start-instances` + wait for `instance-running`).

## Teardown

```
INSTANCE_ID=i-xxxx
aws ec2 modify-instance-attribute --instance-id $INSTANCE_ID --no-disable-api-termination
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws iam detach-role-policy --role-name ff-training-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam detach-role-policy --role-name ff-training-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
aws iam delete-role-policy --role-name ff-training-ec2-role --policy-name ff-training-workload
aws iam remove-role-from-instance-profile --instance-profile-name ff-training-ec2-profile --role-name ff-training-ec2-role
aws iam delete-instance-profile --instance-profile-name ff-training-ec2-profile
aws iam delete-role --role-name ff-training-ec2-role
aws ec2 delete-security-group --group-id <sg-id>
# break-glass key + log group left in place intentionally
```

## What lives where

| File | Purpose |
|------|---------|
| `launch-instance.sh` | Bootstrap: quota check, IAM, SG, key, `run-instances`. Idempotent. |
| `user-data.sh` | Cloud-init (first boot): NVMe mount, ECR credsStore, `ff-train` helper, breadcrumb. |
| `iam-trust-policy.json` | `ec2.amazonaws.com` assume-role trust. |
| `iam-instance-policy.json` | Inline least-priv policy (S3 r/w, ECR read, CW Logs). |
| `cloudwatch-agent.json` | Ships `/var/log/ff-train/*.log` to `/ff/training` log group. |
| `auto-shutdown.sh` + `.service` + `.timer` | Idle watcher; stops (not terminates) the instance. |

## Cost guardrails

Add AWS Budgets alerts at $100/mo and $300/mo on the `Project=fantasy-predictor` tag so a forgotten running instance isn't a surprise.

## Why EC2, not Batch, is the active path

See [`../../docs/ec2_design.md`](../../docs/ec2_design.md) — tl;dr: eliminate 3-5 min of scale-to-zero cold start so "push to main → fresh model" is sub-15-minute. Batch stays on standby (`docs/batch_design.md`) and is reactivated by flipping `BATCH_ACTIVE=true`.

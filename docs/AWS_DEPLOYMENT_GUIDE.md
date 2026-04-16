# Production Deployment Guide: Fantasy Football Predictor on AWS

## Architecture

```
[User] → www.yourdomain.com
              ↓
        [Namecheap DNS]
          CNAME www → ALB DNS name
          URL redirect @ → www
              ↓
        [ACM Certificate (free SSL)]
              ↓
        [Application Load Balancer]
          port 443 (HTTPS) → forward to target group
          port 80  (HTTP)  → redirect to 443
              ↓
        [ECS Fargate Service]
          auto-scaling: 1-3 tasks
              ↓
        [Docker Container]
          gunicorn → Flask app.py
          (6 position models loaded in-memory)
              ↑
        [ECR Repository] ← stores Docker images
              ↑
        [GitHub Actions CI/CD]
          push to main → build → push → deploy
```

### Why each component

| Layer | Choice | Why | Alternative considered |
|-------|--------|-----|----------------------|
| **Container runtime** | Docker + gunicorn | Portable, reproducible deploys. Gunicorn runs multiple workers for concurrent requests (Flask dev server is single-threaded). | Bare Python on EC2 — fragile, "works on my machine" problems |
| **Compute** | ECS Fargate | Serverless containers — no EC2 instances to patch/manage. Pay per vCPU-hour. Auto-scaling built in. | App Runner (simpler but less control), EC2 (cheaper but manual scaling), Lambda (cold starts kill model loading) |
| **Load balancer** | ALB | Handles HTTPS termination, health checks, distributes traffic across tasks. Required for ACM cert attachment. | No LB + public IP on Fargate task — no SSL termination, no health-check routing |
| **SSL** | ACM | Free, auto-renewing certificates. Attaches directly to ALB. | Let's Encrypt — free but requires manual renewal or Certbot setup |
| **DNS** | Namecheap (keep existing) | No migration needed, just add CNAME + redirect. Free. | Route 53 ($0.50/mo) — only needed for alias records or advanced routing |
| **CI/CD** | GitHub Actions | Already have a workflow for tests. Natural to add deploy step. | AWS CodePipeline — more complex, no benefit over Actions here |
| **Image registry** | ECR | Native ECS integration, no auth complexity. | Docker Hub — works but adds cross-service auth overhead |

---

## Step-by-Step Implementation

### Phase 1: Dockerize the App

#### 1a. Create `Dockerfile`
```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "app:app"]
```

**Sizing notes:**
- `--workers 2` — one worker per vCPU (we're using 0.5 vCPU, so 2 is fine for I/O-bound Flask)
- `--timeout 120` — generous timeout since model loading on first request could take a moment
- `python:3.12-slim` — ~150 MB base vs ~900 MB for full image

#### 1b. Create `.dockerignore`
```
.git
__pycache__
*.pyc
venv/
.claude/
docs/
instructions/
tests/
*.md
.github/
```

#### 1c. Add `/health` endpoint to `app.py`
```python
@app.route('/health')
def health():
    return jsonify({"status": "ok"})
```
The ALB pings this every 30 seconds. If it fails 3 times, the task is replaced.

#### 1d. Update `requirements.txt`
Add `gunicorn==22.0.0`.

#### 1e. Local Verification
```bash
docker build -t fantasy-predictor .
docker run -p 8000:8000 fantasy-predictor
# Visit http://localhost:8000 — confirm the dashboard loads
# Visit http://localhost:8000/health — confirm {"status": "ok"}
```

---

### Phase 2: AWS Infrastructure Setup

**Prerequisites:**
- AWS CLI installed and configured (`aws configure`)
- Your AWS account ID (run `aws sts get-caller-identity --query Account --output text`)

#### 2a. Create IAM Role for ECS

ECS needs a role to pull images from ECR and write logs to CloudWatch.

```bash
# Create the ecsTaskExecutionRole (skip if it already exists)
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

#### 2b. Create ECR Repository
```bash
aws ecr create-repository --repository-name fantasy-predictor --region us-east-1
```
Save the repository URI from the output (e.g., `123456789.dkr.ecr.us-east-1.amazonaws.com/fantasy-predictor`).

#### 2c. Push Docker Image to ECR
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Tag and push
docker tag fantasy-predictor:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/fantasy-predictor:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/fantasy-predictor:latest
```

#### 2d. Create CloudWatch Log Group
```bash
aws logs create-log-group --log-group-name /ecs/fantasy-predictor --region us-east-1
```

#### 2e. Create ECS Cluster
```bash
aws ecs create-cluster --cluster-name fantasy-cluster --region us-east-1
```

#### 2f. Set Up Networking (Security Groups)

Use the **default VPC** — every AWS account has one.

```bash
# Get default VPC ID
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

# Get subnet IDs (need at least 2 in different AZs for ALB)
aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].[SubnetId,AvailabilityZone]" --output table
# Pick 2 subnets from different AZs, save as SUBNET_A and SUBNET_B

# Create ALB security group — allows public HTTP/HTTPS traffic
ALB_SG=$(aws ec2 create-security-group \
  --group-name fantasy-alb-sg \
  --description "ALB - public HTTP/HTTPS" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

aws ec2 authorize-security-group-ingress --group-id $ALB_SG --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $ALB_SG --protocol tcp --port 443 --cidr 0.0.0.0/0

# Create ECS security group — only accepts traffic from ALB
ECS_SG=$(aws ec2 create-security-group \
  --group-name fantasy-ecs-sg \
  --description "ECS tasks - ALB traffic only" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

aws ec2 authorize-security-group-ingress --group-id $ECS_SG \
  --protocol tcp --port 8000 --source-group $ALB_SG
```

#### 2g. Create Application Load Balancer + Target Group

```bash
# Create ALB
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name fantasy-alb \
  --subnets $SUBNET_A $SUBNET_B \
  --security-groups $ALB_SG \
  --scheme internet-facing \
  --type application \
  --query "LoadBalancers[0].LoadBalancerArn" --output text)

# Save the ALB DNS name — you'll need this for Namecheap
aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN \
  --query "LoadBalancers[0].DNSName" --output text

# Create target group
TG_ARN=$(aws elbv2 create-target-group \
  --name fantasy-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --query "TargetGroups[0].TargetGroupArn" --output text)

# Create HTTP listener (temporary — redirects to HTTPS once cert is ready)
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN
```

At this point you can test by hitting the ALB DNS name on port 80 — it should forward to your container once the ECS service is created.

#### 2h. Register ECS Task Definition

Create `ecs-task-definition.json` (replace `ACCOUNT_ID` with your actual account ID):
```json
{
  "family": "fantasy-predictor",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fantasy-predictor",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fantasy-predictor:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fantasy-predictor",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Sizing rationale:**
- **512 CPU** (0.5 vCPU) — your models are CPU-only, inference is lightweight
- **1024 MB RAM** — 6 PyTorch models (~5 MB each on disk, ~50-100 MB loaded) + Flask + pandas = ~400-600 MB. 1 GB gives comfortable headroom
- **startPeriod: 60** — container gets 60 seconds to load models before health checks begin failing
- If you see OOM kills in CloudWatch, bump to `"memory": "2048"`

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

#### 2i. Create ECS Service

```bash
aws ecs create-service \
  --cluster fantasy-cluster \
  --service-name fantasy-service \
  --task-definition fantasy-predictor \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_A,$SUBNET_B],securityGroups=[$ECS_SG],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TG_ARN,containerName=fantasy-predictor,containerPort=8000"
```

**Verify it's working:**
```bash
# Watch the service stabilize
aws ecs describe-services --cluster fantasy-cluster --services fantasy-service \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}"

# Check target group health
aws elbv2 describe-target-health --target-group-arn $TG_ARN

# Test via ALB DNS name
curl http://$(aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN \
  --query "LoadBalancers[0].DNSName" --output text)/health
```

---

### Phase 3: Domain & SSL Setup

#### 3a. Request ACM Certificate

```bash
CERT_ARN=$(aws acm request-certificate \
  --domain-name yourdomain.com \
  --subject-alternative-names "*.yourdomain.com" \
  --validation-method DNS \
  --region us-east-1 \
  --query "CertificateArn" --output text)

# Get the CNAME validation record
aws acm describe-certificate --certificate-arn $CERT_ARN \
  --query "Certificate.DomainValidationOptions[0].ResourceRecord"
```

This returns something like:
```
Name:  _abc123.yourdomain.com
Value: _def456.acm-validations.aws
```

#### 3b. Add ACM Validation Record in Namecheap

In Namecheap → **Advanced DNS**:
1. Add a **CNAME Record**:
   - **Host:** `_abc123` (just the subdomain part, Namecheap appends the domain)
   - **Value:** `_def456.acm-validations.aws.` (include trailing dot)
   - **TTL:** Automatic

Wait 5-30 minutes for ACM to validate. Check status:
```bash
aws acm describe-certificate --certificate-arn $CERT_ARN \
  --query "Certificate.Status"
# Should change from "PENDING_VALIDATION" to "ISSUED"
```

#### 3c. Upgrade ALB to HTTPS

Once the certificate is issued:

```bash
# Delete the temporary HTTP-only listener
HTTP_LISTENER=$(aws elbv2 describe-listeners --load-balancer-arn $ALB_ARN \
  --query "Listeners[?Port==\`80\`].ListenerArn" --output text)
aws elbv2 delete-listener --listener-arn $HTTP_LISTENER

# Create HTTPS listener
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=$CERT_ARN \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN

# Create HTTP → HTTPS redirect
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions '[{"Type":"redirect","RedirectConfig":{"Protocol":"HTTPS","Port":"443","StatusCode":"HTTP_301"}}]'
```

#### 3d. Point Domain to ALB in Namecheap

Get your ALB DNS name:
```bash
ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN \
  --query "LoadBalancers[0].DNSName" --output text)
echo $ALB_DNS
# e.g., fantasy-alb-123456789.us-east-1.elb.amazonaws.com
```

In Namecheap → **Advanced DNS**, add two records:

| Type | Host | Value | TTL |
|------|------|-------|-----|
| **CNAME** | `www` | `fantasy-alb-123456789.us-east-1.elb.amazonaws.com.` | Auto |
| **URL Redirect (301)** | `@` | `https://www.yourdomain.com` | — |

**Why this pattern:**
- DNS spec forbids CNAME on the root domain (`@`), so `www` gets the CNAME pointing to the ALB
- The root domain uses Namecheap's built-in URL redirect to send visitors to `www`
- Result: both `yourdomain.com` and `www.yourdomain.com` work, with HTTPS everywhere

**DNS propagation:** Usually 5-30 minutes, can take up to 48 hours.

---

### Phase 4: CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECS

on:
  push:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: fantasy-predictor
  ECS_CLUSTER: fantasy-cluster
  ECS_SERVICE: fantasy-service
  TASK_FAMILY: fantasy-predictor

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest RB/tests/

  deploy:
    needs: test  # only deploy if tests pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, push image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Deploy to ECS
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          # Get current task def, swap image, register new revision
          TASK_DEF=$(aws ecs describe-task-definition \
            --task-definition $TASK_FAMILY --query taskDefinition)
          NEW_TASK_DEF=$(echo $TASK_DEF | jq \
            --arg IMG "$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" \
            '.containerDefinitions[0].image = $IMG |
             del(.taskDefinitionArn,.revision,.status,.requiresAttributes,.compatibilities,.registeredAt,.registeredBy)')
          aws ecs register-task-definition --cli-input-json "$NEW_TASK_DEF"
          aws ecs update-service \
            --cluster $ECS_CLUSTER \
            --service $ECS_SERVICE \
            --force-new-deployment
```

**GitHub Secrets to add** (repo → Settings → Secrets and variables → Actions):

| Secret | Value | How to get it |
|--------|-------|--------------|
| `AWS_ACCESS_KEY_ID` | Access key for deploy user | Create an IAM user with ECR + ECS permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret | Same IAM user |

**IAM policy for the deploy user** (minimum permissions):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Cost Estimate

| Component | Monthly Cost |
|-----------|-------------|
| ECS Fargate (0.5 vCPU, 1 GB, always-on) | ~$15 |
| ALB (fixed hourly + LCU) | ~$18 |
| ACM certificate | Free |
| ECR storage (~500 MB image) | ~$0.05 |
| CloudWatch Logs | ~$1 |
| Namecheap DNS | Free (already paying for domain) |
| **Total** | **~$34/month** |

**Cost-saving options:**
- **Fargate Spot** — add `capacityProviderStrategy` for up to 70% savings (tasks may be interrupted with 2 min warning, fine for non-critical)
- **Scheduled scaling** — scale to 0 during off-season (June-August) via `aws application-autoscaling`
- **Reserved pricing** — commit to 1-year Fargate Savings Plan for ~40% discount

---

## Checklist

### Phase 1 — Containerize
- [ ] Create `Dockerfile`
- [ ] Create `.dockerignore`
- [ ] Add `gunicorn` to `requirements.txt`
- [ ] Add `/health` endpoint to `app.py`
- [ ] `docker build && docker run` — verify locally

### Phase 2 — AWS Infrastructure
- [ ] Create IAM `ecsTaskExecutionRole`
- [ ] Create ECR repository
- [ ] Push Docker image to ECR
- [ ] Create CloudWatch log group
- [ ] Create ECS cluster
- [ ] Create ALB + ECS security groups
- [ ] Create ALB + target group
- [ ] Create temporary HTTP listener
- [ ] Register ECS task definition
- [ ] Create ECS service
- [ ] Verify: ALB DNS name returns your dashboard

### Phase 3 — Domain & SSL
- [ ] Request ACM certificate
- [ ] Add validation CNAME in Namecheap
- [ ] Wait for certificate to be issued
- [ ] Upgrade ALB listeners (HTTPS + redirect)
- [ ] Add CNAME `www` → ALB DNS name in Namecheap
- [ ] Add URL redirect `@` → `https://www.yourdomain.com` in Namecheap
- [ ] Verify: `https://www.yourdomain.com` loads with padlock

### Phase 4 — CI/CD
- [ ] Create IAM deploy user + save credentials
- [ ] Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to GitHub Secrets
- [ ] Create `.github/workflows/deploy.yml`
- [ ] Push to main → verify auto-deploy works

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `Dockerfile` | **Create** | Container image definition |
| `.dockerignore` | **Create** | Exclude dev files from image |
| `requirements.txt` | **Modify** | Add `gunicorn` |
| `app.py` | **Modify** | Add `/health` endpoint |
| `ecs-task-definition.json` | **Create** | ECS Fargate task config (can delete after registering) |
| `.github/workflows/deploy.yml` | **Create** | CI/CD pipeline |

---

## Troubleshooting

| Symptom | Check |
|---------|-------|
| ECS task keeps restarting | `aws logs tail /ecs/fantasy-predictor --follow` — likely model loading error or OOM |
| ALB returns 502 | Target group health check failing — verify `/health` endpoint works, `startPeriod` is long enough |
| ACM stuck on PENDING_VALIDATION | CNAME record in Namecheap may be wrong — verify exact name/value from `aws acm describe-certificate` |
| Domain not resolving | DNS propagation — wait up to 48 hours. Test with `dig www.yourdomain.com` |
| Container OOM killed | Bump task definition memory from 1024 to 2048, re-register, force new deployment |

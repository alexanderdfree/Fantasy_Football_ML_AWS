# Pre-pulled AMI operator workflow

The bake pins an available x86_64 AL2023 ECS GPU source AMI and an ECR image
digest. It disables ECS on the temporary builder, pulls the image, verifies its
layers, removes registry credentials and builder registration state, then
snapshots the stopped host. The builder is terminated on success or failure.
A CPU builder is sufficient for pulling; GPU behavior is checked in the canary.

```sh
FF_WARM_AMI_BUILDER_TYPE=m7a.large FF_WARM_AMI_MANIFEST=bake.json \
  bash infra/batch/build-warm-ami.sh ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ff-training:FULL_SHA
python infra/batch/warm_ami.py check --input bake.json
python infra/batch/warm_ami.py canary --input bake.json --output smoke.json --smoke
python infra/batch/warm_ami.py canary --input bake.json --output canary.json
python infra/batch/warm_ami.py assess --input canary.json
python infra/batch/warm_ami.py activate --input canary.json --output activation.json
python infra/batch/warm_ami.py rollback --input activation.json --output rollback.json
```

`warm-ami.yml` provides these operations as a dispatch-only workflow. Each run
keeps its evidence under `experiments/warm-ami/control/<workflow-run>-<attempt>/`
and as a workflow artifact. Supply the preceding manifest, canary or activation
JSON URI for the next operation. Canaries checkpoint their controller state to
S3 before creating or submitting work. A local operator can enable the same
recovery journal with `--checkpoint-prefix experiments/warm-ami/control/<id>`.

Canaries create temporary **Spot** environments and queues derived from the
current GPU fleet's network, instance role and launch-template settings. Each
arm is capped at one 4-vCPU instance, with zero minimum capacity. Stock and warm
arms use the same source AMI lineage, image digest, sealed data and seed. An RB
job runs first on a new host; additional jobs compare all six positions over
three pairs on each of L4 (`g6.xlarge`) and A10G (`g5.xlarge`). The ordinary
training launcher and checksum-verified artifact receipts remain in use.
Models are published only under the canary's experiment prefix.

The controller records ECS `pullStartedAt`/`pullStoppedAt`, EC2 instance/AMI
identity, Batch queue-plus-provision time, container startup, execution and total
turnaround. These are distinct measurements: queue time is not image-pull time.
See the [ECS Task API](https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_Task.html).
Activation requires all of the following on each GPU family:

- Three successful paired cold starts, each on a distinct host.
- Identical reported model and cohort metrics within small floating-point
  tolerance, with finite MAE/RMSE and all six positions represented.
- At least 60 seconds median paired image-pull savings and no regression in
  median total turnaround.
- Matching current dependency recipe and ECR dependency-layer digests.

The one-pair smoke validates infrastructure and RB execution; it can never
authorize activation. A missing dataset, insufficient Spot capacity, missing
timing evidence or failed metric comparison leaves the current fleet unchanged.
Use `--data-release` for an explicit compatible private release when current
main has no published matching producer snapshot. Never relabel old inputs.

Activation waits for an idle valid fleet, creates a numeric launch-template
version that preserves existing settings, and changes only its AMI selection.
It preserves purchase mode, minimum/maximum capacity, instance types and network
settings. The activation receipt records the prior numeric template for rollback.
Running jobs are never deliberately terminated by activation; the Batch update
policy allows existing work to finish. See [infrastructure updates](https://docs.aws.amazon.com/batch/latest/userguide/infrastructure-updates.html).

Canary cleanup checks ownership tags before terminating its unfinished jobs or
deleting its queues, environments and launch templates. If a controller or CI
runner is interrupted, download its latest state and run:

```sh
python infra/batch/warm_ami.py cleanup --input canary.json
```

Keep the previous AMI while the candidate is in service so rollback remains
available. `check` compares dependency layers rather than only application
source: code-only image rebuilds can reuse the baked dependencies; requirements,
CUDA/base image or Docker dependency-layer changes require a new bake and canary.
No scheduled rebuild or permanent warm pool is enabled by this workflow.

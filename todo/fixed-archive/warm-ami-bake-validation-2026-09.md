### [FIXED] Warm AMI builder used an obsolete family and malformed SSM commands

**File(s):** `infra/batch/build-warm-ami.sh`, `infra/batch/warm_ami.py`,
`.github/workflows/warm-ami.yml`; warm AMI PR pending.

**What:** The existing builder selected AL2 while the fleet used AL2023. Its
AWS CLI shorthand flattened multiline SSM commands. The selected AL2023 image
also lacked the newer `cloud-init clean --machine-id` option.

**Fix:** Pin and verify an AL2023 source plus an immutable image, send SSM
parameters as JSON, use supported cleanup commands, and preserve layer/recipe
evidence. Add temporary Spot canaries, separate pull-time measurements, numeric
launch-template activation and recorded rollback with ownership-checked cleanup.

**Lesson:** A successful image build is not evidence of a working fresh host or
a cold-start improvement. Measure the actual image pull and execute all six
position pipelines before enabling the candidate.

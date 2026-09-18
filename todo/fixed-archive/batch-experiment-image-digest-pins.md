### [FIXED] A/B job revisions retained mutable image tags

**Files:** `src/tuning/launch_ab.py`, `src/scripts/resolve_training_image.py`,
`tests/tuning/test_launch_ab.py`, `tests/scripts/test_resolve_training_image.py`.
Extracted from repair implementation commit `15b61c88c213dd84f22d25d79597b34b9c85c57a`.

**What:** An immutable Batch job-definition revision still referenced a mutable
ECR source-SHA tag. A tag replacement could change the image bytes used by a
queued or retried experiment without changing its recorded source label.

**Fix:** An optional `--image-digest` is verified against the selected ECR tag,
then registered as `repository:<full-source-sha>@<digest>` under `ff-ab-job`.
The source resolver retains the explicit source tag for producer compatibility;
the run manifest records the digest. Mocked tests cover matching pins, drift
rejection before mutations, revision reuse, and legacy tag-only behavior.
The original repair implementation also ran in isolated Spot Batch smoke jobs;
this extraction adds no model training, selection, or configuration changes.

**Lesson:** Record the code revision and immutable image bytes separately.
Keep experimental definitions isolated from production definition names.

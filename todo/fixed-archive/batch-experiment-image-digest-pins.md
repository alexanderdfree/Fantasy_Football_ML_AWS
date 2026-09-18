### [FIXED] A/B job revisions retained mutable image tags

**Status:** Implemented on the unmerged model-repair diagnostic branch.

**Files:** `src/tuning/launch_ab.py`, `src/scripts/resolve_training_image.py`,
`tests/tuning/test_model_default_repair.py`; implementation commit `15b61c88`.

**What:** An immutable Batch job-definition revision still referenced a mutable
ECR source-SHA tag. A tag replacement could change the image bytes used by a
queued or retried experiment without changing its recorded source label.

**Fix:** An optional `--image-digest` is verified against the selected ECR tag,
then registered as `repository:<full-source-sha>@<digest>` under `ff-ab-job`.
The source resolver retains the explicit source tag for producer compatibility;
the run manifest records the digest. Pure registration/source-identity tests
and actual Spot Batch smoke jobs exercised the path.

**Lesson:** Record the code revision and immutable image bytes separately.
Keep experimental definitions isolated from production definition names.

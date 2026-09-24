### [FIXED] Independent experiment launches repeated allocation setup and lost partial progress

**File(s):** `src/tuning/campaign*.py`, `src/tuning/study_checkpoint.py`, the
existing A/B and tuner adapters; campaign PR pending.

**What:** Separate A/B, tuning and benchmark requests each paid startup costs.
Retrying whole workflows could rerun completed work or add to a study budget.

**Fix:** Freeze campaign identity, reuse one allocation per position/resource
group, isolate step processes, and checkpoint completed A/B units, benchmark
folds and SQLite studies. Verify outputs before skipping work. Conditional
submission intents prevent blind duplicate allocations after transport failures.

**Lesson:** Reuse the allocation while retaining explicit execution identities
and per-workload numerical contracts. Count trial attempts and active time
across retries; a successful process exit alone does not prove durable output.

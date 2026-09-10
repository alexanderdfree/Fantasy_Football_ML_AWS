> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Train freshness check anchored on `now()` instead of train-start
- **File:** `.github/workflows/train-ec2.yml` (PR #197, `1f5cd68`).
- **What:** A workflow step that verified "all six positions produced a fresh tarball" computed its freshness threshold from `date -u +%s` *at check time*. Because the check ran after training completed, the threshold tightened as the job ran longer — a slow training run could fail its own freshness check by waiting for itself. The threshold needs to be anchored on when the training step *started*, not when the check runs.
- **Fix:** Captured the train-start timestamp into an output and referenced it in the freshness check.
- **Lesson:** Time-based freshness windows in CI must use the train-job start time as origin, not `now()`. Any computation that uses "current time" mid-workflow drifts in the wrong direction relative to the work it's validating.

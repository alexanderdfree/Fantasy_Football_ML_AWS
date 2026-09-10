### [FIXED] Delayed upcoming refreshes and artifact replacement reduced availability

**File(s):** `.github/workflows/refresh-upcoming-week.yml`, `src/serving/upcoming_artifact.py`, `src/serving/upcoming_week.py`, `src/serving/frontend/src/lib/upcomingWeek.js`, `infra/aws/task-role-policy.json` (PR pending; baseline `3fca0605`).

**What:** The September 1–9 investigation found 45 actual scheduled runs against 72 nominal three-hour ticks. All 45 published, but the median interval between created runs was 4h48m. Runtime/browser pickup added up to 10/5 minutes. Unsuccessful training notifications could replace a useful pending refresh before their job was skipped. Downloaded bytes replaced the cache without validation, and cold containers had no older-version recovery.

**Fix:** Run hourly at minute 17, isolate ineligible events from the useful concurrency group, and retain one active/latest pending build. Retry transient S3 failures three times; upload retries reuse the built bytes. Validate artifacts before atomic installation, retain usable local projections on failures, and inspect a bounded recent version list when cold. Poll S3 conditionally every minute and revalidate visible browser views every minute. Grant version reads only for this snapshot. Keep the four-hour stale warning and the existing compatible-data/publication gates.

**Evidence:** Regression cases cover malformed payloads, 304 responses, interrupted streams, upload retry exhaustion, timestamp retention, previous-version recovery and resumption, worker exclusion, filesystem failures and browser visibility. Read-only production rehearsal uses the actual six-position, 519-row-per-format artifact; no feature/model recipe changes are required.

**Lesson:** A freshness threshold must not erase usable data. Protect the installed artifact and recovery path independently of a best-effort scheduler; do not claim an hourly cron guarantees hourly execution.

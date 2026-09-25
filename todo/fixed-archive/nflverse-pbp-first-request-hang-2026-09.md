### [FIXED] refresh-splits died seven times on the first play-by-play request of the D/ST pass

- **File(s):**
  - `src/data/nfl_source.py` (`_load_pbp_with_retry`, `_retryable_download_error`)
  - `tests/test_nfl_source.py`

  PR pending (branch `claude/refresh-pbp-retry-416f26`); the earlier attempt #1631
  raised `NFLREADPY_TIMEOUT` from 30 s to 120 s and changed nothing.
- **What:** Seven `refresh-splits.yml` runs since 2026-09-23 (three before #1631,
  four after: 35824714961, 35930141193, 35943368836, 35965267838, 36139406185,
  36144522438, 36173338378) failed in `load_dst_scoring_events` with
  `ConnectionError: Failed to download .../play_by_play_{2012|2013}.parquet: ...
  Read timed out. (read timeout=N)`. Always the first-submitted season of the
  D/ST pass; never the red-zone pass minutes earlier, which fetched every season
  on the same session. urllib3 only appends `(read timeout=N)` while waiting for
  response headers, and nflreadpy 0.1.5 uses one module-level `requests.Session`
  with no retries, so the request never got a response: the session's first use
  after roughly nine idle minutes was a dead pooled keep-alive connection. With no
  data release for `935a6e2a` (#1595), train-batch waited 3600 s for compatible
  data and failed, Deploy to ECS kept the old deployment, and the hourly
  upcoming-week refresh refused to run, freezing live forecasts at 11:20Z on
  2026-09-25.
- **Fix:** `nfl_source.pbp_data` retries a hung or reset season load up to twice
  after a 5 s pause on a fresh connection (urllib3 discards the failed one).
  nflreadpy wraps a 404 in the same builtin `ConnectionError`, so a 4xx response
  on the cause chain is raised at once. Unit tests cover retry-then-succeed,
  giving up, the 404/5xx split and non-retried errors.
- **Lesson:**
  - Read the failing request's phase before tuning a timeout: `(read timeout=N)`
    is the header wait, so a longer timeout only makes a dead connection fail
    later. Retry on a fresh socket instead.
  - A producer-path merge is shipped only when its data release, retrain,
    snapshot and deploy have all landed; check all four before reporting it.
  - Follow-ups not in this record's PR: `NFLREADPY_CACHE=filesystem` for the
    refresh step would remove 25 repeated per-season downloads (policy pinned by
    `tests/test_refresh_splits_workflow.py`), and the other `nfl_source` loaders
    have no retry if the idle session hits them first.

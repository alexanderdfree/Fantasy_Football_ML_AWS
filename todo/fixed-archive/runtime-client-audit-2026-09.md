### [FIXED] Web and iOS clients retained stale responses, filters and comparison claims

**File(s)**: `src/serving/app.py`, `src/serving/timeline.py`,
`src/serving/routes.py`, `src/serving/frontend/`, and `ios/Sources/`.
Defects reproduced against `92be2873` during the 2026-09-10 audit.

**What**:

- Non-API HTTP exceptions became server errors; API errors lost HTTP headers.
- Late Wiki responses replaced newer navigation. Removed filter controls kept
  filtering invisibly, Ridge-only qualifying projections were dropped, and an
  empty saved column selection was reset on reload.
- iOS live results remained behind a disk snapshot, refreshes retained old
  scoring caches, and old player/detail or snapshot/live responses overwrote
  newer state. A failed refresh could resurrect an older disk snapshot.
- iOS History used a commit as a run identity. Repeated runs collided.
- A canceled Next Week load stayed failed after tab re-entry. Unstructured
  manual retries could later replace newer results in History, Upcoming,
  Architecture, Comparison, the Wiki index and Wiki documents.
- Nested Wiki navigation fetched documents without displaying them; local
  heading links opened Safari and cross-document links discarded anchors.
- Native favorite/underdog display used the opposite of the API margin sign.
- Architecture and comparison copy described obsolete loss, checkpoint and
  cohort semantics. Timeline counted beating both experts with one comparison.
- Timeline selected winners on each source's own coverage and compared the two
  experts on separate samples. It also graded projected-component forecasts
  against full fantasy actuals, so unprojected WR rushing could change a winner.
- The Wiki render cache attached a later file edit's mtime to older HTML,
  retaining that stale page indefinitely. Timeline retained a failed format's
  error after returning to a successfully cached format.
- Season Leaders selected its accuracy winner on different source samples and
  full fantasy actuals, including stats outside the projected component set.

**Fix**: Preserve HTTP exception responses and headers. Track request identity
through successful and failed async completions, preserve offline data on
failure, expire stale live formats, and refresh the currently selected format.
Reset unavailable filters, include Ridge in the threshold predicate and retain
explicit empty preferences. Use compound native run identities and convert
the API's positive-favorite margin to a conventional betting line. Update
rendered descriptions from current configuration and comparison contracts.
Keep canceled loads retryable and guard late completions by request generation.
Use working native document navigation, retain canonical Wiki anchors, and keep
external links in the system browser.
Timeline reuses canonical regular-season filtering, projected-component truth,
source exclusions and one common player-week intersection. All displayed MAEs,
the winner and the two-expert edge share that sample. The table reports common
rows against total regular-season rows; missing intersections remain unavailable.
Wiki cache entries retain the version observed before reading. A successful
Timeline cache hit clears the previous request error. Season Leaders receives
separate canonical comparison actuals and source exclusions, grades every
candidate on common rows, and displays their count. Its Avg Actual retains full
fantasy totals; older snapshots without comparison truth show unavailable.

**Validation**: Exact before/after browser flows exercised the committed bundle;
the frontend was rebuilt from authored sources. Node tests use out-of-order
responses and threshold controls. Simulator tests use isolated URL sessions
and disk caches, covering stale success/failure, source transitions, format
changes during refresh and offline recovery. The final native review passed
39 unit tests and 14 real UI scenarios on iPhone 17 Pro / iOS 26.5. The UI
fixtures verified request activation before judging refresh behavior; one
missed gesture was rerun against identical app source. Controls cover real
errors, manual retry, hidden-tab scoring, once-loaded caches and Back/external
navigation. No physical-device claim is made.
Timeline tests reproduce a sparse-coverage winner reversal and an expert edge
that incorrectly changed from a three-point loss to a three-point win. A cached
2025 WR stat line verifies that unprojected rushing cannot affect comparison
truth under PPR, Half-PPR or Standard. All six positions, missing components,
infinite forecasts, absent sources and postseason exclusion are covered.
The follow-up browser replay verifies both sparse and complete samples, a real
WR rushing-stat control, an older snapshot, and PPR→failed Half-PPR→cached PPR.
The visible sparse winner changes from Ridge (1.00) to NN (0.50 on 1/2 rows),
while the complete-sample result stays unchanged. Wiki tests edit the actual
Markdown file during rendering and verify the following request recovers.

**Lesson**: A request's completion order does not establish freshness. Cache
mode, request identity and visible controls must agree with the data displayed.
Comparison claims also require the same actual-stat basis and player-week
sample for every source, with missing coverage visible to the reader.

Aggregate failed-position sentinel recovery is excluded because PR #1560
opened with that exact repair. Web History publication/refresh work belongs
to PR #1559; the native History identity defect is a separate consumer.

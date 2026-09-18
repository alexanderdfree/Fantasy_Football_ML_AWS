### [FIXED] iOS client retained stale responses, colliding run identities and broken Wiki navigation

**File(s)**: `ios/Sources/Stores/{Architecture,Comparison,History,PlayerDetail,Upcoming,Wiki}Store.swift`,
`ios/Sources/Views/Wiki/WikiDocView.swift`, `ios/Sources/Views/Wiki/WikiIndexView.swift`,
`ios/Sources/Views/Upcoming/VegasLineView.swift`, `ios/Sources/DesignSystem/Formatters.swift`,
`ios/Sources/Models/BenchmarkHistory.swift`, `ios/Sources/Persistence/SnapshotCache.swift`.
Defects reproduced against `92be2873` during the 2026-09-10 audit (PR #1565,
`codex/audit-runtime-correctness`). This is the native slice of that PR,
extracted onto main after #1566 had already landed the `SnapshotStore`
request-generation guards, offline retention and `APIProviding` injection.

**What**:

- iOS History used a commit as a run identity. Repeated runs collided.
- A canceled Next Week load stayed failed after tab re-entry. Unstructured
  manual retries could later replace newer results in History, Upcoming,
  Architecture, Comparison, the Wiki index, Wiki documents and player detail.
- Nested Wiki navigation fetched documents without displaying them; local
  heading links opened Safari and cross-document links discarded anchors.
- Native favorite/underdog display used the opposite of the API margin sign.

**Fix**: Use compound native run identities (`run_id`, falling back to
`git_hash|timestamp|pr_number`) and convert the API's positive-favorite margin
to a conventional betting line. Keep canceled loads retryable and guard late
completions by request generation in every remaining store, keeping the
`any APIProviding` injection main already had. Use working native document
navigation (`.task(id:)` keyed on the slug, index rows push `WikiDocView`
directly), retain canonical `#wiki:slug:anchor` destinations, scroll plain
heading fragments within the document, and keep external links in the system
browser. `SnapshotCache` accepts a directory so store tests can use an isolated
disk cache. Not carried from #1565: the `SnapshotStore` rewrite (main's #1566
version is a superset: `liveModeGeneration`, `isStale`), the `APIClient`
session parameter (already on main), the static `ComparisonView` copy (main
renders server-supplied subset titles, cohort definitions and sample basis).

**Validation**: `ios/Tests/StoreRegressionTests.swift` uses an isolated
`URLProtocol` session and a temporary disk cache to cover retry ordering for
every store, canceled Upcoming reloads, player-detail scoring races, late live
responses versus a newer snapshot, benchmark run identity and the spread sign;
`ios/Tests/WikiNavigationTests.swift` covers canonical and encoded Wiki
fragments versus external URLs. Five source tests were dropped: three
duplicated `ClientStoreTests` and two asserted #1565's reset of `usingSnapshot`
during a refresh, which #1566's stale-labeling contract replaces. The source PR's
native review passed 39 unit tests and 14 real UI scenarios on iPhone 17 Pro /
iOS 26.5 (request activation verified before judging refresh behaviour; no
physical-device claim). The extracted slice passed `bash ios/scripts/test.sh`
on iPhone 17 Pro / iOS 26.2 with Xcode 27.0: 55 tests, 0 failures (20
`StoreRegressionTests`, 3 `WikiNavigationTests`, plus the existing suites), and
`python -m ios.scripts.generate_client_fixtures --check` is unchanged.

**Lesson**: A request's completion order does not establish freshness. Cache
mode, request identity and visible controls must agree with the data displayed.
When extracting a slice of a superseded PR, diff each overlapping file against
main's version and port only what main lacks; a test that encodes the source's
replaced semantics is evidence of the divergence, not coverage to carry.

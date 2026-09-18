### [FIXED] Web client retained stale Wiki responses, hidden filters and mismatched comparison claims

**File(s)**: `src/serving/frontend/src/components/FilterBar.jsx`,
`src/serving/frontend/src/lib/latestRequest.js`,
`src/serving/frontend/src/lib/predictionFilters.js`,
`src/serving/frontend/src/lib/wikiLinks.js`,
`src/serving/frontend/src/views/NextWeek.jsx`,
`src/serving/frontend/src/views/SeasonLeaders.jsx`,
`src/serving/frontend/src/views/Wiki.jsx`,
`src/serving/frontend/tests/viewState.test.js` and the rebuilt
`src/serving/static/js/app.js`. Defects reproduced against `92be2873` during
the 2026-09-10 runtime audit (PR #1565); this record covers the web-client
slice extracted onto main. Server, Timeline and iOS findings from that audit
belong to their own records.

**What**:

- Late Wiki responses replaced newer navigation: a slow or failed document load
  could overwrite the page the reader had since navigated to, and a response
  arriving after unmount still updated state. Nested navigation from a cached
  page did not cancel the earlier in-flight request.
- A filter control removed from the bar by a new data slice (Age/Class for DST)
  kept filtering invisibly in manual mode.
- Next Week's minimum-points threshold ignored Ridge, dropping kickers whose
  only qualifying projection was the Ridge forecast.
- Season Leaders reset an explicitly saved empty column selection on reload.
- Season Leaders selected its accuracy winner on each source's own row sample
  against full fantasy actuals, including stats outside the projected component
  set that ADR-0024 reserves for cross-source comparison.
- Plain in-document heading links inside Wiki pages navigated to `#heading`,
  which App's unknown-route fallback treated as the homepage; encoded and
  literal anchors resolved differently across reloads.

**Fix**: `createLatestRequest` tracks a request generation through success,
failure and cancellation, so only the latest navigation (including cache hits)
updates the Wiki view and unmount discards pending completions. The filter bar
resets any control that leaves the bar, in auto-fit or manual mode.
`meetsMinimumProjection` includes Ridge in the threshold predicate.
`loadVisibleColumnKeys` retains an explicit empty selection. `sliceAccuracy`
grades every candidate source on one common row intersection, using the
server's `comparison_actual` under the contract's declared basis
(`contract.comparison.actual_basis`) and each source's shared-component
`<source>_comparison_pred`; display forecasts never substitute, excluded
sources cannot win, and the readout shows common rows against cohort rows
(older snapshots without comparison truth show unavailable). Avg Actual keeps
full fantasy totals. `wikiLinkTarget` routes plain heading fragments through
the canonical `#wiki:<slug>:<anchor>` route and decodes anchors consistently.

The comparison grading was reconciled to main's contract rather than carried
verbatim: #1566 introduced the per-row `<source>_comparison_pred` fields and
#1574 moved the basis to `shared_projected_components_v2`, so the source
branch's `<source>_pred` against a literal `shared_projected_components_v1`
check would have reported every row unavailable on main.

**Follow-up (2026-09-18, PR #1616):** the Season Leaders filtered-slice readout
(`FilterSliceStats`, `.filters-stats-row`) was removed at the owner's request,
together with `sliceAccuracy` and its `viewState.test.js` cases. The readout
sentences above describe the pre-#1616 UI; the server-side per-row
`comparison_*` fields and the Lesson still stand.

**Validation**: `node --test tests/*.test.js` (`npm run test:unit`, the command
`tests.yml` runs) covers out-of-order and cancelled requests, the Ridge
threshold, sparse versus complete comparison samples, display-forecast
non-substitution, legacy and other-basis rows, excluded sources and Wiki anchor
routing. `npm run build` regenerated the committed bundle from sources and a
second build left the tree clean; `tests/test_app.py` serves the rebuilt
bundle. No browser replay was performed for this extraction.

**Lesson**: A request's completion order does not establish freshness, and
visible controls must agree with the data displayed. A client-side comparison
claim needs the same actual-stat basis and player-week sample for every source,
taken from the exported contract rather than a literal, with missing coverage
visible to the reader.

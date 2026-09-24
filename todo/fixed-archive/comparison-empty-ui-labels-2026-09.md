### [FIXED] Optional comparison tables stayed loading and Timeline labels assumed two experts

**Files:** `src/serving/frontend/src/views/Comparison.jsx`,
`src/serving/frontend/src/views/Timeline.jsx`, the rebuilt dashboard bundle,
and `src/serving/frontend/tests/contracts.spec.js`. Extracted from the UI hunks
of #1595 at `1442812b82f826b72ddc5acc4adade4cea1d6a4e`.

**What:** A loaded response without an optional weekly-reference or ranking
table still rendered its loading indicator. Timeline assumed exactly two
experts and showed an undefined label for a source without a display name.

**Fix:** Distinguish an in-flight response from a loaded empty optional table.
Render the names supplied by Timeline, fall back to source keys, and use
wording valid for the supplied expert count. Existing backend source groups,
scoring, cohorts, reference recipes and metric values are unchanged.

**Validation:** The loading/label browser regressions failed before the fix.
The rebuilt bundle passes all 13 browser contracts and 21 frontend unit tests,
using local fixtures with external requests blocked. No model fitting ran.

**Lesson:** Missing optional content after a successful response is an empty
state, not a pending request. UI labels should describe the response rather
than assume a fixed number of comparison sources.

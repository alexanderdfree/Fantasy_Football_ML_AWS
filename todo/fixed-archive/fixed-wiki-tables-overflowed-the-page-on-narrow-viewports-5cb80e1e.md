> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Wiki tables overflowed the page on narrow viewports
- **Files:** `src/serving/templates/`, wiki-tab CSS (PR #194, `5b2d880`).
- **What:** The Wiki tab (PR #138, `ce4543e`) renders repo markdown into the app. Markdown tables can be arbitrarily wide; without a container constraint they pushed the page layout past the viewport on narrow screens.
- **Fix:** Wrap each markdown table in an `overflow-x: auto` container.
- **Lesson:** When rendering external/user-provided markdown, native markdown→HTML doesn't constrain table width — wrap tables in a scrollable container at render time.

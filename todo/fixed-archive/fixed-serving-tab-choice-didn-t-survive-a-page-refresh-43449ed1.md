> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Serving tab choice didn't survive a page refresh
- **Files:** `src/serving/templates/`, JS tab-switch handler (PR #198, `d92362c`).
- **What:** The dashboard's tab switcher (predictions / wiki / etc.) held active state in JS memory, so refreshing the page reset to the default tab. Deep links to a specific tab didn't work.
- **Fix:** Mirror the active tab into `location.hash`; restore the tab from the hash on page load.
- **Lesson:** When a single-page-app has multiple visible "modes," put the mode in the URL hash, not in JS memory. Cheap to add and gives you deep linking for free.

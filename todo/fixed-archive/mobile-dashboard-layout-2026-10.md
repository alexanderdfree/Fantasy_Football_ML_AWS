### [FIXED] Mobile dashboard controls exceeded the viewport

- **File(s):** `src/serving/static/css/style.css`,
  `src/serving/frontend/src/{App,Header,PlayerModal}.jsx`, and
  `src/serving/frontend/tests/mobile.spec.js`. Reproduced on base commit
  `0125b68bb0ad9884fc3516edf392d19da6c73d09`.
- **What:** At 320–390px widths the mandatory Position filter and the combined
  badge/menu row widened the document to 450px. At 820px the header widened it
  to 899px. Deep links and browser navigation could select a tab outside the
  visible navigation strip. Player dialogs lacked space for long names beside
  the portrait and close button. Populated performance charts also preserved
  their canvas minimum width on 320px screens.
  Live Wiki links containing long repository paths widened the page to 337px,
  and long architecture feature names were clipped inside the accordion.
- **Fix:** Let the header wrap; use a compact mobile control grid with touch
  targets and readable inputs. Bound live filter groups and arrange menus in
  two columns while preserving intrinsic measurement widths. Keep each popover
  inside the viewport. Reveal the active tab by scrolling only its navigation
  strip. Wrap dialog names, shrink mobile portraits, reserve close-button space,
  constrain dialog height to the dynamic viewport, and lock background scrolling.
  Let chart grid items shrink while wide tables keep their local scrolling.
  Wrap long Wiki prose paths and architecture feature chips without changing
  preformatted code blocks' local horizontal scrolling.
  Rebuild the committed React bundle. Rendered browser checks cover populated
  views from 320px through desktop widths and exercise filtering, menu toggles,
  table scrolling, navigation, and player details.
- **Lesson:** Wrapping the outer flex row cannot constrain a nonshrinking child.
  Test the document and popover bounds with actual controls and populated data,
  and keep adaptive controls' hidden measurement copies intrinsic.

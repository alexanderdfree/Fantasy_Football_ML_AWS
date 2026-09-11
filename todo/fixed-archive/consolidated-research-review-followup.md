### [FIXED] Consolidated research dataset, population and execution contracts

**File(s):** `src/tuning/ab_oline_confirm.py`,
`src/tuning/ab_qb_context_receivers.py`, the other consolidated experiment specs,
`src/tuning/ab_harness.py`, `src/tuning/launch_ab.py`, and
`.github/workflows/ab-batch.yml` (PR #1479 follow-up; reviewed research revision
`a002e00b3ac2`, restacked research revision `a20231a7`).

**What:** The O-line injector discarded selected input frames; regret used
different source populations and accepted duplicate identities; injury team
aliases lost historical QB absences. Research imports violated the new package
boundary, and eager-only custom cohorts were unsafe under automatic stacking.

**Fix:** Preserve the selected dataset/context roots, compute ranking/regret on
one finite population with unique identities, normalize injury team names, use
canonical evaluation primitives and explicit actual-column identities, and
declare eager-only execution capabilities. Reject unsupported explicit stacking
before launch and retain safe historical plus-prefixed workflow variant names.

**Validation:** Regression cases cover the actual context-aware harness, common
populations, duplicate keys, team aliases, package imports and launcher guards.
The legacy RotoWire slate's provenance remains historical/unverified; this fix
does not relabel old evidence as a current-policy result or enable a feature.

**Lesson:** An experiment's selected data and sample identity must survive
execution refactors; changing model coverage must not change the optimum used
to compare its decisions.

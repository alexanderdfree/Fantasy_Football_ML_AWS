### [FIXED] Comparison coverage, FFToday cache identity and analysis reporting seeds (from #1565)

**File(s)**: `src/prediction/comparison.py`, `src/analysis/fftoday_loader.py`,
`src/analysis/analysis_tabpfn_benchmark.py`, `src/analysis/analysis_k_signal_floor.py`,
`src/analysis/audit_depth_alignment.py`, `src/analysis/build_comparison_summary.py`,
`src/analysis/expert_uncertainty.py`, `src/analysis/significance.py`,
`src/analysis/cohort_analysis.py`. Defects reproduced against `92be2873` during the
2026-09-10 audit (PR #1565, `codex/audit-runtime-correctness` @ `7af61481`). This
record covers only the serving/analysis slice extracted onto main after
#1566/#1574/#1572/#1569 superseded the remainder of that audit branch.

**What**:

- A comparison cohort with shared-component actuals but no finite forecast
  source rendered as an unavailable cohort with every metric blank and no
  reason, indistinguishable from missing actual components.
- FFToday caches keyed sparse season/week requests by min/max/count, so a
  sampled pull satisfied a later contiguous request; partial fetches after a
  transient failure were persisted as complete; custom-roster joins overwrote
  the default joined cache; and a joined cache (matched rows only) was reused
  at a stricter threshold sharing its rounded filename, with literal-null IDs
  counted as matches.
- The TabPFN benchmark graded projected-component RotoWire forecasts against
  full fantasy actuals, dropped the requested seed for skill positions, and
  resumed from a seed-agnostic cache.
- The kicker signal-floor expanding baseline could seed a player's first game
  with the previous group's history, and a benchmark path outside the project
  root raised. Depth-chart alignment merged before computing transitions, so a
  missing chart week erased a one-game replacement from the transition test.
- Expert-reliability actuals were always scored in PPR; significance and the
  late-week ablation ignored the requested seed.

**Fix**: Report `predictions_missing` coverage for actual-only cohorts. Use
canonical season/week cache selectors, require completed-fetch metadata before
reusing a cache, keep roster overrides out of the default joined cache, and
validate joined caches against the original projection denominator and valid
identities (`_joined_cache_is_valid` over `src.data.identity.valid_player_ids`).
Grade the TabPFN RotoWire section on the shared projected components for truth
and forecasts, forward `seed`/`config` to every runner, and version the resumable
cache by seed. Lag expanding means within each group, use `os.path.relpath` for
external benchmark paths, compute starter transitions before the chart merge,
thread `scoring_format` through `_position_actuals`, and pass the requested seed
to pipeline runners.

**Validation**: `pytest tests/test_app.py tests/test_app_comparison.py
tests/test_app_timeline.py tests/test_wiki_cache.py tests/analysis` → 599 passed;
with `tests/test_comparison_paired.py tests/contracts/test_api_contract.py
tests/serving/test_comparison_snapshot.py tests/prediction` → 694 passed. New
tests cover the route's `predictions_missing` reason, HTML 404/405 status and
`Allow` retention, wiki cache versioning across an edit, FFToday partial-fetch
recovery, legacy partial caches, custom-roster isolation, sparse season/week
keys, joined-cache identity and denominator validation, TabPFN seed forwarding
and seed-specific caches, the K baseline lag, depth-alignment transitions,
format-aware reliability, and seeded significance runs.

**Lesson**: A cache hit requires the complete request identity, including the
denominator a threshold is judged against. Comparison claims need the same
actual-stat basis for every source, with missing coverage visible to the reader.

Not carried from #1565 (already on main via #1566/#1574, or left with the held
data/model PRs): HTTP-exception and wiki-cache fixes, Sleeper cache selectors,
timeline/tier/top-N/artifact-eval scoring, K/DST native frames for the
data-completeness, covariate-shift and ablation diagnostics, and the NFL.com
joined-cache and roster-lookup identity changes under `src/data/`.

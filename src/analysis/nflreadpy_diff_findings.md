# `nflreadpy` vs `nfl_data_py`: historical-data diff (validation of #383)

Findings from [`analysis_nflreadpy_diff.py`](analysis_nflreadpy_diff.py), run over the
full project range (`nfl_data_py 0.3.3` vs `nflreadpy 0.1.5`).

> **Status note.** This started as a "should we migrate?" investigation. While it
> ran, **PR #383 (merged 2026-05-29) already migrated the feed** `nfl_data_py →
> nflreadpy` behind the [`src/data/nfl_source.py`](../data/nfl_source.py) pandas
> shim and removed `nfl_data_py` entirely. So this is now an **independent
> post-hoc validation** of that migration's data-equivalence — and it both
> **corroborates and refines** #383's own pre-merge A/B.

Reproduce (note: `nfl_data_py` was removed from the project in #383 — install it
ad-hoc alongside `nflreadpy` to run the comparison):

```bash
pip install nfl_data_py            # no longer a project dep post-#383
python -m src.analysis.analysis_nflreadpy_diff      # full range, all 9 sources
# per-source tables land in analysis_output/nflreadpy_diff/{source}.md (+ SUMMARY.md)
```

---

## TL;DR

The hypothesis — *"both pull from nflverse, so the historical data should be the
same"* — **holds for 8 of the 9 sources** the pipeline consumes: they are
value-identical, differing only cosmetically (polars→pandas dtype widths) or by a
column rename. The 9th, **weekly player stats**, differs in **< 0.07 % of
player-weeks** — a few hundred nflverse stat-corrections out of 70,414 rows, plus
`fantasy_points` (never a training target) and a broader row scope (new release
is a superset).

**Why they differ at all:** the packages read *different release assets*.
`nfl_data_py` (frozen at 0.3.3) reads the **legacy** schema; `nflreadpy` reads the
**current** one. For weekly that's legacy `player_stats` vs new
`stats_player_week` — the same split the loader's old 2025+ branch straddled.
#383 unified everything on the new release.

**Verdict on #383: data-safe, confirmed.** Migrating ≤2024 weekly to the new
release changes the raw-stat *targets* for at most a few dozen player-weeks per
column (all corrections). The one open item is process, not data — see
[§ Open item](#open-item-confirm-the-post-383-retrain).

---

## Per-source results (full range)

| Source | nfl_data_py → nflreadpy | Verdict | Consumed-column value diff |
|---|---|---|---|
| **weekly** | `import_weekly_data` → `load_player_stats` | `VALUE-DELTA` | 13/15 cols differ, all **< 0.07 %** of 70,414 rows; B superset (+162,319 rows, every A key ⊆ B) |
| **rosters** | `import_seasonal_rosters` → `load_rosters` | `ROW-DELTA` | `position`/`team`/`jersey_number` **0-mismatch** after `gsis_id→player_id`; +12 player-seasons in B |
| **schedules** | `import_schedules` → `load_schedules` | `DTYPE-ONLY` | 0/10 — identical values, 13 dtype-width diffs |
| **snap_counts** | `import_snap_counts` → `load_snap_counts` | `IDENTICAL` | `offense_pct` 0-mismatch over 324,611 rows |
| **injuries** | `import_injuries` → `load_injuries` | `IDENTICAL` | `practice_status`/`report_status` 0-mismatch over 76,469 rows |
| **depth_charts** | `import_depth_charts` → `load_depth_charts` | `IDENTICAL` | `formation`/`depth_team` 0-mismatch (≤2024) over 476,152 rows |
| **ids** | `import_ids` → `load_ff_playerids` | `DTYPE-ONLY` | pfr↔gsis bridge (`pfr_id`) 0-mismatch on 7,964 shared gsis_ids |
| **teams** | `import_team_desc` → `load_teams` | `DTYPE-ONLY` | identical column set; `team_logo_espn` 0-mismatch |
| **pbp** | `import_pbp_data` → `load_pbp` | `DTYPE-ONLY` | **0/19** consumed cols differ across **675,997** plays; all keys shared |

`DTYPE-ONLY` = same values, different storage width (`nfl_data_py` `float32`/`int32`
from its `downcast`; `nflreadpy` `float64`/`int64` via polars→pandas). The shim's
`_to_pandas()` uses numpy-backed conversion to preserve NaN semantics, and the
loader re-casts anyway — so these are inert. This confirms the shim's
"pass through unchanged" claim for 6 of the sources, and its `gsis_id→player_id`
alias for rosters.

---

## Weekly deep-dive (the only real diff)

Over 70,414 shared player-weeks (2012–2024), consumed-column mismatches:

| column | mismatches | frac | nature |
|---|---|---|---|
| `interceptions`, `rushing_tds` | 0 | 0 % | identical |
| `passing_yards` | 2 | 0.003 % | stat correction |
| `attempts` | 2 | 0.003 % | |
| `receiving_tds` | 3 | 0.004 % | |
| `receptions`, `targets`, `carries` | 4 each | 0.006 % | |
| `sack_fumbles_lost` | 7 | 0.010 % | fumble reclassification |
| `passing_tds` | 8 | 0.011 % | |
| `rushing_yards` | 12 | 0.017 % | |
| `rushing_fumbles_lost` | 23 | 0.033 % | fumble reclassification |
| `receiving_yards` | 31 | 0.044 % | |
| `receiving_fumbles_lost` | 47 | 0.067 % | fumble reclassification |
| `fantasy_points` | 149 | 0.211 % | **downstream of the above** |

Interpretation:

* **Fumble attribution** is the largest driver — the new release re-buckets a
  handful of fumbles between `sack_/rushing_/receiving_fumbles_lost`. These feed
  fantasy scoring in [`aggregate_targets`](../shared/aggregate_targets.py)
  (−2 pts each), which is why most `fantasy_points` deltas are exactly ±2.0.
* **`fantasy_points` is never a training target** (raw-stat targets are — CLAUDE.md),
  so its 0.21 % drift is diagnostic only.
* **Volume-stat deltas** (`receiving_yards` max |Δ|=56, `rushing_yards` max |Δ|=37)
  are single plays re-attributed in the corrected release, concentrated in older
  seasons. No structural NaN differences (1-sided NaN = 0 everywhere).
* **Row scope:** the new release is a superset (+162,319 player-weeks: defenders,
  linemen, etc.), all inert after the pipeline's position filter. `old_unmatched=0`
  — no historical player-week is dropped.

---

## Corroboration & refinement of #383's A/B

#383's PR body reported its own pre-merge A/B. This independent full-range scan:

* **Corroborates** the headline — `passing_yards` differs in exactly **2 rows**
  (#383 reported "2/89,835"); `interceptions`, `rushing_tds` are identical;
  `old_unmatched=0`; only corrections, no regressions.
* **Refines** #383's claim that *"all TD/reception/INT/attempt/carry/target columns
  [are] identical."* That is *near*-true, not literally true: per-column over the
  full range there are **single-to-low-double-digit** drifts in `receptions` (4),
  `targets` (4), `carries` (4), `receiving_tds` (3), `passing_tds` (8),
  `rushing_yards` (12), `receiving_yards` (31) — each **< 0.05 %**, all
  corrections. The conclusion (data-safe) is unchanged; the measurement is just
  more precise.
* **`sack_yards`** — #383 flagged this as the one used QB-feature input that drifts
  (~7 % of rows, a legacy-vs-modern release refinement). It isn't in this harness's
  consumed-weekly set; #383's characterization stands and is consistent with the
  "older-season recomputation" pattern seen here.

The shim ([`src/data/nfl_source.py`](../data/nfl_source.py)) matches this analysis
exactly: the 4-column `_WEEKLY_RENAME`, the `gsis_id→player_id` rosters alias, the
`load_pbp(...).select(cols)` projection (replacing `downcast=`), and the ≤2024-legacy
/ ≥2025-ESPN depth split are all the right reconciliations for the diffs measured
here.

---

## Open item: confirm the post-#383 retrain

#383 states there's "no train/serve skew" because *"models retrain on the new data
via CI `refresh-splits` + Batch on merge."* That shifts ≤2024 weekly targets for a
few dozen player-weeks, which under CLAUDE.md is a target-affecting change requiring
a `benchmark_history/` diff. At the time of writing, the latest `benchmark_history`
entry is `da770b3` (the earlier docs-only PR), **before** #383 — so the post-#383
Batch retrain's benchmark has not yet been captured/committed. Worth confirming the
retrain completed and metrics held (expected impact: negligible, given < 0.07 %
row drift on the one source that moves). This is a process check, not a data risk.

### [FIXED] Experiments lost seeds, native frames, matched controls and scoring identity

**File(s)**: `src/analysis/`, `src/tuning/feature_groups.py`,
`src/tuning/ablate_backbone_norm.py`, `src/tuning/ablate_injury_features.py`,
`src/tuning/ablate_ridge_pca.py`, `src/tuning/ablate_rb_gate.py`,
`src/tuning/ablate_scheduler_type.py`, `src/shared/error_analysis.py`,
`src/shared/registry.py`, and offensive-position `targets.py` files.
Defects reproduced against `92be2873` during the 2026-09-10 audit.

**What**:

- TabPFN/significance entrypoints discarded requested seeds; cached TabPFN
  results also conflated seeds. Native K/DST diagnostics used generic player
  splits, and ablations called incompatible self-loading runner signatures.
- Leave-one-group-out effects used other dropped groups instead of baseline.
  Multi-position normalization reports lost positions; filtered seed lists
  were paired by order; several reports discarded measured standard deviation.
- Scheduler subsets crashed without required comparisons, or used the wrong
  comparator variance. A malformed Markdown separator broke its report table.
- Missing chart weeks erased actual starter transitions from alignment checks.
  A global shift leaked one kicker's future expanding mean into another player.
- Non-PPR artifact/fresh/reliability paths mixed scoring formats or component
  bases. Disjoint valid selections produced NaN F1, and external benchmark
  paths crashed final metadata serialization.
- Fixed quartile labels conflicted with duplicate quantile edges. K nested
  inference omitted configured head widths. Two-point conversion adjustments
  falsely warned about correct canonical targets and could hide corruption.

**Fix**: Preserve requested seeds and cache them explicitly; share native frame
preparation and K history closure semantics inside analysis-only helpers.
Keep supplied KEEP/CUT/validation/test frames and all six supported positions.
Use same-position, same-seed controls and measured uncertainty. Handle missing
comparison arms explicitly. Calculate starter adjacency before chart joining,
shift expanding statistics within player groups, and use the requested format
with canonical shared projected components. Preserve missing-data status,
return zero F1 for valid zero overlap, support external result paths, and keep
tied quantile values together. Mirror K head widths and validate target
decomposition against the actual upstream scoring contract.

**Validation**: Original-code controls reproduce each defect. Four real
CONFIG_TINY KEEP/CUT cells ran for K/DST with attention enabled and finite
predictions. Analysis and legacy tuning suites passed 323 and 307 unit tests
respectively before combined delivery checks. Six-position checkpoint tests
verify training/inference shapes; source-table controls retain healthy PPR,
complete coverage, correctly paired seeds and real-corruption detection.

**Lesson**: A correctly shaped report can still describe a different seed,
dataset, scoring basis or comparison. Follow the real caller and preserve
identity through preparation, training, caching and aggregation.

Model-impact corrections to hurdle means, trade histories, signed-stat
activity filtering and validation-loss weighting are tracked separately.

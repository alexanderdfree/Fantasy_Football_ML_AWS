# Core-pool `-jN` benchmark measurements (raw per-position data)

Raw per-position `elapsed_sec` from the #670 core-pool launch-bound investigation,
recovered from a `git stash` on `feat/local-core-pool` (commit `fa5d7f2`,
2026-05-31) that was never committed. The **conclusion** lives in
[gpu_launch_bound_levers.md](gpu_launch_bound_levers.md) (the `-j6 242s < -j3 244s
< -j2 269s` total-wall-clock sweep + the LightGBM-stage cuts); this preserves the
**per-position breakdown** behind it, which that doc doesn't carry.

Local box: RTX 5080 (sm_120) + 9950X3D, eager FP16 (the pre-2026-06-22-flip
regime). These were dev-iteration runs; only `-j2` recorded a `total_wall_sec`
(269.5 s, matching the documented `-j2 269`). `-j6` — the 242 s winner — is not in
this set.

| Position | core-pool ON¹ (14:05) | `-j3` (14:22) | `-j2` (14:30) |
|----------|---------------------:|-------------:|-------------:|
| QB  | 114.5 |  60.0 |  44.5 |
| RB  | 208.6 | 110.5 |  98.0 |
| WR  | 211.1 | 134.0 |  98.0 |
| TE  | 241.1 | 174.5 | 128.5 |
| K   |  89.5 |  48.5 |  40.0 |
| DST | 241.6 | 179.0 | 126.5 |
| **total_wall_sec** | — | — | **269.5** |

¹ "core-pool ON" was the run's own label (no explicit `-jN`); its higher
per-position times are consistent with maximal fan-out (each position shares fewer
cores). Note the per-position time *drops* as concurrency drops (fewer parallel
positions → more cores each → faster CPU stage), yet **total** wall-clock *rises*
(`-j2` 269 > `-j6` 242) — because the GPU-launch-bound NN training, not the CPU
stage, sets the per-position wall (`max(CPU ≈ 6–11 s, GPU ≈ 200 s)`), and more
concurrent processes fill the GPU's idle gaps. This is the per-position evidence
for the launch-bound conclusion.

**Provenance.** 3 JSON runs
(`benchmark_history/2026-05-31T{14-05-08,14-22-21,14-30-27}_fa5d7f2.json`) from a
stash on `feat/local-core-pool` — transient local data, never on `main` (which
carries only the docs-only skip-marker for `fa5d7f2`). Extracted here rather than
committed as `benchmark_history/` rows, which would be orphan History entries on a
docs-only commit.

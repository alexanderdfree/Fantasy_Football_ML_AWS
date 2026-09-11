# QB role reconstruction validation

The QB feature contract preserves the owner-intended historical participation
proxy while constructing upcoming roles from eligible pregame depth. The proxy
approximates unavailable historical injury/role information; it is not an as-of
source. Historical benchmarks and prediction-time replay therefore remain separate
information regimes.

## Evaluation protocol

- Full QB `POSITION_CONFIG`, all four model families, seeds 42/123/7, CPU FP32/eager.
- The existing parallel A/B harness isolates model outputs and uses a shared core
  pool. No short-model or `CONFIG_TINY` substitutes are used.
- Historical training/validation reconstructs participation using the corrected
  input data and weekly roster-status interpretation.
- Prediction-time replay marks each held-out week's QB rows as upcoming, supplies
  pregame depth and existing roster/injury exclusions, and builds role magnitudes
  from strictly earlier games. Current-week statistics cannot select the QB.
- Paired metrics use the same 686 regular-season 2025 player-weeks and corrected
  projected scoring components. The added 687th QB observation is excluded from
  paired deltas; canonical native benchmark entries retain their full cohort.
- The archived pregame reference defines the expected-starter cohort; it never
  supplies a model feature or selects a model-specific evaluation pool.

## Validated reconstruction

The investigation completed 21 cells: six verified controls and 15 new full
pipeline cells. All four model families improved overall MAE and RMSE on every
tested seed for the reconstruction rebuilt from corrected sources.

| Model | Current MAE | Reconstructed MAE | Current RMSE | Reconstructed RMSE | Paired RMSE change ± seed SD |
|---|---:|---:|---:|---:|---:|
| Ridge | 6.0299 | 5.6570 | 7.4359 | 7.1165 | −0.3194 ± 0.0000 |
| Plain NN | 5.8071 | 5.5533 | 7.2558 | 7.0526 | −0.2031 ± 0.0034 |
| Attention NN | 5.8301 | 5.5907 | 7.3287 | 7.1457 | −0.1830 ± 0.0431 |
| LightGBM | 5.7286 | 5.5875 | 7.1891 | 7.0634 | −0.1258 ± 0.0106 |

On the fixed 98-row historical inheritor subgroup, plain-NN RMSE improved by
0.5861 and attention RMSE by 0.6361. Expected-starter changes were smaller:
LightGBM MAE was effectively flat, and attention's RMSE change was within seed
variation. Week-1 results were noisy. Seed standard deviations are not confidence
intervals, and one held-out season does not establish live-season performance.

Restoring proxy training while leaving roster-only prediction features did not
recover the RMSE improvement. Delaying the depth input by one available week gave
back part of the gain. The newer reconstruction agreed with the archived pregame
reference's leading QB in 540/544 team-weeks; the lagged version matched 521/544,
versus 491/544 for the current roster/prior-role rule. This is role-selection
agreement, not direct model-versus-expert accuracy.

## Boundaries and reproducibility

The 2025 depth adapter selects snapshots no later than the start of the game day.
Weekly injury/roster records lack a complete publication-timestamp audit, so the
replay is an archived-feed reconstruction rather than a fully timestamp-certified
backtest. Planned rest and late lineup changes remain limitations. Missing usable
depth retains the prior-role fallback and emits a warning.

Mutation checks overwriting current/future attempts and expected points left
prediction features unchanged at Week 1 and Week 13. The live source audit found
2026-season charts for all 32 teams and all 87 parsed active QBs at its timestamp;
coverage does not prove every coaching decision was current.

The original frozen production data and saved controls are under
`/tmp/training-data-fix-validation-1e2d/`. The investigation's scripts, individual
predictions, paired metrics and source-coverage evidence are under
`/tmp/qb-role-reconstruction.Q88Gcc/`. The frozen input fingerprint is
`5814b8b9333b54d5e7950d768613942852e3fe7ce186f906ac17f9c9fe60b0bb`.
Datasets and model weights are intentionally not committed.

## Final implementation validation

Source `c924a1c512a3c05ac3fb52045cb0b65e421b5442` was run through six fresh full
pipeline cells: historical-proxy and pregame-replay modes, each at seeds 42/123/7.
Every saved prediction from the pregame implementation exactly matches the
investigated candidate for all 687 observations. Recomputing the paired 686-row
comparison reproduces the table above.

- [Seed 42 native benchmark](../benchmark_history/2026-09-10T23-46-39_c924a1c5_qb_role_native_seed42.json)
- [Seed 123 native benchmark](../benchmark_history/2026-09-10T23-49-26_c924a1c5_qb_role_native_seed123.json)
- [Seed 7 native benchmark](../benchmark_history/2026-09-10T23-49-27_c924a1c5_qb_role_native_seed7.json)
- [Seed 42 pregame replay](validation/qb-role-reconstruction/2026-09-10T23-50-13_c924a1c5_qb_role_replay_seed42.json)
- [Seed 123 pregame replay](validation/qb-role-reconstruction/2026-09-10T23-50-11_c924a1c5_qb_role_replay_seed123.json)
- [Seed 7 pregame replay](validation/qb-role-reconstruction/2026-09-10T23-50-58_c924a1c5_qb_role_replay_seed7.json)

These entries carry the actual numerical-source fingerprint, source commit, input
fingerprint, configuration and information regime. They retain all 687 test
observations; the paired report excludes the additional row. Only the native
historical benchmarks are in the published `benchmark_history/` feed. Pregame
replay evidence stays under `todo/validation/` so History's record/delta comparisons
do not mix it with native historical runs.

The full feature replay covered 95,652 rows. All columns outside the two
availability fields match the frozen reference exactly. Non-QB availability
values match the pre-change implementation exactly when evaluated in the same
process. Four comparisons against the older saved file differ only by existing
unordered-sum rounding (at most `3.6e-15`); float32 arrays are identical. Both QB
information modes also match their investigated feature arrays at float32 precision.
The non-QB check covers the raw builder's RB/WR/TE/K rows; DST uses its separate
data path and does not consume these QB availability features.

`ruff check .`, `ruff format --check .`, and `git diff --check` passed. The focused
feature/upcoming tests passed (71 tests), and the full unit suite passed with
4,323 tests and 2 skips. Local IPC was enabled for the suite's socket tests.
The History loader was exercised against the final evidence layout: it exposes
only the three native runs for this source, while replay records remain outside
the feed. History, documentation-route and benchmark-fingerprint checks passed
again after that layout change (38 tests).
Final implementation data, scripts, individual predictions and equality checks
are under `/tmp/qb-role-pr-validation.lw91SX/`.

Production GPU validation remains a separate pre-publication check; this change
does not deploy or promote models itself. The source change participates in the
existing producer fingerprint, so merged data/training workflows rebuild the
compatible feature release rather than reusing stale baked availability columns.

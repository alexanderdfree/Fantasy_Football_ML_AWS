# Model-default repair campaign

The owner-approved protocol is preserved byte-for-byte in `protocol.json`.
SHA-256: `caae4d8eea1ba16b8a4dcf6baf7fcec07db2b150950ae3f7ef0244b6018984e3`.
It was recovered from task `01a092d4-70d0-7332-b17f-56381d150b44` on 2026-09-17.

The 2026-09-17 execution uses the same pinned main baseline and PR heads.
All training, including smoke runs and test fixtures that fit models, must run
on AWS Batch. Use the existing Spot fleet; local checks may inspect data,
evaluate mathematical functions, or test orchestration with fake runners only.

Promotion requires each affected model to improve both paired mean MAE and
RMSE in each of 2024 and 2025, with neither metric worsening in either
`elite_top24` or `weekly_reference_top24`. Missing coverage, incomplete seeds,
incompatible provenance, or a failed metric keeps the affected PR open.
Development uses only 2022 and 2023. Confirmation is retrospective.

This diagnostic branch leaves WR's producer configuration equal to current main;
the experiment mutator explicitly enables magnitude scaling in corrected arms.
This retains the original frozen data release without relabeling data produced
by another recipe. It is not a proposal to merge a changed production default.

The experiment keeps #1479 on hold and #1534 default-off. Existing PRs #1575
and #1568 remain unmerged until their final exact revisions pass all gates.

## Development findings

The 36-cell WR weight screen completed with matching source/data/input/row
identities and saved-inference parity. None of the four weight candidates
improves both neural models' MAE and RMSE in both development years while
preserving prior-season top-24 errors.

The broader observed-range checks exposed seven cells with count-likelihood
gradient discrepancies above the predeclared `1e-4` scaled-error tolerance;
the largest was approximately `2.09e-4`. These small discrepancies do not by
themselves explain the forecast regression. The next bounded candidate uses
FP64 inside the same zero-truncated NB probability calculation, returning an
FP32 loss and retaining FP32 model parameters and optimizer state. It keeps
the original weights and corrected inference math. It must pass actual CUDA
numerical and forecast gates; no production default is changed by this spec.

## Reference preflight, 2026-09-17

The current reference builder produced 6,662 matched 2024 pregame rows.
NFL.com returned 404 for Week 18; QB, RB, WR and TE therefore have only
Weeks 1–17. ESPN K and RotoWire DST cover all 18 weeks. The reference file
SHA-256 is `d55fae2966a4c8b0ca9046c1a54b0986aac0ce768b40b64b1dc127dfe540e2ce`.
Confirmation and promotion remain blocked. Development diagnostics may proceed.
The missing week must not be dropped or filled from a different reference recipe.

## Execution

Use `src.tuning.ab_wr_default_repair` first with only `baseline` and `corrected`,
then screen weights only if observed count numerics pass. Use
`src.tuning.ab_selector_trajectories` for the three checkpoint policies on
identical QB/WR trajectories. Both specs default to 2022 development and accept
`FF_REPAIR_ORIGIN=2023`; they refuse confirmation years.

The guarded selector searches the full declared trajectory budget. The legacy
anchor keeps its original stop; unrestricted RMSE keeps its own shadow stop.
The first `selector-dev-*-8cfd3959` grid incorrectly limited guarded eligibility
to the unrestricted RMSE stop. Those guarded results are superseded and cannot
be used for candidate admission; five of the twelve trajectories change their
eligible checkpoint under the corrected search. No policy was promoted.

Submit through `src.tuning.launch_ab`, passing environment options to both the
submitter and the container. Pin `FF_DATA_RELEASE` to the protocol release,
use the exact SHA-tagged branch image plus its verified `--image-digest`,
`FF_AMP_DTYPE=fp32`, and an isolated
`ab_runs/model-default-repair-20260917` prefix. Start with one position and seed.
The observer writes content-addressed rows, raw count parameters, full-precision
cohort metrics, restored checkpoint evidence and saved-inference comparisons.

`src.tuning.repair_gate` checks the complete paired confirmation matrix and
fails closed. Development results cannot satisfy this gate. Its tests use fake
metrics, chronological frames and numerical functions; they never fit models.

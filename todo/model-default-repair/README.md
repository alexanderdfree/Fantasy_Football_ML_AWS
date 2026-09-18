# Model-default repair evidence

This directory archives the 2026-09-17 campaign. This extraction includes only
the read-only reporter and promotion checker, their no-fit tests, and evidence.
Experimental trainers and selection policies remain on the diagnostic branch
at `98f0cd22c1b9fdf49cc6b805b52559c6f05d2399`; they are not enabled on main.

The [completed development report](results.md) records 66 successful AWS Batch
cells and no qualifying repair. Both model PRs remain held. The report includes
paired results, immutable provenance, numerical checks and reproduction commands.
Immutable S3 copies are listed in the [publication index](evidence/publication.json).

The owner-approved protocol is preserved byte-for-byte in `protocol.json`.
SHA-256: `caae4d8eea1ba16b8a4dcf6baf7fcec07db2b150950ae3f7ef0244b6018984e3`.
It was recovered from task `01a092d4-70d0-7332-b17f-56381d150b44` on 2026-09-17.

The 2026-09-17 execution used the pinned baseline and PR heads in the report.
All training, including smoke runs and test fixtures that fit models, must run
on AWS Batch. Use the existing Spot fleet; local checks may inspect data,
evaluate mathematical functions, or test orchestration with fake runners only.

Promotion requires each affected model to improve both paired mean MAE and
RMSE in each of 2024 and 2025, with neither metric worsening in either
`elite_top24` or `weekly_reference_top24`. Missing coverage, incomplete seeds,
incompatible provenance, or a failed metric keeps the affected PR open.
Development uses only 2022 and 2023. Confirmation is retrospective.

The diagnostic branch left WR's producer configuration equal to its pinned main
baseline; the experiment mutator enabled magnitude scaling in corrected arms.
This retains the original frozen data release without relabeling data produced
by another recipe. It is not a proposal to merge a changed production default.

PRs #1575 and #1568 remain held by the metric gate; #1534 remains default-off
and dependent on #1575. Since the campaign, #1479 and #1578 have merged into
main (`f3730605` and `110f21fe`). This dated evidence does not establish their
current status, and it does not approve any model PR for merging.

## Development findings

The 36-cell WR weight screen completed with matching source/data/input/row
identities and saved-inference parity. None of the four weight candidates
improves both neural models' MAE and RMSE in both development years while
preserving prior-season top-24 errors.

The broader observed-range checks exposed seven cells with count-likelihood
gradient discrepancies above the predeclared `1e-4` scaled-error tolerance;
the largest was approximately `2.09e-4`. These small discrepancies do not by
themselves explain the forecast regression. The bounded candidate tested used
FP64 inside the same zero-truncated NB probability calculation, returning an
FP32 loss and retained FP32 model parameters and optimizer state. It kept
the original weights and corrected inference math. It passed the CUDA numerical
checks but failed the forecast gate; see the completed report.

## Reference preflight, 2026-09-17

The current reference builder produced 6,662 matched 2024 pregame rows.
NFL.com returned 404 for Week 18; QB, RB, WR and TE therefore have only
Weeks 1–17. ESPN K and RotoWire DST cover all 18 weeks. The reference file
SHA-256 is `d55fae2966a4c8b0ca9046c1a54b0986aac0ce768b40b64b1dc127dfe540e2ce`.
Confirmation and promotion were blocked at the campaign decision.
The missing week must not be dropped or filled from a different reference recipe.

## Historical execution

The specs below belong to the pinned diagnostic source, not this extraction.
They document the completed campaign; this report does not authorize new runs.

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

## Read-only tools

`src.analysis.model_default_repair_report` reproduces the development summaries
from downloaded, content-addressed manifests; commands are in [results.md](results.md).
`src.tuning.repair_gate` checks a complete paired confirmation matrix and fails
closed. Its CLI takes `records.json --candidate NAME --affected affected.json
--output decision.json`; the affected file maps positions to changed model
families. A passing result is evidence only, never merge authorization.
Development results cannot satisfy this gate. Tests use synthetic metrics and
manifest files only; they never import experimental trainers or fit models.

Extraction review hardened both tools to reject model-specific row filtering
and incompatible paired execution. Each model's sample count must equal the
declared protected-cohort count; overall counts must match the baseline Ridge
sample count across models and arms. Recorded eager/device/seed overrides,
actual neural AMP/graph settings, and TF32 settings must be complete and paired.
Run/job IDs may differ. Archived JSON reports remain byte-identical historical
outputs; the extracted tools enforce these additional checks on fresh inputs.
All 66 content-addressed campaign manifests were downloaded and hash-verified
for this review. The stricter reporter reproduced identical parsed JSON for
all three archived reports, with no provenance errors or qualifying repairs.

The earlier CPU-era #1575 evidence remains available at its
[immutable source report](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/blob/2a97c93fe493692e765969bb6d3b2eda1b16ac2a/todo/inheritance-reception-fix-validation.md).
Those historical results are not AWS confirmation or evidence of accepted
production defaults. All future fitting, including smoke tests, stays on Batch.

> Reference recipe note (2026-09-18, PR #1595): `weekly_reference_top24` now keys on
> `shared_components_v4` (all-zero provider rows are missing forecasts). Records produced
> under the v3 artifact carry a different `cohort_hash`, so a confirmation grid must be
> re-run at one SHA after the v4 reference is published; `protocol.json` is byte-pinned
> and still names the v3 recipe.

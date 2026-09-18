# Model-default repair results — 2026-09-17

## Decision

Keep PRs #1575 and #1568 open. No candidate passed the complete development gate. No model policy, production default, merge, or deployment was approved by these results. #1534 remains dependent on #1575 with flags default-off; #1479 remains on hold.

All 66 decision-bearing development cells completed successfully in 16 AWS Batch Spot jobs: 36 WR weight cells, 12 corrected selector trajectories, and 18 numerical-repair cells. Separate smokes preceded expansion. All model fitting ran on AWS Batch. Local validation comprised 47 no-fit tests plus two import-boundary checks; lint, formatting, and diff checks passed.

## Protocol and provenance

- Development scoring years: 2022 and 2023; train through T−2, checkpoint validation on T−1. Seeds: 42, 123, 7. Production loaders/configuration and eager FP32 model execution with TF32 and CUDA graphs were retained. The numerical candidate used FP64 only inside the probability calculation, returning an FP32 loss.
- Baseline: `a9ae2a12a9ea3408d133d5bc231948cbea9b694a`. Current main advanced to `6d67c9871130b85804083a47dbb208e1f3dcffa7` only for setup-uv action updates; model source, configuration, and dependency manifests are identical to the pinned baseline.
- Frozen data release: `4d196674393af2bbdc09e4269117bfe84ef7d98cad3243b27b982c6270dfa6d7`. The original protocol remains byte-identical with SHA-256 `caae4d8eea1ba16b8a4dcf6baf7fcec07db2b150950ae3f7ef0244b6018984e3`.
- Every completed cell had saved-inference parity and immutable evidence. Paired NN arms matched transformed inputs, player-weeks, scored truth, source, data and hardware; Ridge/LightGBM controls remained equivalent. Selector comparisons used identical trajectories and independently rescored restored checkpoints.
- Raw rows, count parameters, validation truth, cross-head errors, checkpoint states, manifests and run records remain under `s3://ff-predictor-training/ab_runs/model-default-repair-20260917/`. The JSON reports beside this document list content-addressed manifest proofs.

| Study | Worker source SHA | Immutable image digest |
|---|---|---|
| WR weights | `8cfd395973b4a2d278b5536e32e389f55741c671` | `sha256:330b4f6008e5475646ae0fcf2e639a431da13a0bb0dc040b496cb55a9d754a71` |
| Selector trajectories | `7d0132c80e5621cccad6f74947e2141cff0c298d` | `sha256:d8a44cf7ac6c604e1f0f45a656213c63cd4b59e1e6c6ea11acaddb82a962b225` |
| Numerical repair | `50ac1d1349bb5026bcce97eff1ee78d836dc58aa` | `sha256:2e29647b5ea2ac17583e6dc9c14bcf581dc57c599b452969e4c8d75632125b28` |

## Results

Deltas below are candidate minus matched baseline, averaged across the three seeds. Negative is better. Full precision, paired deltas and sample standard deviations are retained in the JSON reports; displayed rounding does not determine eligibility. These are retrospective development results, not prospective or confirmation evidence.

### WR weight screen

All four one-axis candidates failed. Attention MAE increased in both years for every candidate. Plain NN and protected-cohort results also prevented promotion. Corrected probability math remained enabled in candidate arms.

| Candidate | Year | Model | Δ MAE | Δ RMSE | Elite Δ MAE | Elite Δ RMSE |
|---|---:|---|---:|---:|---:|---:|
| corrected | 2022 | nn | -0.004171 | +0.001941 | -0.020109 | +0.029681 |
| corrected | 2022 | attn_nn | +0.082663 | -0.004447 | -0.046637 | +0.012484 |
| corrected | 2023 | nn | +0.003956 | +0.012265 | +0.026659 | +0.029711 |
| corrected | 2023 | attn_nn | +0.056668 | -0.028493 | -0.012565 | -0.063408 |
| gate_half | 2022 | nn | -0.004171 | +0.001941 | -0.020109 | +0.029681 |
| gate_half | 2022 | attn_nn | +0.074904 | +0.024161 | +0.008306 | +0.063306 |
| gate_half | 2023 | nn | +0.003956 | +0.012265 | +0.026659 | +0.029711 |
| gate_half | 2023 | attn_nn | +0.063731 | -0.007764 | -0.017106 | -0.046275 |
| gate_double | 2022 | nn | -0.004171 | +0.001941 | -0.020109 | +0.029681 |
| gate_double | 2022 | attn_nn | +0.093795 | +0.011436 | -0.010923 | +0.082669 |
| gate_double | 2023 | nn | +0.003956 | +0.012265 | +0.026659 | +0.029711 |
| gate_double | 2023 | attn_nn | +0.058850 | -0.019904 | -0.017571 | -0.052701 |
| reception_half | 2022 | nn | -0.007115 | +0.061854 | +0.038461 | +0.192092 |
| reception_half | 2022 | attn_nn | +0.094498 | +0.009641 | -0.039287 | +0.018534 |
| reception_half | 2023 | nn | -0.018414 | +0.106586 | +0.092392 | +0.282818 |
| reception_half | 2023 | attn_nn | +0.097705 | -0.025061 | +0.032574 | -0.068474 |
| reception_double | 2022 | nn | -0.002367 | -0.008015 | +0.017889 | -0.021416 |
| reception_double | 2022 | attn_nn | +0.100001 | +0.011745 | -0.014820 | +0.023204 |
| reception_double | 2023 | nn | +0.000467 | +0.010714 | +0.043897 | +0.010076 |
| reception_double | 2023 | attn_nn | +0.047856 | -0.024985 | -0.022298 | -0.076922 |

### Checkpoint selection

The guarded search used the full declared budget (QB 300 epochs, WR 250), with its MAE anchor frozen at the original legacy stopping point. Unrestricted RMSE retained its own shadow stop. The first `selector-dev-*-8cfd3959` grid had an overly restrictive guarded horizon and is superseded; none of its guarded conclusions is used here. The corrected 12-cell grid was rerun with restored-weight and inference checks.

Guarded QB selection improved overall MAE and RMSE in both years, but 2022 elite MAE increased by 0.008747, failing the protected-cohort gate. WR lacked a strictly better feasible validation checkpoint in two seeds in 2022 and all three in 2023. Baseline fallback was not counted as an improvement.

| Policy | Position | Year | Δ MAE | Δ RMSE | Elite Δ MAE | Elite Δ RMSE | Result |
|---|---|---:|---:|---:|---:|---:|---|
| rmse | QB | 2022 | +0.011251 | -0.008810 | +0.025281 | +0.000595 | Fail |
| rmse | QB | 2023 | -0.066846 | -0.069350 | -0.105783 | -0.077028 | Pass this case |
| rmse | WR | 2022 | +0.081190 | +0.002602 | -0.041231 | +0.028578 | Fail |
| rmse | WR | 2023 | +0.039050 | -0.024347 | -0.061195 | -0.091451 | Fail |
| guarded | QB | 2022 | -0.021447 | -0.027681 | +0.008747 | -0.010722 | Fail |
| guarded | QB | 2023 | -0.101765 | -0.065711 | -0.128644 | -0.028572 | Pass this case |
| guarded | WR | 2022 | — | — | — | — | No qualifying checkpoint: seeds 123, 7 |
| guarded | WR | 2023 | — | — | — | — | No qualifying checkpoint: seeds 42, 123, 7 |

### Numerical correction

The expanded weight study exposed small observed-range FP32 likelihood-gradient discrepancies above the declared 1e−4 scaled-error tolerance. One isolated numerical correction was evaluated at the original weights. Across its six fitted cells, the maximum scaled value/gradient discrepancy was 4.63e−8; CUDA graph capture and saved-inference parity passed. This numerical improvement did not satisfy the forecast gate. No weight combinations, new calibration model, fantasy-total training, or test-derived offsets were introduced.

| Year | Model | Δ MAE | Δ RMSE | Elite Δ MAE | Elite Δ RMSE |
|---:|---|---:|---:|---:|---:|
| 2022 | nn | -0.004171 | +0.001941 | -0.020109 | +0.029681 |
| 2022 | attn_nn | +0.102548 | +0.012149 | -0.026822 | +0.032785 |
| 2023 | nn | +0.003956 | +0.012265 | +0.026659 | +0.029711 |
| 2023 | attn_nn | +0.054223 | -0.012946 | -0.019325 | -0.057805 |

## Confirmation and delivery holds

The canonical 2024 reference recipe is incomplete: NFL.com has no Week 18 offensive archive through the current loader, so QB/RB/WR/TE cover only Weeks 1–17. ESPN K and RotoWire DST cover all 18 weeks. The preflight produced 6,662 rows with no duplicate player-weeks; the reference artifact SHA-256 is `d55fae2966a4c8b0ca9046c1a54b0986aac0ce768b40b64b1dc127dfe540e2ce`.

No candidate qualified to freeze for confirmation. Combined-candidate promotion, all-six-position 2024/2025 confirmation, production guarded-selector integration and merges were therefore not advanced. The missing reference week must not be omitted or substituted. Any next repair round needs a new bounded proposal; these results do not authorize additional tuning against confirmation years.

The numerical function and checkpoint-policy helpers remain experimental code on `codex/model-default-repair-20260917`. The existing PR heads are unchanged; their descriptions carry the results and holds. No claim of metric neutrality, improved production accuracy, or completed six-position confirmation is made.

## Reproduce the summaries

Read-only aggregation from the immutable manifest downloads (no model fitting):

```sh
python -m src.analysis.model_default_repair_report --kind weights --input-dir <manifest-directory> --pattern "wr-dev-*-8cfd3959-*-manifest.json" --output weights.json
python -m src.analysis.model_default_repair_report --kind selector --input-dir <manifest-directory> --pattern "selector-budget-dev-*-7d0132c8-*-manifest.json" --output selector.json
python -m src.analysis.model_default_repair_report --kind numerical --input-dir <manifest-directory> --pattern "numeric-dev-*-50ac1d13-*-manifest.json" --output numerical.json
```

The reporter validates content addresses, complete grids, paired provenance, controls and cohort identity. `src.tuning.repair_gate` separately requires all six positions and both confirmation seasons, immutable source/data/candidate pins, complete paired seeds, inference parity, lower overall MAE and RMSE per affected model and year, and non-increasing protected-cohort errors.

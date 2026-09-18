### [FIXED] WR inheritance magnitude collapsed to a binary flag under z-score clipping

**File(s):** `src/shared/feature_build.py` (`MagnitudePreservingScaler`,
`make_nn_scaler`), `src/shared/pipeline.py` (`_scale_xs` and its four call
sites), `src/shared/position_config.py`, `src/shared/position_pipeline.py`,
`src/scripts/feature_manifest.py` (`nn_magnitude_features`), `src/wr/config.py`.
Component 1 of #1575 (`codex/fix-inheritance-reception` @ `2a97c93f`), isolated
onto main as a held draft PR (PR pending).

**What:** On the sealed September baseline, all 92 positive WR
`inherited_opportunity` rows were mapped to +4 by z-score clipping, collapsing
34 distinct float32 input values into one. Both NN branches (the base net and
the attention static branch, which carries the "contextual" category) therefore
saw inheritance as a binary flag, not a magnitude. Ridge and LightGBM read the
raw column and were never affected.

**Fix:** Fit a bounded magnitude-preserving scaler on training-only nonzero
inheritance values, `4*x/(s+abs(x))` with `s` the median nonzero absolute
training value (1 when training has none), and pickle it with the model. Zero
stays zero, the map is monotone and bounded, and validation/test values never
influence the scale. Activation is explicit per position:
`nn_magnitude_features=("inherited_opportunity",)` for WR only; every other
position keeps `()`, which builds a plain `StandardScaler` and takes the
byte-identical legacy path (RB production run vs `origin/main`: bit-identical
artifacts and benchmark results). Ordinary features and legacy
`StandardScaler` artifacts retain their prior transformation. Base-NN,
attention, CV and serving use the same fitted scaler contract. The merged
`_scale_xs` composes with #1534's bounded-flag override (disjoint columns; the
override rewrites fitted stats that the magnitude columns bypass) and rejects a
missing or misaligned column list when magnitude scaling is requested instead
of degrading silently.

**Evidence:** `tests/shared/test_magnitude_scaler.py` pins monotone bounded
magnitudes, train-only scales, pickle round-trip, in-place transform safety,
legacy-artifact behavior, policy composition, WR-only activation and the
legacy-path byte identity. Source A/B (WR, seeds 42/123/7, NVIDIA L4
FP32/TF32, 2025 test season, `magnitude_only` arm, Δ = variant − baseline):
attention ΔMAE −0.0011 ± 0.0150 (+1/−2 seeds), ΔRMSE +0.0027 ± 0.0148; base NN
ΔMAE +0.0092 ± 0.0043 (worse in 3/3 seeds), ΔRMSE −0.0038 ± 0.0048; inheritor
cohort (n=92) attention ΔMAE −0.0662 ± 0.0595; archived pregame reference
top-24 (n=432) attention ΔMAE −0.0051 ± 0.0725. Neutral within the seed band
and not an improvement on both metrics, so the PR stays held behind the
dual-metric + protected-cohort gate (`todo/model-default-repair/README.md`).

**Lesson:** A rare continuous feature can become a flag after generic z-score
clipping. Check observed input distinguishability, not only presence in an
allowlist. Correctness and metric improvement are separate claims: a
preprocessing contract fix still needs the promotion gate before it changes a
production default.

### [FIXED] Gated hurdle head reported the untruncated NB mean as the reception expectation

**Status: HELD.** Isolated from #1575 (component 2 of 3) as a draft PR. The
correction is mathematically right and it **regresses** the attention forecast,
so it must not change a production default until it passes the dual-metric +
protected-cohort gate (`todo/model-default-repair/README.md`). If it ever
merges, ship it default-off (`nn_correct_ztnb_mean=False`).

**File(s):** `src/shared/neural_net.py` (`ztnb2_conditional_mean`, `GatedHead`
version buffer + legacy-compatible load, corrected forward expectation,
`load_warm_start_state`), `src/shared/position_config.py`,
`src/shared/position_pipeline.py`, `src/shared/registry.py`
(`nn_correct_ztnb_mean`; served `head_losses` / `correct_ztnb_mean` kwargs),
`src/shared/pipeline.py` (both warm-start sites), `src/scripts/feature_manifest.py`,
`src/tuning/ab_inheritance_reception.py` (`legacy` vs `expectation_only` arms),
`tests/shared/test_ztnb_reception_expectation.py`.

**What:** The RB/WR/TE attention `receptions` head is a hurdle: a BCE gate on
`y > 0` plus a zero-truncated NB-2 likelihood on the positives
(`hurdle_negbin`). The value branch fits the **untruncated** NB-2 mean `mu`,
but the reported expectation was `sigmoid(gate) * mu`, omitting the truncation
normalisation `1 / (1 - P_NB(0))`. With `mu=1`, `alpha=1` and `gate=0.75` the
fitted distribution's expectation is 1.50; the head reported 0.75.

**Fix:** Report `sigmoid(gate) * mu / (1 - P_NB(0))` using torch `log1p` /
`expm1` arithmetic in at least FP32 (`ztnb2_conditional_mean`), gated by
`GatedHead(correct_ztnb_mean=...)`. Each gated head persists a
`_ztnb_mean_version` buffer: absent/0 keeps the legacy law on load (old
artifacts are not reinterpreted, and re-save as version 0), 1 enables the
corrected mean, any other value fails loudly. The forward branches on the
load-time Python flag, never on a device `.item()`, so CUDA-graph capture and
stacked (`vmap`) training keep working. Warm starts (`load_warm_start_state`)
reuse legacy weights but keep the new fit's requested mode. The factory wires
the correction only where `head_losses[name] == "hurdle_negbin"`; serving
rebuilds from the same `head_losses` / `correct_ztnb_mean` served kwargs. Other
loss families, gated TD outputs and target/loss-weight definitions are
unchanged; the base NN has no gated head and is byte-identical.

**Measured regression (why it is held):**

- WR, 3 seeds (42/123/7), NVIDIA L4 FP32/TF32, CUDA graphs, 2025 test season,
  `expectation_only` arm (Δ = variant − legacy, mean ± sd; negative is
  better): attention ΔMAE **+0.0698 ± 0.0165** (3/3 seeds worse), ΔRMSE
  +0.0027 ± 0.0059, Δbias **+0.1994 ± 0.0834** (3/3); base NN exactly 0.0000
  (no gated head). Archived pregame top-24 (n=432): ΔMAE −0.0080 ± 0.0187,
  ΔRMSE +0.0060 ± 0.0164; inheritor cohort (n=92): ΔMAE +0.0009 ± 0.0054.
  Source: `s3://ff-predictor-training/ab_runs/wr-components-20260911/`
  (image `40469debf92297780a5523d8f868ccbc4decc733`, inference parity 0.0 in
  every cell).
- CPU cross-position, eager FP32, 3 seeds, `expectation_only` attention ΔMAE:
  RB +0.0368 ± 0.0157, TE +0.0389 ± 0.0141, WR +0.0827 ± 0.0110; base NN
  exactly 0.0 in every cell (`benchmark_history/ablations/inheritance_reception_cpu_1b29796b.json`
  on `origin/codex/fix-inheritance-reception`; the paired metrics are
  summarised in the [validation record](../inheritance-reception-fix-validation.md)).
- The 2026-09-17 repair campaign (`todo/model-default-repair/results.md`)
  kept this correction enabled across its 36 WR weight-screen cells and 18
  numerical-correction cells; none passed the gate.

**Validation:** unit tests compare the conditional mean against SciPy's NB
survival function across `(mu, alpha)` pairs down to `mu=1e-5` with finite,
positive gradients; FP16/BF16 inputs are promoted to FP32; the reported
expectation equals the fitted probability mass; a saved corrected head reloads
corrected, a legacy state dict (no buffer) reloads legacy and re-saves as
version 0, versions 99/0.5 are rejected; a warm start keeps the requested mode
in both directions; all six factory/serving configurations agree (RB/WR/TE
correct only `receptions`, QB/DST none, K nested); stacked forward and
gradients work. Containment benchmark (RB/K, CPU eager, seed 42, no thread
caps, `benchmark_history/2026-09-18T08-58-33_4c0994f1.json` vs the
`origin/main` 12da0f92 baseline, 76 artifact files): K fitted state and every
metric row identical (49/49 attention + 30/30 base-NN tensors, scalers, Ridge,
LightGBM); RB Ridge/LightGBM/base NN (38/38 tensors) identical; the RB
attention checkpoint's 79 shared tensors are bit-identical with only the three
new `_ztnb_mean_version` buffers (`receptions`=1, TD heads=0), so the training
trajectory did not move and the delta is purely the corrected reported
expectation: attention MAE 3.806 → 3.844, RMSE 5.818 → 5.775, `receptions`
MAE 0.913 → 0.942 (single seed, CPU, not evidence; direction matches the
3-seed CPU RB result above).

**Lesson:** A distribution's latent rate is not necessarily its reported mean;
verify output expectations against probability mass and preserve semantics in
the model artifact. Correctness and metric improvement are separate claims:
the gate and rate were co-trained under the legacy reporting, so the corrected
expectation shifts the attention forecast upward (positive bias) and needs
retuning evidence before it can ship.

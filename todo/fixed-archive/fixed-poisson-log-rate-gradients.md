### [FIXED] Sparse Poisson heads lost corrective gradients behind nonnegative clamps

- **File(s):** `src/shared/neural_net.py`, `training.py`, `pipeline.py`,
  `position_config.py`, `position_pipeline.py`, `registry.py`; regression tests
  in `tests/shared/test_poisson_heads.py`; comparison spec
  `src/tuning/ab_poisson_log_rate.py`. PR pending.
- **What:** The September 10 architecture audit replayed the July 13 stable
  artifacts and found attention fumbles identically zero for all 1,643 RB,
  2,498 WR, and 1,281 TE holdout rows, plus all 544 DST safety predictions.
  WR's base NN also predicted zero fumbles throughout. Forward hooks showed
  negative preactivations, not display rounding. For raw output -1 and target
  1, the rate-space Poisson loss was 18.42 but the clamp passed zero gradient.
- **Fix:** Ungated Poisson heads fit log-rates directly; predictions remain
  nonnegative raw counts via `exp(log_rate)`. Use `log_input=True` consistently
  in eager, captured-training and captured-validation losses. Initialize only
  these heads from TRAIN event means, leaving other head initialization and
  loss weights unchanged. Resolve target families in both training factories
  and serving; persist a per-head marker so old checkpoints keep their clamp
  semantics and new checkpoints retain the log-rate interpretation. The
  `nn_poisson_log_rate=False` override supplies the unchanged A/B baseline.
- **Validation:** Unit probes cover negative-log-rate gradients down to -100,
  analytical Poisson loss/gradient parity, all three network variants, all six
  positions' training/serving checkpoint round trips, and stacked execution.
  The first full CPU RB seed-42 comparison changed attention FP MAE
  4.028 → 4.025 and base-NN FP MAE 4.114 → 4.119, with identical Ridge output
  metrics. Production GPU and multi-seed validation is pending before merge.
- **Lesson:** A head can pass shape, finite-output and non-negativity tests while
  being unable to learn positive events. Test corrective gradients and track
  sparse-head bias/zero fraction alongside full fantasy error. Sparse-event
  MAE rewards an all-zero predictor and is insufficient by itself. This is
  a Poisson rate parameterization fix, not the rejected global Softplus or
  hurdle-Poisson change; see ADR-0005 for the compatibility contract.

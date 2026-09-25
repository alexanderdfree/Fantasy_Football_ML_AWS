### [FIXED] Validation loss averaged batch means, over-weighting a short final batch

**File(s):** `src/shared/training.py` (`MultiHeadTrainer.train` validation
reduction — the `_GraphedValPass` graphed prefix and the eager tail),
`src/tuning/ab_ensemble_seeds.py::stacked_val_losses`; regression tests in
`tests/shared/test_validation_reduction.py`; real-CUDA acceptance probe
`src/analysis/verify_validation_reduction.py`. Isolated from #1565
(`codex/audit-runtime-correctness` @ `7af61481`, commit `f21d4f4c`), which
reproduced the defect against `0f0fec55` in the 2026-09-10 audit.

**What:** The validation pass summed per-batch *mean* losses (the eager body's
`loss` from `MultiTargetLoss._compute_loss_components`, and the graphed prefix's
`gval.loss_sum`, which accumulates K batch means) and divided by the batch
count, so under the val loader's `shuffle=False, drop_last=False` batching a
short final batch carried the weight of a full one. The same four observations
(squared errors 1, 1, 1, 100) reported `val_loss` 25.75 at batch sizes 2 or 4
but 50.5 at batch size 3, purely from the partition. The per-target
`val_loss_{t}` components used the same reduction, and the stacked tuner's
`stacked_val_losses` averaged each member's loss over batches the same way.
The quantity feeds `history["val_loss"]`, the `epoch_callback` (Optuna
`trial.report` pruning and the `min(val_loss)` objective in
`src/tuning/tune_nn.py`) and `ReduceLROnPlateau.step` when
`scheduler_type="plateau"`. Early stopping and best-checkpoint selection use
the loss-weighted `val_mae_{t}`, which was already computed over the
concatenated per-sample predictions and is unchanged.

**Fix:** Accumulate sample sums — the eager tail multiplies each batch's mean by
its row count; the graphed prefix multiplies `gval.loss_sum` / `gval.comp_sums`
by the fixed batch size `gval._bs` and counts `gval._n_fixed` rows — and divide
by the observation count (`n_val_samples`) for both the combined loss and the
per-target components. `stacked_val_losses` weights each vmapped member loss by
the batch's row count. The empty-loader guards keep their `0.0` / `inf`
contracts; training gradients, weighted-MAE checkpoint selection and the
graph's baked buffers are untouched.

**Validation:** `tests/shared/test_validation_reduction.py` pins the sample mean
across batch sizes 2/3/4/8 on both the `DataLoader` and `_GPUResidentBatcher`
paths, executes the real `_GraphedValPass` body on CPU for the prefix/tail
arithmetic, covers the stacked path and the `train_stacked` callback, and
checks that changing the validation partition does not change training
gradients or the checkpoint. `src/analysis/verify_validation_reduction.py` is
the real-CUDA acceptance probe (sm_80+, graph capture engaged); its L4 run is
recorded in the #1565 cell
`s3://ff-predictor-training/ab_runs/model-corrections-e3ed317f-20260910/cells/RB-verify-42.json`
(`validation_reduction.passed_cases = 3`; batch-3 prefix 3 rows / tail 1 row;
loss 103.0 at every partition). Every production config uses
`cosine_warm_restarts` or `onecycle`, so trained weights are expected to be
unchanged; the corrected value changes tuning objectives, pruning and the
recorded `val_loss` history. RB/K CPU eager seed-42 smoke against main
`12da0f92` (`benchmark_history/2026-09-18T09-06-52_8630f74f.json`): all 76
saved artifacts (NN/attention weights, scalers, Ridge and LightGBM models)
and the benchmark results are bit-identical; only per-checkout provenance
digests differ, and the epoch logs show the same `MAE wtd` and early-stop
epochs with a changed `Val:` value. Held behind the dual-metric +
protected-cohort gate (`todo/model-default-repair/README.md`) until the
isolated campaign cell reports.

**Lesson:** A per-batch mean is only an unbiased epoch mean when every batch has
the same size; reduce validation metrics over observations, and when a CUDA
graph sums batch means, convert the prefix back to a sample sum before adding
the ragged tail.

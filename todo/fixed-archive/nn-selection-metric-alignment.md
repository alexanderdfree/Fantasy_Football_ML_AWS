### [FIXED] NN checkpoint and tuning metrics differed from reported fantasy-point RMSE

**Files:** `src/shared/training.py`, `src/shared/pipeline.py`, `src/tuning/tune_nn.py`, `src/tuning/ab_ensemble_seeds.py`, `src/batch/train.py`; PR #1568 supersedes the partial selector change in `3960ee66`.

**What:** The trainer selected weighted raw-stat MAE/RMSE while Optuna/pruning used mixed validation loss and reporting used aggregated PPR fantasy-point error. Loss weights balanced stat heads but did not represent scoring utility or cross-stat error cancellation.

**Fix:** Keep raw-stat losses; select checkpoints, prune and tune on validation PPR fantasy-point RMSE. Use the canonical aggregator on both predictions and actuals, pool validation rows before roots, and average stacked member RMSEs only afterwards. Version tuning study namespaces and persist selection identity/epoch/curve through Batch merge and history. Retain explicit legacy selectors for matched A/Bs and multi-format/stat diagnostics.

**Lesson:** Match the objective's aggregation, scoring format and population to the reported metric while keeping validation and test separate. An aligned metric is not evidence of improved held-out accuracy; compare matched production runs across seeds before merging.

**Ridge/LightGBM extension (PR #1568):** Ridge's target-specific alphas now minimize joint out-of-fold PPR RMSE with fixed special heads included. LightGBM fits raw-stat regressors, selects their prefixes jointly on validation PPR RMSE, persists the selected counts, and trims unused trees on save. Its Optuna objective/pruner uses mean fold/seed PPR RMSE in a fresh study namespace. The holdout comparison keeps the configured fitting loss and selection policy when applying tuned hyperparameters; metadata survives CPU split/merge and history publication. Fitting the full LightGBM tree budget can cost more than per-head early stopping and must be measured.

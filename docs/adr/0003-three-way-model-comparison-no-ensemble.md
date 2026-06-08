# ADR-0003: Three-way model comparison, no ensemble

**Status:** Accepted

**Decision.** Train and report Ridge (L2 linear), multi-head NN, and LightGBM independently per position. Do not ensemble or stack.

**Context.** A core goal is comparing multiple model architectures quantitatively. Ensembling would dominate any single model's MAE, but it would also muddle the question the project is trying to answer.

**Options considered.**

| Option | What it answers | What it costs |
|---|---|---|
| Ensemble (weighted average) | "Lowest MAE possible" | Loses the per-architecture comparison |
| Stacking (meta-model) | Same as ensemble + meta-risk | Extra CV pass, minimal gain at this sample size |
| Independent comparison (chosen) | "Which architecture wins, and why" | Leaves some accuracy on the table |

**Chosen: independent comparison.** Ridge is the honest baseline (same feature matrix, same scaling, just L2 regression). The NN is the headline "custom architecture" deliverable. LightGBM is the "what could a boosted tree do with the same inputs" reference point. Reporting all three per-position surfaces real trade-offs: Ridge wins on stability and interpretability; NN pulls ahead on WR/TE where interactions matter; LightGBM is competitive where there's enough data and falls apart on K/DST.

**Rejected.** Ensembling was considered and rejected because it would obscure exactly the finding the project is trying to produce. (A future production system would of course blend these — that's a follow-up, not this ADR.)

**References.** [src/shared/models.py](../../src/shared/models.py) (`RidgeMultiTarget`, `LightGBMMultiTarget`), [src/shared/neural_net.py](../../src/shared/neural_net.py), [src/shared/pipeline.py](../../src/shared/pipeline.py). LightGBM added in commit `f343c20`.

## Addendum (2026-06-08): TabPFN as an opt-in, default-off comparison variant — upgraded v2 → TabPFN-3

`TabPFNMultiTarget` ([src/shared/models.py](../../src/shared/models.py)) adds [TabPFN](https://www.nature.com/articles/s41586-024-08328-6) — a *pretrained* transformer foundation model for tabular data — as a per-target regressor that mirrors the `LightGBMMultiTarget` interface and slots into the same holdout-path comparison (`train_tabpfn` flag). **This does not change the decision above:** TabPFN is one more *independent* column, not an ensemble/stack; the per-architecture comparison is preserved.

It ships **disabled for every position** (`PositionConfig.train_tabpfn=False`), and `tabpfn` is intentionally **not** in any requirements file — so production never trains, serves, or displays it, and the import is lazy (CI without the package is unaffected). It exists as ready-to-use infrastructure, not a live model.

**Pinned to the TabPFN-3 line (2026-06-08).** Originally added on the open-weights **v2** line (`tabpfn==2.2.1`); upgraded to **TabPFN-3** — the `tabpfn` 8.x package default (`model_path='auto'`) and the genuine latest in the line (v2 → 2.5 → 2.6 → 3). TabPFN-3 is *faster and stronger* than the intermediate 2.5 (~20× faster forward pass via KV-cached predict + row-chunking; pareto-dominant on the TabArena benchmark — [report](https://priorlabs.ai/technical-reports/tabpfn-3)). Crucially it runs every position **in-regime**: its envelope (≤~1M rows; up to ~2k features at lower row counts) covers the largest split (WR, 25,344 rows × 112 feats), so the v2-era `ignore_pretraining_limits` hack and the PCA-under-cap knob are no longer needed (kept only for unusually wide inputs). The 8.x `TabPFNRegressor` takes the same kwargs the wrapper passes except `n_jobs`, which 8.x renamed to `n_preprocessing_jobs` (CPU-preprocessing parallelism) — the wrapper was updated to the new name; otherwise the model code is unchanged.

**Pilot finding (TabPFN-3, WR, single seed, 2026-06-08).** With `train_tabpfn=True` on WR, TabPFN-3 posted the **best total FP MAE of any model — 3.912** (holdout), *beating* the Attention NN incumbent (3.989), LightGBM (4.105), the base NN (4.065), and Ridge (4.119) — a flip from the v2 pilot, where TabPFN was 2nd (4.004) and lost to the Attention NN. It was best per-target on **receiving TDs (0.241), yards (19.124), and receptions (1.257)** (2nd on fumbles), best on the weekly backtest (3.942 MAE), and 2nd on Top-12 hit rate (0.366 vs LightGBM's 0.370). In-regime it is also fast — **~90 s to fit vs ~355 s in the v2 pilot** (and faster than this run's Attention NN at ~145 s), running the 25,344-row set inside its envelope (no `ignore_pretraining_limits`). Ridge (4.119) and LightGBM (4.105) reproduced their v2-pilot numbers exactly, so data-identity holds — only the model column moved. **Caveats:** single seed (the NN baselines wobble ~0.03 run-to-run, so treat the ~0.08 margin over the Attention NN as suggestive, not multi-seed-confirmed), and TabPFN is *transductive* (a deployed artifact carries the train set and runs a forward pass per inference).

**Why still off.** It now *wins* the pilot, so performance is no longer the reason — the **license** is: TabPFN-3 weights are non-commercial (below), so it **cannot be served** regardless of accuracy. It stays a benchmark-only comparison column. (Were it servable, this result would warrant a multi-seed, all-position evaluation before promoting it.)

**License — now non-commercial; never servable.** The v2 weights were under the *Prior Labs License* (Apache-2.0 + §10 attribution), which *permitted* serving with an attribution badge. **TabPFN-2.5/2.6/3 weights ship under a non-commercial license** (`tabpfn-3-license-v1.0`: research / limited internal evaluation only — **no commercial or production use**), and downloads require accepting the license + a Prior Labs account token (`TABPFN_TOKEN`; headless via [ux.priorlabs.ai](https://ux.priorlabs.ai)). So this variant is now **internal-benchmark-only and cannot be promoted to the served product** — the open-weights **v2** checkpoint remains the only TabPFN that could legally be served. Acceptable because the variant is dormant by design.

**To enable later (benchmark only):** set `train_tabpfn=True` on a position's `POSITION_CONFIG`, `pip install tabpfn` (8.x → TabPFN-3), accept the license and export `TABPFN_TOKEN`. Locally, run via a uv overlay without mutating the shared venv: `TABPFN_TOKEN=… uv run --no-sync --project <repo> --with tabpfn python -m src.wr.run_pipeline`. To pin the intermediate 2.5 instead, point `model_path` at the `Prior-Labs/tabpfn_2_5` checkpoint in `_new_regressor` — but 3 dominates 2.5 for 112-feature / 25k-row data.

## Changelog

- 2026-06-08 · Added `TabPFNMultiTarget` as an opt-in, default-off 5th model variant (dormant infrastructure; not in prod). Decision unchanged — independent comparison, no ensemble. (PR for the TabPFN model variant)

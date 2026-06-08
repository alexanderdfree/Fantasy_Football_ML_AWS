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

## Addendum (2026-06-08): TabPFN as an opt-in, default-off 5th model variant

`TabPFNMultiTarget` ([src/shared/models.py](../../src/shared/models.py)) adds [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6) — a *pretrained* transformer foundation model for small tabular data — as a per-target regressor that mirrors the `LightGBMMultiTarget` interface and slots into the same holdout-path comparison (`train_tabpfn` flag). **This does not change the decision above:** TabPFN is one more *independent* column, not an ensemble/stack; the per-architecture comparison is preserved.

It ships **disabled for every position** (`PositionConfig.train_tabpfn=False`), and `tabpfn` is intentionally **not** in any requirements file — so production never trains, serves, or displays it, and the import is lazy (CI without the package is unaffected). It exists as ready-to-use infrastructure, not a live model.

**Why off (pilot finding, WR, single seed).** With `train_tabpfn=True` on WR, TabPFN reached **4.004** total FP MAE — 2nd of five, beating Ridge (4.119), LightGBM (4.105), and the base NN (4.026), and **best of all models on receptions** (1.259) — but it did **not** beat the Attention NN incumbent (3.956). Two confounds: (1) the WR training set is **25,344 rows**, ~2.5× TabPFN v2's ~10k-sample design limit, so it ran via `ignore_pretraining_limits=True` (outside its trained regime); (2) it took **~355 s vs ~13 s** for the NN (in-context attention over a 25k-row context), and it is *transductive* — a deployed artifact would carry the training set and run a forward pass per inference. Competitive-but-not-winning plus that cost ⇒ not worth enabling now. The natural follow-up (untested) is TabPFN in its ≤10k regime via subsampling / post-hoc ensembling, which would be far faster and may close the gap.

**To enable later:** set `train_tabpfn=True` on a position's `POSITION_CONFIG` (optionally `tabpfn_pca_components` / `tabpfn_ignore_pretraining_limits` for wide/large inputs) and `pip install "tabpfn==2.2.1"`. Use the open-weights **2.x** line — the 8.x package gates weight downloads behind a Prior Labs account token (`TABPFN_TOKEN`). **License:** TabPFN is under the *Prior Labs License* (Apache-2.0 + §10). Internal benchmarking/testing is exempt, but if it is ever enabled in the **served** product, §10 requires prominently displaying **"Built with PriorLabs-TabPFN"** in the UI and shipping a copy of the license.

## Changelog

- 2026-06-08 · Added `TabPFNMultiTarget` as an opt-in, default-off 5th model variant (dormant infrastructure; not in prod). Decision unchanged — independent comparison, no ensemble. (PR for the TabPFN model variant)

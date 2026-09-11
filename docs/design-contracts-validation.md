# Design-contract restructuring validation

## Consolidation replay, 2026-09-11

The evaluation/Timeline consolidation has been ported to the new prediction and
data owners. The artifact-only runtime now evaluates available Timeline groups
without importing Torch, scikit-learn or plotting libraries: NumPy scoring loads
no execution backend, and Tensor calls still dispatch to the already-loaded
Torch implementation. The scoring operations themselves are unchanged.

All six recorded seed-42 fitted bundles were replayed through this consolidated
code using their original captured inputs. Every raw-stat prediction and PPR
total matched exactly (maximum difference zero), including K nested history and
DST opponent history. This is inference replay of existing fits, not a new
training comparison or GPU validation. The original three-seed training evidence
below remains dated evidence for the restructuring.

The consolidated tree passed 4,930 unit tests (2 skipped), 37 focused runtime,
wire-contract and scoring-boundary checks, 9 browser tests and 26 native-client
tests. The runtime probe covers 21 endpoint paths plus available Timeline
groups/scoring formats with ML imports forbidden. The compact replay/source
record is `benchmark_history/audits/2026-09-11-contract-consolidation.json`.

## Original production comparison

Measured on 2026-09-10 for ADR-0027. The comparison uses the corrected numerical
pipeline on main `b9d24f9259c7fd261ab1a4e77d4212d821726420`. Main's later history
record and live roster/practice/QBR fixes through `3fca0605` are integrated and
do not change this numerical baseline. Earlier comparisons against `e5ac3a56`
remain historical evidence; they do not establish parity against the corrected
data in #1564.

## Production preparation

Both sides start from the corrected, sealed raw source used by #1564. In isolated
directories, the actual `load_raw_data → preprocess → build_features → temporal_split`
path rebuilds the numerical splits, including injury and weekly-roster inputs.
Native K/DST preparation uses the same frozen historical dependencies. Provider
and HTTP fetching are forbidden. All 25 raw inputs remain unchanged, including
the archived evaluation reference, which is intentionally reused byte-for-byte.
A historical comparison build must not refresh its archived pregame cohort.
Full reference regeneration would require an uncaptured historical Sleeper ID
response, so this validation explicitly covers offline numerical rebuilding and
replay with the preserved reference, not a new upstream reference refresh.
The original source-directory fingerprint is
`2a49e6c9acbbc1cba5bc945c40fc9a207c72760be9d6cd054baa99044e9a3a44`.

The separate producer rebuilds differ in two training rows and one validation
row of `inherited_opportunity`, by at most `3.552713678800501e-15`; the test split
is exact. These differences disappear in the actual production float32 casts.
For **all six positions**, exact array hashes confirm identical ordered features,
row counts, production configuration, `X_train`/`X_val`/`X_test`, and every raw-stat
target array, with no nonfinite values. Ridge, LightGBM and neural model factories
receive these same float32 inputs; subsequent float64 conversion cannot introduce
an input difference. Equal metrics alone were not used to infer input equality.

## Production-pipeline comparison

The baseline and candidate run the actual six `POSITION_CONFIG` recipes with
CPU eager execution, seeds 42/123/7 and the existing parallel A/B harness with
three workers. Neither side uses `CONFIG_TINY` or stacked training. The selected
Python 3.12.12 environment uses the checked-in numerical pins, including NumPy
2.5.3, pandas 3.0.5, Torch 2.14, scikit-learn 1.9, SciPy 1.18.1 and LightGBM 4.7.
The environment uses the repository's repaired single macOS OpenMP runtime.

All **18 candidate cells** match their corresponding baseline's full ordered
prediction tables exactly: **599,415 prediction values**, with maximum absolute
difference **0**. This includes the raw-stat predictions and scoring totals.
All reported model metrics, rankings and cohort blocks also match.

| Position | Held-out rows per seed | Prediction columns | Fitted replay columns |
|---|---:|---:|---:|
| QB | 687 | 29 | 28 |
| RB | 1,759 | 29 | 28 |
| WR | 2,768 | 21 | 20 |
| TE | 1,660 | 21 | 20 |
| K | 543 | 21 | 20 |
| DST | 544 | 45 | 44 |

All six seed-42 fitted bundles then replayed exported raw heads and PPR totals
bit-for-bit through `ModelBundle`, `Predictor` and the shared frame adapter.
This includes K nested kick history and DST opponent history. Replay deliberately
sets ambient `FF_NN_NORM=layer` while loading saved `backbone_norm=batch`
constructors; the saved architecture wins without modifying the environment.
The fitted replay excludes the pipeline's separate `pred_baseline` forecast and
auxiliary gate logits, which are not exported fitted-model target predictions.

## Durable evidence and limits

The three `benchmark_history/2026-09-10T21-18-0*_115c4aa8_design-contracts_b9_seed*.json`
records contain actual pipeline summaries, separate baseline/candidate file and
seal hashes, preparation checks, replay results and the final content fingerprints.
Each side's 183 audited parquet reads stayed within the declared input/provider
roots, with zero violations; both provider-cache files and every side's input
inventory stayed unchanged throughout fitting. Before/after source fingerprints
also match. The pre-PR gate matches those fingerprints to the committed tree.

Their `git_hash` identifies the checkout during the run (`115c4aa8`);
`code_fingerprints` bind the tested numerical content, including integration edits
subsequently consolidated into this PR. No timing or performance claim is made
from the concurrently executed validation work.
Superseded V6 records and their detailed report are preserved with the local
validation artifacts and excluded from the new comparison's evidence.

This establishes CPU eager behavior for the recorded inputs. It does not establish
GPU parity, a performance improvement, or a live AWS rollout. Publication,
retention, provider replay, deployment-failure and client tests cover separate
contracts; they do not substitute for the production-pipeline comparison.

# Audit model-fix isolation design (2026-09-18)

Status: **design only**. No AWS Batch job was dispatched for this document. Building
the diagnostic branch, the spec and the gate, and launching cells, each need an
explicit owner go.

## Why

PR #1565 (`codex/audit-runtime-correctness`) bundled ~8 unconditional model and
data-preparation fixes with tooling, serving, client and analysis repairs. Its only
metric evidence is a 36-cell CPU *bundle* comparison at `e3ed317f`
(`benchmark_history/audits/2026-09-10-model-intermediate-comparison.json`) whose
`limitations` record "attention aggregate MAE increases for RB/WR/TE". That
comparison predates #1566, #1574 and #1534 on main and has no per-fix arms, so it
cannot say which fix regresses. The neutral slices of #1565 were split into
independent PRs; the fixes below were split into one held draft PR each and need
paired, per-fix evidence before any of them can pass the repair gate
(`todo/model-default-repair/README.md`: both MAE and RMSE improve per affected
model, `elite_top24` and `weekly_reference_top24` non-worse).

## Fixes and their blast radius

| Arm | Fix (source hunks) | Positions | Families | Ridge sentinel |
|---|---|---|---|---|
| `fix1` | Hurdle conditional mean `sigmoid(gate)*E[Y|Y>0]` (`src/shared/neural_net.py` GatedHead, `count_math.py`, `registry.py` `head_losses`) — same correction as #1575's flag-gated `nn_correct_ztnb_mean`; measured there (WR attention MAE +0.070±0.017, 3/3 seeds) | RB/WR/TE (`receptions: hurdle_negbin`) | nn, attn_nn | identical |
| `fix2` | Count-likelihood precision rewrite (`training.py` `negbin2/ztnb2/ztp_log_prob`, `count_math.py`) | RB/WR/TE (hurdle value losses only; `poisson_nll` heads use torch) | nn, attn_nn | identical |
| `fix3` | Validation loss weighted by observation count (`training.py` `n_val_samples`, `gval._bs`) | all six | nn, attn_nn | identical |
| `fix4` | LightGBM `subsample_freq=1` when `subsample<1` (`models.py`) | all six | lgbm | identical |
| `fix5` | WR/TE team-stint share-window reset (`src/wr/features.py`, `src/te/features.py`) | WR, TE | all four | **moves** |
| — | K/DST fold-local context imputation | K, DST CV/rolling-origin only; production `run()` unchanged | — | no cells needed (no-fit identity proof) |
| — | Reporting truth (`actual_projected_total`) and season-aware weekly ranking | reported numbers only | — | no cells needed (fixed-weight replay proof) |

The FP16 promotion inside `fix2` is a production no-op: CUDA training runs FP32 with
TF32 (`src/shared/utils.py::amp_dtype`). Only the algebraic rewrite is measurable.

## Design: one image, module switches, one fan-out

Fixes 1–5 are unconditional code changes, so `ab_harness` config mutators cannot
toggle them. Per-fix images would put baseline and candidate on different Spot hosts
(the diversified `ff-gpu-spot` CE mixes g6 L4 and g5 A10G) and could straddle a
`refresh-splits` release, breaking pairing. Instead:

1. Diagnostic branch off `origin/main` (never off the PR head `7af61481`) carrying
   only the five hunk sets above plus one commit adding `src/shared/audit_switches.py`
   (five booleans, default off = main). Guarded sites: `GatedHead.conditional_mean`,
   `ztnb2_log_prob`/`ztp_log_prob` legacy-vs-stable dispatch, the validation weight
   (`n_batch_samples` vs 1), `subsample_freq`, and the stint-group choice in both
   `_compute_features`. No producer path (`src/data/release.py::DATA_PRODUCER_PATHS`)
   is touched, so the current sealed release resolves. The ON path must equal each
   held PR's code verbatim (reviewer checks the diff).
2. One image: `gh workflow run batch-image.yml --ref <diagnostic-branch>` (SHA tag
   only; no `:latest`, no production job-definition registration).
3. Spec `src/tuning/ab_audit_fix_isolation.py` (`SUPPORTS_STACKED=False`, seeds
   42/123/7): arms `baseline`, `baseline_rep` (identical to baseline; sizes
   same-host run-to-run noise), `fix1`…`fix5`, `stack` (all five on). Every arm's
   `cfg_mutator`, baseline included, sets all five switches explicitly because a
   position's cells run sequentially in one process. `expect_ridge_identical`:
   True for `fix1`–`fix4` and `baseline_rep`, False for `fix5`, None for `stack`.
   `metric_fn` = `default_metric_fn` + the `elite_top24`/`weekly_reference_top24`
   blocks lifted into metrics + the saved-artifact inference-parity check for
   RB/WR/TE (`src/analysis/artifact_eval.py::build_test_df_from_artifacts`).
4. Launches (each under the 120-cell guard), after one real 2-cell smoke:

   | Launch | Positions | `--only` | Cells |
   |---|---|---|---|
   | A | WR TE | `baseline_rep fix1 fix2 fix3 fix4 fix5 stack` | 48 |
   | B | RB | `baseline_rep fix1 fix2 fix3 fix4 stack` | 21 |
   | C | QB K DST | `baseline_rep fix3 fix4 stack` | 45 |

   114 cells on six Spot hosts (24 of the 64-vCPU quota); ~1.5 min per cell on
   L4 (2026-09-18 production record: 18–45 s NN phase + 2–20 s CPU phase per
   position), longest job under one hour, roughly US$1–2 total.

   ```sh
   export FF_DATA_RELEASE=<current sealed release id>
   python -m src.tuning.launch_ab --spec src.tuning.ab_audit_fix_isolation \
     --positions WR TE --only baseline_rep fix1 fix2 fix3 fix4 fix5 stack \
     --image-sha <40-hex> --image-digest <sha256:...> --env FF_AMP_DTYPE=fp32 \
     --cuda-graph auto --s3-prefix ab_runs/audit-fix-isolation-<date> --run-id A-wr-te
   ```

   Optional phase 2 if `stack − Σ singles` exceeds the seed band for a family:
   `fix1+fix2` and `fix1+fix3` on RB/WR/TE (18 cells).

## Gate

A thin no-fit checker (`src/analysis/audit_fix_gate.py`, to be written with the spec)
mirrors `src/tuning/repair_gate.py` identity checks: equal `dataset_id`,
`evaluation_data_id`, `code_id`, GPU name/sm/graph flag, cohort `n` and
`cohort_hash`, complete sample counts (`src/analysis/repair_evidence.py`). Per
affected (position, family): mean paired ΔMAE < 0 and ΔRMSE < 0 on `all`, ≤ 0 on
both protected cohorts. Unaffected families must sit within the `baseline_rep`
noise floor. Outcomes: improves → the held PR may merge with the metric claim;
within noise → stays held under the strict policy unless the owner relaxes it;
regresses → held or default-off behind a flag (the #1534 pattern).

`repair_gate.promotion_gate` cannot be used verbatim: it requires 2024 and 2025
origins, the repair branch's observer `execution` blocks and a frozen candidate
hash. The 2024 offense reference is `partial` (NFL.com Week 18), so the frozen
2024 confirmation cannot pass for QB/RB/WR/TE regardless; record that instead of
substituting a recipe.

## Provable without fitting

- Producer fingerprint of the diagnostic image equals main's
  (`src.data.release.producer_fingerprint(data_producer_hashes(Path('.')))`).
- K/DST fold imputation: `_prepare_position_data` outputs under `run()`'s config
  hash-identical on main and the fix branch; the change is CV-only.
- Reporting truth / weekly ranking: `build_test_df_from_artifacts` replay on synced
  artifacts gives identical predictions and only re-scored reports; the ranking
  replay is identical on a single-season frame and differs on a two-season frame.
- `fix1` reporting component: the same replay on RB/WR/TE isolates the fixed-weight
  expectation change from the checkpoint-selection change.

# [PRIORITY] Run the attention-architecture ablation harness (built, NOT yet run)

**Priority handoff.** PR #662 added `src/tuning/ablate_attn_arch.py` — a config-toggle ablation
for the seven default-OFF attention extensions (PRs #109/#112/#115/#116/#117/#120/#121, all on
`main`, none enabled by any position, none ever benchmarked). The harness is smoke-verified but
**deliberately not run** (the PR's scope was "build it, don't run it"). This note is the
execution follow-up: run it, read the decision table, then adopt winners / delete losers.

## The seven flags + tier prior
The full prior lives in the harness module docstring; in short:
- **Tier 1 (most promising):** `alibi` (#117) — recency dominates fantasy output; a games-ago
  decay is a strong, parameter-free prior. Tested both stacked on the production positional
  encoding (`alibi`) and replacing it (`alibi_only`).
- **Tier 2 (cheap regularisers):** `seqdrop` (#112), `temp` (#109), `swiglu` (#115).
- **Tier 3 (needs work):** `entropy` (#116 — coeff wants a sweep, better as a `tune_nn` axis),
  `condq` (#121 — likely redundant with the opp-defense attention branch).
- **Tier 4 (skeptical):** `selfattn` (#120) — collides with the "larger regressed on 15K-sample
  positions" stop-rule; expect a regression, confirm once, then delete the dead code.

## How to run
**Preferred (GPU-default N=24 stacked).** This ablation is now an `ab_harness` spec —
[src/tuning/ab_attn_arch.py](../src/tuning/ab_attn_arch.py). Run it on the Spot fleet (the
production metric path) with `python -m src.tuning.launch_ab --spec src.tuning.ab_attn_arch`, or
locally with `python -m src.tuning.ab_attn_arch --positions RB` (eager 3-seed on CPU; add
`--stacked-seeds` on a CUDA box). The `entropy` and `selfattn` arms are dropped from the stacked
spec (vmap side-channels — `selfattn`'s `nn.MultiheadAttention` SDPA `attn_bias` errors on every
real-GPU stacked seed, run_id `ab_attn_arch-20260615T072824Z-b5b46ea`) — use the eager
`ablate_attn_arch` recipe below for them (and for the per-target Δ table); `selfattn` is the
Tier-4 "confirm once eagerly, then delete the dead code" arm below:

1. **Data prereq.** Needs a complete local `data/raw` (the harness calls the real pipeline). If
   `data/splits` is a stale symlink (missing recently-added feature columns → `KeyError`), rebuild
   a LOCAL splits dir — the fresh-splits recipe is in the `[PRIORITY] PCA-before-Ridge` entry of
   [TODO.md](../TODO.md) (rm the symlink, `mkdir -p data/splits`, build_features → temporal_split,
   reusing cached `data/raw`). Verify a single `python -m src.rb.run_pipeline` smoke first.
2. **Screen on RB first** (the documented ablation workhorse), Tier 1–2 subset, ≥8 seeds (project
   floor for small NN deltas — single-seed overall-MAE is noise):
   ```
   python -m src.tuning.ablate_attn_arch --flags alibi,alibi_only,seqdrop,temp,swiglu \
       --seeds 42,7,123,5,99,17,31,8 --position RB
   ```
3. **Promote survivors to QB** (binding small-sample / overfit stress test), and run the full set
   incl. the skeptical ones once for the record: `--position QB` (drop `--flags` for all 7).
4. If CPU-bound runs oversubscribe (joblib×BLAS), cap threads:
   `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 …`.

## How to read the table
- **Headline:** attention-NN FP MAE, paired Δ vs baseline (mean±std across seeds). Negative = better.
- **Ridge sentinel MUST be OK** (identical Ridge MAE across variants per seed) — a mismatch means
  the runs saw different data/seed and the deltas are meaningless.
- **Per-target Δ** surfaces where a flag moves things (e.g. ALiBi should help recency-heavy yards).
- **Verdict** classifies each flag FLAT / PROMISING / REGRESSION vs a 0.02 FP-MAE noise floor.
  Single-seed runs are directional only — re-run ≥8 seeds before promoting.

## Decision → action
- **Winner** (beats baseline beyond noise on RB AND holds on QB): enable that flag in the
  position's `POSITION_CONFIG` (`src/{pos}/config.py`), re-run `python -m src.{pos}.run_pipeline`,
  diff `benchmark_history/`, then add a changelog line to
  [docs/adr/0004-attention-over-game-history.md](../docs/adr/0004-attention-over-game-history.md)
  + [docs/adr/CHANGELOG.md](../docs/adr/CHANGELOG.md). Editing a position config fires a GPU
  retrain — **not** `[docs-only]`.
- **Flat / regression** (esp. `selfattn`): leave OFF, or **delete the dead code** (the flag's
  branch in `src/shared/neural_net.py` + its tests). Deletion is numerically inert — confirm via
  the Ridge-identity tell and mark the commit training-skipped. Record a `[TESTED, REJECTED]`
  archive entry with the table.

## Stop-rules / gotchas
- Don't headline a single-seed overall-MAE win or a best-model flip (auto-memory
  `feedback_nn_seed_sensitive_overall_mae`).
- `entropy`'s sign is ambiguous and its single 0.01 probe is a screen, not a verdict — the real
  sweep belongs in `tune_nn` (Tier 3).
- A reduced-config proxy can flip the sign of a small effect — the harness already uses production
  NN config, so don't "speed it up" by shrinking epochs.

**Status:** harness merged (PR #662); this is the run + adopt/delete follow-up. Pick up here.

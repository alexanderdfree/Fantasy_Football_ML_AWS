# AGENTS.md

Orientation file for AI coding agents (Claude Code, OpenAI Codex, …). Human-facing docs live elsewhere — this file surfaces the conventions, gotchas, and "before you touch X, read Y" rules that aren't obvious from a first pass through the tree. It is the **single source of truth** for cross-agent project knowledge; `CLAUDE.md` imports it (`@AGENTS.md`) and adds only Claude-Code-specific machinery.

## Orient yourself first
- **[README.md](README.md)** — overview, architecture diagram, eval results.
- **[SETUP.md](SETUP.md)** — install, first-time data pull, how to run everything locally. If you need a command, it's probably here.
- **[TODO.md](TODO.md)** — open issues; the **Fixed archive** (root-cause + lesson for every non-trivial bug ever squashed) is split out to **[todo/fixed-archive.md](todo/fixed-archive.md)** to keep the live tracker lean. **Read the archive before proposing changes near anything it mentions** — most "obvious" fixes have been tried and it explains why they were wrong. **Update it as you ship**: move a resolved Open item into [todo/fixed-archive.md](todo/fixed-archive.md) (or add a fresh archive entry) using the existing `### [FIXED] Title` + **File(s)/What/Fix/Lesson** format.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the project's ADR: an **index** (Context, System Overview, a decision table, cross-cutting consequences, references) that links out to **per-decision files in [docs/adr/](docs/adr/)** (`00NN-<slug>.md`, one decision each, with rejected alternatives). The terse changelog is [docs/adr/CHANGELOG.md](docs/adr/CHANGELOG.md); the frozen pre-split history is [docs/architecture-history.md](docs/architecture-history.md). **Living docs** — update whenever a non-trivial change touches or adds an architectural decision (see "When making changes").

## Project shape (six-position symmetry)
Each of `src/qb/ src/rb/ src/wr/ src/te/ src/k/ src/dst/` follows the same template:

```
src/{pos}/
  config.py        # hyperparams (Ridge alpha grids, NN dims, loss weights, Huber deltas, LightGBM params)
  data.py          # loading + temporal split specifics
  features.py      # position-specific feature engineering
  targets.py       # raw-stat target definitions
  run_pipeline.py  # exposes run() and run_cv()
```

Tests for each position live under `tests/{pos}/`.

Shared plumbing is in [src/shared/](src/shared/): `pipeline.py` (train/eval loop), `models.py` (single-target building blocks `RidgeModel` / `ElasticNetModel` / `SeasonAverageBaseline` / `LastWeekBaseline` plus the multi-target wrappers `RidgeMultiTarget`, `ElasticNetMultiTarget`, `LightGBMMultiTarget`, plus `TwoStageRidge` and gated-ordinal classifiers), `neural_net.py` (attention and gated NN heads), `aggregate_targets.py` (raw-stat → fantasy-point scoring), `training.py`, `evaluation.py`, `backtest.py`. The root-level `models/` directory is a separate placeholder for trained-model artifacts that load from S3 — different beast.

The rest of `src/` groups by purpose:
- `src/data/` — cross-position data loading + temporal split (per-position `data.py` files wrap these): `loader.py`, `nflcom_loader.py`, `preprocessing.py`, `redzone_pbp.py`, `split.py`.
- `src/features/engineer.py` — cross-position feature engineering coordinator.
- `src/shared/evaluation.py` — position-aware visualization/aggregation layer plus the `compute_metrics(y_true, y_pred)` helper used by backtest and pipeline.
- `src/serving/` — Flask app + assets.
- `src/batch/` — training orchestration (AWS Batch path). New tuner/ablation files go in `src/tuning/`, **never** here — files under `src/batch/` trigger a full 6-position retrain via [src/scripts/scope_positions.py](src/scripts/scope_positions.py), except files whose name contains `tune`/`ablate` and the exact helper basenames `launch.py` / `benchmark.py` (job submission / read-only aggregation). PR [#280](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/280) burned ~4 GPU-jobs from a tuner-only change accidentally placed here.
- `src/benchmarking/`, `src/tuning/` — Optuna + ablations.
- `src/analysis/` — post-hoc analyses.
- `src/scripts/` — operator CLIs.
- `src/config.py` — global constants (`SEASONS`, `POSITIONS`, scoring dicts, `TOP_K_RANKING`). Distinct from per-position `src/{pos}/config.py`, which holds model hyperparams.

All six positions train an attention NN (DST landed via `cc0c627`, K via `801b61a`). There is no "skill-positions-only" carve-out anymore — if you're adding an NN-related knob, wire it through every position.

**Adding a new position**: copy an existing folder under `src/`, rename files/constants, wire it into `src/batch/train.py` and the position list in `.github/workflows/_detect-positions.yml` (shared by `train-batch.yml` (active path) and `train-ec2.yml` (rollback path)). Also update [src/scripts/scope_positions.py](src/scripts/scope_positions.py) — the canonical path → positions mapping (contract-tested by [tests/scripts/test_scope_positions.py](tests/scripts/test_scope_positions.py)) used by both training workflows' `detect` job. Add tests under `tests/{pos}/`.

## Platform & hardware targets (autodetect, then optimize per-arch)

This project is trained / tuned / benchmarked / tested across six environments. **Any platform-specific optimization or fix must autodetect the platform and branch with awareness of all of them — never hardcode for the box you happen to be on.** Reuse the detection primitives below instead of ad-hoc `platform.system()` / `torch.cuda.is_available()` sniffing, and keep the project's "autodetect by default, env-var override" shape (the `FF_DEVICE` / `LGBM_N_JOBS` pattern) so CI and reproducibility runs can still pin behaviour.

| Platform | OS | Device | GPU arch / sm | AMP dtype | torch wheel | CPU | Key gotcha |
|---|---|---|---|---|---|---|---|
| Apple Silicon MacBook | macOS (arm64) | CPU default · **MPS opt-in** | — | FP32 (no AMP) | `cpu` | M-series | MPS unproven for this small model; `FF_DEVICE=mps` to benchmark; CPU = CI-identical |
| PC (RTX 5080) | Windows 11 | CUDA | Blackwell **sm_120** | FP16 def · BF16 opt-in | **cu128** | 9950X3D 16C/32T | `OPENBLAS_NUM_THREADS=1` **REQUIRED** (crash, not perf); `LGBM_N_JOBS=16` |
| PC (RTX 5080) | WSL2 (Linux) | CUDA | Blackwell **sm_120** | FP16 def · BF16 opt-in | **cu128** | 9950X3D | no OPENBLAS crash; still cap BLAS for throughput; `scripts/wsl-env.sh` |
| AWS g4dn.xlarge | Linux | CUDA | Turing **sm_75** (T4) | FP16 (BF16 opt-in→FP16) | cu126 | 4 vCPU | EC2 rollback path AND **the live AWS Batch fan-out as of 2026-06-07** (CE `instanceTypes=["g4dn.xlarge"]`, verified via describe-compute-environments) — so graphs/`torch.compile`/BF16 are all gated off (sm_75) on production Batch; the FP16 lowest-common-denominator |
| AWS g6.xlarge | Linux | CUDA | Ada **sm_89** (L4) | FP16 def · BF16 opt-in | cu126 | 4 vCPU | **Intended** Batch fan-out per design docs, but **NOT deployed** — the live CE is still g4dn/T4 (see row above); an L4 migration (g6 or fractional **g6f**) to unlock CUDA graphs is under evaluation; BF16 measured-worse (#640) |

**Reuse these primitives — don't reinvent detection:**
- **`detect_platform()`** ([src/shared/platform_detect.py](src/shared/platform_detect.py)) — the canonical capability report (`backend` cuda/mps/cpu, `gpu_name`, `compute_capability`, `sm`, `supports_bf16`, `os`, `is_wsl`, `cpu_count`, `recommended_cuda_wheel`). Reporting-only; branch new per-arch logic off this.
- **`requested_device()` / `cuda_enabled()` / `mps_enabled()`** ([src/shared/utils.py](src/shared/utils.py)) — the device resolver: the operator's `FF_DEVICE` (`auto`/`cpu`/`cuda`/`mps`, set by `run_pipeline --device`) layered on top of detection. Single source of truth, consumed by `_nn_device()` ([src/shared/pipeline.py](src/shared/pipeline.py)). `auto` is CUDA-or-CPU and **never** MPS, so the default path stays byte-identical to CI.
- **`_gpu_resident_device()` + `_autocast()`** ([src/shared/training.py](src/shared/training.py)) — the GPU-resident batcher and AMP are **CUDA-only by design**; off-CUDA (MPS/CPU) falls through to the DataLoader + FP32 path.
- **`amp_dtype()` / `requested_amp_dtype()`** (`FF_AMP_DTYPE`, [src/shared/utils.py](src/shared/utils.py)) — the AMP-precision resolver layered on `cuda_enabled()`: FP16 on every CUDA GPU by default, `None` (AMP off) off-CUDA. `FF_AMP_DTYPE=bf16` opts into BF16 **sm_80+ only** (degrades to FP16 on the T4); `fp32` disables AMP (#640).
- **`_maybe_compile()` (`FF_COMPILE`) + TF32** ([src/shared/pipeline.py](src/shared/pipeline.py)) — `torch.compile` is opt-in and compute-capability-gated (sm_80+; off by default after the T4 +32% regression, D12), and TF32 for FP32 matmuls auto-enables on sm_80+. Both are speed-only per-arch knobs (#641).
- **`cuda_graph_enabled()` (`FF_CUDA_GRAPH`) + `_maybe_graph_model()`** ([src/shared/utils.py](src/shared/utils.py), [src/shared/training.py](src/shared/training.py)) — **autodetect-ON CUDA-graph capture of the NN's fwd+bwd for sm_80+** (g6/L4, 5080) via `make_graphed_callables`; `FF_CUDA_GRAPH` is now a **force-off override** (`=0`/`false`/`off`), *not* the trigger (reversed from opt-in 2026-06-05, PR #874 follow-up). **~1.5-1.8× on the launch-bound GPU branch (both base + attention NN) but the ONE speed knob that is NOT numerically inert**: per-step is bitwise-exact, but FP16+GradScaler amplifies the multi-step trajectory ~0.5% worst-target. **A deliberate, owner-approved per-arch metric-path divergence** — the sm_80+ training path is intentionally non-byte-identical to CPU/CI and benchmark history rebaselines graphed-vs-graphed from the cutover; CPU/MPS and the T4 (sm_75) stay eager and K's nested trainer no-ops capture. Use `FF_CUDA_GRAPH=0` for a bit-comparable eager A/B. Full root-cause + the LN/FP32/det-stop dead-ends in [todo/gpu_launch_bound_levers.md](todo/gpu_launch_bound_levers.md) (Lever A); kept investigation knobs: `FF_NN_NORM` (BN→LN backbone; overlaps `src/tuning/ablate_backbone_norm.py`), `FF_FORCE_DROPOUT_ZERO`, `FF_NN_FIXED_EPOCHS`.
- **CPU/thread knobs**: `_lgbm_n_jobs()` (`LGBM_N_JOBS`, [src/shared/models.py](src/shared/models.py)), `_default_n_jobs()` ([src/tuning/tune_lgbm.py](src/tuning/tune_lgbm.py)). Per-platform CPU/BLAS setup is documented in [SETUP.md](SETUP.md) (Windows / WSL2 sections) + [scripts/wsl-env.sh](scripts/wsl-env.sh) — cross-reference it, don't duplicate.
- **Install**: per-platform torch wheels already exist — [requirements-dev.txt](requirements-dev.txt) (cpu), [requirements-gpu.txt](requirements-gpu.txt) (cu128 / Blackwell sm_120), [src/batch/Dockerfile.train](src/batch/Dockerfile.train) (cu126 / T4·L4). Extend these; don't add ad-hoc pins.

**Platform stop-rules (decided — don't relitigate without new evidence):**
- **The training metric path is deliberately NOT per-arch.** The default AMP dtype (`amp_dtype()`) is FP16 on *every* CUDA GPU and AMP-off (FP32) off-CUDA, so CPU/CI and cross-GPU benchmark numbers stay comparable. BF16 is **opt-in only** (`FF_AMP_DTYPE=bf16`, sm_80+) and stays non-default for two measured reasons: it hung the T4 (sm_75 has no BF16 Tensor Cores, #293 → #301) **and** a deterministic 5080 A/B showed it *regresses* high-magnitude heads (QB `passing_yards` +2.2–3.1%, #640) — BF16 trades mantissa bits the model needs. The **T4/g4dn stays a live target** (EC2 rollback, intentionally divergent hardware). "Optimize per-arch" applies to speed knobs that don't change numerics — threads, DataLoader, GPU-resident batcher, wheels, TF32 (auto sm_80+), and the opt-in `torch.compile` (`FF_COMPILE`, sm_80+, #641) — **not** silently to the canonical metric path. The **one deliberate exception is `FF_CUDA_GRAPH`**, which as of 2026-06-05 autodetects ON for sm_80+ and *does* move the metric path per-arch — but that's an explicit, owner-approved divergence with a graphed benchmark rebaseline (see the `cuda_graph_enabled()` primitive above + ADR-0017), not a silent one. Re-proposing a per-arch *training-dtype default* still needs a benchmark-comparability argument, not "the GPU supports it." (GPU dtype × compute-capability and CUDA-wheel-per-arch details are in **Operating lessons** below.)
- **MPS is opt-in, never the Mac default.** `detect_platform()` reports it and `FF_DEVICE=mps` runs on it, but `auto` stays CUDA-or-CPU: no proven speedup for this small attention model, it breaks CPU/CI byte-identity, and it risks silent op-fallback. Flip the default only after a Mac A/B (`time` default vs `FF_DEVICE=mps` on one position) justifies it.
- **Windows `OPENBLAS_NUM_THREADS=1` is correctness, not perf** — without it the Ridge-PCA alpha CV segfaults (`0xC0000005`). Never drop it on **native** Windows; WSL2 / Linux / macOS need it only for throughput (`detect_platform().is_wsl` distinguishes WSL2 from native Windows).
- **Editing `src/shared/` fires a 6-position retrain** (path-based `detect` job). A numerically-inert detection refactor still triggers it — confirm the no-op via the Ridge-identity tell (identical deterministic Ridge MAE ⟺ identical data/code path), or mark the commit training-skipped.

## Conventions that bite if ignored

### Raw-stat targets, never fantasy-point targets
Every position predicts raw NFL stats (yards, TDs, receptions, etc.). Fantasy points are computed *after* prediction via `src.shared.aggregate_targets.predictions_to_fantasy_points(pos, preds)`. If you find yourself training a model directly on `fantasy_points`, stop — you'll break scoring-format flexibility and regress the ~1.9 pt/game double-count fix documented in TODO.md's archive.

### Feature whitelist is explicit, not inferred
`POSITION_CONFIG.include_features` in QB/RB/WR/TE config files (a kwarg on the `PositionConfig` dataclass, backed by a module-level `_INCLUDE_FEATURES` dict) is an opt-in list. K/DST use the same explicit-whitelist rule through `_SPECIFIC_FEATURES`, `_CONTEXTUAL_FEATURES`, and `_ALL_FEATURES` instead of `include_features`. New columns must be added explicitly — the training code will *not* pick them up automatically. This prevents silent feature leakage. When you add a feature, update both the feature-engineering file *and* the relevant config whitelist, then update the test fixture (`tests/conftest.py` or `tests/{pos}/conftest.py`).

### `CONFIG_TINY` is the test fixture, not production
Each `src/{pos}/config.py` exports **two** config shapes that look identical at a glance and have opposite values for the same toggle:

- `CONFIG_TINY = {...}` — a small dict literal near the module top with shrunken `nn_epochs`, no LightGBM, attention often disabled. Used by `tests/{pos}/` for fast unit runs. Dict-literal syntax (`"train_lightgbm": False`).
- `POSITION_CONFIG = PositionConfig(...)` — the production config object consumed by AWS Batch via `build_pipeline_config(pos, POSITION_CONFIG)` in `run_pipeline.py`. Kwarg syntax (`train_lightgbm=True`).

`grep "train_lightgbm" src/k/config.py` returns **both** entries with opposite booleans. When checking what production actually runs, always read `POSITION_CONFIG` (kwarg form, lower in the file) — never the dict-literal form.

### Attention static-feature whitelist is separate per position
The attention NN's static branch reads a *second*, smaller allowlist: `POSITION_CONFIG.attn_static_features` (commit `2500ecc`), a kwarg on the per-position `PositionConfig` dataclass. It is defined per position (QB/RB/WR/TE derive it from an `ATTN_STATIC_CATEGORIES` subset of `_INCLUDE_FEATURES`; DST/K enumerate it directly). The static branch is **deliberately non-temporal**.

**Never propose adding rolling / ewma / trend / L3 / L5 / L8 (or any windowed) features to `ATTN_STATIC_FEATURES`.** Temporal signal already feeds the NN through `ATTN_HISTORY_STATS` via the per-game attention sequence. Mixing windowed features into the static branch re-creates the double-counting this design exists to prevent. If you see attention NN losing to ridge/LightGBM on some target and you're tempted to "promote the rolling stats LGBM uses" into the static branch — stop. That's the wrong reach. The architectures differ, not the input availability: LGBM consumes rolling stats as flat columns split by trees; the attention NN consumes the *same underlying signal* as a 17-game sequence through attention. Eligible reaches for closing such a gap:

1. Add **non-temporal** features to `ATTN_STATIC_FEATURES` — prior-season aggregates, matchup, contextual, weather/Vegas, role/depth, season-to-date rates, interactions.
2. Add new **per-game** stats to `ATTN_HISTORY_STATS` — red-zone splits, share-style measures, game-script not already in the 17-game sequence.
3. Retune the loss head — Huber δ + matching `LOSS_WEIGHTS = 2.0 / δ` (see next section).
4. Change a head's parametric form — gated/two-stage for sparse counts (but note `hurdle_poisson` was tried and reverted for RB sparse counts in PR [#219](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/219)).
5. NN architecture — `d_model`, `n_heads`, dropout. Larger has been tried and regressed on 15K-sample positions; verify against benchmark before merging.

Adding a feature to `INCLUDE_FEATURES` does **not** feed it into attention — add it to `ATTN_STATIC_FEATURES` (if non-temporal) or `ATTN_HISTORY_STATS` (if per-game) too if that's what you want.

### Loss weights are tuned inverse-to-Huber-delta
`LOSS_WEIGHTS` ≈ `2.0 / HUBER_DELTAS[target]` for every Huber head. The rationale is baked into QB's config comment ([src/qb/config.py](src/qb/config.py)): the yards heads keep the 2.0/δ rebalance (without it FP MAE regressed 6.33 → 6.63 and fumbles_lost R² went negative); QB's count heads (TDs/INTs/fumbles) moved to Poisson NLL with weight 1.0, so they no longer use a Huber delta at all. If you retune a Huber delta, re-derive the matching loss weight — don't change one without the other.

### `non_negative_targets` is per-head, not global
The NN clamps outputs to ≥ 0 per head. **All six positions set `nn_non_negative_targets=set(_TARGETS)` explicitly** (clamp every head) in their `POSITION_CONFIG`; the `PositionConfig` field default is `field(default_factory=set)` — empty (no clamp) — so a future position that forgets to set it would silently disable non-negativity. The `MultiHeadNet`-level default of `None` (which clamps every head) is the *fallback* shape; production never hits it. If a position ever adds a signed head (e.g. a bonus that can go negative), pass a set that *excludes* that head rather than flipping the behaviour globally. If you construct `MultiHeadNet(...)` anywhere outside the `build_multihead_net*` factories in `src/shared/neural_net.py`, mirror the `non_negative_targets=cfg.get("nn_non_negative_targets")` kwarg — the CV path was missed once (see TODO.md archive).

### Always diff training vs inference paths
The training pipeline in `src/shared/pipeline.py` and the serving code in `src/serving/app.py` both build features. They have drifted silently in the past (weather/Vegas merge in training but not serving; scaler clip in one path but not the other). If you touch feature building in either, check the other.

### Merge-key-correct ≠ source-semantics-correct
A feature merged on the current `(player_id, season, week)` with no `.shift()` can still be stale: the upstream *source* may label a snapshot by the wrong week. Grepping `.shift()`/`.diff()` proves the *code* doesn't lag — it says nothing about the data source. The legacy (≤2024) nflverse depth chart labeled "week W" actually reflected week W-1's lineup; `_fetch_depth` applies `week -= 1` (REG-only) to realign it ([#595](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/595)). For any "known-before-kickoff" feature (depth chart, weather, lines, injuries), audit alignment against an independent ground truth (e.g. does the chart's rank-1 QB match who *actually* started week W?), restricted to transition rows where stale-by-1 separates from current — don't trust the merge key. The reusable diagnostic is [src/analysis/audit_depth_alignment.py](src/analysis/audit_depth_alignment.py).

### Use `torch` ops inside NN training paths, not `numpy`
Anything that runs inside the forward pass, loss, or an `aggregate_fn` callback must stay in `torch` to preserve gradients. `np.digitize`/`np.clip`/`np.where` on tensors silently breaks autograd — call `torch.bucketize`/`torch.clamp`/`torch.where` instead. Note that `torch.bucketize(..., right=False)` and `np.digitize(..., right=False)` use opposite edge-inclusion conventions; verify boundaries when porting.

### Don't commit data or large binaries
Datasets (`*.parquet`, `*.csv`), model weights, and demo media (`.mov`/`.mp4`) never live in git. Training data loads via `nflreadpy` (through the `src/data/nfl_source.py` shim) at workflow runtime; demo videos go to YouTube and are linked from [README.md](README.md). For new CI data dependencies, fetch in the workflow step — do not stash a file in the repo to "make CI green."

### Stop rules — things that have been tried and reverted
These have all been attempted, shipped, and reverted. Re-proposing them costs a round-trip; don't.

- **Shared-venv CI optimization** — reverted in PRs [#110](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/110) / [#111](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/111) (2026-04-23). Artifact download (~25s/shard) is slower than the warm `uv` install (~10s). Wall-clock is the metric, not compute.
- **Module-level pre-warm under gunicorn `--preload`** — reverted in PRs [#148](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/148) / [#149](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/149) (2026-04-27). The bind happens *after* preload import; a slow pre-warm causes ALB TCP-refused → unhealthy. Use a `post_fork` hook or a background thread instead.
- **Training models directly on `fantasy_points`** — cross-link to "Raw-stat targets" above. Re-introducing this regresses the ~1.9 pt/game double-count fix in TODO.md's archive.
- **Promoting rolling / L3 / L5 / L8 / ewma / trend features into `ATTN_STATIC_FEATURES`** — cross-link to "Attention static-feature whitelist" above. The static branch is deliberately non-temporal; temporal signal already feeds attention through `ATTN_HISTORY_STATS`. Do not pitch this as a way to "close the gap to LightGBM" — the input availability is not the gap; the architecture is.
- **Adding loss-config knobs (`HUBER_DELTAS`, `LOSS_WEIGHTS`, `head_losses`, `gated_targets`) to [src/tuning/tune_nn.py](src/tuning/tune_nn.py)'s search space** — cross-link to "Loss weights are tuned inverse-to-Huber-delta" above. `LOSS_WEIGHTS ≈ 2.0 / HUBER_DELTAS` is a coupling, not two independent axes; Optuna sampling them independently produces inconsistent pairs and searching deltas + deriving weights also blows up dimensionality past what ~30 trials resolve. Hand-tune loss-config via the [src/tuning/ablate_rb_gate.py](src/tuning/ablate_rb_gate.py) pattern instead (hardcoded variants, side-by-side decision table).
- **Rookie draft-capital / NFL-combine features** — investigated, fully implemented, and reverted 2026-05-29 (see [TODO.md](TODO.md) `[TESTED, REJECTED] Draft-capital / combine rookie cold-start features`). Combine athletic testing carries no marginal signal beyond draft position; draft capital (`log(pick)`) *does* have real rookie signal but is **benchmark-flat** — best-model overall MAE is unchanged at every skill position because the benefit concentrates in LightGBM (the best model only for RB), and rookies are ~14% of rows so any gain is invisible in overall MAE. Don't re-propose without first adding a tracked rookie-subgroup metric, or scoping narrowly to RB / LightGBM-only.
- **Per-arch training-dtype default (BF16, etc.)** — cross-link to "Platform & hardware targets" above. FP16 is the deliberate default on every CUDA GPU; BF16 is opt-in only (`FF_AMP_DTYPE=bf16`, sm_80+) because it hung the T4 (#293 → #301) **and** a deterministic 5080 A/B measured it regressing high-magnitude heads (#640). "The GPU supports it" is not sufficient — bring a benchmark-comparability argument.

## Running code

Commands live in [SETUP.md](SETUP.md). Shortcuts:
- `python -m src.benchmarking.benchmark [POS ...]` — benchmark & refresh artifacts (writes a `{run_id}.json` file under `benchmark_history/`).
- `python -m src.{pos}.run_pipeline` — single position, full local run.
- `pytest -m unit` — fast subset, runs in seconds. `pytest` for the full suite (requires `data/splits/*.parquet`).
- `ruff check . && ruff format --check .` — lint/format gate used by CI.

## CI & training

- `tests.yml` — ruff + pytest on push/PR. Installs via `uv` (migrated in `3c897d8`) and shards pytest across `QB/RB/WR/TE/K/DST/shared` matrix jobs (per-position paths under `tests/{pos}/`; the `shared` shard runs `tests/` excluding the per-position dirs). Each shard uploads coverage to Codecov under a matching flag; the project target is **80% per component/flag** (see [codecov.yml](codecov.yml)). Diagnostic CLIs (`src/qb/diagnose_outliers.py`, `src/rb/analyze_errors.py`, `src/wr/benchmark_ridge_variants.py`) are excluded from the coverage denominator. If `Run Tests` silently stops firing on rapid force-push cadence (occasional GitHub Actions bug), run `pytest` locally and merge with `gh pr merge --squash`.
- `batch-image.yml` → `train-batch.yml` OR `train-ec2.yml` — image build triggers training; which workflow fires is controlled by the `BATCH_ACTIVE` repo variable, currently `true` (default since 2026-05-20). `true` → parallel Spot fan-out via `train-batch.yml` (six g6.xlarge Spot instances, one per position; **measured 2026-05-21: ~10 min "Submit Batch jobs and wait" as the dominant step (the prior post-train "Refresh ECS service" step — formerly ~3 min after ALB target-group tuning, ~8 min before — was removed from this workflow in PR #330; the serving container's in-flight manifest poller now picks up new artifacts, so the Batch path no longer blocks on an ECS rollover)**, vs. the original ~25–30 min design estimate. See [docs/batch_design.md](docs/batch_design.md), D13). `false` (rollback) → warm-EC2 path via `train-ec2.yml` (~120 min sequential; see [docs/ec2_design.md](docs/ec2_design.md), D7/D9). `workflow_dispatch` on either workflow bypasses the gate (break-glass). Both paths use the same `detect` job (diff the merge commit, retrain only changed positions); the path → positions mapping is centralized in [src/scripts/scope_positions.py](src/scripts/scope_positions.py) and contract-tested by [tests/scripts/test_scope_positions.py](tests/scripts/test_scope_positions.py) — touch both when changing the global-trigger list. AWS quotas: g4dn.xlarge OD = 4 vCPU (one instance, EC2 path); Spot G+VT = 24 vCPU (six instances, Batch path) — sized exactly for the fan-out.
- `deploy.yml` — ECS Flask deploy.

AWS-side operational facts (GPU quota, training path, CI anomaly, etc.) are in **Operating lessons** below.

## Worktree workflow

This repo is regularly worked from agent worktrees (`.claude/worktrees/<name>` or `~/.codex/worktrees/<id>/Final-Project`) where the parent worktree holds `main`. These quirks apply to any agent:

- **Codex startup should go through `scripts/codex-fresh-worktree.sh`.** A Codex `SessionStart` hook can warn but cannot move the active session cwd. The launcher reuses only a clean Codex-owned worktree under `${CODEX_HOME:-~/.codex}/worktrees/*/Final-Project`; otherwise it creates a fresh `codex/session-<id>` worktree from `origin/main`, links ignored `data/raw` and `data/splits` from the main checkout when present, and starts Codex with `--cd` there.
- **Edit files in the worktree, not the parent checkout.** Plan files and search tools often report repo-relative or *parent*-absolute paths (`/…/Final-Project/src/foo.py`). Writing those verbatim silently edits the parent (`main`'s checkout) instead of this feature branch — `git status` in the worktree stays clean and any benchmark re-run uses the *unchanged* code (MAE Δ=0.0000 on every target is the late smell). Re-prefix to the active worktree path (Claude: `/…/Final-Project/.claude/worktrees/<name>/…`; Codex: `~/.codex/worktrees/<id>/Final-Project/…`), then `grep` the new symbol in the worktree file to confirm the edit landed. Claude Code and Codex both have deterministic guard hooks for this (`CLAUDE.md`, `CODEX.md`).
- **`gh pr merge --delete-branch` fails** in a worktree (it tries to `git checkout main`, which is held by the parent). Use `gh pr merge <N> --squash` then `git push origin --delete <branch>` separately. Local feature branch can stay.
- **"Is X on `main`?" / dead-link checks** must read `origin/main:<path>` via `git fetch origin main --quiet && git show origin/main:<path>` — never `cat <path>` in the worktree, which lags `main`.

## When making changes
- **Open a PR, wait for green CI, then merge.** Push to a feature branch, open the PR with `gh pr create`, then `gh pr checks <N> --watch` until every check passes before running `gh pr merge <N> --squash`. Don't merge with red or pending checks; if a check fails, fix the underlying issue rather than bypassing with `--admin`. Exception: the `Run Tests` silent-stop bug noted in the CI section — run `pytest` locally and merge.
- **`[docs-only]` commit-subject opt-in for comment/docstring/import-reorder PRs.** When every change in the PR is non-behavioural (a comment text fix, a docstring update, an `is*` typo, ruff I001 import reordering) and you are 100% certain there is no metric or runtime impact, include `[docs-only]` in at least one commit **subject line** (the squash-merge subject becomes the PR title, and squash bodies preserve constituent subjects as `* `-prefixed bullets — both count; prose in commit *bodies* does not, consumers use a subject-line awk filter). It's respected by: `tests.yml`'s `detect` (empty shard matrix; `tests-pass` green via `skipped`), `batch-image.yml`'s `check-docs-only` (skips `build-and-push`), `_detect-positions.yml` (empty `positions` → training skip), and the Claude/Codex pre-PR hooks (early-exits the ruff/pytest/benchmark gates). `lint` + `detect` still run. [deploy.yml](.github/workflows/deploy.yml) is **not** gated by the tag (its `paths:` filter on `docs/**`+`README.md`+wiki sources is the gate — docs render in the in-app wiki, so they need redeploy). Trust contract — CI can't verify the assertion; the author owns correctness. Two traps: keep the literal tag out of your OWN commit subject / PR title when the PR *touches* docs-only machinery (else it skips its own matrix, #293), and serving display strings (dict values rendered into HTTP responses, e.g. `POSITION_INFO` "formula" fields) are behavioral — **not** docs-only even if no test asserts them.
- Respect the **[Fixed archive](todo/fixed-archive.md)** — it encodes the project's accumulated "already tried" knowledge.
- **Update the ADR + decision log alongside non-trivial changes.**
  - **ADR (per-decision files in [docs/adr/](docs/adr/)):** if your change touches an existing decision, edit that decision's `docs/adr/00NN-<slug>.md` (`Decision`/`Context`/`Chosen`/`Rejected`/`References`/`Consequence`) and append a dated line to its `## Changelog` section (create it if absent). Introducing a new decision of similar weight? Add `docs/adr/00NN-<slug>.md` (next free number) + a row in the index table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — it auto-registers in the in-app wiki via the `docs/adr/*.md` glob. **Either way, add one terse line to [docs/adr/CHANGELOG.md](docs/adr/CHANGELOG.md)** (`YYYY-MM-DD · summary · (PR #N) · → ADR-00NN`) — never a verbose paragraph. Superseding a decision? Flip the old file's `**Status:**` to superseded and add a new file — don't rewrite the original. Per-decision files stay well under the Read-tool cap; never re-monolith them.
  - **Decision log ([TODO.md](TODO.md) Open list; Fixed archive at [todo/fixed-archive.md](todo/fixed-archive.md)):** for non-trivial bug fixes, move the corresponding `Open` entry into [todo/fixed-archive.md](todo/fixed-archive.md) using the existing `### [FIXED] Title` + **File(s)** (paths + commit SHA) / **What** / **Fix** / **Lesson** format. If the bug wasn't tracked, add a fresh archive entry anyway.
  - **Skip both** for truly trivial changes (typos, formatting, lockfile bumps, comment-only tweaks). When in doubt, lean toward writing the entry — a thin archive is the failure mode, not over-documentation.
- Update tests and fixtures when you change feature lists or targets (archive has multiple entries where this was missed).
- Don't add error handling, fallbacks, or validation for cases that can't happen. One exception: network/data-source boundaries are real and should be defensive.
- **For NN/feature/loss/target changes, run the actual pipeline before merging.** `pytest -m unit` and CI tests don't catch metric regressions. Run `python -m src.{pos}.run_pipeline` on the affected position and diff `benchmark_history/` output against the prior run. The K refactor regression and the QB metrics-label bug both shipped because tests passed without the pipeline being run. **This applies to *investigating* a feature, not just merging it** — see "Validation proxy must match production" in Operating lessons. For subgroup analysis, per-row test predictions are already on `result["test_df"]` (`pred_{model}_total`) — slice those, don't reimplement the models.
- **Large (>10-item) parallel cleanups: partition into file-disjoint bundles, draft commits per bundle, then open one PR per risk tier** (safest → highest-risk; the 113-finding remediation → 3 PRs [#312](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/312)/[#314](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/314)/[#315](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/pull/315) is the canonical example). **File-disjointness is for parallelism, not correctness** — it does NOT protect against an API-signature change in shared code that a per-position bundle still calls (the 2026-05-21 Tier A `_train_nn` signature conflict), so when a bundle changes a shared signature, grep for every caller first. Operator-only CLI scripts (`diagnose_outliers.py`, `analyze_errors.py`, `audit_features.py`) need at least an import-smoke test so signature drift fails the unit shard, not PR review.

## Operating lessons (any agent)

Hard-won lessons from prior sessions. One line each; provider-neutral. (Claude Code also keeps these in its auto-memory; this section is the cross-agent source of truth.)

### ML modeling & investigation method
- **Validation proxy must match production.** A reduced/unregularized model (low-`n_estimators` LightGBM, bare Ridge) can give the **wrong sign** vs the tuned production model — trust the real pipeline. Per-row preds for subgroup slicing are on `result["test_df"]` (`pred_{model}_total`); don't reimplement the models.
- **Single-seed NN overall-MAE is noise.** Judge a targeted NN fix by its subgroup metric's *direction* across ≥2 seeds; don't headline an overall-MAE win / best-model flip from one seed (#596: seed-42 "beats LGBM" was flat at seed 123).
- **Default to 3 seeds for FP-MAE A/Bs, reported mean±std** (8 is too slow, especially for a tuning sweep's many variants). 3 seeds can still fake a *borderline* result (backbone-norm: 3-seed −0.022±0.019 → 8-seed +0.007±0.034 = noise) — so when a delta lands inside the seed band and the decision hinges on it, bump to 5–8 (via `--seeds`) before concluding rather than headlining the 3-seed call. Local runs ~1.5 min/variant.
- **Subgroup error = bias, not MAE.** Skewed FP targets mean low-scoring slices (returners, role players) have lower MAE regardless of model quality; judge "is the model worse on X" by bias. (MAE-delta on a *fixed* ablation slice is fine.)
- **Ridge MAE is the data-identity tell.** Identical deterministic Ridge MAE across positions ⟺ two runs trained on identical data, whatever the commit label — use it to confirm a data/code change actually moved the inputs.
- **Analysis must match production NaN handling.** Feature stats (corr/VIF/cond) must impute NaN→0 like `feature_build.py:110`, not `dropna` (dropna kept only ~52–59%, veteran-heavy, in #594).
- **Diagnostic data path must match the loader.** Load a feature the loader's way (reuse its fetch/normalize), not a raw `nfl_source.*` shim that lags schema changes; a "0 for season Y" is a schema red flag (#588→#592/#593).
- **Verify a data effect with an A/B, not column profiling.** A test-constant / float-noise feature can map to ~−4σ post-scaler. Offline split-build changes are invisible to a runtime code A/B — run the data A/B.
- **Training-data ablations via `run(train,val,test,seed)` frame injection** — filter train/val, same seed, compare on `result["test_df"]`; read-only in `src/analysis/`, no retrain.
- **Check imputation-branch reachability before "fixing" it.** An upstream fillna/lag/dropna may kill the NaN before the branch runs; NaN-count the column the fn receives on the real artifact — if 0, it's dead code (benign, Δ0 retrain) (#608→#609).
- **Trust structural-equivalence.** A default-guarded new path inert on the non-prevailing device + one sampled position byte-identical ⇒ redundant to run the other positions.
- **Pipeline-hang phase misread:** `[timing] phase=X` + silence ≠ a hang in X+1 — CPU/GPU branches run concurrently; cross-check matching CPU-branch logs first.
- **Prove a guard FIRES (positive control) + read the right data stage.** Run a regression guard on the known-bad case first. Built splits drop no-snap rows, so "who was listed" must come from the `rosters` cache (incl. benched), not splits/weekly (#611).
- **GPU-guarded code (`if torch.cuda.is_available()`) is invisible to CPU unit tests** — treat as untested; Batch dry-run before merge.
- **The attention NN is learned-query pooling (`AttentionPool`), NOT a transformer** — the `SelfAttentionBlock` encoder is gated off (`attn_self_layers=0`, set by no position). Verify the knob before describing it.
- **Projection sources may include roster placeholders.** A new expert source may return a row for every rostered player; unprojected placeholders (no stat line) scored 0.0 deflate the expert MAE and can flip a conclusion — filter to genuine projections.
- **Don't close over pipeline fns in a factory** — tests monkeypatch `src.{pos}.run_pipeline.run_pipeline`; define `run()` locally per position.
- **Refactor LOC estimate:** a foundation PR (adds the abstraction) is LOC-positive; only the migration PR saves LOC — don't promise net drops from the foundation alone.

### Project facts & infrastructure
- **AWS GPU quota (us-east-1):** 4 vCPU G OD (one g4dn.xlarge) + Spot G+VT 24 vCPU (six instances) as of 2026-04-20 — sized exactly for the six-position fan-out; check before relying.
- **Training path is flag-selected:** `BATCH_ACTIVE` repo var picks `train-batch.yml` (Spot fan-out, default since 2026-05-20) vs `train-ec2.yml` (warm OD rollback); check S3 for prod artifacts.
- **GPU dtype × compute-capability:** `autocast(dtype=)` accepting an arg ≠ kernels support it — T4 (sm_75) = FP16 only; BF16 needs sm_80+. Check the matrix before enabling AMP.
- **GPU arch is per-CUDA-wheel, not per-version:** sm_xx support comes from the cuXXX wheel build (`2.11.0+cu128` adds Blackwell sm_120) — verify the wheel index; don't trust a version→GPU claim.
- **AWS Batch UserData must be MIME multipart, not raw bash**, or the compute environment flips INVALID (recovery needs republish + disable→update→enable).
- **Local benchmark runs must mirror to S3** (`benchmark.py::_maybe_upload_to_s3`, env-gated, `--no-sync` opt-out) so they reach the website History tab; don't lift that helper into `src/shared/` (fires a 6-position retrain).
- **Diagnostics live in `src/analysis/`** (import shared helpers); editing `src/shared/` or `src/{pos}/` to wire in read-only tooling fires a 6-position retrain — expose a predicate fn for later drop-in.
- **Split `fantasy_points` is skill-only** (≈0 for K, absent for DST); per-position label analysis from splits is QB/RB/WR/TE-only — K/DST totals exist only post-pipeline on `result["test_df"]`.
- **`/health` must distinguish cold start (200) from affirmative failure (503):** predicate is `errors AND not loaded`, not `not loaded` alone.
- **Injury/return:** models over-predict 2+ wk returners (bias, not MAE); attention PE is slot-indexed (`arange(seq_len)`) so gap-blind — attention ties plain NN on returners, LGBM not actually robust (#623).

### Git / PR / CI workflow
- **Check `git log origin/main` at the START of planning a feature AND again before the PR** (rebase is the backstop): the worktree base lags main; if a planned fix already shipped, pivot to the gap it left (#383/#516).
- **Before driving a PR in a hot shared file** (TODO.md, configs, tune_nn.py), `gh pr list` for concurrent OPEN PRs — a parallel session can fully supersede yours (#634⊃#629). Distinct from the merged-on-main check.
- **Regression on commit X?** `git log origin/main --grep=X` FIRST — a later PR may already have reverted it (#189).
- **Attribute regressions by reading every PR in `git log baseline..HEAD`,** not the most thematic suspect — 2nd-order bugs co-occur.
- **`gh api` paginates ~30/page** — "are these ALL the open X?" enumeration needs `--paginate` (#319 missed pages 2+).
- **Match image-tag / `head_sha` → PR on the full SHA** (positions 1–6), not 2 chars; use the workflow log's "HEAD is now at" as ground truth.
- **After `gh pr merge --squash`, verify the squash commit contains the latest fix** (`git show <sha> -- <file>`) — the repo can diverge from live.
- **Never chain a destructive action past a gate.** Don't `test && commit && push` or `merge && delete` in one batch — a masked-exit merge failure + unconditional branch-delete CLOSED #622 and auto-closed #627. Run the gate alone, verify state==MERGED before deleting; prefer `gh pr merge --squash --auto`.
- **After a conflict-resolution edit, `grep -c '<<<<<<<' <file>` must be 0** before `git add` + `rebase --continue` — an edit can silently fail and leave markers.
- **Verify checkout landed before rebase.** `git checkout <branch>` fails silently if that branch is in another worktree, and a following `git rebase` runs on the WRONG branch — rebase a sibling branch in its worktree (`git -C <path>`) or `--detach origin/<branch>`.
- **Stacked PRs:** verify the GH base retarget before deleting a merged base; rebase to fire CI on a base-swap; brief reviewers with an explicit `gh pr diff`.
- **CI `Run Tests` can silently stop firing on rapid force-push** (GitHub Actions bug) — run `pytest` locally and `gh pr merge --squash`.
- **A repo-tracked file isn't shipped until merged.** After implementing, check branch state vs origin/main, then commit → PR → merge; don't stop at file-on-disk.
- **Don't `--no-verify`** (skip hooks) even on merge-resolution commits.
- **Copilot "encountered an error / rate limit" review comments are infra noise** — don't address or reply.

### Verification, audit & communication
- **Audit by running the test, not grepping.** "Every position satisfies Y" claims need an actual run — symbol grep misses `LOSS_WEIGHTS=` vs `loss_weights=` kwarg forms.
- **Check ALL config layers for "is X implemented?"** In Batch/ECS a per-submission `submit_job(...)` override can invalidate the resource default.
- **Changing an endpoint contract:** grep the route path AND the handler name (`def health`, `/health`), not just the data structures read.
- **Enumerate before filtering in tree-wide sweeps.** One unfiltered `grep -rn <term> .`, categorize EVERY hit, THEN filter to verify — a `grep -v` as the sweep tool gives phantom completeness (#637 missed user-facing wiki refs).
- **Default investigation sweep target is the property-extremum case,** not the example the user happened to name.
- **Surface gate friction with options** (eat the cost / authorized bypass / fix the gate) instead of silently bypassing — the user often picks "fix the gate."
- **Surface user-requested scope that a mid-implementation audit shows infeasible** — pause and offer options, don't unilaterally drop it.
- **Goal governs over named mechanism.** A user may name a gate ("make pre-pr require signoff") but mean a goal ("signoff before merge"); two phrasings can be ONE checkpoint — confirm the mapping at plan time (#642).
- **Validate a tentatively-proposed mechanism before building.** A user's "maybe use X?" is a hypothesis, not a decision; for a *new subsystem* present a short architecture tradeoff (2–3 options, the axes that matter, a recommendation) and confirm before implementing. Matching the repo's existing patterns is a point *in favor*, not a reason to skip the comparison. Distinct from "Goal governs over named mechanism" (mechanism→goal mapping) — here the user names a backend/mechanism tentatively and wants the design-fit validated; building straight from the hint (S3 memory-sync) drew an explicit "confirm it's the right architecture first."
- **Verify a library version / `requires-python` via `pypi.org/pypi/<pkg>/json`,** not install output.
- **File existence is filesystem, not git.** An untracked parent-checkout file is invisible to origin/main + the worktree path — `find` broadly before crying hallucination.
- **Trace artifact lifecycle.** Adding a file to a dir that's later uploaded? Trace every call between write→upload — a `_replace_*`/`_reset_*` may wipe the destination.
- **Don't act on garbled/truncated tool output** — re-verify with one clean sequential call (a Read or a tiny script) before reporting a number (#622 cited a wrong σ + a false regen "success").
- **A skill/hook file existing ≠ it firing** — confirm the trigger path before reporting a mechanism "done."

### Environment (macOS / worktree / venv / pytest)
- **macOS has no `timeout`, and `cmd | tail` makes the exit code tail's (0)** — redirect to a file and verify the artifact, not the reported code.
- **`uv pip install <pkg>` from outside a uv-project wipes the `.venv`** — use requirements.txt + reinstall, or `uv run --with`.
- **Don't symlink the `.venv` in a worktree** (site-packages resolves wrong) — use an absolute python path. Data symlinks ARE fine.
- **Fresh worktree: symlink the parent's `data/{splits,raw}`** (`rm -rf` the dir first) instead of a slow first pull.
- **A fresh worktree has no `.venv` and the parent's can be stale** — an env with full deps (e.g. the miniforge3 base) mirrors the pre-PR gate; verify there.
- **Symlinked parent `data/splits` lag main's feature whitelist** → the pipeline fails loud (`KeyError: N whitelisted cols missing`); rebuild splits locally, or verify non-model code with synthetic tests — don't diagnose via column-diff (pipeline-time features are absent by design) (#656).
- **The pre-PR unit gate's `pytest -m unit` includes data-dependent tests that read `data/raw/*.parquet`;** a partial worktree `data/raw` flakes by xdist ordering — populate it from the parent (`cp -R`), don't retry-as-flake (#662).
- **The pre-PR freshness gate uses strict `>` mtime;** `git stash pop`/`rebase`/`checkout --` bump mtimes and break it — order operations or `touch`.
- **Parquet `ArrowInvalid` in `pytest -m unit -q` but the single test passes** → xdist race; retry the full suite ONCE before env forensics.
- **Stalled pytest with no `=== passed ===` footer** → re-run the suspect file in isolation FAST; don't reverse-engineer the dots.
- **Handoff/plan docs go at repo root or the tracked `todo/` folder** (linked from TODO.md, `[docs-only]`), NOT `.claude/plans/` (gitignored) or `docs/` (fires deploy + public wiki).
- **Splitting/renaming a tracked doc can break a test/CI allowlist that names its path as a string literal** — grep the OLD path across `tests/`/`.github/`/`.claude/`, extend the allowlist + repoint refs in the same PR, `git add` the new file before the coupled test (#661).

## Tool capabilities differ between agents

Neither agent should assume the other's tools. **Claude Code** has claude.ai account connectors (Canva, AWS, Gmail, GitHub, …), browser-preview tools, and sub-agent / Workflow orchestration. **Codex** has the OpenAI plugin set (github, browser, data-viz, codex-security, …) and a `node_repl` MCP server. There is no local MCP-server config shared between them. When you reference a capability, name it concretely so the other agent knows whether it applies.

## Codex specifics

The Claude-Code machinery (hooks, skills, the scheduled audit routine, sub-agent contracts) lives in `CLAUDE.md`; the Codex-side machinery lives in [`CODEX.md`](CODEX.md) plus [`.codex/`](.codex/). Keep both files in sync when you change a shared discipline.

| Discipline (what it enforces) | Claude Code mechanism | Codex equivalent |
|---|---|---|
| Auto-format Python after an edit | `ruff-format.sh` (PostToolUse) | [`.codex/hooks/ruff-format.sh`](.codex/hooks/ruff-format.sh) (PostToolUse on `apply_patch`/edit aliases) |
| Pre-PR gate: `ruff check` + `ruff format --check` + `pytest -m unit` + benchmark-freshness for pipeline edits | `pre-pr.sh` (PreToolUse on `gh pr create`) | [`.codex/hooks/pre-pr.sh`](.codex/hooks/pre-pr.sh) wraps the Claude gate with the Codex project root |
| Block edits that target the parent checkout from inside a worktree | `guard-worktree-path.sh` (PreToolUse) | [`.codex/hooks/guard-worktree-path.sh`](.codex/hooks/guard-worktree-path.sh) parses Codex `apply_patch` headers / edit paths |
| Env bootstrap (venv + deps + `PYTHONPATH`) at session start | `session-start.sh` (SessionStart) | [`.codex/hooks/session-start.sh`](.codex/hooks/session-start.sh) adds context only; Codex hooks cannot persist shell exports |
| Start in a clean fresh worktree | Claude starts sessions directly in its chosen worktree | [`scripts/codex-fresh-worktree.sh`](scripts/codex-fresh-worktree.sh) creates/reuses a clean Codex worktree, then launches `codex --cd` |
| Post-PR: rebase → review → apply nits → merge on green | `post-pr-create.sh` + `/review` | [`.codex/hooks/post-pr-create.sh`](.codex/hooks/post-pr-create.sh) emits a compact pointer to the prompt-backed workflow; review equivalent is `scripts/codex-review-quiet.sh --base origin/main` |
| Scope-creep check before opening a PR | `pre-pr-judge` skill | `/prompts:pre-pr-judge` from [`.codex/prompts/pre-pr-judge.md`](.codex/prompts/pre-pr-judge.md) after `scripts/bootstrap-codex-local.sh` |
| Capture prompt-lessons after a non-routine session | `post-session-critique` skill | `/prompts:post-session-critique` from [`.codex/prompts/post-session-critique.md`](.codex/prompts/post-session-critique.md) after `scripts/bootstrap-codex-local.sh` |
| Triage the `claude-audit` issue backlog into tier-by-risk PRs | `solve-issues` skill | `/prompts:solve-issues` from [`.codex/prompts/solve-issues.md`](.codex/prompts/solve-issues.md) after `scripts/bootstrap-codex-local.sh` |
| Sync auto-memory across machines (S3) | `scripts/agent-memory-sync.sh claude ...` / `scripts/claude-memory-sync.sh` + Claude hooks | `scripts/agent-memory-sync.sh codex ...` / `scripts/codex-memory-sync.sh` + Codex hooks; S3 prefix is separate from Claude |

Codex local setup: run [scripts/bootstrap-codex-local.sh](scripts/bootstrap-codex-local.sh) to install/update the user-home prompt copies, restart Codex, then review/trust the project hooks with `/hooks`. Custom prompts are user-home scoped (`$CODEX_HOME/prompts`), so the tracked [`.codex/prompts/`](.codex/prompts/) files are templates, not directly loaded by Codex. On a fresh machine, `scripts/bootstrap-codex-local.sh --with-memory-sync` also does a best-effort initial pull from the Codex memory S3 prefix.

**Memory sync is now agent-aware, not shared-state.** [scripts/agent-memory-sync.sh](scripts/agent-memory-sync.sh) syncs Claude and Codex memories to separate S3 subfolders (`claude-memory/<repo>/memory/` and `codex-memory/<repo>/memories/`). Claude's memory is project-scoped; Codex memory is still a global local store under `${CODEX_HOME:-~/.codex}/memories`, so the script syncs only the markdown memory tree and excludes Codex's `.git` internals / SQLite runtime state. `SessionStart` hooks pull the respective agent prefix; `Stop` hooks push both local memory trees if present so cross-agent memory updates made on one machine are not stranded. Durable project guidance still belongs in this file, not only in synced incidental memory.

# Model and feature contracts

Read only the sections relevant to the task. [AGENTS.md](../AGENTS.md) supplies the shared entrypoint; current code/config and linked decisions supply operational state. Dated measurements describe their recorded regime, not a promise about today.

## Conventions that bite if ignored

### Raw-stat targets, never fantasy-point targets
Every position predicts raw NFL stats (yards, TDs, receptions, etc.). Fantasy points are computed *after* prediction via `src.shared.aggregate_targets.predictions_to_fantasy_points(pos, preds)`. Training directly on `fantasy_points` breaks scoring-format flexibility and regresses the ~1.9 pt/game double-count fix in [todo/fixed-archive.md](../todo/fixed-archive.md).

### Feature whitelist is explicit, not inferred
`POSITION_CONFIG.include_features` in QB/RB/WR/TE (a kwarg on `PositionConfig`, backed by a module-level `_INCLUDE_FEATURES` dict) is an opt-in list. K/DST use the same explicit-whitelist rule via `_SPECIFIC_FEATURES`/`_CONTEXTUAL_FEATURES`/`_ALL_FEATURES`. New columns must be added explicitly — training code won't pick them up (prevents silent leakage). When you add a feature, update the feature-engineering file *and* the relevant config whitelist, then the test fixture (`tests/conftest.py` or `tests/{pos}/conftest.py`).

### `CONFIG_TINY` is the test fixture, not production
Each `src/{pos}/config.py` exports **two** config shapes that look identical at a glance and have opposite values for the same toggle:

- `CONFIG_TINY = {...}` — a small dict literal near the module top with shrunken `nn_epochs`, no LightGBM, attention often disabled. Used by `tests/{pos}/` for fast unit runs. Dict-literal syntax (`"train_lightgbm": False`).
- `POSITION_CONFIG = PositionConfig(...)` — the production config object consumed by AWS Batch via `build_pipeline_config(pos, POSITION_CONFIG)` in `run_pipeline.py`. Kwarg syntax (`train_lightgbm=True`).

`grep "train_lightgbm" src/k/config.py` returns **both** entries with opposite booleans. When checking what production actually runs, always read `POSITION_CONFIG` (kwarg form, lower in the file) — never the dict-literal form.

### Attention static-feature whitelist is separate per position
The attention NN's static branch reads a *second*, smaller allowlist: `POSITION_CONFIG.attn_static_features` (commit `2500ecc`), a kwarg on `PositionConfig`, defined per position (QB/RB/WR/TE derive it from an `ATTN_STATIC_CATEGORIES` subset of `_INCLUDE_FEATURES`; DST/K enumerate it directly). The static branch is **deliberately non-temporal**.

**Never add rolling / ewma / trend / L3 / L5 / L8 (or any windowed) features to `ATTN_STATIC_FEATURES`.** Temporal signal already feeds the NN through `ATTN_HISTORY_STATS` via the per-game attention sequence; mixing windowed features into the static branch re-creates the double-counting this design prevents. If the attention NN loses to ridge/LightGBM on a target, don't "promote the rolling stats LGBM uses" — the architectures differ, not the input availability (LGBM splits rolling stats as flat columns; the NN consumes the *same signal* as a 17-game sequence). Eligible reaches for that gap:

1. Add **non-temporal** features to `ATTN_STATIC_FEATURES` — prior-season aggregates, matchup, contextual, weather/Vegas, role/depth, season-to-date rates, interactions.
2. Add new **per-game** stats to `ATTN_HISTORY_STATS` — red-zone splits, share-style measures, game-script not already in the sequence. The mirror of the static-branch rule applies here: the token must be a **raw per-game signal genuinely absent from the sequence**, *not* a windowed/expanding-mean-derived aggregate. Routing a role/inheritance signal (an expanding-mean of `snap_pct_raw`) through `ATTN_HISTORY_STATS` is doubly wrong — it averages an already-averaged quantity, *and* the event it encodes (a spot-start) is already in the existing per-game usage tokens (`snap_pct_raw`, `game_carry_share`, carries, production), so the branch gains nothing and the redundant token slightly hurts (tested-rejected, RB 3-seed, [src/tuning/ab_history_token.py](../src/tuning/ab_history_token.py): −0.32 FP / ~3σ on the ascension cohort vs the static-only arm). The *current-week* value of such a signal is a legitimate **static** feature (reach #1) — it describes the upcoming game's vacancy, which is in no past-game sequence.
3. Retune the loss head — the per-head δ error scale + matching `LOSS_WEIGHTS = 1 / δ` (next section).
4. Change a head's parametric form — gated/two-stage for sparse counts (but `hurdle_poisson` was tried and reverted for RB sparse counts, PR #219).
5. NN architecture — `d_model`, `n_heads`, dropout. Larger regressed on 15K-sample positions; verify against benchmark first.

Adding a feature to `INCLUDE_FEATURES` does **not** feed it into attention — also add it to `ATTN_STATIC_FEATURES` (non-temporal) or `ATTN_HISTORY_STATS` (per-game).

### Loss weights are tuned inverse-to-Huber-delta
`LOSS_WEIGHTS` ≈ `1 / HUBER_DELTAS[target]` for every MSE yards head (rationale in QB's config comment, [src/qb/config.py](../src/qb/config.py)): PR #870 switched every position's yards heads from Huber to MSE (to stop discounting the elite upper tail) and re-derived each weight to 1/δ — gradient-matched at the characteristic error e≈δ to the old 2.0/δ Huber rebalance, which mattered (without it FP MAE regressed 6.33 → 6.63 and fumbles_lost R² went negative). `huber_deltas` is retained only as the characteristic error scale the weights derive from; MSE heads ignore it at loss time. Count heads (TDs/INTs/fumbles) use Poisson NLL with weight 1.0, so they use no delta. Retuning a delta means re-deriving its loss weight — don't change one without the other.

### `non_negative_targets` is per-head, not global
The NN clamps outputs to ≥ 0 per head. **All six positions set `nn_non_negative_targets=set(_TARGETS)` explicitly** in their `POSITION_CONFIG`; the `PositionConfig` field default is `field(default_factory=set)` (empty, no clamp), so a position that forgets it would silently disable non-negativity. The `MultiHeadNet`-level default of `None` (clamps every head) is the *fallback*; production never hits it. If a position adds a signed head, pass a set that *excludes* it rather than flipping behaviour globally. If you construct `MultiHeadNet(...)` outside the `build_multihead_net*` factories in `src/shared/neural_net.py`, mirror the `non_negative_targets=cfg.get("nn_non_negative_targets")` kwarg — the CV path was missed once (see [todo/fixed-archive.md](../todo/fixed-archive.md)).

### Always diff training vs inference paths
The training pipeline in `src/shared/pipeline.py` and the serving code in `src/serving/app.py` both build features. They have drifted silently in the past (weather/Vegas merge in training but not serving; scaler clip in one path but not the other). If you touch feature building in either, check the other.

### Merge-key-correct ≠ source-semantics-correct
A feature merged on the current `(player_id, season, week)` with no `.shift()` can still be stale: the upstream *source* may label a snapshot by the wrong week. Grepping `.shift()`/`.diff()` proves the *code* doesn't lag — it says nothing about the source. The legacy (≤2024) nflverse depth chart labeled "week W" actually reflected week W-1's lineup; `_fetch_depth` applies `week -= 1` (REG-only) to realign it (#595). For any "known-before-kickoff" feature (depth chart, weather, lines, injuries), audit alignment against an independent ground truth (does the chart's rank-1 QB match who *actually* started week W?), restricted to transition rows where stale-by-1 separates from current — don't trust the merge key. Reusable diagnostic: [src/analysis/audit_depth_alignment.py](../src/analysis/audit_depth_alignment.py).

### Use `torch` ops inside NN training paths, not `numpy`
Anything that runs inside the forward pass, loss, or an `aggregate_fn` callback must stay in `torch` to preserve gradients. `np.digitize`/`np.clip`/`np.where` on tensors silently breaks autograd — call `torch.bucketize`/`torch.clamp`/`torch.where` instead. Note that `torch.bucketize(..., right=False)` and `np.digitize(..., right=False)` use opposite edge-inclusion conventions; verify boundaries when porting.

### Don't commit data or large binaries
Datasets (`*.parquet`, `*.csv`), model weights, and demo media (`.mov`/`.mp4`) never live in git. Training data loads via `nflreadpy` (through the `src/data/nfl_source.py` shim) at workflow runtime. For new CI data dependencies, fetch in the workflow step — do not stash a file in the repo to "make CI green."

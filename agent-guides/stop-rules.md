# Tested and rejected approaches

Read the relevant subsystem, including its exceptions and reopening conditions:
[modeling/features](#modeling-and-features), [GPU execution](#gpu-execution),
[CI/serving](#ci-and-serving). These are rejected, conditional or superseded
approaches; they were not all shipped and reverted. Historical measurements are
retained in linked records and apply to their recorded regime.

<a id="stop-rules-things-that-have-been-tried-and-reverted"></a>
<a id="stop-rules--things-that-have-been-tried-and-reverted"></a>

## Modeling and features

- **Training models directly on `fantasy_points`** — see [Raw-stat targets](modeling.md#raw-stat-targets-never-fantasy-point-targets); regresses the ~1.9 pt/game double-count fix in [todo/fixed-archive.md](../todo/fixed-archive.md).
- **Promoting rolling / L3 / L5 / L8 / ewma / trend features into `ATTN_STATIC_FEATURES`** — see [the attention whitelist](modeling.md#attention-static-feature-whitelist-is-separate-per-position); the static branch is deliberately non-temporal. Not a way to "close the gap to LightGBM" — the gap is architecture, not input availability.
- **Routing a role / inheritance / "spot-start" signal through `ATTN_HISTORY_STATS`** — tested-rejected (RB 3-seed, [src/tuning/ab_history_token.py](../src/tuning/ab_history_token.py), 2026-06-07). The history branch already encodes a past spot-start via the existing per-game usage tokens (`snap_pct_raw`, `game_carry_share`, carries, production); a derived inheritance token (an expanding-mean of `snap_pct_raw`) is *averaging an average* and re-encodes signal already there → −0.32 FP / ~3σ worse on the ascension cohort than the static-only arm, MAE flat. The *current-week* value belongs in the **static** path (`INCLUDE_FEATURES` + `ATTN_STATIC_FEATURES`), where it's genuinely new (the upcoming game's vacancy is in no past sequence). See "Attention static-feature whitelist" reach #2.
- **Adding loss-config knobs (`HUBER_DELTAS`, `LOSS_WEIGHTS`, `head_losses`, `gated_targets`) to [src/tuning/tune_nn.py](../src/tuning/tune_nn.py)'s search space** — see [loss weights](modeling.md#loss-weights-are-tuned-inverse-to-huber-delta). `LOSS_WEIGHTS ≈ 1/HUBER_DELTAS` (2.0/δ in the pre-#870 Huber era) is a coupling, not two independent axes; sampling them independently produces inconsistent pairs and blows up dimensionality past what ~30 trials resolve. Hand-tune via the [src/tuning/ablate_rb_gate.py](../src/tuning/ablate_rb_gate.py) pattern (hardcoded variants, decision table).
- **Rookie draft-capital / NFL-combine features** — investigated, implemented, reverted 2026-05-29 (see [the rookie-feature record](../todo/fixed-archive/tested-rejected-draft-capital-combine-rookie-cold-start-features-benchmark-f-3be5bb00.md)). Combine testing carries no marginal signal beyond draft position; draft capital (`log(pick)`) *does* have real rookie signal but is **benchmark-flat** — the gain concentrates in LightGBM (best model only for RB) and rookies are ~14% of rows, so it's invisible in overall MAE. Don't re-propose without a tracked rookie-subgroup metric, or scope to RB / LightGBM-only.

- **Dormant attention-architecture extensions:** `attn_learn_temperature`,
  `attn_history_dropout`, `attn_use_swiglu_encoder`, `attn_entropy_coeff`,
  `attn_use_alibi_bias` (including `alibi_only`) and `attn_self_layers` stay
  default-OFF; do not re-propose without a tracked subgroup metric. The
  `selfattn` trials regressed and destabilized small positions. The exception is
  `attn_condition_queries_on_static` (`condq`), enabled for RB/WR/TE by owner
  decision; QB/K/DST remain OFF absent a tracked metric. Its RMSE screen wins
  did **not** transfer to the eager FP16 retrain `ac3686f` (RB/TE flat, WR worse).
  Retention is a forward bet on matchup features
  ([#1210](https://github.com/alexanderdfree/Fantasy_Football_ML_AWS/issues/1210)),
  not a measured production win. Re-screen in the current production regime
  when those features land; the historical FP16 evidence is not today's FP32
  regime. Judge tail effects on RMSE and the eager pipeline, not MAE alone.
  Forward `condq` on both the training config and
  `registry._{flat,nested}_attn_kwargs_static` serving paths: its `cond_proj`
  layer changes checkpoint shape. Full per-position results, accepted tradeoffs
  and rejection evidence are in [ADR-0004](../docs/adr/0004-attention-over-game-history.md#changelog)
  and [the experiment record](../todo/fixed-archive/tested-attention-architecture-default-off-extensions-prs-109-121-6-rejected-f1bf1cff.md).

## GPU execution

- **Per-architecture training-dtype defaults** require the
  [platform policy's comparability argument](platform.md#device-and-dtype-policy).
  GPU support alone is insufficient; FP16/BF16 remain opt-in.
- **Stacked and eager runs are never seed-by-seed comparable.** Sub-ULP kernel
  differences amplified by Adam fork trajectories even when the stacking
  machinery is bitwise-correct. Production training stays eager. Comparative
  tuning/A/Bs may stack under the owner-approved regime, but compare stacked
  against stacked and rebaseline before shipping to eager production. Keep
  `_ens{N}x{E}` studies and artifacts separate from coexisting eager history.
  The [Lever C evidence](../todo/gpu_launch_bound_levers.md#stacked-seed-evidence)
  retains the initial rejection, reversal, parity gates and width measurements.
- **Stacking is GPU-gated and width-coupled.** Local CUDA tuning/`ab_harness`
  defaults to `DEFAULT_STACKED_SEEDS=24`; CPU/MPS use lean 3-seed eager runs,
  and K/DST fall back to eager because they cannot vmap. Batch `launch_ab`
  defaults eager unless `--stacked-seeds` is explicit. Check
  [resolve_default_stacked_seeds](../src/tuning/ab_ensemble_seeds.py) and the
  actual entrypoint before launching. Overrides include `--stacked-seeds 0`,
  `--no-stacked-seeds` and `FF_TUNE_STACKED_SEEDS=0`. Do not narrow the default
  below the measured ~9-seed L4 crossover without new evidence; those historic
  eager-FP16/full-graph timings are not a universal hardware threshold.
- **Do not finish stacking deliberately eager ablations without reconfirming.**
  `attn_arch` and `scheduler_type` have stackable
  [ab_attn_arch](../src/tuning/ab_attn_arch.py) /
  [ab_scheduler_type](../src/tuning/ab_scheduler_type.py) specs; the former omits
  entropy (vmap side-channel), the latter plateau (`ReduceLROnPlateau` rejected
  by `train_stacked`). Their legacy `ablate_*` runners remain the eager/per-head
  table path. `rb_gate` needs per-head MAE/gate AUC unavailable through the
  stacked harness's `pred_attn_nn_total` (D/E were reverted `hurdle_poisson`).
  `batch_lr` measures throughput the fixed-epoch FP32/vmap regime cannot assess.
  `backbone_norm` forces LN, `ridge_pca` is not an NN ablation, and
  `min_games`/`injury_features` change data: all remain eager. The full rationale
  is retained with [Lever C](../todo/gpu_launch_bound_levers.md#stacked-seed-evidence).
- **In-process base-NN/attention-NN overlap (`FF_NN_OVERLAP`) is rejected.**
  Concurrent CUDA-graph capture conflicts across streams; graphs-off overlap
  is slower than sequential graphs-on and threads share global RNG. The
  [2026-06-22 RB/5080 measurements](../todo/gpu_launch_bound_levers.md#within-position-overlap-evidence)
  retain the 1.58× graphs-off mechanism result and ~6.7× domination by graphs.
  Do not re-propose the in-process flag. Process-based overlap and its Linux
  NVIDIA MPS variant are a separate untested track with their own gate;
  NVIDIA MPS is unavailable on WSL2/native Windows.
- **Epoch-boundary work is closed pending a material-overhead profile.**
  Val-tail padding and batching/deferring validation-MAE transfers change
  early-stopping/model selection; they are metric changes. Per-epoch `randperm`
  transfer is safe but negligible. The [2026-06-22 gate evidence](../todo/gpu_launch_bound_levers.md#epoch-boundary-evidence)
  retains the 1.5–5.7 s/position, <3–5% overhead and <0.3 s maximum savings on an
  orchestration-bound run. K/DST graphing remains blocked by the positional,
  all-tensor `make_graphed_callables` contract versus nested keyword histories
  and `None` leaves. Reopen only with a profile showing materially large
  epoch-boundary overhead.

## CI and serving

- **Shared-venv CI optimization** — reverted in #110 / #111 (2026-04-23). Artifact download (~25s/shard) is slower than the warm `uv` install (~10s). Wall-clock is the metric, not compute.
- **Module-level pre-warm under gunicorn `--preload`** — reverted in #148 / #149 (2026-04-27). The bind happens *after* preload import; a slow pre-warm causes ALB TCP-refused → unhealthy. Use a `post_fork` hook or a background thread instead.
- **Building the upcoming-week artifact inside the serving container** — shipped in #1069, reverted to a CI build in #1076 (2026-06-08). A 2-worker serving task OOMs (worker SIGKILL) running `load_raw_data` + `build_features` + inference and attempts a runtime PBP download (which SSL-failed in-container); raising the task to 4 vCPU/8 GB did **not** fix it — it's an architectural mismatch, not sizing. Build the artifact in a scheduled CI job ([.github/workflows/refresh-upcoming-week.yml](../.github/workflows/refresh-upcoming-week.yml)) and have serving only **download** it from S3 (`sync_artifact_from_s3`). General rule: heavy `load_raw_data`/`build_features`/inference work doesn't belong in the serving container — build artifacts in CI, serve them. See [docs/adr/0018-live-upcoming-week-predictions-espn.md](../docs/adr/0018-live-upcoming-week-predictions-espn.md).

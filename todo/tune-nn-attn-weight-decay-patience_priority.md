# Plan: add `attn_weight_decay` + `attn_patience` to the Optuna attention-NN search space

## Context

The Optuna attention tuner (`src/tuning/tune_nn.py`) currently samples 6 attention knobs
(`attn_d_model`, `attn_n_heads`, `attn_encoder_hidden_dim`, `attn_dropout`, `attn_lr`,
`attn_batch_size`) plus 6 base-NN knobs. Two optimizer/regularization knobs that the
attention trainer actually consumes are **not** searched:

- **`attn_weight_decay`** — an asymmetry, not a documented exclusion: the base NN's
  `nn_weight_decay` *is* tuned ([tune_nn.py:185](../src/tuning/tune_nn.py#L185)) but the
  attention branch's weight decay is pinned at the per-position default (`5e-5` for
  QB/RB/WR/K, `3e-4` for TE/DST). Reads like a v1-scoping omission. **Clean addition.**
- **`attn_patience`** — early-stopping patience (default `20` everywhere). The user asked
  to add it; included with a caveat (below) and a tight range.

Both cfg keys already flow end-to-end, so this is contained to the tuner + its tests:
`position_pipeline.py:235-238` sets `cfg["attn_weight_decay"]` / `cfg["attn_patience"]`,
and the attention trainer reads them at
[pipeline.py:768-769](../src/shared/pipeline.py#L768)
(`weight_decay=cfg.get("attn_weight_decay", cfg["nn_weight_decay"])`,
`patience=cfg.get("attn_patience", cfg["nn_patience"])`). Overriding the cfg keys in
`_sample_overrides` is therefore sufficient — no pipeline or position-config changes.

**Scope note (no retrain, no prod-metric change):** this edits the *offline, opt-in*
(`workflow_dispatch`-only) tuner's search space. It does not touch any `POSITION_CONFIG`,
model, feature, target, or loss code, so production training metrics are unchanged and the
"run the pipeline + diff `benchmark_history/`" rule does **not** apply. The tuner emits
paste-ready constants; an operator hand-pastes winners into `src/{pos}/config.py` later.

### ⚠️ `attn_patience` is weakly identified by this objective

The trial objective is `min(val_loss)` over the training trajectory
([tune_nn.py:258](../src/tuning/tune_nn.py#L258)). Patience controls only *when*
early-stopping halts; the LR schedule is a fixed function of epoch. So larger patience just
lets the deterministic trajectory run more epochs, and `min` over a longer prefix is always
≤ `min` over a shorter one → **the objective is monotone non-increasing in patience**. TPE
will drift `attn_patience` toward the top of its range regardless of generalization value.
Mitigations baked into this plan: keep the range **tight (15–30)** and well below
`nn_epochs` (the `HyperbandPruner.max_resource` cap, [tune_nn.py:569](../src/tuning/tune_nn.py#L569)),
and document the caveat in a code comment so the next reader doesn't treat the tuned
patience as a meaningful optimum. If on reflection this isn't worth the search budget, drop
`attn_patience` and ship only `attn_weight_decay`.

## Changes

### 1. `src/tuning/tune_nn.py` — search space + paste-ready mapping

In `_sample_overrides` ([tune_nn.py:172-187](../src/tuning/tune_nn.py#L172)), add to the
returned dict, in the attn cluster (after `attn_batch_size`):

```python
"attn_weight_decay": trial.suggest_float("attn_weight_decay", 1e-5, 1e-3, log=True),
# attn_patience is monotone-favored by the min(val_loss) objective (longer run =>
# lower running-min); kept to a tight band so the inevitable drift-to-max stays sane.
"attn_patience": trial.suggest_int("attn_patience", 15, 30),
```

Rationale for ranges: `attn_weight_decay` mirrors `nn_weight_decay`'s `1e-5–1e-3` log range
and brackets both current defaults (`5e-5`, `3e-4`). `attn_patience` 15–30 brackets the
default `20` and stays far under `nn_epochs`.

In `_PARAM_TO_CONST` ([tune_nn.py:278-291](../src/tuning/tune_nn.py#L278)), add two entries so
the new params appear in `_format_config_lines` output (keep them in the attn block):

```python
"attn_weight_decay": "ATTN_WEIGHT_DECAY",
"attn_patience": "ATTN_PATIENCE",
```

`_format_value` already handles float (`5e-05` via `:.6g`) and int (`str()`), so the
rendered lines `{POS}_ATTN_WEIGHT_DECAY = 5e-05` / `{POS}_ATTN_PATIENCE = 22` need no
formatter change.

Optionally update the module docstring's MVP framing (line ~17) to note weight-decay +
patience are now searched — light touch, not required for correctness.

### 2. `tests/tuning/test_tune_nn.py` — mirror the search-space contract

- Add `"attn_weight_decay"` and `"attn_patience"` to `_EXPECTED_KEYS`
  (~[test_tune_nn.py:45-58]); this is what `test_sample_overrides_returns_every_cfg_key`
  asserts against.
- In `test_sample_overrides_ranges` (~lines 104-116), add bound assertions mirroring the
  existing ones: `1e-5 <= o["attn_weight_decay"] <= 1e-3`, and
  `15 <= o["attn_patience"] <= 30` with `isinstance(o["attn_patience"], int)`.
- In `test_format_config_lines_roundtrips_through_eval` (~lines 149-177), add the two new
  constants (e.g. `RB_ATTN_WEIGHT_DECAY`, `RB_ATTN_PATIENCE`) to the post-`eval()` namespace
  assertions so the round-trip check covers them.

### 3. `docs/ARCHITECTURE.md` — one-line ADR update

Add a dated line to the `Update history` block noting the D15 attention-tuner search space
gained `attn_weight_decay` + `attn_patience` (reference the PR). No new D-entry — this is an
extension of D15, not a new decision. No `TODO.md` archive entry (enhancement, not a bug).

## Not in this plan (answers to "anything else?")

- **Scheduler search is done elsewhere:** `src/tuning/tune_nn.py` now samples
  `scheduler_type` plus matching conditional keys (`cosine_t0`/`cosine_t_mult`/
  `cosine_eta_min` or `onecycle_max_lr`/`onecycle_pct_start`) and validates stale or
  mismatched scheduler payloads before training/config output. Do not duplicate that work here.
- **Cheap toggles:** `attn_positional_encoding`, `attn_project_kv` (1 categorical dim each).
- **`attn_max_seq_len`** (history window) — real capacity knob, borderline-structural, needs
  a shape-handling check first.
- **Excluded by stop-rule/coupling/correctness (do not add):** loss-config
  (`huber_deltas`, `loss_weights`, `head_losses`, `gated_targets`, `attn_gate_weight`),
  structural features (`attn_static_features`, `attn_history_stats`), `nn_non_negative_targets`.
- **Dimensionality budget:** 12 → 14 dims is fine for ~30 trials + HyperbandPruner. The
  scheduler axis now lives in the tuner, so revisit `--n-trials` before combining all axes in a
  production retune.

## Verification

1. `ruff check . && ruff format --check .`
2. `pytest tests/tuning/test_tune_nn.py -q` — fast, no data/GPU needed (builds TPE studies
   and calls `_sample_overrides` directly). Confirms `_EXPECTED_KEYS`, the range assertions,
   and the config-line round-trip all pass with the two new params.
3. *(Optional end-to-end smoke, needs data + a real NN train)* `python -m src.tuning.tune_nn QB --n-trials 2`
   — confirms the new keys survive a live trial (sampled → cfg override → attention trainer
   reads `attn_weight_decay`/`attn_patience`) and that `--print-best` renders the two new
   `QB_ATTN_*` lines.
4. **Gate note:** if `pre-pr.sh`'s benchmark-freshness gate (B2) fires on this tuner-only
   change, it's a false positive — no production model/feature/target/loss was touched, so
   there's nothing to re-benchmark. Surface it rather than running a full sweep.

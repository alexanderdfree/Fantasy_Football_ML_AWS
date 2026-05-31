# Batch-size audit: NN tuning vs. dataset size + BatchNorm sensitivity

> **Status: AUDIT COMPLETE (read-only), no code changed.** Committed handoff doc so
> another session can pick up the one actionable follow-up (a QB `attn_batch_size`
> A/B). The corresponding `[PRIORITY]` note is in [TODO.md](../TODO.md)'s Open section.
>
> **Relationship to [`increase_batch_size_plan.md`](increase_batch_size_plan.md):** these
> are two ends of the *same* hyperparameter. That doc widens the Optuna grid *up*
> (add a 1024 rung) for the large-N positions; this audit looks at *current
> production* `attn_batch_size` and finds it's already too *large* (update-starved)
> for the small-N positions' attention pass. Read both together. The √-scaling /
> "LR moves with batch" reasoning in that doc applies verbatim to any change here.

## Context

Started from the conceptual question "why wouldn't I want full batch size if I have
enough VRAM?" The grounded answer: VRAM only removes the *memory* constraint. Batch
size independently sets (a) **update density** — steps/epoch = `⌊N/B⌋` with
`drop_last=True`; (b) **gradient-noise regularization** (~`1/√B`); and (c)
project-specific, **BatchNorm statistic quality**. On small datasets all three favor
*smaller* batches, so "I have spare VRAM" is necessary-but-not-sufficient for going
bigger. This audit turns that into a per-position check of the production configs.

Two architecture facts that shape the audit:
- **Both NN passes flow through the BatchNorm backbone.** `_build_backbone` (with
  `nn.BatchNorm1d`, [`src/shared/neural_net.py:180-204`](../src/shared/neural_net.py))
  is shared by `MultiHeadNet` (main pass, `nn_batch_size`/`nn_lr`) **and**
  `MultiHeadNetWithHistory` (attention pass, `attn_batch_size`/`attn_lr`) — the
  attention model's *static* branch still goes through BN. So `attn_batch_size` has
  the same low-noise/low-update exposure as `nn_batch_size`. (The attention encoder
  itself uses LayerNorm, which is batch-size-invariant.) Two separate training passes:
  `_train_nn` / `_train_attention_nn` in [`src/shared/pipeline.py`](../src/shared/pipeline.py).
- **`drop_last=True` on train for both passes** ([`src/shared/training.py`](../src/shared/training.py)),
  so every train batch is exactly `B` — BN never sees a size-1 batch (no crash), the
  only effect is statistic *quality* at very few batches/epoch. This is the same tail
  tracked by the `[LOW] drop_last=True` item in [TODO.md](../TODO.md).

The flag direction here is "batch too **large** relative to N" (too few updates, too
little SGD/BN noise → under-regularized + under-trained), *not* "batch too small for
BN" — no position is near the small-batch-BN-instability regime (the smallest batch in
production is 128).

## Findings (executed 2026-05-31)

### Diagnostics — exact N (post `MIN_GAMES` filter), steps/epoch for both passes

| Pos | N_train | nn_bs | main steps/ep | main B/N | attn_bs | attn steps/ep | attn B/N |
|-----|--------:|------:|--------------:|---------:|--------:|--------------:|---------:|
| QB  | 6,305  | 128 | 49 | 2.0% | 256 | **24** ⚠ | 4.1% |
| RB  | 15,028 | 256 | 58 | 1.7% | 256 | 58 | 1.7% |
| WR  | 22,690 | 512 | 44 | 2.3% | 256 | 88 | 1.1% |
| TE  | 10,857 | 128 | 84 | 1.2% | 128 | 84 | 1.2% |
| K   | 4,574  | 128 | 35 | 2.8% | 256 | **17** ⚠ | 5.6% |
| DST | 6,238  | 128 | 48 | 2.1% | 128 | 48 | 2.1% |

Thresholds (heuristic): **WATCH** if steps/ep < ~30 or B/N > ~5%; **FLAG** if
steps/ep < ~15 or B/N > ~10%. **Nothing hits FLAG.** Two WATCH items, both the
attention pass at `attn_batch_size=256` on small N: **QB** (24 steps/ep) and **K**
(17 steps/ep + 5.6% B/N — the most starved on both axes).

### Cross-reference — is the attention NN even competitive there? (latest *trained* run `cbbd156`, fantasy-point MAE)

| Pos | ridge | nn | attn_nn | lgbm | best | attn_nn gap to best |
|-----|------:|---:|--------:|-----:|------|--------------------:|
| QB  | 6.176 | 6.018 | 5.93  | 5.891 | lgbm    | +0.039 (strong 2nd) |
| RB  | 4.255 | 4.053 | 3.985 | 3.988 | **attn_nn** | best |
| WR  | 4.177 | 3.979 | 3.957 | 3.951 | lgbm    | +0.006 (~tied) |
| TE  | 3.451 | 3.379 | 3.367 | 3.331 | lgbm    | +0.036 |
| K   | 4.008 | 4.165 | 4.221 | 4.13  | **ridge** | **+0.213 (attn_nn is K's WORST)** |
| DST | 5.203 | 5.115 | 5.107 | 5.271 | **attn_nn** | best |

### Per-position verdict
- **RB / WR / TE / DST — OK.** 44–88 steps/ep; attn NN is best or within noise of best.
  No action. **WR's `nn_bs=512` is vindicated** — 44 steps/ep is healthy and WR ties
  for best MAE; the larger batch buys smoother gradients on 22.7K rows without starving
  updates (exactly what its config comment claims).
- **QB attention pass — WATCH, the higher-value flag.** 24 steps/ep is thin, and here
  it matters: attn_nn is a strong 2nd (+0.039 behind lgbm). The one worth chasing.
- **K attention pass — most starved diagnostically (17 steps/ep, 5.6% B/N) but LOW
  priority to act.** K's attn NN is its *worst* model — Ridge wins by 0.213. So the
  starved batch is *consistent with* why the attn NN is uncompetitive, but a batch fix
  alone won't make it beat Ridge; K ships Ridge regardless.

## Actionable follow-up (only QB worth chasing)

**QB `attn_batch_size` A/B.** Halving 256→128 doubles update density to ~49 steps/ep
(matching QB's own main pass) at 2.0% B/N — plausibly enough to close the 0.039 gap to
LGBM and flip QB's selected model. **Re-tune `attn_lr` alongside** (batch↔LR is coupled;
AdamW → √-scaling, ~`1e-3 × √(128/256)` ≈ `7e-4` as a starting point — but let the A/B
or tuner find it; there is no scaling rule in the code, the two are independent constants).

**Validated path (no blind config edit):**
- Hardcoded A/B in the [`src/tuning/ablate_rb_gate.py`](../src/tuning/ablate_rb_gate.py)
  pattern — `attn_bs ∈ {256, 128}` × re-tuned `attn_lr`, decision table on **attn_nn
  MAE**, **≥8 seeds** (small NN deltas are seed-noisy — see auto-memory
  `feedback_ablation_seed_count_for_small_deltas`; a 3-seed read can fake a borderline
  result). Or let [`src/tuning/tune_nn.py`](../src/tuning/tune_nn.py) (already searches
  `attn_batch_size`/`attn_lr` jointly) run on QB.
- Judge on QB **attn_nn** MAE direction across seeds, not a single-seed overall-MAE flip
  (auto-memory `feedback_nn_seed_sensitive_overall_mae`).

**K:** only if you want to understand *why* its attn NN trails Ridge — same A/B, expect
the gap to narrow, not close. Not a production tune.

## Out of scope / stop-rules respected
- **No `POSITION_CONFIG` edit without a validating pipeline run.** Editing
  `src/qb/config.py` fires a full GPU retrain of QB via
  [`scope_positions.py`](../src/scripts/scope_positions.py); it is **not** `[docs-only]`
  and a reduced-config proxy can give the wrong sign (auto-memory
  `feedback_validation_proxy_must_match_production_model`).
- Not touching `ATTN_STATIC_FEATURES` (no windowed features — CLAUDE.md stop-rule);
  not adding loss-config knobs to the tuner search.

## Verification / how the numbers were derived (all read-only, no network)
- **QB/RB/WR/TE N:** read `data/splits/train.parquet`, filter by position, apply the
  train-only `MIN_GAMES_PER_SEASON=6` filter ([`pipeline.py:508`](../src/shared/pipeline.py)).
  QB pre-filter = 7,376 (matches an independent earlier read), post-filter = 6,305.
- **K N:** read the reconstructed weekly cache `data/raw/kicker_pbp_2015_2024.parquet`
  (the loader's train path adds no row drops before the filter), season ≤ 2023, K
  `min_games=4` → 4,574.
- **DST N:** 2 team-rows per REG game (the `pts_allowed` base of
  [`build_data`](../src/dst/data.py); left-joins drop nothing, teams always clear ≥6) →
  6,238 for train seasons 2012–2023.
- **Caveat:** local `data/splits` may slightly lag `main`'s seasons; benchmark is
  `cbbd156` (the most recent *trained* run — newer commits are docs-only,
  `training_skipped`). Order-of-magnitude steps/epoch and the model rankings are robust
  to this.

## Critical files (for the QB A/B, if pursued)
- [`src/tuning/ablate_rb_gate.py`](../src/tuning/ablate_rb_gate.py) — A/B pattern to copy.
- [`src/tuning/tune_nn.py`](../src/tuning/tune_nn.py) — joint `attn_batch_size`/`attn_lr` search.
- [`src/qb/config.py`](../src/qb/config.py) — `attn_batch_size` / `attn_lr` (only after a validated win).
- [`src/shared/pipeline.py`](../src/shared/pipeline.py), [`src/shared/neural_net.py`](../src/shared/neural_net.py) — the two-pass training + shared BN backbone.

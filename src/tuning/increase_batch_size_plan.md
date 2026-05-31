# Increase NN batch size + adjust LR (via the Optuna search)

> **Status: PLANNED, not implemented.** This is a committed handoff doc so another
> session can pick the work up. The corresponding priority note is the
> `[PLANNED]` entry at the top of [TODO.md](../../TODO.md)'s Open section.
> No code in `tune_nn.py` has been changed yet.

## Context

Goal: train the attention NNs with larger batches to (a) cut wall-clock — the T4 / RTX 5080 are heavily under-utilized — and (b) hopefully hold or improve MAE, while answering "how do we adjust the learning rate accordingly."

Two facts about *this* codebase reframe the textbook question:

1. **Optimizer is AdamW** ([`src/shared/pipeline.py:213`](../shared/pipeline.py)), loss is **mean-reduced** (`MultiTargetLoss` in [`src/shared/training.py`](../shared/training.py)). The linear LR-scaling rule (Goyal et al.) is an SGD result and does **not** apply. For Adam-family the grounded heuristic is **√-scaling**: `lr_new = lr_old · √(B_new / B_old)`.
2. **Batch size and LR are already *jointly* searched by Optuna** ([`src/tuning/tune_nn.py:179-186`](tune_nn.py)): `nn_batch_size`/`attn_batch_size ∈ {128,256,512}`, `nn_lr`/`attn_lr ∈ [1e-4, 5e-3]` (log). The TPE sampler learns the batch↔LR coupling empirically — strictly better than hardcoding a √-rule. So **"adjust LR accordingly" = widen the LR ceiling so the larger-batch optimum isn't clipped**, not "apply a formula." (The √-rule is the fallback only if you ever hard-pin batch in `POSITION_CONFIG` instead of searching.)

Decisions (from user):
- **Extend the Optuna grid; do not re-run the tuner now.**
- **Add one shared rung → `1024`** (`[128,256,512,1024]` for all six positions; keep `_sample_overrides(trial)` position-agnostic).
- **Leave the objective as pure `min(val_loss)`** — larger batch is selected only when it's also most accurate; speedup is a free-if-competitive bonus, not forced. No speed tiebreak, no hard-pin.

## How far can we go? — the VRAM answer

**VRAM is not the constraint here, by ~2 orders of magnitude.** The attention NN is tiny:
- Params ≈ **69K** (WR & RB). AdamW state (param+grad+2 moments, fp32) ≈ **1.1 MB**.
- The whole train set lives on the GPU once (`_GPUResidentBatcher`): WR ≈ **38 MB**, RB ≈ **54 MB** (rows × [static 41 + seq17×(hist 13–37 + opp 6)] × 4B). All splits < ~110 MB.
- Per-step activations (fp16 under AMP), seq_len=17, d_model=32, n_heads=2: ≈ **2 MB at batch 1024**, ≈ 18 MB at batch 8192.

**Peak VRAM at batch 1024 ≈ 150–250 MB of 16 GB. Even full-batch (B≈25K) stays under ~1 GB.** Corollary worth stating plainly: *increasing batch size will barely move GPU/VRAM utilization* — this model is too small to saturate a T4/5080 regardless. Larger batches help wall-clock via fewer kernel launches and fewer Python epoch-loop iterations, not via memory pressure.

## How far can we go? — the binding constraint (statistics + drop_last)

The real ceiling is **steps-per-epoch** and the **`drop_last=True` tail** ([`src/shared/training.py`](../shared/training.py), already a tracked LOW item in [TODO.md](../../TODO.md)), both worse for smaller positions. Because `_sample_overrides(trial)` is **position-agnostic** (one grid for all six positions), the ceiling must be safe for the *smallest* NN position, QB (~7.7K train rows):

| pos | train rows | steps/epoch @512 | @1024 | @2048 | avg drop_last waste @1024 / @2048 |
|-----|-----------:|-----:|-----:|-----:|-----:|
| QB  | ~7.7K  | 15 | 7  | 3  | ~6.7% / ~13% |
| TE  | ~12.6K | 24 | 12 | 6  | ~4%  / ~8%  |
| RB  | ~17.3K | 33 | 16 | 8  | ~3%  / ~6%  |
| WR  | ~25.7K | 50 | 25 | 12 | ~2%  / ~4%  |

**Conclusion: add one rung → `1024`.** Safe for all six (QB still ≥7 steps/epoch, ≤7% tail). `2048` only makes sense for WR/RB and would starve QB/TE (3–6 steps/epoch, ~13% tail) — so it belongs in a *position-aware* grid, not the shared one.

## Recommended change

**1. Extend both batch grids** in [`src/tuning/tune_nn.py`](tune_nn.py) `_sample_overrides` (lines 180, 186):
```python
"attn_batch_size": trial.suggest_categorical("attn_batch_size", [128, 256, 512, 1024]),
"nn_batch_size":   trial.suggest_categorical("nn_batch_size",   [128, 256, 512, 1024]),
```

**2. Widen the LR ceiling** so the co-searched LR can reach the larger-batch optimum — this is the "adjust LR accordingly" step (lines 179, 184): `5e-3 → 1e-2` (log-scale; modest +0.3 dex). Rationale: √-scaling 512→1024 ≈ 1.41×; cosine positions (QB/RB/WR/DST) use `nn_lr`/`attn_lr` as the *peak* LR, so a ~7e-3 optimum at batch 1024 would otherwise be clipped at 5e-3.
```python
"attn_lr": trial.suggest_float("attn_lr", 1e-4, 1e-2, log=True),
"nn_lr":   trial.suggest_float("nn_lr",   1e-4, 1e-2, log=True),
```
Add a short comment block at the search-space site recording the 1024-rung rationale (VRAM is slack; ceiling = QB steps/epoch + drop_last) and the LR-ceiling↔batch coupling, mirroring the existing explanatory-comment style in this file.

**3. Update the search-space test** [`tests/tuning/test_tune_nn.py`](../../tests/tuning/test_tune_nn.py) — both assertions break and must change:
- lines 109-110: `in (128, 256, 512)` → `in (128, 256, 512, 1024)`
- lines 105, 107: `<= 5e-3` → `<= 1e-2` (widening the search **up** *does* break the old `<=` bound).

**4. Docs.** One-line `Update history` entry in [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) if an NN-tuning D-entry exists; update the `drop_last=True` LOW item in [TODO.md](../../TODO.md) to note the @1024 tail (~7% QB) now that 1024 is reachable.

### Known consequence (accepted)
The Optuna objective stays **`min(val_loss)` only** ([`tune_nn.py:258`](tune_nn.py)) — speed is not in it. So 1024 *wins only if it is also the most accurate*. If a larger batch is marginally less accurate (the common small-data outcome) it won't be selected and there's **no speedup** — accepted: the grid extension never *hurts* (it can only add a better option), and if 1024 does win it's both faster and more accurate. Note separately: even when 1024 wins, expect only a **modest wall-clock gain** and **little change in GPU/VRAM utilization** — this model is too small (~69K params) to saturate a T4/5080 at any batch size; the win is fewer epoch-loop iterations / kernel launches, not memory throughput. If a *guaranteed* speedup is wanted later, that's a separate follow-up (hard-pin a larger batch in `POSITION_CONFIG` with √-scaled LR, validated against the ±2% MAE tolerance) — explicitly out of scope here.

### Out of scope / stop-rules respected
- Not touching `ATTN_STATIC_FEATURES` (no windowed features — CLAUDE.md stop-rule).
- Not adding loss-config knobs to the search (CLAUDE.md stop-rule).
- Not changing `POSITION_CONFIG` production values (the search picks them; we only widen the menu).

## Verification
- `ruff check . && ruff format --check .`
- `pytest tests/tuning/test_tune_nn.py -m unit` — the updated range/grid assertions are the direct gate. (No GPU / full pipeline run; "don't re-run" honored. A real tuner run is a later, separate step the user triggers.)
- Sanity: drive `_sample_overrides` over a handful of trials (as `test_sample_overrides_ranges` does) and confirm 1024 appears and LRs stay ≤ 1e-2.

## Critical files
- [src/tuning/tune_nn.py](tune_nn.py) — `_sample_overrides` (grids + LR range + comment).
- [tests/tuning/test_tune_nn.py](../../tests/tuning/test_tune_nn.py) — range/grid assertions (105, 107, 109, 110).
- [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md), [TODO.md](../../TODO.md) — doc updates.

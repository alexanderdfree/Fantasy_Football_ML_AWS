# Attention NN — Architecture Audit

**Date:** 2026-06-05 · **Scope:** the production attention neural network across all six
positions (QB/RB/WR/TE/K/DST). **Type:** read-only correctness/design audit (not a feature
change). **Sources read:** `src/shared/neural_net.py`, `src/shared/training.py`, NN portions
of `src/shared/pipeline.py`, all six `src/{pos}/config.py`.

**Headline:** the architecture is mature, numerically careful, and consistent across
positions. **No confirmed correctness bugs.** The value of this audit is (1) a live-vs-dormant
capability map, (2) one low-risk numerical-hardening note, and (3) a record of what was
checked and found sound — including subagent-flagged "bugs" that are false positives, recorded
so they don't get re-raised.

Related existing docs (not duplicated here): accuracy/subgroup behaviour is in
[attn_accuracy_findings.md](attn_accuracy_findings.md); the dormant-extension *ablation plan*
is the `[PRIORITY] Run the attention-architecture ablation harness (PR #662)` item in
[../TODO.md](../TODO.md).

---

## 1. What the production attention NN actually is

Production config is `POSITION_CONFIG` (the kwarg form, lower in each `config.py`) — **not**
`CONFIG_TINY` (the test fixture).

```
static features ─► shared backbone [Linear→BN→ReLU→Dropout]×N ─┐
                                                                ├─ per-target concat ─► per-target head
game history [B,17,game_dim] ─► game_encoder ─► +pos_embedding ─┘                         │
                              ─► AttentionPool (per-target × per-head learned queries)     ├─ GatedHead (sparse counts)
                                                                                          └─ Linear head + clamp≥0 (else)
```

- **Core mechanism = learned-query attention pooling** (`AttentionPool`, `neural_net.py:333`),
  **not** a transformer. `n_targets × n_heads` learned query parameters attend over the
  17-game right-padded, newest-first sequence; output is one fixed-size vector per target.
  `attn_self_layers=0` in every position ⇒ the `SelfAttentionBlock` encoder is gated off.
- **Per-target queries** let each target pull its own slice of history (e.g. `rushing_tds`
  on goal-line usage vs `rushing_yards` on carry volume) instead of a shared bottleneck.
- **Per-head loss families** (position-specific, verified):
  - QB: Huber (yards) + Poisson-NLL (TDs/INTs/fumbles); no gated targets.
  - RB/WR/TE: Huber/Poisson + `gated_targets` hurdle heads for sparse counts
    (receptions, rushing/receiving TDs).
  - K: all-Huber. DST: Poisson-NLL + Huber mix.
- **`LOSS_WEIGHTS ≈ 2.0/HUBER_DELTAS`** holds for Huber heads; Poisson heads use weight 1.0
  (e.g. `qb/config.py:234-246`). This coupling is intentional — do not tune the two axes
  independently (see AGENTS.md stop-rule).

## 2. Live vs. dormant capability (main finding)

`AttentionPool`/`MultiHeadNetWithHistory` implement substantial, test-covered machinery that
**no production `POSITION_CONFIG` enables**:

| Capability | cfg key / code | Enabled in any of 6? |
|---|---|---|
| Self-attention encoder stack | `attn_self_layers` / `SelfAttentionBlock` | No (0 everywhere) |
| ALiBi time-decay bias | `attn_use_alibi_bias` | No |
| Learned per-target softmax temperature | `attn_learn_temperature` | No |
| Opponent/context-conditioned queries | `attn_condition_queries_on_static` / `cond_dim` | No |
| Gated fusion (sigmoid-gated history) | `attn_gated_fusion` | No |
| Attention-entropy regularizer | `attn_entropy_coeff` | No |
| K/V projections | `attn_project_kv` | No |
| Opponent-history parallel branch | `use_opp_history` | No |
| SwiGLU game encoder | `attn_use_swiglu_encoder` | No |

**Enabled in production:** static backbone, game encoder, additive learned positional
embedding, AttentionPool, per-target gated/linear heads, and `attn_no_history_embedding` (QB
— learned season-opener embedding).

This is **intentional** per project convention (zero-init-inert, A/B-ready "kept investigation
knobs"); each path is constructed so it is numerically identical to baseline at step 0. It is
*not* a defect — but it is a real maintenance/clarity surface: a reader of `neural_net.py`
should not assume the transformer/ALiBi/temperature/conditioning paths are exercised in
served models. The adopt-or-delete decision for these is already tracked as the PR #662
ablation-harness item in TODO.md; this audit just makes the live/dormant boundary explicit.

## 3. Findings by category

### A. Correctness — no confirmed bugs
- Attention masking + all-padding-row `nan_to_num(0.0)` guards (`neural_net.py:516` in
  `AttentionPool`, `:148` in `SelfAttentionBlock`) correctly handle empty/short histories.
- Non-negativity uses `torch.clamp(min=0.0)`, **not** softplus — this is the documented K
  bias trap (softplus(0)≈0.693 compounds per head); verified correct (`apply_non_negative`).
- Season-opener handling: the `no_history` mask is captured *before* history-dropout, and
  `_apply_history_dropout` has a restore guard so a row that loses all real games to dropout
  is not misclassified as an opener — verified correct.
- `LOSS_WEIGHTS ≈ 2.0/δ` coupling holds for every Huber head across positions.

### B. Numerical stability — one low-risk note
- `GatedHead.value_log_alpha` is an unbounded `Linear` (`neural_net.py:268`). The NB-2
  log-pmf clamps `alpha = exp(log_alpha)` to a **min** of `1e-6` (`training.py:79`) but there
  is **no max clamp**. A runaway positive `log_alpha` → large `alpha` → `r=1/alpha→0` →
  `lgamma(r)→+∞` → `-inf` log-prob / `+inf` NLL. Only reachable on gated targets (RB/WR/TE
  sparse counts); weight decay regularizes `log_alpha` and this has never manifested.
  **Severity: low.** Optional hardening: clamp `log_alpha` to e.g. `[-6, 6]` in
  `GatedHead.forward` (or in the loss). Numerically inert in the normal regime ⇒
  benchmark-safe, but editing `src/shared/` fires a 6-position retrain, so it must go through
  a pipeline run + benchmark diff, not `[docs-only]`. **Not bundled with this doc.**

### C. Cross-position consistency — clean
- All six set `nn_non_negative_targets=set(_TARGETS)` explicitly (the `PositionConfig`
  default is an empty set, so an omission would silently disable clamping — none omit it).
- All six `POSITION_CONFIG`s use `attn_d_model=32`, `attn_n_heads=2`,
  `attn_positional_encoding=True`. K's `CONFIG_TINY` shows `attn_d_model=8/n_heads=1` — the
  expected test-fixture-vs-production divergence (read `POSITION_CONFIG`), not a bug.

### D. Training path — mature
- AMP/GradScaler (scaler enabled for FP16 only; BF16 opt-in sm_80+), grad-clip-*after*-unscale
  (`max_norm=1.0`), GPU-resident batcher, opt-in CUDA-graph capture, determinism gating — all
  correct and heavily commented with PR provenance. No device/dtype mismatch found.

### E. False positives ruled out (do **not** re-raise)
- **"Empty val batch → 0.0 → false best checkpoint."** Wrong. Early stopping uses
  `val_mae_weighted` (`training.py:1049`), and empty-batch val MAE is set to `float("inf")`
  (`:1025`), so `inf < best` is False. The `0.0` is the display-only loss accumulator.
- **"GPU-resident batcher permutation seeding is a bug."** It is an explicitly documented
  design decision (`training.py:358-377`): the permutation is drawn from the CPU RNG so the
  same global seed yields the same batch order across CPU/MPS/CUDA hosts. The comment already
  states it is *not* bit-identical to `RandomSampler` and why that is acceptable.

## 4. Recommendations

1. **Adopt-or-delete the dormant attention extensions** — **DONE (2026-06-19).** Ran the PR #662
   harness (RB stacked N=24 `ab_attn_arch-…b5b46ea` for the six vmap-safe flags; eager RB+QB
   8-seed for the vmap-incompatible `selfattn` via the new `launch_ablate` Batch path, #1180): no
   flag beats baseline beyond the 0.02 FP-MAE noise floor, and `selfattn` regresses on both RB
   (≈15K) and the small-sample QB (RB Δ +0.116 / QB +0.076, worse on 14/16 seeds). **None adopted;
   all kept default-OFF; the scaffolding is retained (owner decision — not deleted).** Full table +
   lesson in [fixed-archive.md](fixed-archive.md). The live-vs-dormant gap this audit surfaced is
   now a documented, deliberate keep.
2. **Optional `log_alpha` clamp** (§3.B) — low-risk hardening; only worthwhile bundled with
   other gated-head work given the retrain cost.
3. **No other action.** Enabling/disabling architecture is an ablation decision requiring
   benchmark evidence, out of scope for an audit.

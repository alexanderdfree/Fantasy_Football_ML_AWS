# ADR Changelog

Terse, chronological log of architecture changes — one line each: `YYYY-MM-DD · summary · (PR #N) · → ADR-00NN`. Full rationale lives in the per-decision
files in this directory; pre-split detail is in [../architecture-history.md](../architecture-history.md).

- 2026-06-01 · D1: benchmark-history CLIs' `--cv` is now a deprecated alias for rolling-origin walk-forward reporting; internal expanding-window CV remains for ad-hoc/tuning callers; no retrain · → ADR-0001
- 2026-05-31 · D4: newest-first attention history ordering (index 0 = most-recent game) across all 3 sequence builders, so the learned positional embedding is recency-indexed; ALiBi formula kept correct (games_ago == position); 6-position retrain · → ADR-0004
- 2026-06-01 · Corrected stale ADR references after config consolidation, unused display constants, and EC2 rollback sequencing docs; documentation-only · → ADR-0002/0004/0006/0007/0009
- 2026-06-01 · Corrected D13 g6 Spot per-run cost estimate to match operator runbook math; documentation-only · → ADR-0013
- 2026-05-31 · Split ARCHITECTURE.md into per-decision ADR files under docs/adr/; froze the prior Update history to architecture-history.md · → index in [../ARCHITECTURE.md](../ARCHITECTURE.md)
- 2026-05-31 · D12: launch-bound NN hot-path cuts (−15.6% QB) + `torch.compile` measured +169%/+62% (stays off on all archs); fused-AdamW MAE cost isolated deterministically to +0.89% base-NN / −0.18% attention · (PR #655) · → ADR-0012
- 2026-05-31 · D1: per-position `min_games_per_season` knob — RB/WR/TE→1, QB deferred; triggers a 6-position retrain · (PR #656) · → ADR-0001
- 2026-05-31 · Distributed 48 decision-scoped entries from the frozen Update history into their per-decision `## Changelog` sections; 13 non-decision entries (serving/analysis/CI/tooling) remain in architecture-history.md · → all ADRs
- 2026-05-31 · Local parallel trainer: work-conserving CPU core pool (`src/shared/core_pool.py`) replaces the static per-slice `LGBM_N_JOBS` (immutable post-spawn) — each position leases `ceil(cores/active)` per CPU stage so cores freed by finished positions widen survivors' thread counts; numerically inert, opt-in via `FF_CORE_POOL_ADDR`; dispatch order from measured `benchmark_history` · (PR #670) · → ADR-0017
- 2026-05-31 · `FF_CUDA_GRAPH` opt-in CUDA-graph NN training (sm_80+, off by default): 1.84× attn-NN speedup but NOT bit-inert (~0.5% eval drift — FP16+GradScaler amplifies a per-step-exact graph; LN/FP32/det-stop fixes all fail/cost the speedup) → local-iteration knob, not for benchmark A/Bs; off-by-default ⇒ production byte-identical · → ADR-0017
- 2026-06-01 · CUDA-graph benchmarkability follow-up: graph-vs-graph clean; BN warmup restore inert; fixed-scale 512 removes GradScaler schedule drift but graph-vs-eager still differs at fixed 2130 steps, so `FF_CUDA_GRAPH=1` A/Bs need graphed rebaselines and remain incomparable to eager baselines · → ADR-0017

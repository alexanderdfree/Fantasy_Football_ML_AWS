# ADR-001: Fantasy Football Predictor — Consolidated Architecture

**Status:** Accepted · **Date:** 2026-04-16 · **Author:** Alex Free

### Update history

The architecture changelog lives under [docs/adr/](adr/): recent changes in [adr/CHANGELOG.md](adr/CHANGELOG.md) (terse) and per-decision detail in each ADR's `## Changelog`. The full pre-2026-05-31 history is frozen verbatim in [architecture-history.md](architecture-history.md).

## Table of Contents

1. [Context](#1-context)
2. [System Overview](#2-system-overview)
3. [Decision Log](#3-decision-log) — full per-decision index table below; files under [docs/adr/](adr/)
4. [Cross-Cutting Consequences](#4-cross-cutting-consequences)
5. [Open Issues / Follow-Ups](#5-open-issues--follow-ups)
6. [References](#6-references)

---

## 1. Context

**Problem.** Predict weekly fantasy football points for individual NFL players across six positions (QB, RB, WR, TE, K, DST) for the 2025 season, using 2012–2024 as training history. Primary output is a per-player point projection (regression); ranking metrics (top-12 hit rate, Spearman correlation) are derived from projections post-hoc.

**Constraints.**
- Solo personal project, ~2 weeks of initial execution.
- Small-sample ML regime: after position filtering and the per-position games-per-season minimum (global default ≥6, with position overrides), roughly 200–600 player-seasons per position — orders of magnitude smaller than datasets most modern NN architectures assume.
- Public data only — `nflreadpy` ([nflverse](https://github.com/nflverse)) weekly stats, rosters, schedules, snap counts. Snap count coverage starts 2012, which bounds the training window.
- Documenting design decisions with technical trade-offs is what this ADR satisfies.

**Scope.**
- In: per-player weekly projections, three scoring formats (Standard / Half-PPR / Full-PPR), a Flask dashboard for lookup, automated training on AWS Batch.
- Out: lineup optimization, DFS salary-aware construction, betting odds beyond what's embedded as features.

**Forces driving the architecture.**
- The positions are not the same sport — QB and DST share maybe 10% of meaningful features. A one-model-fits-all approach under-fits all of them.
- Fantasy points are a *sum* of component stats (rushing yards, TDs, receptions). Modeling the sum directly wastes structure that modeling the components preserves.
- Training is cheap (~2 min on a GPU per position), but inference must run in a standard ECS container with no CUDA. This asymmetry shapes the whole deployment story.

---

## 2. System Overview

```
                    ┌─────────────────────────┐
  nflverse API ───▶ │ Data ingest + features  │ ─┐
  (2012–2025)       │  src/data, src/features │  │
                    └─────────────────────────┘  │
                                                 ▼
                    ┌──────────────────────────────────────────┐
                    │   Three model families (per position)    │
                    │  ┌────────┐ ┌─────────┐ ┌──────────────┐ │
                    │  │ Ridge  │ │ Multi-  │ │ Attention NN │ │
                    │  │ (base) │ │ HeadNet │ │ (all pos.)   │ │
                    │  └────────┘ └─────────┘ └──────────────┘ │
                    │  + LightGBM (all positions)              │
                    └──────────────────────────────────────────┘
                                         │
                       Compared, not     │
                       ensembled ──────▶ │
                                         ▼
         ┌──────────────────────┐  ┌────────────────────┐
         │ 6× g6 Spot (Batch)   │  │ Flask app (serve)  │
         │ Dockerfile.train     │  │ Dockerfile (slim)  │
         │ one position / host  │  │ ECS, CPU-only      │
         │ ~15 min parallel     │  │                    │
         └──────────────────────┘  └────────────────────┘
                    │                          ▲
                    │   model artifacts → S3 → │
                    └──────────────────────────┘
```

> **Rollback path:** the warm-EC2 implementation ([docs/ec2_design.md](ec2_design.md)) stays provisioned and is reactivated by `gh variable set BATCH_ACTIVE --body "false"` on the next push. D13 explains why the active default flipped to Batch; D7/D9 cover the warm-EC2 fallback.

A training run is triggered by a push to `main`, which invokes [`.github/workflows/train-batch.yml`](../.github/workflows/train-batch.yml) (when `BATCH_ACTIVE=true`): the workflow submits six Batch jobs in parallel against the `ff-gpu-spot` Compute Environment — one per position on its own Spot g6.xlarge — blocks until they terminate, verifies fresh `manifest.json` entries landed in S3 per position (the legacy `model.tar.gz` mirror was removed in D13 layer C — freshness is now checked against the manifest and the artifact tarball each manifest points at), and commits a fresh `benchmark_history/{run_id}.json`. When `BATCH_ACTIVE != 'true'` the [warm-EC2 trainer](../.github/workflows/train-ec2.yml) fires instead and loops the six positions sequentially via SSM. The Flask service is built separately and deployed to ECS on every push to `main`; it reads pre-baked models from S3 and serves projections through a dashboard.

---

## 3. Decision Log

Each decision below follows the same structure: what was decided, the forces at play, options considered, the chosen option's trade-offs, why the rejected alternatives were rejected, and references to code.

---

| # | Decision | Status |
|---|---|---|
| D1 | [Temporal split (2012–23 train / 2024 val / 2025 test)](adr/0001-temporal-split.md) | Accepted |
| D2 | [Multi-target decomposition with shared NN backbone](adr/0002-multi-target-decomposition-with-shared-nn-backbone.md) | Accepted |
| D3 | [Three-way model comparison, no ensemble](adr/0003-three-way-model-comparison-no-ensemble.md) | Accepted |
| D4 | [Attention over game history (all positions)](adr/0004-attention-over-game-history.md) | Accepted |
| D5 | [Output-constraint stack](adr/0005-output-constraint-stack.md) | Accepted |
| D6 | [Explicit per-position feature allowlist](adr/0006-explicit-per-position-feature-allowlist.md) | Accepted |
| D7 | [EC2 warm instance over Batch/SageMaker](adr/0007-ec2-warm-instance-over-batch-sagemaker.md) | Accepted |
| D8 | [Two Docker images](adr/0008-two-docker-images.md) | Accepted |
| D9 | [Warm training host](adr/0009-warm-training-host.md) | Accepted |
| D10 | [Trunk-based CI/CD with test-gated deploys](adr/0010-trunk-based-ci-cd-with-test-gated-deploys.md) | Accepted |
| D11 | [Smoke-test gate + always-stable artifact (manifest v2)](adr/0011-smoke-test-gate-always-stable-artifact.md) | Accepted |
| D12 | [Training-step perf composition (torch.compile rejected on T4)](adr/0012-training-step-perf-composition.md) | Accepted |
| D13 | [Spot fan-out via AWS Batch (overrides D7 when BATCH_ACTIVE=true)](adr/0013-spot-fan-out-via-aws-batch.md) | Accepted |
| D14 | [Serving prediction-cache + post_fork pre-warm](adr/0014-serving-prediction-cache-post-fork-pre-warm.md) | Accepted |
| D15 | [Attention-NN hyperparameter tuning via Optuna + Batch Spot fan-out](adr/0015-attention-nn-hyperparameter-tuning-via-optuna-batch-spot-fan-out.md) | Accepted |
| D16 | [External opportunity / quality / value signals (ff_opportunity, ESPN QBR, contracts)](adr/0016-external-opportunity-quality-value-signals.md) | Accepted |
| D17 | [Platform autodetection & per-arch optimization policy](adr/0017-platform-autodetection-per-arch-optimization-policy.md) | Accepted |
| D18 | [Live upcoming-week predictions via ESPN + S3 artifact build](adr/0018-live-upcoming-week-predictions-espn.md) | Accepted |
| D19 | [Split Batch training: GPU NN + c8a CPU Ridge/LightGBM](adr/0019-split-batch-training-gpu-nn-cpu-ridge-lgbm.md) | Accepted |
| D20 | [Batch GPU execution path for the shared A/B harness](adr/0020-batch-gpu-execution-path-for-ab-harness.md) | Accepted |
| D21 | [Systematic knob + feature-family screens as stacked A/B-harness specs](adr/0021-systematic-doe-screens-on-ab-harness.md) | Proposed |

## 4. Cross-Cutting Consequences

**What becomes easier.**
- *Parallel iteration per position.* A change to RB features affects only RB's training job, only RB's models, only RB's tests. D2 (multi-head), D6 (allowlist), and D13 (parallel Batch jobs on the active path) compose into position-independent evolution.
- *Reproducible serving.* The Flask image is immutable and SHA-tagged (D10); models are baked in, not pulled at runtime. No "it worked yesterday" class of bugs.
- *Audit trail for leakage.* The allowlist (D6) plus the temporal split (D1) plus the ±4σ clip (D5) means any new feature has to survive three independent checks before it affects training.
- *Visibility into perf regressions.* Phase-level timings emitted under D12 make a slow-down at any pipeline stage visible in the next benchmark JSON without an explicit perf-test job.
- *Bounded blast radius for a bad artifact.* D11's smoke-test gate + always-stable manifest pointer means a NaN-emitting or shape-mismatched model lands in `current`/`history` but never replaces the live `stable` pointer; rollback is a single `promote.py` invocation against the manifest, no S3 surgery required.

**What becomes harder.**
- *Six configuration surfaces instead of one.* Each position has its own config, targets, loss weights. A framework-level change (e.g., a new regularizer) needs propagation to six places. This is a deliberate trade (D2/D6) but real.
- *Training-inference skew.* The Flask app must run the same preprocessing as the training pipeline, or models get zeros for features they were trained on. This happened once already (weather/Vegas features missing at inference, fixed after the fact — see [TODO.md](../TODO.md)).
- *Two images to maintain.* D8 doubles the Dockerfile surface; a requirements bump in one does not automatically propagate to the other.
- *Two-channel artifact pointer to reason about.* Under D11, "did training succeed?" splits into `current` (always advances) and `stable` (advances on smoke-pass only). Operator scripts and consumers must read the right pointer for their use case — model_sync clients want `stable`, forensics tooling wants `history[]`/`previous`.

**What we'll need to revisit.**
- K-position features use cross-season rolling windows (see [TODO.md](../TODO.md) "Open"). Technically a leakage source; currently justified by the specialist-role-stability argument, but worth re-measuring once the 2025 season completes.
- Single-format (PPR) training models. The scoring-format flexibility is only at the fantasy-points *computation* layer; the models themselves are trained on PPR. Retraining per format is straightforward but not automated.
- No lineup-construction layer. Per-player projections alone aren't a DFS product.

---

## 5. Open Issues / Follow-Ups

Standing follow-ups, mapped to decisions (item 1 is also tracked in the [TODO.md](../TODO.md) "Open" list):

1. **K cross-season rolling leakage** — related to D1 (temporal split) and D5 (per-position non-negative targets). Requires either collecting more K games or accepting the bias.
2. **PPR-only training** — related to D2 (multi-target). Needs a training-matrix flag for scoring format and re-running six position pipelines.
3. **No lineup optimizer** — out of scope for this ADR; tracked as a follow-up project, not a revision of D1–D17.

---

## 6. References

### Source files by subsystem

- **Data & features:** [src/data/loader.py](../src/data/loader.py), [src/data/split.py](../src/data/split.py), [src/data/preprocessing.py](../src/data/preprocessing.py), [src/features/engineer.py](../src/features/engineer.py), [src/shared/weather_features.py](../src/shared/weather_features.py).
- **Models:** [src/shared/models.py](../src/shared/models.py) (Ridge, LightGBM, ordinal, two-stage, baselines), [src/shared/neural_net.py](../src/shared/neural_net.py) (MultiHeadNet, attention, gated TD head).
- **Training:** [src/shared/training.py](../src/shared/training.py) (MultiTargetLoss, trainer, schedulers), [src/shared/pipeline.py](../src/shared/pipeline.py) (pipeline orchestrator).
- **Per-position configs:** `src/qb/config.py`, `src/rb/config.py`, `src/wr/config.py`, `src/te/config.py`, `src/k/config.py`, `src/dst/config.py`.
- **Serving:** [src/serving/app.py](../src/serving/app.py), [Dockerfile](../Dockerfile).
- **Training infra (active, Batch):** [infra/batch/](../infra/batch/), [src/batch/launch.py](../src/batch/launch.py), [src/batch/train.py](../src/batch/train.py), [src/batch/Dockerfile.train](../src/batch/Dockerfile.train), [src/batch/build_and_push.sh](../src/batch/build_and_push.sh), [.github/workflows/train-batch.yml](../.github/workflows/train-batch.yml), [.github/workflows/batch-image.yml](../.github/workflows/batch-image.yml).
- **Training infra (rollback, EC2):** [infra/ec2/](../infra/ec2/), [.github/workflows/train-ec2.yml](../.github/workflows/train-ec2.yml). Reuses the same `src/batch/train.py` entrypoint and Dockerfile.
- **CI:** [.github/workflows/tests.yml](../.github/workflows/tests.yml), [train-batch.yml](../.github/workflows/train-batch.yml), [train-ec2.yml](../.github/workflows/train-ec2.yml), [batch-image.yml](../.github/workflows/batch-image.yml), [deploy.yml](../.github/workflows/deploy.yml).

### Related design docs

- [docs/method_contracts.md](method_contracts.md) — function signatures + data-layer contracts.
- [docs/batch_design.md](batch_design.md) — Batch + Spot fan-out design, cold-start optimizations, cost breakdown (authoritative for D13, the active push-driven path).
- [docs/ec2_design.md](ec2_design.md) — warm-host training design (authoritative for the D7 / D9 rollback path).
- [infra/aws/README.md](../infra/aws/README.md) — ECS + ALB + domain runbook (authoritative for D8 serving ops).
- [infra/ec2/README.md](../infra/ec2/README.md) — EC2 warm-host runbook (authoritative for D9 ops).
- [docs/archive/design_weather_and_odds.md](archive/design_weather_and_odds.md) — weather/Vegas feature rationale (folded into D6).
- [docs/archive/design_lstm_multihead.md](archive/design_lstm_multihead.md) — LSTM exploration, kept as artifact of the rejection under D4.
- [docs/archive/design_xgboost_ensemble.md](archive/design_xgboost_ensemble.md) — ensembling consideration, rejected under D3.
- [docs/expert_comparison.md](expert_comparison.md) — benchmark against published projections (evaluation evidence).
- [TODO.md](../TODO.md) — open issue log; the Fixed archive is at [todo/fixed-archive.md](../todo/fixed-archive.md).

### Commit timeline of inflection points

| Commit | Phase | What changed |
|---|---|---|
| `974f00d` | Prototype | Monolithic local pipeline |
| `f400a5c` | Data | Knowledge cutoff set to 2012 (D1) |
| `99d7086` | Modeling | Attention V4 + model consolidation (D2, D4) |
| `fe507e0` | Modeling | Softplus → clamp output constraints (D5) |
| `18170a6` | Modeling | Gated TD head + allowlist refactor (D5, D6) |
| `f343c20` | Modeling | LightGBM added (D3) |
| `eedacfc` | Infra | SageMaker attempt |
| `57d52f9` | Infra | Pivot to AWS Batch (D7) |
| `0e814a1` | Infra | Docker optimization — two images (D8) |
| `4145257`, `8a50eec` | Infra | Batch cold-start stack (D9) |
| `ffb3119` | Infra | CI/CD + test gating + final batch infra (D10) |
| `cc0c627` | Modeling | DST attention NN + migrate DST targets from 5 mixed-bucket heads to 10 raw stats (D2, D4, D5) |
| `2500ecc` | Modeling | Per-position `POSITION_CONFIG.attn_static_features` allowlist — rolling/EWMA/trend features blocked from the attention-NN static branch (D4, D6) |
| `801b61a` | Modeling | K nested attention — outer over games + inner per-kick pool; attention now covers all six positions (D4) |
| `c7fa2d7` | Infra | Smoke-test gate + manifest v2 (stable/current/previous + history[5]) + S3 bucket versioning (D11) |
| `8c42e88` | Infra | ECS force-rollover after train so promoted artifacts get loaded (D11 closure) |
| `a533990` | Serving | PPR / Half-PPR / Standard end-to-end scoring switch (D2 extension) |
| `20cda09`, `668fa81` | Repo | Layout reorganization; `src/{POS}/` → `src/{pos}/` rename + symbol prefix drop |
| `48ef419` | Training | Feather cache + async DataLoader + cuDNN benchmark + phase timings (D12 wins) |
| `3167b56` | Training | `torch.compile` short-circuit — +32% on T4 (D12 rejection) |
| `0c66171` | Modeling | K/DST eval totals use signed/tiered aggregator (was reporting bogus `total_r2`) |
| `dff43fb` | Modeling | Drop K `ATTN_L1_FEATURES` — K now matches DST/skill convention (D4, D6) |
| `349aa4a` | Training | Static-pad attention (−30% attn train) + disk-backed feature-engineering cache (−86× on prepare_data hits) (D12) |
| `056423b` | Serving | Benchmark History tab — per-PR rows fetched from S3 at boot, auto-updates without redeploy |
| PR #215 | Infra | `infra/batch/` provisioning scripts (CE/JQ/JD/IAM) for Spot fan-out (D13) |
| PR #216 | Infra | `batch-image.yml` cold-start opts — ECR pull-through cache + SOCI v2 lazy-load (D13) |
| PR #217 | Infra | `train-batch.yml` workflow + `train-ec2.yml` gate on `BATCH_ACTIVE` + `launch.py --skip-upload` (D13) |
| PR #1069 | Serving | Live upcoming-week predictions homepage (new default tab) — ESPN-sourced slate, QB/RB/WR/TE projections for the next unplayed week (D18) |

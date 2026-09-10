# Platform policy

## Platform & hardware targets (autodetect, then optimize per-arch)

Changes must account for every supported environment, using autodetection with
explicit overrides rather than hardcoding the current machine. The decision and
its measured history live in [ADR-0017](../docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md).
Use current code/configuration and live infrastructure for effective settings;
this compatibility matrix does not establish today's deployment or quota.

| Environment | Capability / constraint |
|---|---|
| Apple Silicon macOS | CPU default; Apple MPS is opt-in |
| Native Windows, RTX 5080 / sm_120 | `OPENBLAS_NUM_THREADS=1` is required for correctness; 9950X3D has 16 physical cores |
| WSL2, RTX 5080 / sm_120 | Linux BLAS throughput limits; [wsl-env.sh](../scripts/wsl-env.sh) |
| AWS g4dn / T4 / sm_75 | Retired rollback target; keep defensive FP16-only AMP support, no BF16/TF32 or graphs |
| AWS g6 / L4 / sm_89 | Supports BF16/TF32 and CUDA graphs; check live Batch/rollback configuration |
| AWS g5 / A10G / sm_86 | Supports BF16/TF32 and CUDA graphs; check live Batch fleet configuration |

## Device and dtype policy

- Default training uses the FP32 family: FP32 storage plus TF32 matmuls on
  supported CUDA hardware, pure FP32 elsewhere. FP16+GradScaler and BF16 are
  explicit `FF_AMP_DTYPE` opt-ins; BF16 requires sm_80+ and falls back to FP16 on
  T4. Hardware support alone cannot justify changing a training-dtype default:
  bring a benchmark-comparability argument. The measured FP16/BF16 history and
  rejected defaults are retained in [ADR-0017](../docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md).
- Apple MPS is opt-in (`FF_DEVICE=mps`), never the `auto` default. It needs a Mac
  default-vs-MPS A/B before promotion: no demonstrated speedup for this small
  model, CPU/CI byte-identity loss and silent op-fallback remain the constraints.
- Native Windows requires `OPENBLAS_NUM_THREADS=1`; removing it causes Ridge-PCA
  alpha-CV `0xC0000005` crashes. WSL2/Linux/macOS thread caps are for throughput;
  use `detect_platform().is_wsl` to distinguish Windows from WSL2.
- Apply per-architecture speed changes without silently changing model numerics.
  Execute changed GPU paths using [GPU validation](validation.md#gpu-and-batch-validation).
  Shared paths can trigger six-position retraining; inspect
  [scope_positions.py](../src/scripts/scope_positions.py). A no-op claim follows
  [production validation](validation.md#production-path); only the
  [docs-only contract](delivery.md#docs-only-exception) can exempt wholly
  non-behavioral changes. The `training-skipped:` marker was retired in PR #1542.

## CUDA graph comparability

`cuda_graph_enabled()` autodetects ON for supported CUDA sm_80+;
`FF_CUDA_GRAPH=0`/`false`/`off` disables capture for an eager comparison.
The default FP32+TF32 regime was measured per-step bit-exact; graph-on/off can
still differ through dropout-RNG warmup (zero dropout removed that difference).
The **opt-in FP16+GradScaler regime is the owner-approved exception**: its
multi-step trajectory drift requires graphed-vs-graphed rebaselining, not a
comparison against eager history. CPU/MPS and T4 remain eager; K's nested trainer
no-ops capture. Do not generalize this exception to arbitrary dtype changes.

Capture variants, historical speed measurements, default promotions and the
retained investigation knobs (`FF_NN_NORM`, `FF_FORCE_DROPOUT_ZERO`,
`FF_NN_FIXED_EPOCHS`) are recorded in [the GPU investigation](../todo/gpu_launch_bound_levers.md)
and [ADR-0017's changelog](../docs/adr/0017-platform-autodetection-per-arch-optimization-policy.md#changelog).
Stacked/tuning regimes have separate [execution constraints](stop-rules.md#gpu-execution).

## Primitives

Reuse these entrypoints; inspect their implementation before stating defaults:

| Concern | Canonical implementation |
|---|---|
| Capability report | [platform_detect.py](../src/shared/platform_detect.py): `detect_platform()` is reporting-only (`backend`, GPU name/capability/sm, BF16 support, OS/WSL, cores, recommended wheel) |
| Device and dtype overrides | [utils.py](../src/shared/utils.py): `requested_device`, `cuda_enabled`, `mps_enabled`, `amp_dtype`, `requested_amp_dtype`; `FF_DEVICE` overrides detection, `auto` is CUDA-or-CPU |
| Pipeline device / compile / TF32 | [pipeline.py](../src/shared/pipeline.py): `_nn_device`, `_maybe_compile`; `FF_COMPILE` is opt-in and sm_80+-gated (T4 regression in [ADR-0012](../docs/adr/0012-training-step-perf-composition.md)) |
| Residency / AMP / graphs | [training.py](../src/shared/training.py): `_gpu_resident_device`, `_autocast`, `_maybe_graph_model`; residency and AMP are CUDA-only, MPS/CPU use DataLoader + FP32 |
| CPU threads | [models.py](../src/shared/models.py): `_lgbm_n_jobs` / `LGBM_N_JOBS`; [tune_lgbm.py](../src/tuning/tune_lgbm.py): `_default_n_jobs` |
| Wheel installation | [requirements-dev.txt](../requirements-dev.txt), [requirements-gpu.txt](../requirements-gpu.txt), [Dockerfile.train](../src/batch/Dockerfile.train); extend these, do not add ad-hoc pins |

Per-platform installation and CPU/BLAS commands belong in [SETUP.md](../SETUP.md).

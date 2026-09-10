### [FIXED] Failed CUDA optimizer capture retained warmup updates

**File(s)**: `src/shared/training.py::_GraphedFullStep.build`,
`tests/shared/test_cuda_graph.py`,
`src/analysis/verify_cuda_capture_rollback.py`,
`src/tuning/ab_verify_cuda_capture.py`.
Defect reproduced against `92be2873` during the 2026-09-10 audit.

**What**: Priming and warmup execute real parameter, BatchNorm and Adam updates.
Restoration ran only after successful capture, so the existing fallback could
continue from contaminated state. The original failure test raised before any
mutation and therefore could not detect this path.

**Fix**: Restore model state and reset warmup optimizer state in `finally`.
Synchronize failed CUDA work before restoration, clear incomplete graph/LR
metadata and restore the original learning-rate bindings. Successful capture
retains its established reset and scheduler-refresh behavior.

**Validation**: Real Torch CPU computations with injected capture APIs prove
the failure-state bug and its successful-reset control; they do not establish
GPU execution. The dedicated one-cell A/B spec runs real CUDA failures after
prime, warmup and capture-body updates, plus successful capture/replay and LR
refresh comparisons. Its actual CPU schema smoke uses normal QB data and
mocks only the hardware verifier; hardware acceptance requires the real
CUDA artifact, not that smoke or `--dry-run`.

**Lesson**: A fallback test must fail after the side effects it promises to
undo. Guard tests that fail before warmup cannot prove rollback correctness.

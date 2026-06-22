"""Shared utilities — seeding, small helpers."""

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

# Operator-facing compute-device override, set by run_pipeline's ``--device``
# flag (see src/shared/run_pipeline_factory.py). Read from the environment here
# — rather than threaded through ``run_pipeline`` as a parameter — to match the
# project's other env-var knobs (FF_DETERMINISTIC, REQUIRE_GPU, LGBM_N_JOBS) and
# to reach the device helpers deep in the call graph without disturbing the
# monkeypatch-sensitive ``run_pipeline`` signature.
_DEVICE_ENV = "FF_DEVICE"
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")

# Operator-facing AMP-precision override, analogous to FF_DEVICE. ``auto`` picks
# the dtype by GPU compute capability (see ``amp_dtype``); the explicit values
# pin it for A/B testing without code edits.
_AMP_DTYPE_ENV = "FF_AMP_DTYPE"
_VALID_AMP = ("auto", "bf16", "fp16", "fp32")

# Operator-facing CUDA-graph capture override (NOT the trigger): capture
# autodetects ON for CUDA sm_80+; set this falsy to force the eager path —
# see ``cuda_graph_enabled``.
_CUDA_GRAPH_ENV = "FF_CUDA_GRAPH"
# Full-step capture autodetects ON for CUDA sm_80+ (like the model-only
# capture); this env var is a force-off override (set it falsy to force the
# eager path) gating on top of ``cuda_graph_enabled()`` — see
# ``cuda_graph_full_enabled``.
_CUDA_GRAPH_FULL_ENV = "FF_CUDA_GRAPH_FULL"
# Optimizer-tail capture (Lever A3) autodetects ON for CUDA sm_80+ on top of
# full-step capture; this env var is a force-off override gating above
# ``cuda_graph_full_enabled()`` — see ``cuda_graph_opt_enabled``.
_CUDA_GRAPH_OPT_ENV = "FF_CUDA_GRAPH_OPT"


def requested_device() -> str:
    """Operator-requested device: ``"auto"`` (default), ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Sourced from ``$FF_DEVICE`` (set by ``run_pipeline --device``). An unset or
    unrecognised value falls back to ``"auto"`` so a typo can never silently pin
    the wrong device — only the four explicit choices change behaviour.
    """
    val = os.environ.get(_DEVICE_ENV, "auto").strip().lower()
    return val if val in _VALID_DEVICES else "auto"


def cuda_enabled() -> bool:
    """Whether CUDA should be used for this run, honouring ``--device``.

    - ``auto`` (default): CUDA iff ``torch.cuda.is_available()`` — the historical
      behaviour, so Linux/macOS/CI runs are unchanged when the flag is omitted.
      ``auto`` never selects MPS (that stays opt-in via ``mps``), so the default
      path is CUDA-or-CPU and byte-identical to CI.
    - ``cpu``: never CUDA — force the CPU path wherever this is consulted.
    - ``cuda``: require CUDA and raise if torch sees no device, so an explicit
      request fails loudly instead of silently degrading to CPU.
    - ``mps``: not CUDA — see :func:`mps_enabled`.

    Single source of truth for the CPU/CUDA decision shared by ``_nn_device``
    (src/shared/pipeline.py) and ``_gpu_resident_device`` (src/shared/training.py)
    so the NN's device and its batcher path cannot disagree.
    """
    req = requested_device()
    if req in ("cpu", "mps"):
        return False
    available = torch.cuda.is_available()
    if req == "cuda" and not available:
        raise RuntimeError(
            "--device cuda (FF_DEVICE=cuda) was requested, but "
            "torch.cuda.is_available() is False. Install a CUDA-enabled torch "
            "build, or rerun with --device auto (or --device cpu)."
        )
    return available


def requested_amp_dtype() -> str:
    """Operator-requested AMP dtype: ``"auto"`` (default), ``bf16``/``fp16``/``fp32``.

    Sourced from ``$FF_AMP_DTYPE``. Unset/unrecognised → ``"auto"`` so a typo can
    never silently pin the wrong precision.
    """
    val = os.environ.get(_AMP_DTYPE_ENV, "auto").strip().lower()
    return val if val in _VALID_AMP else "auto"


def amp_dtype() -> torch.dtype | None:
    """The autocast dtype for this run, or ``None`` if AMP should be off.

    Layers on :func:`cuda_enabled` so the selection follows the same single
    source of truth as device placement:

    - **Non-CUDA (CPU/MPS — local Mac dev, CI):** ``None``. AMP is off, so those
      runs stay byte-identical to the FP32 path.
    - **CUDA, default (``auto``):** ``None`` — AMP is *off*. The NN trains in
      FP32 storage with TF32 matmuls (``set_float32_matmul_precision("high")``
      auto-enabled on sm_80+ in ``_nn_device``). Flipped from the old FP16
      autocast default 2026-06-22 (owner-approved metric-path change): on this
      launch-bound model FP16 autocast's per-op cast kernels cost more wall-time
      than TF32 saves, and TF32 keeps FP16's 10-bit matmul mantissa plus the
      full FP32 exponent so accuracy is neutral (QB/RB/WR/K/DST A/B, n=8,
      graphs-off). With AMP off there is no ``GradScaler``, so the graphed path
      is per-step bit-exact.

    ``$FF_AMP_DTYPE`` overrides the default for experimentation:

    - ``fp16`` — opt into the old FP16 autocast + ``GradScaler`` path on every
      CUDA device (T4 and Blackwell alike). This was the default before the
      2026-06-22 flip; it remains the proven mixed-precision path.
    - ``bf16`` — opt into BF16, but **only on sm_80+** (Ampere/Ada/Blackwell).
      On Turing (Tesla T4 ``sm_75``) BF16 autocast hung production (PRs
      #293/#301), so this falls back to FP16 there rather than reintroduce the
      hang — the opt-in cannot footgun a T4. A deterministic 5080 A/B also showed
      BF16 *regresses* high-magnitude heads (QB ``passing_yards`` +2.2–3.1%),
      so it is never auto-selected.
    - ``fp32`` — disable AMP entirely (same as the ``auto`` default now; explicit
      for symmetry).
    """
    if not cuda_enabled():
        return None
    req = requested_amp_dtype()
    if req == "fp32":
        return None
    if req == "fp16":
        return torch.float16
    if req == "bf16":
        # Opt-in BF16 is SAFE only on sm_80+. T4 (sm_75) has no BF16 Tensor
        # Cores — BF16 autocast hung it (#293/#301) — so degrade to FP16.
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    # auto (default): AMP OFF -> FP32 storage + TF32 matmuls (sm_80+, via
    # set_float32_matmul_precision("high") in _nn_device). Flipped from FP16
    # default 2026-06-22 (owner-approved metric-path change): on this
    # launch-bound model FP16 autocast's per-op cast kernels cost more than
    # TF32 saves, and TF32 keeps FP16's 10-bit matmul mantissa + full FP32
    # exponent so accuracy is neutral (QB/RB/WR/K/DST A/B, n=8 graphs-off).
    # FP16 remains available via FF_AMP_DTYPE=fp16.
    return None


def mps_enabled() -> bool:
    """Whether Apple MPS should be used for this run.

    MPS is **opt-in**: only ``--device mps`` (``FF_DEVICE=mps``) selects it. It is
    deliberately excluded from ``auto`` because, for this project's small
    attention model, MPS has no proven speedup over the CPU path, it breaks
    byte-identity with the CPU/CI numerics, and it risks silent op-fallback (see
    the *Platform & hardware targets* section of CLAUDE.md — run an A/B before
    relying on it). Raises if MPS is requested but unavailable so an explicit
    request fails loudly instead of silently degrading to CPU.
    """
    if requested_device() != "mps":
        return False
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "--device mps (FF_DEVICE=mps) was requested, but "
            "torch.backends.mps.is_available() is False. MPS needs an Apple "
            "Silicon Mac with an MPS-enabled torch build; rerun with "
            "--device auto (or --device cpu)."
        )
    return True


def cuda_graph_enabled() -> bool:
    """Whether hand-rolled CUDA graph capture is active for NN training.

    **Autodetect-on for CUDA sm_80+** (g5/A10G ``sm_86``, g6/L4 ``sm_89``,
    RTX 5080 ``sm_120``):
    capturing the NN's forward+backward once and replaying it collapses the
    hundreds of thousands of microsecond kernel launches the tiny attention
    model is bottlenecked on (it is GPU-launch-bound, not compute-bound) —
    ~1.5-1.8× on the GPU branch, speeding up both the base/control NN and the
    attention NN. ``$FF_CUDA_GRAPH`` is an **override, not the trigger**: set it
    to a falsy value (``0``/``false``/``no``/``off``) to force the eager path —
    e.g. for a bit-comparable A/B against an eager baseline.

    **Deliberately per-arch / NOT byte-identical** (reverses the original
    off-by-default; ADR-0017). Unlike ``FF_COMPILE``/TF32, graph replay is not
    numerically inert: a single fwd+bwd step is bitwise-exact, but the
    FP16+GradScaler path amplifies sub-ULP graph kernel-ordering differences
    over many steps into a ~0.5% worst-target *eval* drift. So the sm_80+
    training path is intentionally non-byte-identical to CPU/CI, and benchmark
    history rebaselines (graphed-vs-graphed) from the cutover — the speedup was
    prioritised over comparability by owner decision (see ADR-0017 and
    todo/gpu_launch_bound_levers.md). CPU/MPS and the T4 (sm_75, EC2 rollback)
    can't capture, so they always take the eager path: CI stays eager and the
    divergence is confined to sm_80+.
    """
    # Hardware floor: capture only runs on CUDA sm_80+. CPU/MPS and the T4
    # (sm_75) short-circuit to False here, so an explicit FF_CUDA_GRAPH=1 can't
    # force it on unsupported hardware. ``cuda_enabled()`` is evaluated first so
    # ``get_device_capability()`` is never called on a CPU/MPS box.
    if not cuda_enabled() or torch.cuda.get_device_capability()[0] < 8:
        return False
    # Autodetect ON for capable hardware; FF_CUDA_GRAPH is the force-off knob.
    return os.environ.get(_CUDA_GRAPH_ENV, "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def cuda_graph_full_enabled() -> bool:
    """Whether FULL-STEP CUDA graph capture is active for NN training.

    Extends :func:`cuda_graph_enabled`'s model-only fwd+bwd capture to one
    graph covering batch gather + model forward + combined loss (see
    ``_GraphedTrainStep`` in ``src/shared/training.py``), eliminating the
    eager per-step loss/gather kernel launches on the launch-bound host path.

    **Autodetect-ON for CUDA sm_80+** (the production default since 2026-06-15;
    ADR-0017 Changelog), ``FF_CUDA_GRAPH_FULL`` is a **force-off override**
    (``0``/``false``/``no``/``off``) — mirroring :func:`cuda_graph_enabled`. This
    is the **owner-approved second graphed rebaseline**: full-step capture is
    per-step bitwise-exact but the FP16+GradScaler path amplifies the multi-step
    trajectory the same way model-only capture does (~0.5% worst-target eval
    drift, the 2026-06-05 model-only promotion below), so the next 6-position
    retrain rebaselines graphed-full-vs-graphed-full. Validated on Batch before
    promotion (graph-scope determinism + bounded divergence vs model-only).
    Requires ``cuda_graph_enabled()`` (CUDA sm_80+, not force-disabled) —
    full-step capture is a superset, never a substitute, of the base gate, so
    ``FF_CUDA_GRAPH=0`` disables both. K's nested trainer no-ops capture; the
    ensemble/stacked regime sets ``FF_CUDA_GRAPH_FULL=0`` to stay eager.
    """
    if os.environ.get(_CUDA_GRAPH_FULL_ENV, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    return cuda_graph_enabled()


def cuda_graph_opt_enabled() -> bool:
    """Whether OPTIMIZER-TAIL CUDA graph capture (Lever A3) is active.

    Extends :func:`cuda_graph_full_enabled`'s {gather + forward + backward +
    combined-loss} graph to also bake the per-step eager tail —
    ``zero_grad`` -> ``clip_grad_norm_`` -> ``AdamW.step`` -> loss accumulate —
    into one manual ``torch.cuda.CUDAGraph`` over the whole iteration (see
    ``_GraphedFullStep`` in ``src/shared/training.py``). The eager tail is
    8.6% (RB) / 12.7% (WR) / 24.2% (QB) of attn-NN step time; capturing it
    removes those launches on the launch-bound host path.

    **Autodetect-ON for CUDA sm_80+**, ``FF_CUDA_GRAPH_OPT`` a **force-off
    override** (``0``/``false``/``no``/``off``) — mirroring the two graphs it
    builds upon. Unlike full-step capture, this path is **strictly inert**
    (bit-identical to the A2-only path; no rebaseline): the FP32 production
    default (#1311) removed ``GradScaler`` so the optimizer has no inf/NaN skip
    branch, ``AdamW(capturable=True)`` is a math-no-op over identical grads, and
    a graph replay of the optimizer step is bitwise-exact (both verified, local
    Δ=0 gate). Requires ``cuda_graph_full_enabled()`` (A3 ⊆ A2 ⊆ base gate), so
    ``FF_CUDA_GRAPH=0`` or ``FF_CUDA_GRAPH_FULL=0`` cascades it off; K's nested
    trainer no-ops capture.
    """
    if os.environ.get(_CUDA_GRAPH_OPT_ENV, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    return cuda_graph_full_enabled()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # ``torch.manual_seed`` also schedules CUDA seeding, even when the caller
    # forced ``FF_DEVICE=cpu``. Seed the CPU generator directly so CPU-only
    # diagnostics can fork workers without touching CUDA state.
    torch.random.default_generator.manual_seed(seed)
    if cuda_enabled():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timed(phase: str, store: dict | None = None):
    """Emit a [timing] log line with wall-clock seconds spent in a phase.

    Format: ``[timing] phase={phase} seconds={secs:.1f}``. Matches the log
    contract consumed by CloudWatch and the GitHub Actions log-scrape in
    train-ec2.yml — do not change the format without updating those consumers.

    If ``store`` is provided, also record ``store[phase] = round(secs, 1)`` so
    the caller can persist the breakdown alongside its own metrics.
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        secs = time.monotonic() - t0
        print(f"[timing] phase={phase} seconds={secs:.1f}", flush=True)
        if store is not None:
            store[phase] = round(secs, 1)

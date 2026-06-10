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
    - **CUDA, default (``auto``/``fp16``):** ``torch.float16`` on *all* CUDA
      (T4 and Blackwell alike). A deterministic 5080 A/B showed BF16 *regresses*
      high-magnitude heads — QB ``passing_yards`` +2.2–3.1% — because BF16 trades
      mantissa bits (7 vs FP16's 10) the model uses for exponent range it does
      not need (``GradScaler`` already covers FP16 gradient underflow). FP16 also
      runs full-throughput Tensor Cores on Blackwell, so there is no speed reason
      to switch. Hence BF16 is never auto-selected.

    ``$FF_AMP_DTYPE`` overrides the default for experimentation:

    - ``bf16`` — opt into BF16, but **only on sm_80+** (Ampere/Ada/Blackwell).
      On Turing (Tesla T4 ``sm_75``) BF16 autocast hung production (PRs
      #293/#301), so this falls back to FP16 there rather than reintroduce the
      hang — the opt-in cannot footgun a T4.
    - ``fp16`` — force FP16 (same as the default; explicit for symmetry).
    - ``fp32`` — disable AMP entirely (e.g. to measure the TF32 FP32 path).
    """
    if not cuda_enabled():
        return None
    req = requested_amp_dtype()
    if req == "fp32":
        return None
    if req == "bf16":
        # Opt-in BF16 is SAFE only on sm_80+. T4 (sm_75) has no BF16 Tensor
        # Cores — BF16 autocast hung it (#293/#301) — so degrade to FP16.
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    # auto / fp16: FP16 is the proven default on every CUDA device.
    return torch.float16


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

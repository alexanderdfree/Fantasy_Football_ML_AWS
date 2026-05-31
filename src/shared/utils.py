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
_VALID_DEVICES = ("auto", "cpu", "cuda")


def requested_device() -> str:
    """Operator-requested device: ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.

    Sourced from ``$FF_DEVICE`` (set by ``run_pipeline --device``). An unset or
    unrecognised value falls back to ``"auto"`` so a typo can never silently pin
    the wrong device — only the three explicit choices change behaviour.
    """
    val = os.environ.get(_DEVICE_ENV, "auto").strip().lower()
    return val if val in _VALID_DEVICES else "auto"


def cuda_enabled() -> bool:
    """Whether CUDA should be used for this run, honouring ``--device``.

    - ``auto`` (default): CUDA iff ``torch.cuda.is_available()`` — the historical
      behaviour, so Linux/macOS/CI runs are unchanged when the flag is omitted.
    - ``cpu``: never CUDA — force the CPU path wherever this is consulted.
    - ``cuda``: require CUDA and raise if torch sees no device, so an explicit
      request fails loudly instead of silently degrading to CPU.

    Single source of truth for the CPU/CUDA decision shared by ``_nn_device``
    (src/shared/pipeline.py) and ``_gpu_resident_device`` (src/shared/training.py)
    so the NN's device and its batcher path cannot disagree.
    """
    req = requested_device()
    if req == "cpu":
        return False
    available = torch.cuda.is_available()
    if req == "cuda" and not available:
        raise RuntimeError(
            "--device cuda (FF_DEVICE=cuda) was requested, but "
            "torch.cuda.is_available() is False. Install a CUDA-enabled torch "
            "build, or rerun with --device auto (or --device cpu)."
        )
    return available


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

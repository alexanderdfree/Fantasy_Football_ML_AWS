"""Shared utilities — seeding, small helpers."""

import random
import time
from contextlib import contextmanager

import numpy as np
import torch


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

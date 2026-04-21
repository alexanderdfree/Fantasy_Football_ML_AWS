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
def timed(phase: str):
    """Emit a [timing] log line with wall-clock seconds spent in a phase.

    Format: ``[timing] phase={phase} seconds={secs:.1f}``. Matches the log
    contract consumed by CloudWatch and the GitHub Actions log-scrape in
    train-ec2.yml — do not change the format without updating those consumers.
    """
    t0 = time.monotonic()
    try:
        yield
    finally:
        print(f"[timing] phase={phase} seconds={time.monotonic() - t0:.1f}", flush=True)

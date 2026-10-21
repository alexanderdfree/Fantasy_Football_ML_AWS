"""Positive control for exact result reuse through the ordinary A/B harness.

Run sequentially within each allocation so the second identical recipe can
reuse the first fit. This validates execution and provenance, not accuracy
improvement. Use ``--fresh`` to force both cells to fit independently.
"""

from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
SEEDS = [42]
VARIANTS = [
    Variant("baseline", label="production recipe, first request"),
    Variant("repeat", expect_ridge_identical=True, label="identical recipe, second request"),
]
SUPPORTS_STACKED = False


if __name__ == "__main__":
    ab_main(__spec__.name)

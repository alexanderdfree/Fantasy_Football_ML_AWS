"""Worked example / copy-me template for the shared A/B harness.

Run it::

    python -m src.tuning.ab_example                          # RB WR, 3 seeds, autodetect -j
    python -m src.tuning.ab_example --positions K --seeds 42 123 -j 2   # fast smoke
    python -m src.tuning.ab_example --list                   # show the grid, run nothing

It demonstrates the two variant shapes and the two Ridge-sentinel directions:

* ``+season_recency`` — a **feature A/B**: a ``frame_injector`` adds a
  pre-kickoff column (``season`` is known before the game, so it is leakage-safe)
  and a ``cfg_mutator`` whitelists it into BOTH model paths
  (``get_feature_columns_fn`` for Ridge/LightGBM/NN-static and
  ``attn_static_features`` for the NN static branch). It declares
  ``expect_ridge_identical=False`` — a real new feature MUST move the
  deterministic Ridge fit, so an unchanged Ridge MAE is the "feature didn't take"
  ``Δ=0`` smell (cache/injection bug).

* ``nn_dropout=0`` — an **NN-only A/B**: a ``cfg_mutator`` flips one neural-net
  knob and nothing else. It declares ``expect_ridge_identical=True`` — Ridge
  never sees the NN config, so a *changed* Ridge MAE would prove the "NN-only"
  change leaked into the shared data path.

This file is intentionally low-stakes science (``season_recency`` is a weak
feature). It exists to be copied — replace the injector/mutator with your real
A/B and keep the structure. See [src/tuning/ab_harness.py](ab_harness.py) for the
full contract and [todo/ab_harness_priority.md](../../todo/ab_harness_priority.md).
"""

from __future__ import annotations

from src.config import SEASONS
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["RB", "WR"]  # the motivating role-inheritance case; override with --positions
SEEDS = [42, 123, 7]


# --------------------------------------------------------------------------- #
# Variant 1 — a feature A/B (frame injection + whitelist)
# --------------------------------------------------------------------------- #
def _inject_season_recency(train, val, test):
    """Add ``season_recency`` (seasons since the first trained season).

    Pre-kickoff and constant within a season, so trivially leakage-safe. A real
    injector computes a derived signal here — and YOU own its leakage-safety; the
    Ridge sentinel checks that the feature *took*, not that it is honest (a
    season-mean leak passes the sentinel and inflated the role-inheritance A/B
    ~60%). Compute cross-player features *within position* if you add them.
    """
    base = SEASONS[0]
    for df in (train, val, test):
        df["season_recency"] = (df["season"] - base).astype("float64")
    return train, val, test


def _whitelist_season_recency(cfg):
    """Whitelist ``season_recency`` into both model paths (in place — the harness
    hands every mutator a private deep copy)."""
    get_cols = cfg["get_feature_columns_fn"]
    cfg["get_feature_columns_fn"] = lambda: [*get_cols(), "season_recency"]
    if "attn_static_features" in cfg:  # K/DST enumerate it; skip if a position omits it
        cfg["attn_static_features"] = [*cfg["attn_static_features"], "season_recency"]
    return cfg


# --------------------------------------------------------------------------- #
# Variant 2 — an NN-only A/B (config mutation, no data change)
# --------------------------------------------------------------------------- #
def _zero_dropout(cfg):
    cfg["nn_dropout"] = 0.0
    return cfg


VARIANTS = [
    Variant("baseline", label="baseline (production config)"),
    Variant(
        "+season_recency",
        cfg_mutator=_whitelist_season_recency,
        frame_injector=_inject_season_recency,
        expect_ridge_identical=False,  # a feature MUST move Ridge
        label="+season_recency feature",
    ),
    Variant(
        "nn_dropout=0",
        cfg_mutator=_zero_dropout,
        expect_ridge_identical=True,  # NN-only — must NOT move Ridge
        label="nn_dropout=0 (NN-only)",
    ),
]


if __name__ == "__main__":
    ab_main(__spec__.name)

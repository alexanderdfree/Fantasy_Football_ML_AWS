"""Factorial WR isolation of the three held NN correctness changes.

All eight recipes retain the production pipeline, architecture, training budget,
seeds, inputs and scoring. The three booleans alone vary. A one-seed baseline /
corrected smoke must pass before running the 24-cell three-seed grid. Capture,
saved-inference parity, durable raw rows and input identities use the existing
merge-readiness observer. No recipe is an acceptance decision.
"""

from __future__ import annotations

import sys
from functools import partial

from src.shared.neural_net import GatedHead, PoissonLogRateHead
from src.tuning import ab_inheritance_reception, ab_merge_readiness
from src.tuning.ab_harness import Variant, ab_main

POSITIONS = ["WR"]
SEEDS = [42, 123, 7]
BASELINE = "baseline"
# Magnitude scaling, corrected reception expectation, Poisson log-rate.
RECIPES = {
    "baseline": (False, False, False),
    "magnitude_only": (True, False, False),
    "expectation_only": (False, True, False),
    "poisson_only": (False, False, True),
    "magnitude_expectation": (True, True, False),
    "magnitude_poisson": (True, False, True),
    "expectation_poisson": (False, True, True),
    "corrected": (True, True, True),
}
_ARM = "unset"


def configure(config, *, arm):
    global _ARM
    magnitude, expectation, poisson = RECIPES[arm]
    # Install the existing observer without altering trainer calls or RNG.
    ab_merge_readiness.baseline(config)
    _ARM = arm
    config["nn_magnitude_features"] = ("inherited_opportunity",) if magnitude else ()
    config["nn_correct_ztnb_mean"] = expectation
    config["nn_poisson_log_rate"] = poisson
    # Reuse the existing positive-control check for magnitude-preserving arms.
    ab_inheritance_reception._ARM = "magnitude_only" if magnitude else "baseline"
    return config


VARIANTS = [
    Variant(
        arm,
        cfg_mutator=partial(configure, arm=arm),
        expect_ridge_identical=None if arm == BASELINE else True,
    )
    for arm in RECIPES
]


def activation(models, arm):
    """Check the fitted head modes, including the gated TD exception."""
    _, expectation, poisson = RECIPES[arm]
    observed = {}
    for family in ("nn", "attn_nn"):
        model = models[family]
        expected_log = {"fumbles_lost", "receiving_tds"} if family == "nn" else {"fumbles_lost"}
        actual_log = {
            target
            for target, head in model.heads.items()
            if isinstance(head, PoissonLogRateHead) and head.uses_log_rate
        }
        if actual_log != (expected_log if poisson else set()):
            raise ValueError(f"{arm}/{family}: wrong fitted Poisson head modes: {actual_log}")
        observed[f"{family}_log_rate_heads"] = len(actual_log)
    receptions = models["attn_nn"].heads["receptions"]
    if not isinstance(receptions, GatedHead) or receptions.correct_ztnb_mean != expectation:
        raise ValueError(f"{arm}: fitted reception expectation does not match the recipe")
    # Plain NN has no GatedHead, so the reception switch must be inactive there.
    if isinstance(models["nn"].heads["receptions"], GatedHead):
        raise ValueError("WR plain NN unexpectedly acquired a gated reception head")
    observed["attn_reception_expectation_corrected"] = int(receptions.correct_ztnb_mean)
    return observed


def metric_fn(result, position):
    if position != "WR" or _ARM not in RECIPES:
        raise ValueError("WR component evidence requires its configured WR cell")
    observed = activation(result.models, _ARM)
    metrics = ab_merge_readiness.metric_fn(result, position, variant=_ARM)
    metrics["component_activation"] = observed
    return metrics


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    return ab_main("src.tuning.ab_wr_components", ["--no-stacked-seeds", *args])


if __name__ == "__main__":
    main()

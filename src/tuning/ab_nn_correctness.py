"""Compare the consolidated NN correctness policies against main's legacy outputs.

Run a real one-position smoke before the full five-position, three-seed grid.
Production recipes supply WR-only magnitude scaling; no other feature or loss
setting is changed. This remains eager so sparse per-head diagnostics survive.
"""

import sys

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.ab_poisson_log_rate import metric_fn as poisson_metrics

POSITIONS = ["QB", "RB", "WR", "TE", "DST"]
SEEDS = [42, 123, 7]


def legacy(config):
    from src.tuning import ab_inheritance_reception

    ab_inheritance_reception._ARM = "baseline"
    config["nn_poisson_log_rate"] = False
    config["nn_correct_ztnb_mean"] = False
    config["nn_magnitude_features"] = ()
    return config


def corrected(config):
    from src.tuning import ab_inheritance_reception

    ab_inheritance_reception._ARM = "corrected"
    config["nn_poisson_log_rate"] = True
    config["nn_correct_ztnb_mean"] = True
    return config


VARIANTS = [
    Variant("baseline", cfg_mutator=legacy),
    Variant("corrected", cfg_mutator=corrected, expect_ridge_identical=True),
]


def metric_fn(result, position):
    metrics = poisson_metrics(result, position)
    if position != "DST":
        from src.tuning.ab_inheritance_reception import metric_fn as inheritance_metrics

        metrics.update(inheritance_metrics(result, position))
        if position == "WR":
            for name in ("nn_encoding", "attn_encoding"):
                encoding = metrics[name]
                if encoding["encoded_unique"] != encoding["raw_unique"]:
                    # The baseline intentionally reproduces the collapse.
                    from src.tuning import ab_inheritance_reception

                    if ab_inheritance_reception._ARM == "corrected":
                        raise ValueError("Corrected WR inheritance magnitudes collapsed")
    return metrics


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    return ab_main("src.tuning.ab_nn_correctness", ["--no-stacked-seeds", *args])


if __name__ == "__main__":
    main()

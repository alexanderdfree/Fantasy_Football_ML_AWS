"""One isolated stable count-likelihood candidate at the original WR weights."""

from functools import partial

from src.tuning.ab_harness import Variant, ab_main
from src.tuning.model_default_repair import configure, origin_frames
from src.tuning.model_default_repair import metric_fn as metric_fn

POSITIONS = ["WR"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = [
    Variant(
        arm,
        cfg_mutator=partial(configure, arm=arm, mode="wr"),
        frame_injector=origin_frames,
        expect_ridge_identical=arm != "baseline",
    )
    for arm in ("baseline", "corrected", "stable_numeric")
]


if __name__ == "__main__":
    ab_main("src.tuning.ab_wr_numerical_repair")

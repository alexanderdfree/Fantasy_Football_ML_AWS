"""PR1608/1613 paired historical production checks; AWS Spot Batch only."""

from src.tuning.ab_harness import ab_main
from src.tuning.correctness_development import metric_fn as metric_fn
from src.tuning.correctness_development import variants

POSITIONS = ["K", "DST"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = variants(native=True, controls=True)

if __name__ == "__main__":
    ab_main("src.tuning.ab_correctness_native")

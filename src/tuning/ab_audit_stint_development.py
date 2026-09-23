"""PR #1607 historical WR/TE feature rebuilds, with attention as a control."""

from src.tuning.ab_harness import ab_main
from src.tuning.audit_development import metric_fn as metric_fn
from src.tuning.audit_development import variants

POSITIONS = ["WR", "TE"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = variants("stint_reset")


if __name__ == "__main__":
    ab_main("src.tuning.ab_audit_stint_development")

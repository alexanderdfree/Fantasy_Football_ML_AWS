"""PR #1613 paired historical development, with unchanged-main inference means."""

from src.tuning.ab_harness import ab_main
from src.tuning.audit_development import metric_fn as metric_fn
from src.tuning.audit_development import variants

POSITIONS = ["RB", "WR", "TE"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = variants("count_precision")


if __name__ == "__main__":
    ab_main("src.tuning.ab_audit_count_development")

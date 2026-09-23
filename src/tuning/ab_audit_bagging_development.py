"""PR #1606 historical development on the shared skill-position providers."""

from src.tuning.ab_harness import ab_main
from src.tuning.audit_development import metric_fn as metric_fn
from src.tuning.audit_development import variants

POSITIONS = ["QB", "RB", "WR", "TE"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = variants("bagging")


if __name__ == "__main__":
    ab_main("src.tuning.ab_audit_bagging_development")

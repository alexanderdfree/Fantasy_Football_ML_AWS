"""PR #1606 historical development using native K/DST dataset providers."""

from src.tuning.ab_harness import ab_main
from src.tuning.audit_development import metric_fn as metric_fn
from src.tuning.audit_development import variants

POSITIONS = ["K", "DST"]
SEEDS = [42, 123, 7]
SUPPORTS_STACKED = False
VARIANTS = variants("bagging", native=True)


if __name__ == "__main__":
    ab_main("src.tuning.ab_audit_native_bagging_development")

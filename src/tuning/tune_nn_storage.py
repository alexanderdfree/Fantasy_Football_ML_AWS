"""Storage names for the current ``tune_nn`` Optuna search space.

This module is intentionally dependency-free so Batch aggregation utilities can
share the same S3/local naming contract without importing ``src.tuning.tune_nn``
and pulling in Optuna.
"""

SEARCH_SPACE_VERSION = "scheduler_v2"
S3_PREFIX = f"tune_nn/{SEARCH_SPACE_VERSION}"


def study_name(pos: str) -> str:
    return f"nn_{SEARCH_SPACE_VERSION}_{pos.lower()}"


def study_db_path(pos: str) -> str:
    return f"tune_nn_{SEARCH_SPACE_VERSION}_{pos.lower()}.db"


def s3_key_prefix(pos: str) -> str:
    return f"{S3_PREFIX}/{pos.lower()}"

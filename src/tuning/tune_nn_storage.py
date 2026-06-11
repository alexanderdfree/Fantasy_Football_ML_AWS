"""Storage names for the current ``tune_nn`` Optuna search space.

This module is intentionally dependency-free so Batch aggregation utilities can
share the same S3/local naming contract without importing ``src.tuning.tune_nn``
and pulling in Optuna.
"""

SEARCH_SPACE_VERSION = "scheduler_v2"
MPS_GRAPH_SEARCH_SPACE_VERSION = f"{SEARCH_SPACE_VERSION}_mps_graph"
MPS_SEARCH_SPACE_VERSION = f"{SEARCH_SPACE_VERSION}_mps"
GRAPH_SEARCH_SPACE_VERSION = f"{SEARCH_SPACE_VERSION}_graph"


def resolve_search_space_version(
    parallel_backend: str = "thread", *, cuda_graph: bool = False
) -> str:
    """Storage namespace for the execution profile.

    The sampled search space is still scheduler_v2, but CUDA-graph/MPS tuning
    follows a different training trajectory from the eager local default. Keep
    those studies separate so Batch graph-enabled results never resume from an
    older eager study DB.

    The thread backend honors ``cuda_graph`` too: since the 2026-06-05
    autodetect-ON cutover, sm_80+ boxes run graphed by default, so a
    thread-backend tune on such a box must not resume (or pollute) the eager
    ``scheduler_v2`` study — graphed runs compare to graphed runs (ADR-0017).
    """
    if parallel_backend == "mps":
        return MPS_GRAPH_SEARCH_SPACE_VERSION if cuda_graph else MPS_SEARCH_SPACE_VERSION
    return GRAPH_SEARCH_SPACE_VERSION if cuda_graph else SEARCH_SPACE_VERSION


def s3_prefix(version: str = SEARCH_SPACE_VERSION) -> str:
    return f"tune_nn/{version}"


def study_name(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"nn_{version}_{pos.lower()}"


def study_db_path(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"tune_nn_{version}_{pos.lower()}.db"


def s3_key_prefix(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"{s3_prefix(version)}/{pos.lower()}"

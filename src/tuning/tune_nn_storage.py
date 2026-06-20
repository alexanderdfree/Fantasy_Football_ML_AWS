"""Storage names for the current ``tune_nn`` Optuna search space.

This module is intentionally dependency-free so Batch aggregation utilities can
share the same S3/local naming contract without importing ``src.tuning.tune_nn``
and pulling in Optuna.
"""

SEARCH_SPACE_VERSION = "scheduler_v2"

# Root namespace for the attention game-history-branch tuner (``tune_nn
# --scope history``): it adds attn_max_seq_len + per-game token bundles to the
# search space and freezes the static backbone, so its trials must NOT mix with
# the default ``scheduler_v2`` study (Optuna rejects a param-space mismatch in
# one study). The graph/mps/full suffixing below applies to this root too — a
# graphed history tune is still a different trajectory from an eager one.
HISTORY_SEARCH_SPACE_VERSION = "history_v1"

# Search-space roots selectable by ``--scope``. ``resolve_search_space_version``
# applies the execution-profile (mps/graph/full) suffixes to whichever root.
SCOPE_ROOTS: dict[str, str] = {
    "full": SEARCH_SPACE_VERSION,
    "history": HISTORY_SEARCH_SPACE_VERSION,
}


def resolve_search_space_version(
    parallel_backend: str = "thread",
    *,
    cuda_graph: bool = False,
    full_graph: bool = False,
    root: str = SEARCH_SPACE_VERSION,
) -> str:
    """Storage namespace for the execution profile.

    ``root`` selects the sampled search space (``scheduler_v2`` for the default
    full scope, ``history_v1`` for ``--scope history``); the mps/graph/full
    suffixes below are applied to it. The sampled search space is otherwise
    scheduler_v2, but CUDA-graph/MPS tuning follows a different training
    trajectory from the eager local default. Keep those studies separate so
    Batch graph-enabled results never resume from an older eager study DB.

    The thread backend honors ``cuda_graph`` too: since the 2026-06-05
    autodetect-ON cutover, sm_80+ boxes run graphed by default, so a
    thread-backend tune on such a box must not resume (or pollute) the eager
    ``scheduler_v2`` study — graphed runs compare to graphed runs (ADR-0017).

    ``cuda_graph`` must be the trainer's *actual* capture decision —
    ``src.shared.utils.cuda_graph_enabled()`` on the box that trains, or, for
    the Batch launcher's submit-side prediction, the CLI bool it also injects
    as ``FF_CUDA_GRAPH`` — never a raw env-truthy read. Post-cutover the env is
    a force-OFF override only: an sm_80+ box with it unset trains graphed, and
    a sub-sm_80 box with ``FF_CUDA_GRAPH=1`` trains eager.

    ``full_graph`` (FF_CUDA_GRAPH_FULL: gather+forward+loss in one capture)
    appends ``full`` to the graph namespaces — yet another trajectory regime,
    same separation rationale, same resolved-decision rule (the trainer's
    ``cuda_graph_full_enabled()``, or the launcher's submit-side CLI bool). It
    composes only WITH ``cuda_graph`` (the trainer's full-step gate requires
    the base gate), so full-without-graph resolves to the plain namespace
    rather than inventing an unreachable one.
    """
    if parallel_backend == "mps":
        base = f"{root}_mps_graph" if cuda_graph else f"{root}_mps"
    else:
        base = f"{root}_graph" if cuda_graph else root
    if full_graph and cuda_graph:
        return f"{base}full"
    return base


def s3_prefix(version: str = SEARCH_SPACE_VERSION) -> str:
    return f"tune_nn/{version}"


def study_name(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"nn_{version}_{pos.lower()}"


def study_db_path(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"tune_nn_{version}_{pos.lower()}.db"


def s3_key_prefix(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"{s3_prefix(version)}/{pos.lower()}"

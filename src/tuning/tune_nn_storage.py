"""Storage names for the current ``tune_nn`` Optuna search space.

This module is intentionally dependency-free so Batch aggregation utilities can
share the same S3/local naming contract without importing ``src.tuning.tune_nn``
and pulling in Optuna.
"""

# v3 preserves the sampled parameters but reports sample-weighted validation
# loss. v2 averaged batch means, overweighting ragged tails; resuming those
# trials would mix different objectives in TPE/pruning and best-trial selection.
SEARCH_SPACE_VERSION = "scheduler_v3"

# Shared metadata stays importable on orchestration runners without torch.
ENSEMBLE_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_STACKED_SEEDS = 24
DEFAULT_CUDA_GRAPH = True
DEFAULT_CUDA_GRAPH_FULL = True
DEFAULT_PARALLEL_BACKEND = "auto"


def stacked_default_seed_list(n: int = DEFAULT_STACKED_SEEDS) -> list[int]:
    """The canonical ensemble seeds, available without importing the trainer."""
    return list(range(42, 42 + n))


# Root namespace for the attention game-history-branch tuner (``tune_nn
# --scope history``). The v2 isolation still searches ONLY attn_max_seq_len + the
# per-game token bundles and freezes the entire production recipe (sizing, lr,
# batch, scheduler, static backbone), so its trials must NOT mix with the
# default full study OR the v1 history studies (Optuna rejects a
# param-space mismatch in one study; v1 also co-sampled lr/sizing, which
# confounded its objective — GH #1239). The graph/mps/full suffixing below
# applies to this root too. v3 also separates sample-weighted objective values
# from history_v2's batch-weighted trials; the isolated parameter space is unchanged.
HISTORY_SEARCH_SPACE_VERSION = "history_v3"

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

    ``root`` selects the sampled search space and objective (``scheduler_v3``
    for full scope, ``history_v3`` for ``--scope history``); the mps/graph/full
    suffixes below are applied to it. CUDA-graph/MPS tuning follows a different
    training trajectory from the eager local default. Keep those studies separate so
    Batch graph-enabled results never resume from an older eager study DB.

    The thread backend honors ``cuda_graph`` too: since the 2026-06-05
    autodetect-ON cutover, sm_80+ boxes run graphed by default, so a
    thread-backend tune on such a box must not resume (or pollute) the eager
    study — graphed runs compare to graphed runs (ADR-0017).

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


def resolve_batch_storage_versions(
    positions,
    *,
    parallel_backend: str = DEFAULT_PARALLEL_BACKEND,
    cuda_graph: bool = DEFAULT_CUDA_GRAPH,
    cuda_graph_full: bool = DEFAULT_CUDA_GRAPH_FULL,
    stacked_seeds: int | None = None,
    stacked_epochs: int = 30,
    scope: str = "full",
) -> dict[str, str]:
    """Resolve launch_tune namespaces, including eager K/DST fallback jobs."""
    width = DEFAULT_STACKED_SEEDS if stacked_seeds is None else stacked_seeds
    backend = "mps" if parallel_backend == "auto" else parallel_backend
    versions = {}
    for pos in positions:
        pos_width = width if pos.upper() in ENSEMBLE_POSITIONS else 0
        stacked = pos_width >= 2
        version = resolve_search_space_version(
            backend,
            cuda_graph=cuda_graph and not stacked,
            full_graph=cuda_graph_full and not stacked,
            root=SCOPE_ROOTS[scope],
        )
        versions[pos] = version + (f"_ens{pos_width}x{stacked_epochs}" if stacked else "")
    return versions


def study_name(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"nn_{version}_{pos.lower()}"


def study_db_path(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"tune_nn_{version}_{pos.lower()}.db"


def s3_key_prefix(pos: str, version: str = SEARCH_SPACE_VERSION) -> str:
    return f"{s3_prefix(version)}/{pos.lower()}"

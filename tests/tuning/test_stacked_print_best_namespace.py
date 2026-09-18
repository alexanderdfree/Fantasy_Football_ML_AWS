import os

import pytest

from src.tuning import tune_nn
from src.tuning.ab_ensemble_seeds import ensemble_env
from src.tuning.tune_nn_storage import study_db_path, study_name

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["thread", "mps"])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("scope", ["full", "history"])
@pytest.mark.parametrize("width", [0, 24])
@pytest.mark.parametrize("graph", [False, True])
def test_print_best_reads_actual_training_namespace(
    monkeypatch,
    backend,
    n_jobs,
    scope,
    width,
    graph,
):
    graph_env = "1" if graph else "0"
    monkeypatch.setenv("FF_CUDA_GRAPH", graph_env)
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", graph_env)
    monkeypatch.setenv("FF_COMPILE", "0")
    monkeypatch.setattr(tune_nn, "_cuda_graph_enabled", lambda: os.getenv("FF_CUDA_GRAPH") == "1")
    monkeypatch.setattr(
        tune_nn, "_cuda_graph_full_enabled", lambda: os.getenv("FF_CUDA_GRAPH_FULL") == "1"
    )
    if width:
        with ensemble_env(7):
            trained_version = tune_nn._resolve_storage_version(backend, scope)[0] + f"_ens{width}x7"
    else:
        tune_nn._force_eager_for_concurrent_thread_trials(backend, n_jobs)
        trained_version = tune_nn._resolve_storage_version(backend, scope)[0]
    monkeypatch.setenv("FF_CUDA_GRAPH", graph_env)
    monkeypatch.setenv("FF_CUDA_GRAPH_FULL", graph_env)
    requested = {}

    def storage(path, timeout):
        requested["db"] = path
        return object()

    def load(**kwargs):
        requested["study"] = kwargs["study_name"]
        raise LookupError("no training or filesystem reads in this namespace control")

    monkeypatch.setattr(tune_nn, "_make_storage", storage)
    monkeypatch.setattr(tune_nn.optuna, "load_study", load)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_nn",
            "QB",
            "--print-best",
            "--scope",
            scope,
            "--parallel-backend",
            backend,
            "--n-jobs",
            str(n_jobs),
            "--stacked-seeds",
            str(width),
            "--stacked-epochs",
            "7",
        ],
    )
    tune_nn.main()
    assert requested == {
        "db": study_db_path("QB", trained_version),
        "study": study_name("QB", trained_version),
    }

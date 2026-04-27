"""Coverage tests for ``WR/benchmark_ridge_variants.py``.

The file is a 280-line diagnostic CLI that compares six Ridge configurations
on the same WR split. These tests exercise the three entry points
(``_condition_number``, ``_run_variant``, ``main``) with stubbed parquet
reads + stubbed heavy ML calls so the whole module runs in under a second.

We do NOT assert on numerical correctness — the goal is branch coverage of
the printing / variant-selection / PCA-autoselection logic, not regression
testing of a fabricated dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------------
# _condition_number — tiny unit test
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_condition_number_nonzero_on_full_rank():
    """On a full-rank matrix condition number is finite and >= 1."""
    from src.wr.benchmark_ridge_variants import _condition_number

    X = np.random.default_rng(0).normal(size=(30, 4))
    cond = _condition_number(X)
    assert np.isfinite(cond)
    assert cond >= 1.0


@pytest.mark.unit
def test_condition_number_infinite_on_rank_deficient():
    """A column duplicated → smallest singular value ≈ 0 → cond = inf."""
    from src.wr.benchmark_ridge_variants import _condition_number

    base = np.random.default_rng(0).normal(size=(30, 3))
    X = np.hstack([base, base[:, :1]])  # column 3 == column 0
    cond = _condition_number(X)
    assert np.isinf(cond) or cond > 1e10


# --------------------------------------------------------------------------
# _run_variant — mock RidgeMultiTarget, _tune_ridge_alphas_cv, compute_metrics
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_run_variant_builds_metrics_dict(monkeypatch):
    """``_run_variant`` must call the tuner, fit+predict, and return the
    per-target + total metrics dict. We replace the heavy pieces with
    lightweight stubs so the test runs in milliseconds."""
    import src.wr.benchmark_ridge_variants as bench

    TARGETS = bench.TARGETS

    # Stub the CV alpha tuner — returns a flat {target: alpha} dict.
    monkeypatch.setattr(bench, "_tune_ridge_alphas_cv", lambda *a, **k: {t: 1.0 for t in TARGETS})

    # Stub RidgeMultiTarget to a trivial model that returns zeros for every target.
    class _FakeRidge:
        def __init__(self, *args, **kwargs):
            self.init_args = (args, kwargs)

        def fit(self, X, y_dict):
            self._n_train = len(X)

        def predict(self, X):
            return {t: np.zeros(len(X), dtype=np.float32) for t in TARGETS}

    monkeypatch.setattr(bench, "RidgeMultiTarget", _FakeRidge)
    monkeypatch.setattr(
        bench, "compute_metrics", lambda y_true, y_pred: {"mae": 1.5, "r2": 0.1, "rmse": 2.0}
    )

    # Minimal X / y inputs.
    n = 20
    rng = np.random.default_rng(0)
    feature_cols = ["f0", "f1", "f2"]
    X_train = pd.DataFrame(rng.normal(size=(n, 3)), columns=feature_cols)
    X_test = pd.DataFrame(rng.normal(size=(n, 3)), columns=feature_cols)

    y_train_dict = {t: rng.normal(size=n) for t in TARGETS}
    y_test_dict = {t: rng.normal(size=n) for t in TARGETS}
    y_train_dict["total"] = sum(y_train_dict[t] for t in TARGETS)
    y_test_dict["total"] = sum(y_test_dict[t] for t in TARGETS)

    pos_train = pd.DataFrame({"season": rng.integers(2020, 2024, n)})

    result = bench._run_variant(
        "test-variant",
        feature_cols,
        X_train,
        X_test,
        y_train_dict,
        y_test_dict,
        pos_train,
        pca_n=None,
    )
    assert result["name"] == "test-variant"
    assert result["n_features"] == 3
    assert result["pca_n"] is None
    assert "metrics" in result
    assert "total" in result["metrics"]
    assert set(result["metrics"].keys()) == set(TARGETS) | {"total"}


# --------------------------------------------------------------------------
# main — stub parquet reads, feature builders, _run_variant to canned output
# --------------------------------------------------------------------------


@pytest.fixture()
def _stub_main(monkeypatch, tmp_path):
    """Stub every heavy call in ``main()`` so it runs in-process in < 1s."""
    import src.wr.benchmark_ridge_variants as bench

    TARGETS = bench.TARGETS

    # 30 fake WR-season rows with the few columns main touches.
    rng = np.random.default_rng(42)
    n = 30

    # Features that both the fake frame and get_feature_columns() agree on.
    # ``is_home`` is in ``EXTRA_DROPS`` so main's aggressive-cols branch will
    # drop it, exercising that code path.
    feature_cols = ["is_home", "season", "week"]

    def _fake_frame():
        feats = {
            "player_id": [f"WR{i:02d}" for i in range(n)],
            "position": ["WR"] * n,
            "season": rng.integers(2020, 2024, n).astype(np.float32),
            "week": rng.integers(1, 18, n).astype(np.float32),
            "recent_team": ["KC"] * n,
            "is_home": rng.integers(0, 2, n).astype(np.float32),
        }
        for col in bench.SPECIFIC_FEATURES:
            feats[col] = rng.normal(size=n)
        for t in TARGETS:
            feats[t] = rng.normal(size=n)
        return pd.DataFrame(feats)

    splits_dir = tmp_path
    for name in ("train", "val", "test"):
        _fake_frame().to_parquet(splits_dir / f"{name}.parquet")

    monkeypatch.setattr(bench, "SPLITS_DIR", str(splits_dir))
    # MIN_GAMES_PER_SEASON filter would wipe the fake data — drop the threshold.
    monkeypatch.setattr(bench, "MIN_GAMES_PER_SEASON", 0)

    # Feature helpers: pass through so main() keeps the full frame.
    monkeypatch.setattr(bench, "filter_to_position", lambda df: df)
    monkeypatch.setattr(bench, "compute_targets", lambda df: df)
    monkeypatch.setattr(
        bench,
        "add_specific_features",
        lambda tr, va, te: (tr.copy(), va.copy(), te.copy()),
    )
    monkeypatch.setattr(
        bench,
        "fill_nans",
        lambda tr, va, te, specs: (tr.copy(), va.copy(), te.copy()),
    )
    # Restrict the feature whitelist so aggressive_cols ends up non-empty after
    # EXTRA_DROPS — we want both the base and aggressive branches in main to fire.
    monkeypatch.setattr(bench, "get_feature_columns", lambda: feature_cols)

    # _run_variant is the inner loop — return canned metrics so main() just
    # drives the aggregation + printing.
    def _canned(name, cols, X_train, X_test, y_train, y_test, pos_train, pca_n=None):
        mae_by_pca = {None: 3.0, 30: 2.7, 50: 2.8, 80: 2.85}.get(pca_n, 3.0)
        metrics = {}
        for t in TARGETS:
            metrics[t] = {"mae": 1.0, "r2": 0.05, "rmse": 2.0}
        metrics["total"] = {"mae": mae_by_pca, "r2": 0.2, "rmse": 4.0}
        return {
            "name": name,
            "n_features": len(cols),
            "pca_n": pca_n,
            "cond_number": 1e5,
            "best_alphas": {t: 1.0 for t in TARGETS},
            "metrics": metrics,
            "elapsed": 0.01,
        }

    monkeypatch.setattr(bench, "_run_variant", _canned)
    return bench


@pytest.mark.unit
def test_main_runs_and_prints_best_variant(_stub_main, capsys):
    """``main()`` must execute all six variants + emit the comparison tables."""
    _stub_main.main()
    out = capsys.readouterr().out
    assert "WR RIDGE VARIANT COMPARISON" in out
    assert "PER-TARGET R" in out  # Unicode squared may render differently
    assert "Best variant:" in out
    # PCA autoselect branch must fire when variant 6 ("aggressive_drops+pcr")
    # picks the best PCA from variants 2-4.
    assert "[auto] Using PCA(" in out


# NOTE: The final ``if __name__ == '__main__': main()`` line is left uncovered
# intentionally — ``runpy.run_path`` re-imports the module in a fresh namespace,
# which reverts every module-level monkeypatch and would require driving the
# real main() end-to-end (parquet reads + Ridge tuning). That's 1 stmt out of
# 114; the rest of the module (incl. all of ``main()``'s body) is covered by
# the direct ``bench.main()`` call above.

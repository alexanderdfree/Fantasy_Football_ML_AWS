"""Unit tests for src.tuning.ablation_runner."""

from __future__ import annotations

import json

import pytest

from src.tuning import ablation_runner as ar

pytestmark = pytest.mark.unit


def _ok_job(job: ar.AblationJob) -> dict:
    return {
        "metrics": {"value": float(job.seed), "nested": {"mae": float(job.seed) + 0.5}},
        "timings": {"elapsed": float(job.seed) / 10.0},
        "metadata": {"seen_label": job.label},
    }


def _bad_job(job: ar.AblationJob) -> dict:
    raise RuntimeError(f"boom {job.variant}")


def _job(seed: int, variant: str = "baseline", run_fn=_ok_job) -> ar.AblationJob:
    return ar.AblationJob(
        position="QB",
        seed=seed,
        variant=variant,
        label=variant,
        run_fn=run_fn,
        base_cfg={"x": 1},
        metadata={"run_kind": "test"},
    )


def test_parse_seed_list_and_variant_selection():
    assert ar.parse_seed_list("42, 7,123") == [42, 7, 123]
    with pytest.raises(ValueError, match="at least one seed"):
        ar.parse_seed_list("")

    available = {"baseline": object(), "alt": object()}
    assert ar.select_variants(None, available, ("baseline",)) == ["baseline"]
    assert ar.select_variants("all", available, ("baseline",)) == ["baseline", "alt"]
    assert ar.select_variants("baseline,alt", available, ("baseline",)) == ["baseline", "alt"]
    with pytest.raises(ValueError, match="unknown variant"):
        ar.select_variants("missing", available, ("baseline",))


def test_run_grid_serial_preserves_order_and_metadata():
    results = ar.run_grid([_job(2, "a"), _job(1, "b")])

    assert [(r.seed, r.variant) for r in results] == [(2, "a"), (1, "b")]
    assert results[0].metrics["value"] == 2.0
    assert results[0].timings["elapsed"] == 0.2
    assert results[0].metadata["run_kind"] == "test"
    assert results[0].metadata["seen_label"] == "a"


def test_run_grid_captures_job_errors():
    result = ar.run_grid([_job(42, "bad", _bad_job)])[0]

    assert result.error == "RuntimeError: boom bad"
    assert result.metrics == {}
    assert "traceback" in result.metadata


def test_run_grid_parallel_path_can_preserve_or_completion_order(monkeypatch):
    submitted = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, job):
            submitted.append((self.max_workers, job.seed))
            return FakeFuture(fn(job))

    monkeypatch.setattr(ar, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(ar, "as_completed", lambda futures: list(reversed(list(futures))))

    jobs = [_job(1, "a"), _job(2, "b")]
    preserved = ar.run_grid(jobs, max_workers=2, preserve_order=True)
    completed = ar.run_grid(jobs, max_workers=2, preserve_order=False)

    assert [(r.seed, r.variant) for r in preserved] == [(1, "a"), (2, "b")]
    assert [(r.seed, r.variant) for r in completed] == [(2, "b"), (1, "a")]
    assert submitted[:2] == [(2, 1), (2, 2)]


def test_mean_std_and_paired_deltas():
    results = ar.run_grid(
        [
            _job(1, "baseline"),
            _job(1, "alt"),
            _job(2, "baseline"),
            _job(2, "alt"),
        ]
    )
    # Make alt worse by one point on both paired seeds.
    adjusted = []
    for result in results:
        metrics = dict(result.metrics)
        if result.variant == "alt":
            metrics["value"] += 1.0
        adjusted.append(
            ar.AblationResult(
                result.position,
                result.seed,
                result.variant,
                metrics,
                result.timings,
                result.metadata,
            )
        )

    assert ar.mean_std([1.0, 3.0]) == {"mean": 2.0, "std": pytest.approx(1.41421356), "n": 2}
    assert ar.paired_deltas(
        adjusted,
        variant="alt",
        baseline_variant="baseline",
        metric_key="value",
        position="QB",
    ) == [1.0, 1.0]


def test_format_dry_run_table_contains_grouped_counts():
    text = ar.format_dry_run_table([_job(1, "baseline"), _job(2, "alt")])

    assert "Planned ablation jobs: 2" in text
    assert "QB" in text
    assert "alt,baseline" in text


def test_write_history_payload_shape(tmp_path):
    result = ar.run_grid([_job(42)])[0]
    path = ar.write_history(
        "unit_ablation",
        [result],
        metadata={"summary": {"QB": {"ok": True}}},
        history_dir=str(tmp_path),
    )

    with open(path) as f:
        payload = json.load(f)

    assert payload["kind"] == "ablation"
    assert payload["name"] == "unit_ablation"
    assert payload["summary"] == {"QB": {"ok": True}}
    assert payload["results"][0]["position"] == "QB"
    assert payload["results"][0]["metrics"]["value"] == 42.0

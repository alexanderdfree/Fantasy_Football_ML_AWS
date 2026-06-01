import json
import os

import pytest

from src.analysis.cuda_graph_gradscale import (
    FIXED_SCALE_ENV,
    GRAPH_ENV,
    GRAPH_RESTORE_BN_ENV,
    INIT_SCALE_ENV,
    VARIANT_ENVS,
    _patched_env,
    first_trace_diff,
    summarize_trace,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.mark.unit
def test_summarize_trace_reports_first_change_and_skip(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_jsonl(
        path,
        [
            {"kind": "meta"},
            {
                "kind": "step",
                "step": 0,
                "scale": 65536.0,
                "next_scale": 65536.0,
                "scale_changed": False,
                "skipped": False,
            },
            {
                "kind": "step",
                "step": 1,
                "scale": 65536.0,
                "next_scale": 32768.0,
                "scale_changed": True,
                "skipped": True,
            },
        ],
    )

    out = summarize_trace(path)

    assert out["n_steps"] == 2
    assert out["first_scale_change_step"] == 1
    assert out["first_skip_step"] == 1
    assert out["initial_scale"] == 65536.0
    assert out["final_scale"] == 32768.0


@pytest.mark.unit
def test_first_trace_diff_ignores_metadata_and_names_field(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    common = {
        "kind": "step",
        "step": 0,
        "scale": 65536.0,
        "next_scale": 65536.0,
        "scale_changed": False,
        "skipped": False,
    }
    _write_jsonl(left, [{"kind": "meta", "label": "left"}, common])
    _write_jsonl(right, [{"kind": "meta", "label": "right"}, {**common, "next_scale": 32768.0}])

    out = first_trace_diff(left, right)

    assert out is not None
    assert out["index"] == 0
    assert out["left"]["next_scale"] == 65536.0
    assert out["right"]["next_scale"] == 32768.0


@pytest.mark.unit
def test_fixed_scale_variants_cover_eager_and_bn_restore():
    assert VARIANT_ENVS["eager_fixed_scale"] == {
        GRAPH_ENV: "0",
        FIXED_SCALE_ENV: "1",
    }
    assert VARIANT_ENVS["graph_fixed_scale_restore_bn"] == {
        GRAPH_ENV: "1",
        FIXED_SCALE_ENV: "1",
        GRAPH_RESTORE_BN_ENV: "1",
    }


@pytest.mark.unit
def test_patched_env_manages_fixed_scale_init(monkeypatch):
    monkeypatch.setenv(INIT_SCALE_ENV, "8192")

    with _patched_env({FIXED_SCALE_ENV: "1"}):
        assert FIXED_SCALE_ENV in os.environ
        assert INIT_SCALE_ENV not in os.environ

    assert os.environ[INIT_SCALE_ENV] == "8192"

"""Unit tests for src/tuning/aggregate_results.py.

S3 is mocked — the tests exercise the merge + format logic against local
fixture files.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from src.tuning import aggregate_results

pytestmark = pytest.mark.unit


def _write_pos_result(tmp_path, pos, val_loss, d_model, n_heads, lr):
    """Drop a tune_nn_{pos}_results.json into ``tmp_path``."""
    payload = {
        pos: {
            "best_trial": 0,
            "best_val_loss": val_loss,
            "best_params": {
                "attn_d_model": d_model,
                "attn_n_heads": n_heads,
                "attn_lr": lr,
                "nn_backbone_layers": [128, 64],
            },
            "n_trials": 30,
            "elapsed_seconds": 1234.5,
        }
    }
    path = tmp_path / f"tune_nn_{pos.lower()}_results.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_merge_results_combines_multiple_positions(tmp_path):
    p1 = _write_pos_result(tmp_path, "QB", 100.5, 32, 2, 0.001)
    p2 = _write_pos_result(tmp_path, "RB", 80.2, 24, 4, 0.0005)
    merged = aggregate_results._merge_results([p1, p2])
    assert set(merged) == {"QB", "RB"}
    assert merged["QB"]["best_val_loss"] == 100.5
    assert merged["RB"]["best_params"]["attn_d_model"] == 24


def test_merge_results_skips_non_dict_payloads(tmp_path, capsys):
    """Malformed file (not a top-level dict) is warned about, not raised."""
    bad = tmp_path / "tune_nn_qb_results.json"
    bad.write_text(json.dumps(["a", "b"]))  # list, not dict
    good = _write_pos_result(tmp_path, "RB", 80.0, 24, 2, 0.001)
    merged = aggregate_results._merge_results([str(bad), good])
    assert "QB" not in merged
    assert "RB" in merged
    assert "skipping" in capsys.readouterr().out


def test_format_markdown_summary_includes_each_position():
    merged = {
        "QB": {
            "best_val_loss": 100.5,
            "best_trial": 7,
            "n_trials": 30,
            "elapsed_seconds": 1234.5,
            "best_params": {"attn_d_model": 32, "attn_n_heads": 2, "attn_lr": 0.001},
        },
        "RB": {
            "best_val_loss": 80.2,
            "best_trial": 12,
            "n_trials": 30,
            "elapsed_seconds": 1000.0,
            "best_params": {"attn_d_model": 24, "attn_n_heads": 4, "attn_lr": 0.0005},
        },
    }
    s = aggregate_results._format_markdown_summary(merged)
    assert "| QB |" in s
    assert "| RB |" in s
    assert "100.5" in s
    # Markdown table header is present.
    assert "| Position |" in s
    # Compact param fingerprint format: "d_model / n_heads / lr".
    assert "32 / 2 / 0.001" in s
    assert "24 / 4 / 0.0005" in s


def test_format_markdown_summary_handles_empty_input():
    s = aggregate_results._format_markdown_summary({})
    assert "No per-position results" in s


def test_collect_local_files_skips_missing(tmp_path, capsys):
    """Position with no file is skipped with a warning, not an error."""
    _write_pos_result(tmp_path, "QB", 100.0, 32, 2, 0.001)
    paths = aggregate_results._collect_local_files(str(tmp_path), ["QB", "RB"])
    assert len(paths) == 1
    assert "qb" in paths[0]
    assert "WARNING" in capsys.readouterr().out


def test_download_skips_404(tmp_path, capsys):
    """If S3 returns 404 for one position the aggregator continues rather
    than aborting — a single Spot failure shouldn't block the rest."""
    from botocore.exceptions import ClientError

    fake_s3 = type("FakeS3", (), {})()
    calls = []

    def _download(bucket, key, local_path):
        calls.append(key)
        if "qb" in key:
            # Pretend QB's file is missing.
            raise ClientError({"Error": {"Code": "404"}}, "GetObject")
        # Write a stub for RB so the function returns it.
        with open(local_path, "w") as f:
            json.dump({"RB": {"best_val_loss": 1.0}}, f)

    fake_s3.download_file = _download

    with patch("boto3.client", return_value=fake_s3):
        paths = aggregate_results._download_from_s3("bucket", ["QB", "RB"], str(tmp_path))
    assert len(paths) == 1
    assert "rb" in paths[0]
    out = capsys.readouterr().out
    assert "no results at s3" in out


def test_main_writes_merged_and_summary(tmp_path, monkeypatch):
    """End-to-end --no-s3 happy path: writes the merged JSON + GITHUB_STEP_SUMMARY."""
    _write_pos_result(tmp_path, "QB", 100.0, 32, 2, 0.001)
    _write_pos_result(tmp_path, "RB", 80.0, 24, 4, 0.0005)

    output = tmp_path / "out.json"
    summary = tmp_path / "summary.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_results",
            "--positions",
            "QB",
            "RB",
            "--no-s3",
            "--local-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )
    aggregate_results.main()

    assert output.exists()
    merged = json.loads(output.read_text())
    assert set(merged) == {"QB", "RB"}

    assert summary.exists()
    text = summary.read_text()
    assert "| QB |" in text
    assert "| RB |" in text

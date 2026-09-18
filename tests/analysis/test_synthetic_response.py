"""Paired model-response reports over hand-built replay directories; no models."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from src.analysis.synthetic_response import build_report, main, pair_replays, render_markdown
from src.shared.aggregate_targets import predictions_to_fantasy_points

pytestmark = pytest.mark.unit

TARGETS = [
    "passing_yards",
    "rushing_yards",
    "passing_tds",
    "rushing_tds",
    "interceptions",
    "fumbles_lost",
]
ACCURACY_WORDS = re.compile(r"\b(mae|rmse|r2|error|bias|actual)\b", re.IGNORECASE)


def _replay(
    directory,
    *,
    name,
    offset=0.0,
    identity="same",
    source="src",
    families=("attn_nn",),
    totals=True,
    excluded=None,
):
    directory.mkdir(parents=True)
    rows = []
    for index in range(4):
        row = {
            "case_id": f"c{index}",
            "case_index": index,
            "donor_player_id": f"p{index % 2}",
            "donor_season": 2022,
            "forecast_week": 5 + index,
            "donor_history_ppg": 18.0 + index,
            "generated_history_ppg": 18.0 + index + offset,
        }
        for family in families:
            for target in TARGETS:
                row[f"pred_{family}_{target}"] = float(index + 1) + offset * (
                    1 if target == "passing_yards" else 0
                )
            raw = {t: np.array([row[f"pred_{family}_{t}"]]) for t in TARGETS}
            if totals:
                row[f"pred_{family}_total"] = float(
                    predictions_to_fantasy_points("QB", raw, "ppr")[0]
                )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(directory / "predictions.parquet", index=False)
    manifest = {
        "recipe": {"position": "QB", "name": name, "mode": "replay"},
        "cohort_name": name,
        "recipe_sha256": name,
        "sampling_identity_sha256": identity,
        "source_values_sha256": source,
        "cohort_manifest_sha256": f"{name}-manifest",
        "history_kind": "transformed" if offset else "donor",
        "fixture": bool(offset),
        "families_excluded": excluded or {},
    }
    (directory / "replay_manifest.json").write_text(json.dumps(manifest))
    return directory


def test_pairs_by_case_index_and_summarizes_the_response(tmp_path):
    baseline = _replay(tmp_path / "base", name="base")
    treatment = _replay(tmp_path / "treat", name="treat", offset=10.0)
    report = build_report(baseline, treatment)
    assert report["report_kind"] == "synthetic_model_response"
    assert report["pairing"]["n_pairs"] == 4 and report["pairing"]["unique_donor_windows"] == 4
    family = report["families"]["attn_nn"]
    assert family["n"] == 4
    delta = family["delta_total"]
    assert delta["mean"] == pytest.approx(0.4) and delta["median"] == pytest.approx(0.4)
    assert delta["std"] == pytest.approx(0.0) and delta["share_positive"] == 1.0
    assert delta["sign_consistency"] == 1.0
    assert family["delta_targets"]["passing_yards"]["mean"] == pytest.approx(10.0)
    assert family["delta_targets"]["rushing_yards"]["mean"] == pytest.approx(0.0)
    assert report["history_ppg"]["generated_delta_mean"] == pytest.approx(10.0)
    assert report["treatment"]["fixture"] is True and report["baseline"]["history_kind"] == "donor"


def test_refuses_different_donors_or_sources(tmp_path):
    baseline = _replay(tmp_path / "base", name="base")
    with pytest.raises(ValueError, match="not sampled from the same donors"):
        pair_replays(_load(baseline), _load(_replay(tmp_path / "other", name="o", identity="x")))
    with pytest.raises(ValueError, match="same source values"):
        pair_replays(_load(baseline), _load(_replay(tmp_path / "src", name="s", source="y")))


def _load(directory):
    from src.analysis.synthetic_response import load_replay

    return load_replay(directory)


def test_non_ppr_totals_are_recomputed_through_shared_scoring(tmp_path):
    baseline = _replay(tmp_path / "base", name="base", totals=False)
    treatment = _replay(tmp_path / "treat", name="treat", offset=10.0, totals=False)
    with pytest.raises(ValueError, match="share no replayed model family"):
        build_report(baseline, treatment)
    baseline = _replay(tmp_path / "base2", name="base")
    treatment = _replay(tmp_path / "treat2", name="treat", offset=10.0)
    report = build_report(baseline, treatment, scoring="standard")
    assert report["scoring_format"] == "standard"
    assert report["families"]["attn_nn"]["delta_total"]["mean"] == pytest.approx(0.4)


def test_report_never_uses_accuracy_vocabulary(tmp_path):
    baseline = _replay(tmp_path / "base", name="base", excluded={"ridge": "history transformed"})
    treatment = _replay(tmp_path / "treat", name="treat", offset=10.0)
    report = build_report(baseline, treatment)
    markdown = render_markdown(report)
    assert not ACCURACY_WORDS.search(json.dumps(report))
    assert not ACCURACY_WORDS.search(markdown)
    # The word appears only in the disclaimer that says what the report is not.
    without_disclaimer = json.dumps({k: v for k, v in report.items() if k != "semantics"})
    assert "accuracy" not in without_disclaimer
    assert markdown.count("accuracy") == 1
    assert report["excluded_families"] == {"ridge": "history transformed"}
    assert "| Attention NN | 4 |" in markdown


def test_cli_writes_json_and_markdown_and_never_overwrites(tmp_path, capsys):
    baseline = _replay(tmp_path / "base", name="base")
    treatment = _replay(tmp_path / "treat", name="treat", offset=10.0)
    output = tmp_path / "report"
    argv = ["--baseline", str(baseline), "--treatment", str(treatment), "--output", str(output)]
    assert main(argv) == 0
    assert capsys.readouterr().out.startswith("# Model response: treat vs base")
    report = json.loads((output / "response_report.json").read_text())
    assert report["families"]["attn_nn"]["delta_total"]["mean"] == pytest.approx(0.4)
    assert (output / "response_report.md").read_text().startswith("# Model response")
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert main(argv[:-2]) == 0

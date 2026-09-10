"""Native K/DST data must reach diagnostic feature and artifact paths."""

import pandas as pd
import pytest

from src.analysis import position_data
from src.config import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS

pytestmark = pytest.mark.unit


def _frames(position, value=1.0):
    return tuple(
        pd.DataFrame(
            {
                "player_id": [position],
                "position": [position],
                "season": [season],
                "week": [1],
                "fantasy_points": [value],
                "native_value": [value],
            }
        )
        for season in (TRAIN_SEASONS[-1], VAL_SEASONS[0], TEST_SEASONS[0])
    )


@pytest.mark.parametrize("position", ["K", "DST"])
def test_native_loader_derives_real_targets_before_splitting(monkeypatch, position):
    import importlib

    data = importlib.import_module(f"src.{position.lower()}.data")
    features = importlib.import_module(f"src.{position.lower()}.features")
    raw = pd.concat(_frames(position), ignore_index=True)
    games = getattr(data, "MIN_GAMES", 1)
    raw = pd.concat([raw.assign(week=week) for week in range(1, games + 1)], ignore_index=True)
    if position == "K":
        raw = raw.assign(fg_yards_made=70.0, pat_made=2.0, fg_missed=1.0, pat_missed=0.0)
        expected_points = 8.0
    else:
        raw = raw.assign(
            points_allowed=21.0,
            yards_allowed=350.0,
            def_sacks=2.0,
            def_ints=1.0,
            def_fumble_rec=0.0,
            def_fumbles_forced=0.0,
            def_safeties=0.0,
            def_tds=0.0,
            def_blocked_kicks=0.0,
            special_teams_tds=0.0,
        )
        expected_points = 3.0
    monkeypatch.setattr(data, "load_data" if position == "K" else "build_data", lambda: raw)

    def compute_features(frame):
        assert len(frame) == 3 * games  # features see full history before splitting
        assert frame["fantasy_points"].eq(expected_points).all()
        frame["feature_ready"] = True

    monkeypatch.setattr(features, "compute_features", compute_features)
    frames = position_data.load_position_frames(position)
    for frame, season in zip(
        frames, (TRAIN_SEASONS[-1], VAL_SEASONS[0], TEST_SEASONS[0]), strict=True
    ):
        assert frame["season"].tolist() == [season] * games
        assert frame["feature_ready"].all()
        assert frame["fantasy_points"].eq(expected_points).all()


@pytest.mark.parametrize("position", ["K", "DST"])
def test_covariate_report_uses_native_frames(monkeypatch, position):
    from src.analysis import covariate_shift as shift
    from src.shared import feature_build, registry

    native = _frames(position, 17.0)
    monkeypatch.setattr(position_data, "load_position_frames", lambda pos: native)

    def ordinary_read(path):
        raise AssertionError("native report must not consume ordinary player splits")

    monkeypatch.setattr(shift, "_load_split", ordinary_read)
    cfg = {
        "filter_fn": lambda frame: frame,
        "compute_targets_fn": lambda frame: frame,
        "get_feature_columns_fn": lambda: ["native_value"],
        "min_games_per_season": 1,
    }
    monkeypatch.setattr(registry, "get_config", lambda pos: cfg)
    monkeypatch.setattr(
        feature_build, "build_position_features", lambda tr, va, te, *a, **kw: (tr, va, te)
    )
    report = shift.shift_report_for_position(position)
    assert (report["n_train"], report["n_val"], report["n_test"]) == (1, 1, 1)
    assert report["features"][0]["mean_z"] == 0.0


@pytest.mark.parametrize("position", ["K", "DST"])
def test_artifact_cli_routes_native_frames_without_ordinary_splits(monkeypatch, position):
    from src.analysis import artifact_eval, cohort_analysis

    native = _frames(position)
    monkeypatch.setattr(position_data, "load_position_frames", lambda pos: native)

    def ordinary_read():
        raise AssertionError("native CLI must not need ordinary player splits")

    monkeypatch.setattr(cohort_analysis, "_load_splits", ordinary_read)
    calls = []

    def build(pos, train, val, test, **kwargs):
        assert pos == position
        assert train is native[0] and val is native[1] and test is native[2]
        calls.append(pos)
        return test

    monkeypatch.setattr(artifact_eval, "build_test_df_from_artifacts", build)
    artifact_eval._main(["--positions", position])
    assert calls == [position]


@pytest.mark.parametrize("position", ["K", "DST"])
def test_topn_report_passes_native_frames_to_artifact_path(monkeypatch, tmp_path, position):
    from src.analysis import cohort_analysis
    from src.analysis import topn_expert_gap as gap

    native = _frames(position, 17.0)
    monkeypatch.setattr(cohort_analysis, "_load_splits", lambda: _frames("QB"))
    monkeypatch.setattr(position_data, "load_position_frames", lambda pos: native)
    monkeypatch.setattr(gap, "build_expert_sources", lambda *args: [])
    monkeypatch.setattr(gap, "render_summary", lambda *args: "test report")
    calls = []

    def load(pos, train, val, test, **kwargs):
        assert train is native[0] and val is native[1] and test is native[2]
        calls.append(pos)
        return gap.PredictionLoad(df=test, mode="artifacts")

    def report(pos, frame, **kwargs):
        assert position in kwargs["min_season"].index
        return [], [], [], []

    monkeypatch.setattr(gap, "load_position_predictions", load)
    monkeypatch.setattr(gap, "build_position_report", report)
    result = gap.run_analysis(positions=[position], out_dir=tmp_path)
    assert calls == [position]
    assert result["prediction_loads"][position]["mode"] == "artifacts"


@pytest.mark.parametrize("position", ["K", "DST"])
def test_tier_report_uses_native_history_for_artifacts_and_prior(monkeypatch, position):
    from src.analysis import artifact_eval
    from src.analysis import tier_expert_comparison as tier

    native = _frames(position, 17.0)
    monkeypatch.setattr(tier, "_load_splits", lambda: _frames("QB", 0.0))
    monkeypatch.setattr(tier, "load_position_frames", lambda pos: native)
    monkeypatch.setattr(tier, "_build_experts", lambda *args: [])
    calls = []

    def build(pos, train, val, test, **kwargs):
        assert train is native[0] and val is native[1] and test is native[2]
        return test

    def compare(pos, frame, prior, *args, **kwargs):
        calls.append(pos)
        assert prior.loc[(position, TEST_SEASONS[0])] == 17.0

    monkeypatch.setattr(artifact_eval, "build_test_df_from_artifacts", build)
    monkeypatch.setattr(tier, "compare_position", compare)
    tier.main(["--positions", position, "--from-artifacts"])
    assert calls == [position]


def test_native_ablation_preserves_full_kick_history_and_config(monkeypatch):
    from src.k import data as kicker_data
    from src.shared import registry

    full = pd.concat(_frames("K"), ignore_index=True)
    splits = _frames("K")
    kicks = pd.DataFrame({"kick_distance": [30.0, 40.0]})
    cfg = {"attn_kick_stats": ["kick_distance"], "attn_max_games": 2, "attn_max_kicks_per_game": 3}
    monkeypatch.setattr(position_data, "_native_frame", lambda pos: full)
    monkeypatch.setattr(position_data, "_native_splits", lambda pos, frame: splits)
    monkeypatch.setattr(registry, "get_config", lambda pos: cfg)

    def load_kicks(frame):
        assert frame is full  # before the K training-row filter, not concatenated splits
        return kicks

    monkeypatch.setattr(kicker_data, "load_kicks", load_kicks)
    actual_splits, actual_cfg = position_data.prepare_native_ablation("K")
    builder = actual_cfg["attn_history_builder_fn"]
    assert actual_splits is splits
    assert builder.keywords["kicks_df"] is kicks
    assert builder.keywords["max_games"] == 2
    assert builder.keywords["max_kicks_per_game"] == 3
    assert "attn_history_builder_fn" not in cfg


@pytest.mark.parametrize("position", ["QB", "RB", "WR", "TE", "K", "DST"])
def test_ablation_runs_keep_and_cut_on_identical_test_rows(monkeypatch, position):
    from src.analysis import cohort_analysis
    from src.shared import pipeline, registry

    frames = tuple(
        pd.concat([frame, frame.assign(week=18)], ignore_index=True) for frame in _frames(position)
    )
    cfg = {"native": position}
    monkeypatch.setattr(position_data, "prepare_native_ablation", lambda pos: (frames, cfg))
    monkeypatch.setattr(
        pipeline,
        "_read_split",
        lambda path: frames[
            {"train.parquet": 0, "val.parquet": 1, "test.parquet": 2}[path.rsplit("/", 1)[-1]]
        ],
    )
    calls = []

    def run(train, val, test, seed=42):
        calls.append((train, val, test, seed))
        return {"test_df": test.assign(pred_ridge_total=test["fantasy_points"] + 1)}

    def native_run(pos, config, train, val, test, seed=42):
        assert pos == position and config is cfg
        return run(train, val, test, seed)

    monkeypatch.setattr(pipeline, "run_pipeline", native_run)
    monkeypatch.setattr(registry, "get_runner", lambda pos: run)
    cohort_analysis.run_ablation([position], "/fixture", 1, 17, seed=7)
    assert len(calls) == 2
    assert calls[0][0]["week"].tolist() == [1, 18]
    assert calls[1][0]["week"].tolist() == [1]
    assert calls[0][1]["week"].tolist() == [1, 18]
    assert calls[1][1]["week"].tolist() == [1]
    assert calls[0][2] is calls[1][2] is frames[2]
    assert calls[0][3] == calls[1][3] == 7


def test_cohort_ablation_cli_forwards_seed(monkeypatch):
    from src.analysis import cohort_analysis

    calls = []
    monkeypatch.setattr(
        cohort_analysis, "run_ablation", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    cohort_analysis.main(["late_week", "--ablation", "--positions", "K", "--seed", "7"])
    assert calls[0][0][0] == ["K"]
    assert calls[0][1] == {"seed": 7}

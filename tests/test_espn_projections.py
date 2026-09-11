"""Historical archive, identity and scoring contracts for ESPN comparisons."""

import copy
import re

import numpy as np
import pandas as pd
import pytest

from src.serving import espn_projections as mod

pytestmark = pytest.mark.unit


def _split(stats, *, season=2025, week=1, source=1, split=1):
    return {
        "seasonId": season,
        "scoringPeriodId": week,
        "statSourceId": source,
        "statSplitTypeId": split,
        "stats": stats,
        "appliedTotal": 999.0,
    }


def _player(pos, splits, pid=123, team=14):
    return {
        "player": {
            "id": pid,
            "fullName": "Historical Player",
            "proTeamId": team,
            "defaultPositionId": pos,
            "stats": splits,
        }
    }


def _joined(pos, stats):
    frame = mod._normalize_season({"players": [_player(pos, [_split(stats)])]}, 2025)
    frame["player_id"] = "P1"
    return frame


@pytest.mark.parametrize("fmt,rec_weight", [("ppr", 1), ("half_ppr", 0.5), ("standard", 0)])
@pytest.mark.parametrize("pos", [2, 3, 4])
def test_offense_preserves_raw_stats_in_every_scoring_format(fmt, rec_weight, pos):
    frame = _joined(
        pos, {"24": 50, "25": 0.5, "42": 20, "43": 0.25, "53": 4, "41": 999, "72": 0.2, "62": 2}
    )
    result = mod.project_espn_to_fantasy(frame, mod._POS_MAP[pos], fmt)
    expected = 2 + 1.5 - 0.4 + 4 * rec_weight + 8
    assert result.espn_pred_total.iloc[0] == pytest.approx(expected)


def test_qb_interceptions_are_stat20_not_two_point_conversions():
    frame = _joined(1, {"3": 250, "4": 2, "19": 9, "20": 1, "24": 30, "25": 1, "72": 0.2})
    assert mod.project_espn_to_fantasy(frame, "QB", "ppr").espn_pred_total.iloc[0] == pytest.approx(
        24.6
    )


def test_kicker_uses_made_yardage_and_signed_misses():
    frame = _joined(5, {"214": 40, "215": 99, "216": 139, "86": 2, "85": 0.5, "88": 0.1})
    assert mod.project_espn_to_fantasy(frame, "K", "ppr").espn_pred_total.iloc[0] == pytest.approx(
        5.4
    )
    assert _joined(5, {"83": 2, "86": 2}).empty  # bins/counts cannot substitute for yardage


def test_dst_touchdowns_are_not_double_counted_and_teams_are_normalized():
    frame = _joined(
        16,
        {
            "120": 10,
            "127": 250,
            "99": 2,
            "95": 1,
            "94": 0.1,
            "93": 0.2,
            "101": 0.3,
            "102": 0.4,
            "105": 99,
        },
    )
    assert frame.espn_id.iloc[0] == "LA"
    assert frame.def_tds.iloc[0] == pytest.approx(0.3)
    assert frame.special_teams_tds.iloc[0] == pytest.approx(0.7)
    # 2 sacks + 2 INT points + 6 TD points + 4 PA bonus + 2 YA bonus.
    assert mod.project_espn_to_fantasy(frame, "DST", "ppr").espn_pred_total.iloc[
        0
    ] == pytest.approx(16)


def test_filters_actuals_season_totals_wrong_year_playoffs_and_placeholders():
    splits = [
        _split({"3": 250}),
        _split({"3": 400}, source=0),
        _split({"3": 4000}, split=0),
        _split({"3": 300}, season=2024),
        _split({"3": 300}, week=0),
        _split({"3": 300}, week=19),
        _split({"210": 1}, week=2),
        _split({"3": 0}, week=3),
    ]
    frame = mod._normalize_season({"players": [_player(1, splits)]}, 2025)
    assert len(frame) == 1
    assert frame.passing_yards.iloc[0] == 250


def test_missing_stock_total_does_not_discard_a_real_raw_projection():
    payload = {"players": [_player(1, [_split({"3": 250})])]}
    payload["players"][0]["player"]["stats"][0]["appliedTotal"] = 0
    assert len(mod._normalize_season(payload, 2025)) == 1


def test_receiving_only_running_back_is_a_genuine_projection():
    frame = _joined(2, {"42": 20, "53": 3})
    assert mod.project_espn_to_fantasy(frame, "RB", "ppr").espn_pred_total.iloc[0] == 5
    assert _joined(2, {"101": 0.1, "114": 20, "210": 1}).empty  # return-only placeholder


@pytest.mark.parametrize(
    "pos,stats,expected", [(3, {"24": 20, "25": 0.5}, 5), (1, {"42": 20, "53": 3}, 5)]
)
def test_projection_outside_modeled_heads_is_not_a_placeholder(pos, stats, expected):
    frame = _joined(pos, stats)
    assert (
        mod.project_espn_to_fantasy(frame, mod._POS_MAP[pos], "ppr").espn_pred_total.iloc[0]
        == expected
    )


def test_old_raw_cache_cannot_hide_rushing_only_receivers(tmp_path):
    pd.DataFrame({"espn_id": ["stale"]}).to_parquet(tmp_path / "espn_projections_v1_2025.parquet")
    raw = mod.load_espn_projections(
        [2025],
        str(tmp_path),
        reader=lambda *a, **k: {"players": [_player(3, [_split({"24": 20, "25": 0.5})])]},
    )
    assert raw.iloc[0].rushing_yards == 20
    assert (tmp_path / "espn_projections_v2_2025.parquet").exists()


def test_excludes_entire_incomplete_week_and_old_week18():
    for year, excluded in [(2023, 1), (2020, 18)]:
        frame = mod._normalize_season(
            {
                "players": [
                    _player(
                        1,
                        [
                            _split({"3": 250}, season=year, week=excluded),
                            _split({"3": 260}, season=year, week=2),
                        ],
                    )
                ]
            },
            year,
        )
        assert frame.week.tolist() == [2]


def test_bad_payloads_fail_without_caching(tmp_path):
    with pytest.raises(ValueError, match="players"):
        mod.load_espn_projections([2025], str(tmp_path), reader=lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="No ESPN"):
        mod.load_espn_projections([2025], str(tmp_path), reader=lambda *a, **k: {"players": []})
    assert not list(tmp_path.glob("*.parquet"))
    for stats in ({"3": float("nan")}, {"3": "broken"}):
        with pytest.raises(ValueError):
            mod._normalize_season({"players": [_player(1, [_split(stats)])]}, 2025)
    player = _player(1, [_split({"3": 250})])
    with pytest.raises(ValueError, match="Duplicate"):
        mod._normalize_season({"players": [player, copy.deepcopy(player)]}, 2025)


def test_per_season_cache_and_week_filter_do_not_poison_later_requests(tmp_path):
    calls = []

    def reader(url, headers):
        year = int(re.search(r"/seasons/(\d+)", url)[1])
        calls.append(year)
        assert "X-Fantasy-Filter" in headers
        return {"players": [_player(1, [_split({"3": 250}, season=year, week=w) for w in (1, 2)])]}

    first = mod.load_espn_projections([2024, 2025], str(tmp_path), weeks=[2], reader=reader)
    assert set(first.week) == {2}
    full = mod.load_espn_projections([2025, 2023, 2024], str(tmp_path), reader=reader)
    assert sorted(calls) == [2023, 2024, 2025]
    assert len(full) == 5  # 2023 Week 1 excluded
    mod.load_espn_projections([2025], str(tmp_path), force_refresh=True, reader=reader)
    assert calls.count(2025) == 2


def test_crosswalk_handles_float_ids_retired_players_and_unmatched_rows(tmp_path):
    payload = {
        "players": [
            _player(1, [_split({"3": 250})], pid=123),
            _player(1, [_split({"3": 200})], pid=456),
            _player(16, [_split({"120": 20, "127": 300})]),
        ]
    }
    result = mod.load_espn_with_gsis_id(
        [np.int64(2025)],
        str(tmp_path),
        reader=lambda *a, **k: payload,
        player_ids_loader=lambda: pd.DataFrame({"espn_id": [123.0], "gsis_id": ["retired-id"]}),
    )
    assert set(result.player_id) == {"retired-id", "LA"}


@pytest.mark.parametrize(
    "seasons,weeks", [([], None), ([2017], None), ([2025], []), ([2025], [19])]
)
def test_invalid_season_and_week_requests(seasons, weeks, tmp_path):
    with pytest.raises(ValueError):
        mod.load_espn_projections(seasons, str(tmp_path), weeks=weeks)

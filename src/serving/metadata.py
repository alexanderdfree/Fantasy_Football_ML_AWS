"""Static position metadata for the serving UI.

``POSITION_INFO`` is the single source of truth for each position's display
label, raw-stat target identity + display order, scoring formulas, and the NN
architecture summary. It drives ``/api/position_details``,
``/api/model_architecture`` and the per-target breakdown columns. Extracted from
``app.py`` during the serving decomposition; ``app.py`` re-exports these names so
``src.serving.app.POSITION_INFO`` / ``_ALL_TARGETS`` access keeps working.
"""

import src.dst.config as dst_cfg
import src.k.config as k_cfg
import src.qb.config as qb_cfg
import src.rb.config as rb_cfg
import src.te.config as te_cfg
import src.wr.config as wr_cfg

POSITION_INFO = {
    "QB": {
        "label": "Quarterback",
        "targets": [
            {"key": "passing_yards", "label": "Passing Yards", "formula": "raw passing yards"},
            {"key": "rushing_yards", "label": "Rushing Yards", "formula": "raw rushing yards"},
            {"key": "passing_tds", "label": "Passing TDs", "formula": "raw passing TD count"},
            {"key": "rushing_tds", "label": "Rushing TDs", "formula": "raw rushing TD count"},
            {"key": "interceptions", "label": "Interceptions", "formula": "raw interception count"},
            {
                "key": "fumbles_lost",
                "label": "Fumbles Lost",
                "formula": ("sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost"),
            },
        ],
        "adjustments": "None - penalties are now direct targets (interceptions, fumbles_lost).",
        "specific_features": qb_cfg.POSITION_CONFIG.specific_features,
        "architecture": {
            "backbone": list(qb_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": qb_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
    "RB": {
        "label": "Running Back",
        "targets": [
            {"key": "rushing_tds", "label": "Rushing TDs", "formula": "raw rushing TD count"},
            {"key": "receiving_tds", "label": "Receiving TDs", "formula": "raw receiving TD count"},
            {"key": "rushing_yards", "label": "Rushing Yards", "formula": "raw rushing yards"},
            {
                "key": "receiving_yards",
                "label": "Receiving Yards",
                "formula": "raw receiving yards",
            },
            {"key": "receptions", "label": "Receptions", "formula": "raw reception count"},
            {
                "key": "fumbles_lost",
                "label": "Fumbles Lost",
                "formula": ("sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost"),
            },
        ],
        "adjustments": "None - fumbles_lost is now a direct target.",
        "specific_features": list(rb_cfg.POSITION_CONFIG.specific_features),
        "architecture": {
            "backbone": list(rb_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": rb_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
    "WR": {
        "label": "Wide Receiver",
        "targets": [
            {"key": "receiving_tds", "label": "Receiving TDs", "formula": "raw receiving TD count"},
            {
                "key": "receiving_yards",
                "label": "Receiving Yards",
                "formula": "raw receiving yards",
            },
            {"key": "receptions", "label": "Receptions", "formula": "raw reception count"},
            {
                "key": "fumbles_lost",
                "label": "Fumbles Lost",
                "formula": ("sack_fumbles_lost + rushing_fumbles_lost + receiving_fumbles_lost"),
            },
        ],
        "adjustments": "None - fumbles_lost is now a direct target.",
        "specific_features": list(wr_cfg.POSITION_CONFIG.specific_features),
        "architecture": {
            "backbone": list(wr_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": wr_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
    "TE": {
        "label": "Tight End",
        "targets": [
            {"key": "receiving_tds", "label": "Receiving TDs", "formula": "raw count"},
            {"key": "receiving_yards", "label": "Receiving Yards", "formula": "raw count"},
            {"key": "receptions", "label": "Receptions", "formula": "raw count"},
            {"key": "fumbles_lost", "label": "Fumbles Lost", "formula": "raw count"},
        ],
        "adjustments": "None - fumbles_lost is now a direct target.",
        "specific_features": list(te_cfg.POSITION_CONFIG.specific_features),
        "architecture": {
            "backbone": list(te_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": te_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
    "K": {
        "label": "Kicker",
        "targets": [
            {
                "key": "fg_yard_points",
                "label": "FG Yard Points",
                "formula": "FG yards made × 0.1",
            },
            {"key": "pat_points", "label": "PAT Points", "formula": "PAT made × 1"},
            {
                "key": "fg_misses",
                "label": "FG Misses",
                "formula": "FG missed (−1 each in total)",
            },
            {
                "key": "xp_misses",
                "label": "XP Misses",
                "formula": "PAT missed (−1 each in total)",
            },
        ],
        "adjustments": "None",
        "formula": "fg_yard_points + pat_points − fg_misses − xp_misses",
        "specific_features": list(k_cfg.POSITION_CONFIG.specific_features),
        "architecture": {
            "backbone": list(k_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": k_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
    "DST": {
        "label": "Defense/Special Teams",
        "targets": [
            {"key": "def_sacks", "label": "Sacks", "formula": "sacks x 1"},
            {"key": "def_ints", "label": "Interceptions", "formula": "INT x 2"},
            {"key": "def_fumble_rec", "label": "Fumble Recoveries", "formula": "fum_rec x 2"},
            {"key": "def_fumbles_forced", "label": "Forced Fumbles", "formula": "forced_fum x 1"},
            {"key": "def_safeties", "label": "Safeties", "formula": "safeties x 2"},
            {"key": "def_tds", "label": "Defensive TDs", "formula": "def_TD x 6"},
            {"key": "def_blocked_kicks", "label": "Blocked Kicks", "formula": "blocked x 2"},
            {"key": "special_teams_tds", "label": "Special Teams TDs", "formula": "ST_TD x 6"},
            {
                "key": "points_allowed",
                "label": "Points Allowed",
                "formula": (
                    "raw PA, tier-mapped at inference "
                    "(0=+10, 1-6=+7, 7-13=+4, 14-20=+1, 21-27=0, 28-34=-1, 35+=-4)"
                ),
            },
            {
                "key": "yards_allowed",
                "label": "Yards Allowed",
                "formula": (
                    "raw YA, tier-mapped at inference "
                    "(<100=+5, 100-199=+3, 200-299=+2, 300-349=0, 350-399=-1, 400-449=-3, 450+=-5)"
                ),
            },
        ],
        "adjustments": "None (PA/YA tier bonuses applied at inference to regressed raw values)",
        "formula": (
            "def_sacks*1 + def_ints*2 + def_fumble_rec*2 + def_fumbles_forced*1 "
            "+ def_safeties*2 + def_tds*6 + def_blocked_kicks*2 + special_teams_tds*6 "
            "+ tier_pa(points_allowed) + tier_ya(yards_allowed)"
        ),
        "specific_features": list(dst_cfg.POSITION_CONFIG.specific_features),
        "architecture": {
            "backbone": list(dst_cfg.POSITION_CONFIG.nn_backbone_layers),
            "head_hidden": dst_cfg.POSITION_CONFIG.nn_head_hidden,
        },
    },
}


# Union of every position's raw-stat target keys (deduped, sorted). Drives the
# per-target breakdown columns pre-declared in the results frame so they survive
# the parquet persist/hydrate round-trip. POSITION_INFO is the single source for
# target identity + display order used by /api/predictions/breakdown.
_ALL_TARGETS = sorted({t["key"] for info in POSITION_INFO.values() for t in info["targets"]})


# Canonical position order for the UI / orchestration loops.
_ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]

# Positions sourced from their own dedicated splits (``_load_k_splits`` /
# ``_load_dst_splits``) and appended to ``results`` separately in
# ``_load_base_data_locked``. They MUST be excluded from the skill-position
# ``test.parquet`` base copy: kickers ALSO appear in the offensive player table
# (with ~0 offensive fantasy_points), so copying them there would double every
# kicker row once the K split is appended — a phantom twin with actual≈0 and
# null preds. See TODO.md Fixed archive ("Kicker rows duplicated in serving").
_APPENDED_POSITIONS = ("K", "DST")

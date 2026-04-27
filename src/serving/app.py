"""Flask web application for the Fantasy Football Points Predictor.

All predictions come from position-specific models (QB, RB, WR, TE, K, DST).
No general cross-position model is used.
"""

import os
import re
import sys
import threading

import markdown

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")

import traceback

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

import src.dst.config as dst_cfg
import src.k.config as k_cfg
import src.qb.config as qb_cfg
import src.rb.config as rb_cfg
import src.te.config as te_cfg
import src.wr.config as wr_cfg
from src.config import SCORING_HALF_PPR, SCORING_STANDARD, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.data.loader import compute_fantasy_points
from src.dst.config import DST_SPECIFIC_FEATURES, DST_TARGETS
from src.dst.data import build_dst_data
from src.dst.features import compute_dst_features
from src.evaluation.metrics import compute_metrics, compute_positional_metrics
from src.features.engineer import (
    build_game_history_arrays,
    build_opp_defense_history_arrays,
    build_opp_defense_per_game_df,
    get_attn_static_columns,
)
from src.k.config import K_SPECIFIC_FEATURES, K_TARGETS

# Per-position imports needed only for /api/model_architecture metadata and
# for K/DST data loaders (these have their own data pipelines, not in registry).
from src.k.data import kicker_season_split, load_kicker_data, load_kicker_kicks
from src.k.features import build_nested_kick_history, compute_k_features
from src.qb.config import QB_SPECIFIC_FEATURES, QB_TARGETS
from src.rb.config import RB_SPECIFIC_FEATURES, RB_TARGETS
from src.shared.aggregate_targets import predictions_to_fantasy_points
from src.shared.artifact_integrity import (
    assert_scaler_matches,
    read_scaler_meta,
    unwrap_state_dict,
)
from src.shared.feature_build import build_position_features, scale_and_clip
from src.shared.model_sync import sync_data_from_s3, sync_models_from_s3
from src.shared.models import LightGBMMultiTarget, RidgeMultiTarget
from src.shared.neural_net import (
    MultiHeadNet,
    MultiHeadNetWithHistory,
    MultiHeadNetWithNestedHistory,
)
from src.shared.registry import INFERENCE_REGISTRY as POSITION_REGISTRY
from src.shared.weather_features import WEATHER_FEATURES_ALL
from src.te.config import TE_SPECIFIC_FEATURES, TE_TARGETS
from src.wr.config import WR_SPECIFIC_FEATURES, WR_TARGETS

sync_data_from_s3()
sync_models_from_s3()

app = Flask(__name__)

_cache = {}
# Serializes lazy model/data loads — Flask dispatches requests on multiple
# threads, so two concurrent first-hit requests would otherwise both see
# _cache as empty and race on duplicate I/O plus .loc-writes into the shared
# results DataFrame. Reentrant because _ensure_metrics nests into
# _ensure_position_loaded.
_cache_lock = threading.RLock()


def _safe_num(v):
    """Convert NaN/inf to None so jsonify produces valid JSON (browsers reject NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _safe_str(v, default=""):
    """Return default for NaN/None/non-string values."""
    if v is None:
        return default
    if isinstance(v, float) and not np.isfinite(v):
        return default
    return str(v)


_VALID_SCORING = ("ppr", "half_ppr", "standard")
_MODEL_PRED_PREFIXES = ("ridge", "nn", "attn_nn", "lgbm")


def _validate_scoring(arg):
    """Return arg if it is a known scoring format, else 'ppr' (default).

    Used at every endpoint that accepts ?scoring=. Silent fallback so a stale
    bookmark doesn't error; bad clients are caught by the unit tests.
    """
    if arg in _VALID_SCORING:
        return arg
    return "ppr"


def _actual_col(fmt):
    """DataFrame column for the actual fantasy-point value in this scoring format.

    PPR is canonical and stored under 'fantasy_points' (no suffix); the other
    two are populated via compute_fantasy_points in _compute_scoring_formats.
    """
    return "fantasy_points" if fmt == "ppr" else f"fantasy_points_{fmt}"


def _pred_col(prefix, fmt):
    """DataFrame column for a model's aggregated prediction in this scoring format."""
    return f"{prefix}_pred_{fmt}"


_PLAYER_ROW_COLS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "week",
    "fantasy_points",
    "fantasy_points_half_ppr",
    "fantasy_points_standard",
    "ridge_pred",
    "nn_pred",
    "attn_nn_pred",
    "lgbm_pred",
    "ridge_pred_ppr",
    "ridge_pred_half_ppr",
    "ridge_pred_standard",
    "nn_pred_ppr",
    "nn_pred_half_ppr",
    "nn_pred_standard",
    "attn_nn_pred_ppr",
    "attn_nn_pred_half_ppr",
    "attn_nn_pred_standard",
    "lgbm_pred_ppr",
    "lgbm_pred_half_ppr",
    "lgbm_pred_standard",
    "headshot_url",
]


def _round_or_none(v):
    """Round a numeric to 2dp, returning None for NaN / missing / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 2)


def _records_to_player_rows(df, scoring="ppr"):
    cols = [c for c in _PLAYER_ROW_COLS if c in df.columns]
    actual_key = _actual_col(scoring)
    pred_keys = {prefix: _pred_col(prefix, scoring) for prefix in _MODEL_PRED_PREFIXES}
    return [
        {
            "player_id": _safe_str(r.get("player_id")),
            "name": _safe_str(r.get("player_display_name")),
            "position": _safe_str(r.get("position")),
            "team": _safe_str(r.get("recent_team")),
            "week": int(r["week"]),
            "actual": _round_or_none(r.get(actual_key)),
            "ridge_pred": _safe_num(r.get(pred_keys["ridge"])),
            "nn_pred": _safe_num(r.get(pred_keys["nn"])),
            "attn_nn_pred": _safe_num(r.get(pred_keys["attn_nn"])),
            "lgbm_pred": _safe_num(r.get(pred_keys["lgbm"])),
            "headshot": _safe_str(r.get("headshot_url", "")),
        }
        for r in df[cols].to_dict(orient="records")
    ]


@app.errorhandler(Exception)
def handle_api_error(e):
    """Return JSON errors for /api/ routes, default HTML for others."""
    if request.path.startswith("/api/"):
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    raise e


# ---------------------------------------------------------------------------
# Position model metadata (static, for the UI)
# ---------------------------------------------------------------------------
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
                "formula": "sack_fumbles_lost + rushing_fumbles_lost",
            },
        ],
        "adjustments": "None - penalties are now direct targets (interceptions, fumbles_lost).",
        "specific_features": QB_SPECIFIC_FEATURES,
        "architecture": {
            "backbone": list(qb_cfg.QB_NN_BACKBONE_LAYERS),
            "head_hidden": qb_cfg.QB_NN_HEAD_HIDDEN,
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
                "formula": "rushing_fumbles_lost + receiving_fumbles_lost",
            },
        ],
        "adjustments": "None - fumbles_lost is now a direct target.",
        "specific_features": list(RB_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(rb_cfg.RB_NN_BACKBONE_LAYERS),
            "head_hidden": rb_cfg.RB_NN_HEAD_HIDDEN,
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
                "formula": "rushing_fumbles_lost + receiving_fumbles_lost",
            },
        ],
        "adjustments": "None - fumbles_lost is now a direct target.",
        "specific_features": list(WR_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(wr_cfg.WR_NN_BACKBONE_LAYERS),
            "head_hidden": wr_cfg.WR_NN_HEAD_HIDDEN,
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
        "specific_features": list(TE_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(te_cfg.TE_NN_BACKBONE_LAYERS),
            "head_hidden": te_cfg.TE_NN_HEAD_HIDDEN,
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
        "specific_features": list(K_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(k_cfg.K_NN_BACKBONE_LAYERS),
            "head_hidden": k_cfg.K_NN_HEAD_HIDDEN,
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
        "specific_features": list(DST_SPECIFIC_FEATURES),
        "architecture": {
            "backbone": list(dst_cfg.DST_NN_BACKBONE_LAYERS),
            "head_hidden": dst_cfg.DST_NN_HEAD_HIDDEN,
        },
    },
}


# ---------------------------------------------------------------------------
# Wiki — render committed markdown docs as in-app HTML pages.
# Slug is the only public identifier; raw paths are never accepted from the
# client, so a path-traversal slug like "../etc/passwd" simply misses the
# registry and 404s. Order in the dict drives sidebar order.
# ---------------------------------------------------------------------------
WIKI_DOCS: dict[str, dict] = {
    "readme": {"name": "Project Overview", "group": "Overview", "path": "README.md"},
    "setup": {"name": "Setup & Local Run", "group": "Overview", "path": "SETUP.md"},
    "attribution": {
        "name": "Data & Tool Attribution",
        "group": "Overview",
        "path": "ATTRIBUTION.md",
    },
    "todo": {"name": "TODO & Bug Archive", "group": "Overview", "path": "TODO.md"},
    "architecture": {
        "name": "ADR-001: System Architecture",
        "group": "Architecture",
        "path": "docs/ARCHITECTURE.md",
    },
    "ec2-design": {
        "name": "EC2 Training Design",
        "group": "Architecture",
        "path": "docs/ec2_design.md",
    },
    "expert-comparison": {
        "name": "Expert Projection Comparison",
        "group": "Architecture",
        "path": "docs/expert_comparison.md",
    },
    "batch-design": {
        "name": "AWS Batch Design (standby)",
        "group": "Design History",
        "path": "docs/batch_design.md",
    },
    "design-lstm-multihead": {
        "name": "LSTM Multi-Head Proposal",
        "group": "Design History",
        "path": "docs/design_lstm_multihead.md",
    },
    "design-weather-and-odds": {
        "name": "Weather & Odds Features",
        "group": "Design History",
        "path": "docs/design_weather_and_odds.md",
    },
    "design-xgboost-ensemble": {
        "name": "XGBoost Ensemble (rejected)",
        "group": "Design History",
        "path": "docs/design_xgboost_ensemble.md",
    },
    "design-doc": {
        "name": "Design Document (rubric spec)",
        "group": "Specification",
        "path": "instructions/DESIGN_DOC.md",
    },
    "method-contracts": {
        "name": "Method Contracts",
        "group": "Specification",
        "path": "instructions/METHOD_CONTRACTS.md",
    },
    "self-assessment": {
        "name": "Self-Assessment Evidence",
        "group": "Specification",
        "path": "instructions/SELF_ASSESSMENT_EVIDENCE.md",
    },
    "infra-ec2": {
        "name": "EC2 Infrastructure",
        "group": "Infrastructure",
        "path": "infra/ec2/README.md",
    },
    "infra-aws": {
        "name": "AWS Serving Infrastructure",
        "group": "Infrastructure",
        "path": "infra/aws/README.md",
    },
}

# Reverse map: normalized repo-relative path -> slug. Used to rewrite intra-wiki
# markdown links (e.g. "[ARCH](docs/ARCHITECTURE.md)") into in-app `#wiki:slug`
# anchors so the JS can swap content without a full reload.
_WIKI_PATH_TO_SLUG = {os.path.normpath(d["path"]): slug for slug, d in WIKI_DOCS.items()}

_WIKI_HREF_RE = re.compile(r'href="([^"]+)"')


def _wiki_rewrite_href(href: str, doc_path: str) -> str:
    """Rewrite an intra-wiki markdown link to a `#wiki:slug[:anchor]` anchor.

    Leaves alone: empty hrefs, pure anchors (`#section`), absolute URLs (`http(s)://`),
    `mailto:`, and any path that doesn't resolve to a registered wiki doc.
    """
    if not href or href.startswith("#") or "://" in href or href.startswith("mailto:"):
        return href
    target, _, anchor = href.partition("#")
    if not target:
        return href
    doc_dir = os.path.dirname(doc_path)
    resolved = os.path.normpath(os.path.join(doc_dir, target))
    target_slug = _WIKI_PATH_TO_SLUG.get(resolved)
    if not target_slug:
        return href
    return f"#wiki:{target_slug}:{anchor}" if anchor else f"#wiki:{target_slug}"


def _render_wiki_doc(slug: str) -> str:
    """Return cached, rendered HTML for the doc at WIKI_DOCS[slug]."""
    cache_key = ("wiki", slug)
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
    meta = WIKI_DOCS[slug]
    # WIKI_DOCS paths are relative to repo root (e.g. "docs/ARCHITECTURE.md").
    # After the src/ migration this module lives at src/serving/app.py, so
    # repo root is two parents up. The Dockerfile copies the same .md files
    # into /app at the same relative paths, so this resolves correctly in
    # both local dev and the deployed container.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    abs_path = os.path.join(repo_root, meta["path"])
    with open(abs_path, encoding="utf-8") as f:
        text = f.read()
    html = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        output_format="html",
    )
    doc_path = meta["path"]
    html = _WIKI_HREF_RE.sub(
        lambda m: f'href="{_wiki_rewrite_href(m.group(1), doc_path)}"',
        html,
    )
    with _cache_lock:
        _cache[cache_key] = html
    return html


def _compute_scoring_formats(df):
    if "fantasy_points_standard" not in df.columns:
        df["fantasy_points_standard"] = compute_fantasy_points(df, SCORING_STANDARD)
    if "fantasy_points_half_ppr" not in df.columns:
        df["fantasy_points_half_ppr"] = compute_fantasy_points(df, SCORING_HALF_PPR)


def _load_k_splits():
    """Load kicker data with features pre-computed on full dataset.

    K uses its own data pipeline because kicking stats (FG/PAT) are only
    available for 2025 in nflverse, and uses within-season temporal splits.
    Also returns the per-kick records dataframe needed by the attention NN's
    nested kick-history builder at inference time.
    """
    k_df = load_kicker_data()
    k_df = POSITION_REGISTRY["K"]["compute_targets_fn"](k_df)
    compute_k_features(k_df)
    kicks_df = load_kicker_kicks(k_df)
    train, val, test = kicker_season_split(k_df)
    return train, val, test, kicks_df


def _load_dst_splits():
    """Load D/ST data with features pre-computed on full dataset.

    D/ST operates at team level (not player level), built from schedule
    scores and opponent offensive stats.
    """
    dst_df = build_dst_data()
    dst_df = POSITION_REGISTRY["DST"]["compute_targets_fn"](dst_df)
    compute_dst_features(dst_df)
    train = dst_df[dst_df["season"].isin(TRAIN_SEASONS)].copy()
    val = dst_df[dst_df["season"].isin(VAL_SEASONS)].copy()
    test = dst_df[dst_df["season"].isin(TEST_SEASONS)].copy()
    return train, val, test


def _apply_position_models(train, val, test, pos, results):
    """Load pre-trained position-specific models and write predictions into
    results. Graceful per-model degradation: a single model's load failure is
    recorded in ``_cache["position_load_errors"]`` and the corresponding
    pred column is NaN'd for this position's rows, but other models still
    load and the caller continues with the remaining five positions.

    Setup failures (feature build, data filter) are unrecoverable and
    propagate — they usually mean base data is missing, which would break
    every position. ``_ensure_position_loaded`` catches these and records
    the position as fully failed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = POSITION_REGISTRY[pos]

    targets = reg["targets"]
    model_dir = reg["model_dir"]

    # Prepare position data
    pos_train = reg["filter_fn"](train)
    pos_val = reg["filter_fn"](val)
    pos_test = reg["filter_fn"](test)

    pos_train = reg["compute_targets_fn"](pos_train)
    pos_val = reg["compute_targets_fn"](pos_val)
    pos_test = reg["compute_targets_fn"](pos_test)

    feature_cols = reg["get_feature_columns_fn"]()
    pos_train, pos_val, pos_test = build_position_features(
        pos_train, pos_val, pos_test, reg, feature_cols
    )

    # DST still applies a post-hoc adjustment (defensive TDs + safeties); K now
    # encodes its miss penalties as signed raw-value heads (see target_signs).
    # QB/RB/WR/TE aggregate raw-stat preds via reg["aggregate_fn"].
    adj_values = None
    if reg.get("compute_adjustment_fn") is not None:
        adj = reg["compute_adjustment_fn"](pos_test)
        adj_values = adj.values
    aggregate_fn = reg.get("aggregate_fn")
    target_signs = reg.get("target_signs")

    X_test_pos = pos_test[feature_cols].values.astype(np.float32)
    pos_index = pos_test.index
    errors = _cache.setdefault("position_load_errors", {})

    def _combine_total(preds: dict, fmt: str = "ppr") -> np.ndarray:
        # QB/RB/WR/TE/DST go through predictions_to_fantasy_points which knows
        # the per-format reception weight; QB/DST values happen to be format-
        # invariant by construction (no reception target / DST tier-mapped
        # aggregator) so the three calls return identical numbers there.
        if aggregate_fn is not None:
            return predictions_to_fantasy_points(pos, preds, scoring_format=fmt)
        # K — sign-vectored sum, no reception target, format-invariant.
        if target_signs is not None:
            total = sum(preds[t] * target_signs.get(t, 1.0) for t in targets)
        else:
            total = sum(preds[t] for t in targets)
        if adj_values is not None:
            total = total + adj_values
        return total

    def _per_format_totals(preds):
        """Return {fmt: total_array} for a single model's preds dict."""
        if preds is None:
            return None
        return {fmt: _combine_total(preds, fmt) for fmt in _VALID_SCORING}

    # Each of the four model blocks below sets its ``*_preds`` / ``*_total``
    # locals on success. On failure, the error is recorded under
    # ``{pos}_{model_type}`` in ``position_load_errors`` and the locals stay
    # None — the results-write block at the bottom NaN's the pred column for
    # this position's rows when that happens.

    ridge_preds = None
    ridge_totals = None
    try:
        ridge = RidgeMultiTarget(target_names=targets)
        ridge.load(model_dir)
        ridge_preds = ridge.predict(X_test_pos)
        ridge_totals = _per_format_totals(ridge_preds)
    except Exception as e:
        errors[f"{pos}_ridge"] = str(e)
        print(f"[app] {pos} ridge load failed: {e!r} — NaN'ing ridge_pred")

    # NN predictions — integrity-check scaler+weights before running inference.
    nn_preds = None
    nn_totals = None
    try:
        nn_scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        nn_meta = read_scaler_meta(f"{model_dir}/nn_scaler_meta.json")
        nn_checkpoint = torch.load(
            f"{model_dir}/{reg['nn_file']}", map_location=device, weights_only=True
        )
        nn_state_dict, nn_hash = unwrap_state_dict(nn_checkpoint)
        assert_scaler_matches(
            pos,
            nn_scaler,
            nn_hash,
            nn_meta,
            feature_cols,
            targets,
            scaler_label="nn_scaler",
        )

        X_test_scaled = scale_and_clip(nn_scaler, X_test_pos)
        nn_model = MultiHeadNet(
            input_dim=len(feature_cols), target_names=targets, **reg["nn_kwargs"]
        ).to(device)
        nn_model.load_state_dict(nn_state_dict)
        nn_preds = nn_model.predict_numpy(X_test_scaled, device)
        nn_totals = _per_format_totals(nn_preds)
    except Exception as e:
        errors[f"{pos}_nn"] = str(e)
        print(f"[app] {pos} nn load failed: {e!r} — NaN'ing nn_pred")

    # Attention NN — enabled per-position via ``reg["train_attention_nn"]``.
    # Flat-history variant for QB/RB/WR/TE/DST; nested per-kick variant for K.
    # Positions without an attention model leave the column as NaN so the
    # frontend renders "--".
    attn_nn_preds = None
    attn_nn_totals = None
    if reg.get("train_attention_nn", False) and reg.get("attn_nn_file"):
        try:
            # K resolves its attention static columns directly from the
            # DataFrame (they live outside the Ridge/base-NN feature list, per
            # src/shared/pipeline.py::attn_static_from_df). Others use the filtered
            # whitelist over the base feature matrix.
            if reg.get("attn_static_from_df", False):
                attn_static_cols = list(reg.get("attn_static_features", []))
                X_test_attn = pos_test[attn_static_cols].to_numpy(dtype=np.float32)
            else:
                attn_static_cols = get_attn_static_columns(
                    feature_cols, reg.get("attn_static_features", [])
                )
                attn_static_col_set = set(attn_static_cols)
                attn_col_idx = [i for i, c in enumerate(feature_cols) if c in attn_static_col_set]
                X_test_attn = X_test_pos[:, attn_col_idx]

            attn_scaler = joblib.load(f"{model_dir}/attention_nn_scaler.pkl")
            attn_meta = read_scaler_meta(f"{model_dir}/attention_nn_scaler_meta.json")
            attn_checkpoint = torch.load(
                f"{model_dir}/{reg['attn_nn_file']}",
                map_location=device,
                weights_only=True,
            )
            attn_state_dict, attn_hash = unwrap_state_dict(attn_checkpoint)
            assert_scaler_matches(
                pos,
                attn_scaler,
                attn_hash,
                attn_meta,
                attn_static_cols,
                targets,
                scaler_label="attention_nn_scaler",
            )

            X_test_attn_scaled = scale_and_clip(attn_scaler, X_test_attn)

            structure = reg.get("attn_history_structure", "flat")
            if structure == "nested":
                # K: build 4-D [N, G, K, kick_dim] history from per-kick records.
                kicks_df = _cache.get("k_kicks_df")
                if kicks_df is None:
                    raise RuntimeError(
                        "K nested attention requires kicks_df cached by _load_k_splits"
                    )
                hist_test, outer_test, inner_test = build_nested_kick_history(
                    pos_test,
                    kicks_df=kicks_df,
                    kick_stats=reg["attn_kick_stats"],
                    max_games=reg["attn_max_games"],
                    max_kicks_per_game=reg["attn_max_kicks_per_game"],
                )
                attn_model = MultiHeadNetWithNestedHistory(
                    static_dim=len(attn_static_cols),
                    kick_dim=hist_test.shape[-1],
                    target_names=targets,
                    **reg["attn_nn_kwargs_static"],
                ).to(device)
                attn_model.load_state_dict(attn_state_dict)
                attn_nn_preds = attn_model.predict_numpy(
                    X_test_attn_scaled, hist_test, outer_test, inner_test, device
                )
            else:
                hist_stats = [s for s in reg.get("attn_history_stats", []) if s in pos_test.columns]
                max_seq_len = reg.get("attn_max_seq_len", 17)
                hist_test, mask_test = build_game_history_arrays(
                    pos_test, history_stats=hist_stats, max_seq_len=max_seq_len
                )

                # Optional opponent-defense attention branch — mirrors the
                # pipeline: aggregate per-(opp_team, season, week) stats from
                # the PRE-FILTER all-position frames, then build a padded
                # sequence of the defense's prior games for each pos_test row.
                # CLAUDE.md rule: keep training and inference feature paths
                # byte-for-byte consistent.
                opp_history_stats = reg.get("opp_attn_history_stats") or []
                opp_hist_test = opp_mask_test = None
                opp_game_dim = None
                if opp_history_stats:
                    opp_max_seq_len = reg.get("opp_attn_max_seq_len", max_seq_len)
                    all_pos = pd.concat([train, val, test], ignore_index=True)
                    opp_def_per_game = build_opp_defense_per_game_df(all_pos)
                    opp_hist_test, opp_mask_test = build_opp_defense_history_arrays(
                        pos_test, opp_def_per_game, opp_history_stats, opp_max_seq_len
                    )
                    opp_game_dim = opp_hist_test.shape[2]

                attn_model = MultiHeadNetWithHistory(
                    static_dim=len(attn_static_cols),
                    game_dim=hist_test.shape[2],
                    target_names=targets,
                    opp_game_dim=opp_game_dim,
                    **reg["attn_nn_kwargs_static"],
                ).to(device)
                attn_model.load_state_dict(attn_state_dict)
                if opp_game_dim is not None:
                    attn_nn_preds = attn_model.predict_numpy(
                        X_test_attn_scaled,
                        hist_test,
                        mask_test,
                        device,
                        X_opp_history=opp_hist_test,
                        opp_history_mask=opp_mask_test,
                    )
                else:
                    attn_nn_preds = attn_model.predict_numpy(
                        X_test_attn_scaled, hist_test, mask_test, device
                    )
            attn_nn_totals = _per_format_totals(attn_nn_preds)
        except Exception as e:
            errors[f"{pos}_attn_nn"] = str(e)
            print(f"[app] {pos} attn_nn load failed: {e!r} — leaving attn_nn_pred NaN")
            attn_nn_preds = None
            attn_nn_totals = None

    # LightGBM — only trained for QB/RB/WR/TE. Same no-column-emitted policy for
    # K/DST as Attention NN.
    lgbm_preds = None
    lgbm_totals = None
    if reg.get("train_lightgbm", False):
        try:
            lgbm_model = LightGBMMultiTarget(target_names=targets)
            lgbm_model.load(model_dir)
            lgbm_preds = lgbm_model.predict(X_test_pos)
            lgbm_totals = _per_format_totals(lgbm_preds)
        except Exception as e:
            errors[f"{pos}_lgbm"] = str(e)
            print(f"[app] {pos} lgbm load failed: {e!r} — leaving lgbm_pred NaN")
            lgbm_preds = None
            lgbm_totals = None

    # Write into results — NaN the pred column when its model failed so the
    # frontend renders "--" instead of a misleading 0.0 (the DataFrame
    # initializes the per-format columns to 0.0 / NaN in _load_base_data_locked).
    # We write three format-specific columns per model AND the legacy unsuffixed
    # column (a PPR alias) so existing fixtures / tests keep working.
    model_totals_pairs = (
        ("ridge", ridge_totals),
        ("nn", nn_totals),
        ("attn_nn", attn_nn_totals),
        ("lgbm", lgbm_totals),
    )
    for prefix, totals in model_totals_pairs:
        legacy_col = f"{prefix}_pred"
        if totals is not None:
            for fmt, arr in totals.items():
                results.loc[pos_index, _pred_col(prefix, fmt)] = np.round(arr, 2).astype(np.float32)
            results.loc[pos_index, legacy_col] = np.round(totals["ppr"], 2).astype(np.float32)
        else:
            for fmt in _VALID_SCORING:
                results.loc[pos_index, _pred_col(prefix, fmt)] = np.nan
            results.loc[pos_index, legacy_col] = np.nan

    # Cache per-target metrics for /api/position_details. Per-target MAEs are
    # raw-stat (yards / TDs / receptions count) and so are format-invariant —
    # only the aggregated "total" row depends on scoring format. We cache the
    # total row three times under target_metrics["total_by_format"][fmt] and
    # let the API endpoint pick the right one.
    target_metrics = {}
    for t in targets:
        if t in pos_test.columns:
            actual_t = pos_test[t].values
            tm = {}
            if ridge_preds is not None and t in ridge_preds:
                tm["ridge_mae"] = round(float(np.mean(np.abs(ridge_preds[t] - actual_t))), 3)
            if nn_preds is not None and t in nn_preds:
                tm["nn_mae"] = round(float(np.mean(np.abs(nn_preds[t] - actual_t))), 3)
            if attn_nn_preds is not None and t in attn_nn_preds:
                tm["attn_nn_mae"] = round(float(np.mean(np.abs(attn_nn_preds[t] - actual_t))), 3)
            if lgbm_preds is not None and t in lgbm_preds:
                tm["lgbm_mae"] = round(float(np.mean(np.abs(lgbm_preds[t] - actual_t))), 3)
            target_metrics[t] = tm
    total_by_format = {}
    for fmt in _VALID_SCORING:
        actual_col = _actual_col(fmt)
        total_actual = pos_test[actual_col].values if actual_col in pos_test.columns else None
        # K/DST splits don't carry the format-suffixed actual columns, but their
        # scoring is format-invariant so fantasy_points is the same value for
        # all three; fall back to it if the suffixed column is missing.
        if total_actual is None and "fantasy_points" in pos_test.columns:
            total_actual = pos_test["fantasy_points"].values
        if total_actual is None:
            continue
        total_tm = {}
        if ridge_totals is not None:
            total_tm["ridge_mae"] = round(
                float(np.mean(np.abs(ridge_totals[fmt] - total_actual))), 3
            )
        if nn_totals is not None:
            total_tm["nn_mae"] = round(float(np.mean(np.abs(nn_totals[fmt] - total_actual))), 3)
        if attn_nn_totals is not None:
            total_tm["attn_nn_mae"] = round(
                float(np.mean(np.abs(attn_nn_totals[fmt] - total_actual))), 3
            )
        if lgbm_totals is not None:
            total_tm["lgbm_mae"] = round(float(np.mean(np.abs(lgbm_totals[fmt] - total_actual))), 3)
        total_by_format[fmt] = total_tm
    # Default "total" key keeps the PPR view for callers that haven't migrated.
    target_metrics["total"] = total_by_format.get("ppr", {})
    target_metrics["total_by_format"] = total_by_format
    _cache.setdefault("position_details", {})[pos] = {
        "n_features": len(feature_cols),
        "n_samples_test": len(pos_test),
        "target_metrics": target_metrics,
    }


_ALL_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]


def _ensure_base_data():
    """Load splits + build empty results frame. Idempotent. No model loads."""
    if _cache.get("base_loaded"):
        return
    with _cache_lock:
        # Re-check under lock: another thread may have populated between our
        # fast-path check and lock acquisition.
        if _cache.get("base_loaded") or "results" in _cache:
            return
        _load_base_data_locked()


def _load_base_data_locked():
    print("Loading data...")

    def _load_reg(path):
        df = pd.read_parquet(path)
        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()
        return df

    train = _load_reg("data/splits/train.parquet")
    val = _load_reg("data/splits/val.parquet")
    test = _load_reg("data/splits/test.parquet")

    for df in [train, val, test]:
        _compute_scoring_formats(df)

    print("Loading kicker data...")
    k_train, k_val, k_test, k_kicks_df = _load_k_splits()
    print("Loading D/ST data...")
    dst_train, dst_val, dst_test = _load_dst_splits()

    keep_cols = [
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
        "headshot_url",
        "fantasy_points",
        "fantasy_points_half_ppr",
        "fantasy_points_standard",
    ]
    keep_cols = [c for c in keep_cols if c in test.columns]
    results = test[keep_cols].copy()

    for pos_test_df in [k_test, dst_test]:
        offset = results.index.max() + 1
        pos_rows = pd.DataFrame(index=range(offset, offset + len(pos_test_df)))
        for col in keep_cols:
            if col in pos_test_df.columns:
                pos_rows[col] = pos_test_df[col].values
            elif col in ("fantasy_points_half_ppr", "fantasy_points_standard"):
                pos_rows[col] = pos_test_df["fantasy_points"].values
            elif col == "headshot_url":
                pos_rows[col] = ""
            else:
                pos_rows[col] = np.nan
        pos_test_df.index = pos_rows.index
        results = pd.concat([results, pos_rows])

    # Initialize per-model, per-format prediction columns. ridge/nn default to
    # 0.0 (every row gets a value once the position loads); attn_nn/lgbm default
    # to NaN so K/DST rows — which have no attn_nn/lgbm models — are correctly
    # excluded from overall MAE in _compute_metrics_locked.
    for fmt in _VALID_SCORING:
        results[_pred_col("ridge", fmt)] = 0.0
        results[_pred_col("nn", fmt)] = 0.0
        results[_pred_col("attn_nn", fmt)] = np.nan
        results[_pred_col("lgbm", fmt)] = np.nan
    # Legacy unsuffixed columns kept as a PPR alias so existing tests / code
    # that reads results["ridge_pred"] doesn't break.
    results["ridge_pred"] = 0.0
    results["nn_pred"] = 0.0
    results["attn_nn_pred"] = np.nan
    results["lgbm_pred"] = np.nan

    _cache["splits"] = {
        "QB": (train, val, test),
        "RB": (train, val, test),
        "WR": (train, val, test),
        "TE": (train, val, test),
        "K": (k_train, k_val, k_test),
        "DST": (dst_train, dst_val, dst_test),
    }
    # K's attention NN needs the raw per-kick records to build nested history
    # at inference — stash here so _apply_position_models can reach it.
    _cache["k_kicks_df"] = k_kicks_df
    _cache["results"] = results
    _cache["positions_loaded"] = set()
    _cache["base_loaded"] = True


def _ensure_position_loaded(pos):
    """Apply position-specific model. Idempotent, thread-safe, degrade-aware.

    ``_apply_position_models`` records per-model failures internally and
    NaN's the affected pred columns — the position still counts as "loaded"
    in that case (the DataFrame rows are there, just with some NaN preds).
    An outer ``try/except`` here catches unrecoverable setup failures
    (feature build, parquet read) and marks the position as fully failed
    so the other five still serve.
    """
    _ensure_base_data()
    if pos in _cache.get("positions_loaded", ()):
        return
    if pos in _cache.get("positions_failed", ()):
        return  # cached hard-failure — don't retry every request
    with _cache_lock:
        if "splits" not in _cache:
            return
        if pos in _cache["positions_loaded"]:
            return
        if pos in _cache.get("positions_failed", set()):
            return
        train, val, test = _cache["splits"][pos]
        print(f"Applying {pos}-specific model...")
        try:
            _apply_position_models(train, val, test, pos, _cache["results"])
        except Exception as e:
            # Anything that slips past the inner per-model try/excepts —
            # typically data-loading or feature-building failures that affect
            # the whole position. Cache so we don't retry on every request,
            # and leave the pred columns untouched (still the default init
            # values from _load_base_data_locked).
            import traceback

            traceback.print_exc()
            _cache.setdefault("positions_failed", set()).add(pos)
            _cache.setdefault("position_load_errors", {})[pos] = repr(e)
            print(f"[app] {pos} fully failed: {e!r} — serving degraded")
            return
        _cache["positions_loaded"].add(pos)


def _ensure_all_positions_loaded():
    """Load every position, best-effort. A per-position failure records in
    ``positions_failed`` but does not re-raise — the remaining positions
    still get loaded. If EVERY position fails, raise a top-level error so
    gunicorn ``--preload`` aborts at boot and ECS blocks the broken
    rollout (preserves the existing fail-loud contract for the
    all-broken case).
    """
    for pos in _ALL_POSITIONS:
        _ensure_position_loaded(pos)
    failed = _cache.get("positions_failed", set())
    if failed and len(failed) == len(_ALL_POSITIONS):
        raise RuntimeError(
            f"All positions failed to load — see position_load_errors: "
            f"{list(_cache.get('position_load_errors', {}).keys())}"
        )


def _degraded_positions() -> list[str]:
    """Positions with any recorded model-load error (fully failed OR partial
    per-model failure). Returned sorted so the frontend banner has a stable
    ordering across requests.
    """
    errs = _cache.get("position_load_errors", {})
    if not errs:
        return []
    degraded: set[str] = set()
    for key in errs:
        for p in _ALL_POSITIONS:
            if key == p or key.startswith(f"{p}_"):
                degraded.add(p)
                break
    return sorted(degraded)


_MODEL_PRED_COLUMNS = [
    ("Ridge Regression", "ridge"),
    ("Neural Network", "nn"),
    ("Attention NN", "attn_nn"),
    ("LightGBM", "lgbm"),
]


def _ensure_metrics():
    if "metrics_by_format" in _cache:
        return
    with _cache_lock:
        if "metrics_by_format" in _cache:
            return
        _ensure_all_positions_loaded()
        _compute_metrics_locked()


def _compute_metrics_locked():
    """Compute overall + per-position MAE/RMSE/R² for every model under each
    of the three scoring formats, caching the results under
    ``_cache["metrics_by_format"][fmt]``. Keeps ``_cache["metrics"]`` as a PPR
    alias for callers that haven't migrated.
    """
    results = _cache["results"]
    metrics_by_format = {}
    for fmt in _VALID_SCORING:
        actual_col = _actual_col(fmt)
        if actual_col not in results.columns:
            continue
        actual_values = results[actual_col].values
        per_format = {}
        for name, prefix in _MODEL_PRED_COLUMNS:
            pred_col = _pred_col(prefix, fmt)
            if pred_col not in results.columns:
                per_format[name] = {"overall": None, "by_position": []}
                continue
            pred_series = results[pred_col]
            # Skip rows where this model has no prediction (K/DST for attn/lgbm).
            available_mask = pred_series.notna().values
            if not available_mask.any():
                per_format[name] = {"overall": None, "by_position": []}
                continue
            y_avail = actual_values[available_mask]
            preds_avail = pred_series.values[available_mask]
            overall = compute_metrics(y_avail, preds_avail)
            pos_df = results.loc[available_mask, ["position"]].copy()
            pos_df["pred"] = preds_avail
            pos_df["actual"] = y_avail
            pos_metrics = compute_positional_metrics(pos_df, "pred", "actual")
            per_format[name] = {
                "overall": {k: round(v, 4) for k, v in overall.items()},
                "by_position": pos_metrics.to_dict(orient="records"),
            }
        metrics_by_format[fmt] = per_format
    _cache["metrics_by_format"] = metrics_by_format
    _cache["metrics"] = metrics_by_format.get("ppr", {})
    print("Ready!")


def _get_data(scoring="ppr"):
    """Full load: all positions + metrics for the requested scoring format."""
    _ensure_metrics()
    metrics = _cache["metrics_by_format"].get(scoring, _cache["metrics_by_format"]["ppr"])
    return _cache["results"], metrics


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    position = request.args.get("position", "ALL")
    week = request.args.get("week", "ALL")
    search = request.args.get("search", "").strip().lower()
    sort_by = request.args.get("sort", "fantasy_points")
    order = request.args.get("order", "desc")
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))

    if position != "ALL":
        _ensure_position_loaded(position)
        df = _cache["results"]
        df = df[df["position"] == position]
    else:
        results, _ = _get_data(scoring)
        df = results
    if week != "ALL":
        try:
            df = df[df["week"] == int(week)]
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid week: {week}"}), 400
    if search:
        df = df[df["player_display_name"].str.lower().str.contains(search, na=False, regex=False)]

    rows = _records_to_player_rows(df, scoring=scoring)

    reverse = order == "desc"
    if sort_by in ("actual", "ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred", "week"):
        rows.sort(key=lambda x: x.get(sort_by) or 0, reverse=reverse)
    else:
        rows.sort(key=lambda x: x.get("actual") or 0, reverse=reverse)

    return jsonify(
        {
            "players": rows,
            "total": len(rows),
            "scoring": scoring,
            # Surfaced so the frontend can render a banner. See
            # static/js/app.js::loadPredictions and _degraded_positions above.
            "degraded_positions": _degraded_positions(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _, metrics = _get_data(scoring)
    return jsonify(metrics)


@app.route("/api/weeks")
def api_weeks():
    _ensure_base_data()
    results = _cache["results"]
    weeks = sorted(results["week"].unique().tolist())
    return jsonify({"weeks": [int(w) for w in weeks]})


@app.route("/api/player/<player_id>")
def api_player(player_id):
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _ensure_base_data()
    results = _cache["results"]
    match = results[results["player_id"] == player_id]
    if match.empty:
        return jsonify({"error": "Player not found"}), 404
    _ensure_position_loaded(match.iloc[0]["position"])
    results = _cache["results"]
    df = results[results["player_id"] == player_id].sort_values("week")

    info = df.iloc[0]
    actual_col = _actual_col(scoring)
    pred_cols = {prefix: _pred_col(prefix, scoring) for prefix in _MODEL_PRED_PREFIXES}
    weekly_cols = ["week", actual_col, *pred_cols.values()]
    weekly_cols = [c for c in weekly_cols if c in df.columns]
    weekly = [
        {
            "week": int(r["week"]),
            "actual": _round_or_none(r.get(actual_col)),
            "ridge_pred": _safe_num(r.get(pred_cols["ridge"])),
            "nn_pred": _safe_num(r.get(pred_cols["nn"])),
            "attn_nn_pred": _safe_num(r.get(pred_cols["attn_nn"])),
            "lgbm_pred": _safe_num(r.get(pred_cols["lgbm"])),
        }
        for r in df[weekly_cols].to_dict(orient="records")
    ]

    return jsonify(
        {
            "player_id": player_id,
            "name": _safe_str(info["player_display_name"]),
            "position": _safe_str(info["position"]),
            "team": _safe_str(info["recent_team"]),
            "headshot": _safe_str(info.get("headshot_url", "")),
            "weekly": weekly,
            "season_avg": _safe_num(round(df[actual_col].mean(), 2)),
            "season_total": _safe_num(round(df[actual_col].sum(), 2)),
            "scoring": scoring,
        }
    )


@app.route("/api/top_players")
def api_top_players():
    position = request.args.get("position", "ALL")
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))

    if position != "ALL":
        _ensure_position_loaded(position)
        df = _cache["results"]
        df = df[df["position"] == position]
    else:
        results, _ = _get_data(scoring)
        df = results

    agg_dict = {
        "avg_actual": (_actual_col(scoring), "mean"),
        "avg_ridge": (_pred_col("ridge", scoring), "mean"),
        "avg_nn": (_pred_col("nn", scoring), "mean"),
        "avg_attn_nn": (_pred_col("attn_nn", scoring), "mean"),
        "avg_lgbm": (_pred_col("lgbm", scoring), "mean"),
        "games": ("week", "count"),
    }
    avg = (
        df.groupby(["player_id", "player_display_name", "position", "recent_team"])
        .agg(
            **agg_dict,
        )
        .reset_index()
    )
    avg = avg[avg["games"] >= 6]
    avg = avg.sort_values("avg_actual", ascending=False).head(25)

    rows = [
        {
            "player_id": _safe_str(r["player_id"]),
            "name": _safe_str(r["player_display_name"]),
            "position": _safe_str(r["position"]),
            "team": _safe_str(r["recent_team"]),
            "avg_actual": _round_or_none(r["avg_actual"]),
            "avg_ridge": _round_or_none(r["avg_ridge"]),
            "avg_nn": _round_or_none(r["avg_nn"]),
            "avg_attn_nn": _round_or_none(r["avg_attn_nn"]),
            "avg_lgbm": _round_or_none(r["avg_lgbm"]),
            "games": int(r["games"]),
        }
        for r in avg.to_dict(orient="records")
    ]

    return jsonify({"players": rows, "scoring": scoring})


@app.route("/api/weekly_accuracy")
def api_weekly_accuracy():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    results, _ = _get_data(scoring)
    actual = results[_actual_col(scoring)].values
    # Per-row abs error; NaN where a model has no prediction (K/DST attn/lgbm)
    # so groupby.mean() excludes those rows from that model's weekly MAE.
    err_df = results.assign(
        _ridge_err=np.abs(actual - results[_pred_col("ridge", scoring)].values),
        _nn_err=np.abs(actual - results[_pred_col("nn", scoring)].values),
        _attn_nn_err=np.abs(actual - results[_pred_col("attn_nn", scoring)].values),
        _lgbm_err=np.abs(actual - results[_pred_col("lgbm", scoring)].values),
    )
    weekly = (
        err_df.groupby("week")
        .agg(
            ridge_mae=("_ridge_err", "mean"),
            nn_mae=("_nn_err", "mean"),
            attn_nn_mae=("_attn_nn_err", "mean"),
            lgbm_mae=("_lgbm_err", "mean"),
        )
        .round(3)
        .sort_index()
    )

    def _series_to_list(s):
        # Convert pandas Series with NaN -> list with None so jsonify works.
        return [None if pd.isna(v) else float(v) for v in s]

    return jsonify(
        {
            "weeks": [int(w) for w in weekly.index],
            "ridge_mae": _series_to_list(weekly["ridge_mae"]),
            "nn_mae": _series_to_list(weekly["nn_mae"]),
            "attn_nn_mae": _series_to_list(weekly["attn_nn_mae"]),
            "lgbm_mae": _series_to_list(weekly["lgbm_mae"]),
            "scoring": scoring,
        }
    )


@app.route("/api/position_details")
def api_position_details():
    scoring = _validate_scoring(request.args.get("scoring", "ppr"))
    _get_data(scoring)  # ensure cache is populated
    details = _cache.get("position_details", {})
    result = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        info = dict(POSITION_INFO[pos])
        pos_details = dict(details.get(pos, {}))
        # Swap target_metrics["total"] for the format-specific cached row.
        # Per-target raw-stat MAE is format-invariant and is left untouched.
        target_metrics = dict(pos_details.get("target_metrics", {}))
        total_by_format = target_metrics.pop("total_by_format", {}) or {}
        if total_by_format:
            target_metrics["total"] = total_by_format.get(scoring, target_metrics.get("total", {}))
        if target_metrics:
            pos_details["target_metrics"] = target_metrics
        info.update(pos_details)
        result[pos] = info
    return jsonify(result)


def _categorize_features(features):
    """Bucket feature names into human-readable categories by prefix."""
    weather_set = set(WEATHER_FEATURES_ALL) | {"game_wind", "game_temp"}
    categories = {
        "rolling": [],
        "prior_season": [],
        "ewma": [],
        "trend": [],
        "share": [],
        "matchup": [],
        "defense": [],
        "weather_vegas": [],
        "contextual": [],
        "other": [],
    }
    contextual_set = {
        "is_home",
        "week",
        "is_returning_from_absence",
        "days_rest",
        "practice_status",
        "game_status",
        "depth_chart_rank",
        "rest_days",
        "div_game",
        "spread_line",
    }
    for f in features:
        if f in weather_set:
            categories["weather_vegas"].append(f)
        elif f.startswith("rolling_"):
            categories["rolling"].append(f)
        elif f.startswith("prior_season_"):
            categories["prior_season"].append(f)
        elif f.startswith("ewma_"):
            categories["ewma"].append(f)
        elif f.startswith("trend_") or f.endswith("_trend"):
            categories["trend"].append(f)
        elif "share" in f or "hhi" in f:
            categories["share"].append(f)
        elif f.startswith("opp_def_") or (f.startswith("opp_") and "def" in f):
            categories["defense"].append(f)
        elif f.startswith("opp_") or f.endswith("_rank_vs_pos"):
            categories["matchup"].append(f)
        elif f in contextual_set:
            categories["contextual"].append(f)
        else:
            categories["other"].append(f)
    return {k: v for k, v in categories.items() if v}


def _position_arch_payload(pos, cfg, specific, targets, include_features, attn_history=None):
    """Build the per-position JSON payload for /api/model_architecture.

    `include_features` may be a categorized dict (QB/RB/WR/TE) or a flat list
    (K/DST contextual); either shape is normalized to categorized groups.
    """

    def get(name, default=None):
        return getattr(cfg, name, default)

    prefix = pos.upper()

    # Scheduler summary string
    scheduler = get(f"{prefix}_SCHEDULER_TYPE", "unknown")
    if scheduler == "cosine_warm_restarts":
        t0 = get(f"{prefix}_COSINE_T0", "?")
        tm = get(f"{prefix}_COSINE_T_MULT", "?")
        em = get(f"{prefix}_COSINE_ETA_MIN", "?")
        scheduler_str = f"CosineAnnealingWarmRestarts(T0={t0}, T_mult={tm}, eta_min={em})"
    elif scheduler == "onecycle":
        mx = get(f"{prefix}_ONECYCLE_MAX_LR", "?")
        ps = get(f"{prefix}_ONECYCLE_PCT_START", "?")
        scheduler_str = f"OneCycleLR(max_lr={mx}, pct_start={ps})"
    elif scheduler == "plateau":
        scheduler_str = "ReduceLROnPlateau"
    else:
        scheduler_str = str(scheduler)

    # Normalize feature groupings
    if isinstance(include_features, dict):
        grouped = {k: list(v) for k, v in include_features.items() if v}
        # Ensure position-specific features surface even if not keyed in dict
        if "specific" not in grouped and specific:
            grouped["specific"] = list(specific)
        flat_features = [f for group in grouped.values() for f in group]
    else:
        flat = list(include_features or [])
        grouped = {"specific": list(specific or [])}
        grouped.update(_categorize_features(flat))
        flat_features = list(specific or []) + flat

    payload = {
        "targets": list(targets),
        "backbone_layers": list(get(f"{prefix}_NN_BACKBONE_LAYERS", [])),
        "head_hidden": get(f"{prefix}_NN_HEAD_HIDDEN"),
        "head_hidden_overrides": dict(get(f"{prefix}_NN_HEAD_HIDDEN_OVERRIDES", {}) or {}),
        "dropout": get(f"{prefix}_NN_DROPOUT"),
        "lr": get(f"{prefix}_NN_LR"),
        "weight_decay": get(f"{prefix}_NN_WEIGHT_DECAY"),
        "batch_size": get(f"{prefix}_NN_BATCH_SIZE"),
        "epochs": get(f"{prefix}_NN_EPOCHS"),
        "patience": get(f"{prefix}_NN_PATIENCE"),
        "scheduler": scheduler_str,
        "attention_enabled": bool(get(f"{prefix}_TRAIN_ATTENTION_NN", False)),
        "lightgbm_enabled": bool(get(f"{prefix}_TRAIN_LIGHTGBM", False)),
        "feature_count": len(flat_features),
        "features": grouped,
    }
    if attn_history:
        payload["features"]["attention_history"] = list(attn_history)
    return payload


@app.route("/api/model_architecture")
def api_model_architecture():
    try:
        positions = {
            "QB": _position_arch_payload(
                "QB",
                qb_cfg,
                QB_SPECIFIC_FEATURES,
                QB_TARGETS,
                getattr(qb_cfg, "QB_INCLUDE_FEATURES", []),
                getattr(qb_cfg, "QB_ATTN_HISTORY_STATS", None),
            ),
            "RB": _position_arch_payload(
                "RB",
                rb_cfg,
                RB_SPECIFIC_FEATURES,
                RB_TARGETS,
                getattr(rb_cfg, "RB_INCLUDE_FEATURES", []),
                getattr(rb_cfg, "RB_ATTN_HISTORY_STATS", None),
            ),
            "WR": _position_arch_payload(
                "WR",
                wr_cfg,
                WR_SPECIFIC_FEATURES,
                WR_TARGETS,
                getattr(wr_cfg, "WR_INCLUDE_FEATURES", []),
                getattr(wr_cfg, "WR_ATTN_HISTORY_STATS", None),
            ),
            "TE": _position_arch_payload(
                "TE",
                te_cfg,
                TE_SPECIFIC_FEATURES,
                TE_TARGETS,
                getattr(te_cfg, "TE_INCLUDE_FEATURES", []),
                getattr(te_cfg, "TE_ATTN_HISTORY_STATS", None),
            ),
            "K": _position_arch_payload(
                "K",
                k_cfg,
                K_SPECIFIC_FEATURES,
                K_TARGETS,
                getattr(k_cfg, "K_CONTEXTUAL_FEATURES", []),
            ),
            "DST": _position_arch_payload(
                "DST",
                dst_cfg,
                DST_SPECIFIC_FEATURES,
                DST_TARGETS,
                getattr(dst_cfg, "DST_CONTEXTUAL_FEATURES", []),
            ),
        }
        return jsonify(
            {
                "overview": {
                    "framework": "PyTorch 2.11 + CUDA 12.6 (AWS Batch)",
                    "device": "CUDA if available, else CPU",
                    "data_splits": "Train 2012-2023, Val 2024, Test 2025 (K uses 2015+)",
                    "ensemble": [
                        "Season-average baseline",
                        "Ridge multi-target",
                        "MultiHeadNet (dense)",
                        "MultiHeadNetWithHistory (attention)",
                        "LightGBM",
                    ],
                },
                "training_loop": {
                    "optimizer": "AdamW",
                    "loss": "MultiTargetLoss: per-target Huber or Poisson NLL + optional BCE on TD gate",
                    "gradient_clip": "clip_grad_norm_(max_norm=1.0)",
                    "feature_scaling": "StandardScaler, clipped to [-4, 4]",
                    "early_stopping": "Best loss-weighted val MAE restored on patience",
                    "checkpoint": "Best state_dict kept in memory, saved as .pt",
                },
                "positions": positions,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/wiki/index")
def api_wiki_index():
    return jsonify(
        [
            {"slug": slug, "name": meta["name"], "group": meta["group"]}
            for slug, meta in WIKI_DOCS.items()
        ]
    )


@app.route("/api/wiki/<slug>")
def api_wiki_page(slug):
    if slug not in WIKI_DOCS:
        return jsonify({"error": "Unknown doc"}), 404
    meta = WIKI_DOCS[slug]
    html = _render_wiki_doc(slug)
    return jsonify({"slug": slug, "name": meta["name"], "group": meta["group"], "html": html})


@app.route("/health")
def health():
    errors = _cache.get("position_load_errors")
    if errors:
        return jsonify({"status": "degraded", "position_load_errors": errors}), 503
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5050, use_reloader=False)

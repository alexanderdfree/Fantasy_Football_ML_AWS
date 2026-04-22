"""Diagnostic: explain outlier QB predictions (Darnold W6, Love W2, Mullens, Hill).

Trains Ridge + NN using the standard QB pipeline prep, then for each target row
emits per-target feature attributions:
  - Ridge: coef * scaled_feature (signed contribution to the raw prediction).
  - NN:    integrated gradients w.r.t. scaled input, 50 interpolation steps.

Also prints (a) dataset-wide receiving contamination stats in QB rows and
(b) the last 8 games (features + raw stats) for each outlier player so we can
see whether rolling features were polluted by non-QB usage.

Usage:
    python -m QB.diagnose_qb_outliers

Writes:
    analysis_output/qb_outlier_diagnostic.md
    analysis_output/qb_outlier_diagnostic.json
"""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from QB.run_qb_pipeline import QB_CONFIG
from shared.models import RidgeMultiTarget
from shared.neural_net import MultiHeadNet
from shared.pipeline import (
    _build_scheduler,
    _prepare_position_data,
    _read_split,
    _tune_ridge_alphas_cv,
)
from shared.training import MultiHeadTrainer, MultiTargetLoss, make_dataloaders
from shared.utils import seed_everything
from src.config import SPLITS_DIR

OUTPUT_DIR = "analysis_output"
MD_PATH = f"{OUTPUT_DIR}/qb_outlier_diagnostic.md"
JSON_PATH = f"{OUTPUT_DIR}/qb_outlier_diagnostic.json"

TOP_N = 15
IG_STEPS = 50


@dataclass
class Target:
    name: str
    season: int | None  # None => pick latest season where the player has a row matching week
    week: int | None  # None => pick the row with the largest pred_total


TARGETS = [
    Target("Sam Darnold", season=None, week=6),
    Target("Jordan Love", season=None, week=2),
    Target("Nick Mullens", season=None, week=None),  # largest predicted row
    Target("Taysom Hill", season=None, week=None),
]


# ---------------------------------------------------------------------------
# Data + training
# ---------------------------------------------------------------------------


def _load_splits():
    train_df = _read_split(f"{SPLITS_DIR}/train.parquet")
    val_df = _read_split(f"{SPLITS_DIR}/val.parquet")
    test_df = _read_split(f"{SPLITS_DIR}/test.parquet")
    return train_df, val_df, test_df


def _train_models(seed=42):
    """Replicate the relevant portion of run_pipeline: Ridge + NN only."""
    seed_everything(seed)

    train_df, val_df, test_df = _load_splits()

    (
        X_train,
        X_val,
        X_test,
        y_train_dict,
        y_val_dict,
        y_test_dict,
        pos_train,
        pos_val,
        pos_test,
        feature_cols,
    ) = _prepare_position_data("QB", QB_CONFIG, train_df, val_df, test_df)

    # --- Ridge ---
    cfg = QB_CONFIG
    targets = cfg["targets"]
    best_alphas = _tune_ridge_alphas_cv(
        X_train,
        y_train_dict,
        pos_train["season"].values,
        targets=targets,
        alpha_grids={t: cfg["ridge_alpha_grids"][t] for t in targets},
        n_cv_folds=cfg.get("ridge_cv_folds", 4),
        refine_points=cfg.get("ridge_refine_points", 5),
    )
    print(f"Ridge alphas: {best_alphas}")
    ridge_model = RidgeMultiTarget(target_names=targets, alpha=best_alphas)
    ridge_model.fit(X_train, y_train_dict)
    ridge_preds = ridge_model.predict(X_test)

    # --- NN ---
    nn_scaler = StandardScaler()
    X_train_s = np.clip(nn_scaler.fit_transform(X_train), -4, 4)
    X_val_s = np.clip(nn_scaler.transform(X_val), -4, 4)
    X_test_s = np.clip(nn_scaler.transform(X_test), -4, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = MultiHeadNet(
        input_dim=X_train_s.shape[1],
        target_names=targets,
        backbone_layers=cfg["nn_backbone_layers"],
        head_hidden=cfg["nn_head_hidden"],
        dropout=cfg["nn_dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        nn_model.parameters(), lr=cfg["nn_lr"], weight_decay=cfg["nn_weight_decay"]
    )
    criterion = MultiTargetLoss(
        target_names=targets,
        loss_weights=cfg["loss_weights"],
        huber_deltas=cfg["huber_deltas"],
    )
    train_loader, val_loader = make_dataloaders(
        X_train_s, y_train_dict, X_val_s, y_val_dict, batch_size=cfg["nn_batch_size"]
    )
    scheduler, sched_per_batch = _build_scheduler(optimizer, cfg, train_loader)
    trainer = MultiHeadTrainer(
        model=nn_model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        target_names=targets,
        patience=cfg["nn_patience"],
        scheduler_per_batch=sched_per_batch,
        log_every=25,
    )
    trainer.train(train_loader, val_loader, n_epochs=cfg["nn_epochs"])
    nn_preds = nn_model.predict_numpy(X_test_s, device)

    return {
        "feature_cols": feature_cols,
        "targets": targets,
        "pos_train": pos_train,
        "pos_val": pos_val,
        "pos_test": pos_test,
        "raw_train": train_df,
        "raw_val": val_df,
        "raw_test": test_df,
        "X_test": X_test,
        "X_test_s": X_test_s,
        "ridge_model": ridge_model,
        "nn_model": nn_model,
        "nn_scaler": nn_scaler,
        "ridge_preds": ridge_preds,
        "nn_preds": nn_preds,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def ridge_attribution(ridge_model, feature_cols, x_raw, target_name):
    """Signed per-feature contribution for a single Ridge target.

    Returns dict with per-feature contributions plus reconciliation.
    """
    submodel = ridge_model._models[target_name]
    # RidgeModel: scaler + (optional pca) + Ridge
    x_scaled = submodel.scaler.transform(x_raw.reshape(1, -1))[0]
    if submodel.pca is not None:
        # Fold PCA loadings into effective coefs on original scaled features.
        effective_coefs = submodel.pca.components_.T @ submodel.model.coef_
        intercept = submodel.model.intercept_ - submodel.pca.mean_ @ effective_coefs
    else:
        effective_coefs = submodel.model.coef_
        intercept = submodel.model.intercept_

    contributions = effective_coefs * x_scaled
    raw_pred = intercept + contributions.sum()

    order = np.argsort(-np.abs(contributions))
    rows = []
    for i in order[:TOP_N]:
        rows.append(
            {
                "feature": feature_cols[i],
                "raw_value": float(x_raw[i]),
                "scaled_value": float(x_scaled[i]),
                "coef": float(effective_coefs[i]),
                "contribution": float(contributions[i]),
            }
        )
    return {
        "target": target_name,
        "intercept": float(intercept),
        "raw_prediction": float(raw_pred),
        "clamped_prediction": float(max(raw_pred, 0)),
        "top_contributions": rows,
        "sum_of_all_contributions": float(contributions.sum()),
    }


def nn_integrated_gradients(nn_model, nn_scaler, feature_cols, x_raw, target_name, device):
    """Integrated gradients w.r.t. scaled input for a single NN target head.

    Baseline is an all-zero scaled vector (= feature means after StandardScaler).
    Sum of attributions + f(baseline) reconciles to f(x).
    """
    x_scaled = np.clip(nn_scaler.transform(x_raw.reshape(1, -1)), -4, 4)[0]
    baseline = np.zeros_like(x_scaled)
    alphas = np.linspace(0.0, 1.0, IG_STEPS + 1)

    nn_model.eval()
    grads_sum = np.zeros_like(x_scaled)
    x_scaled_t = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    baseline_t = torch.tensor(baseline, dtype=torch.float32, device=device)

    for a in alphas:
        interp_vec = (baseline_t + float(a) * (x_scaled_t - baseline_t)).detach().clone()
        interp = interp_vec.unsqueeze(0).requires_grad_(True)
        nn_model.zero_grad(set_to_none=True)
        preds = nn_model(interp)
        preds[target_name].sum().backward()
        grads_sum += interp.grad.detach().cpu().numpy()[0]

    avg_grads = grads_sum / (IG_STEPS + 1)
    attributions = (x_scaled - baseline) * avg_grads

    # Reconcile
    with torch.no_grad():
        f_x = float(nn_model(x_scaled_t.unsqueeze(0))[target_name].item())
        f_base = float(nn_model(baseline_t.unsqueeze(0))[target_name].item())

    order = np.argsort(-np.abs(attributions))
    rows = []
    for i in order[:TOP_N]:
        rows.append(
            {
                "feature": feature_cols[i],
                "raw_value": float(x_raw[i]),
                "scaled_value": float(x_scaled[i]),
                "attribution": float(attributions[i]),
            }
        )
    return {
        "target": target_name,
        "f_baseline": f_base,
        "f_x": f_x,
        "sum_attributions": float(attributions.sum()),
        "ig_reconciliation_error": float(f_x - f_base - attributions.sum()),
        "top_attributions": rows,
    }


# ---------------------------------------------------------------------------
# Row picking
# ---------------------------------------------------------------------------


def _pick_row(pos_test: pd.DataFrame, ridge_totals: np.ndarray, target: Target):
    """Return (positional_index_in_X_test, row, error_msg_or_None).

    pos_test rows align 1:1 with X_test by position (not by index label), so we
    work in positional indices throughout.
    """
    names = pos_test["player_display_name"].str.lower().values
    name_mask = names == target.name.lower()
    if not name_mask.any():
        return None, None, f"no test row for {target.name}"

    weeks = pos_test["week"].values
    seasons = pos_test["season"].values

    mask = name_mask
    if target.week is not None:
        mask = mask & (weeks == target.week)
    if target.season is not None:
        mask = mask & (seasons == target.season)

    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return None, None, (f"no row for {target.name} week={target.week} season={target.season}")

    if target.week is not None:
        # Prefer the most recent season when multiple matched
        row_pos = int(positions[np.argmax(seasons[positions])])
    else:
        # Pick the highest-predicted row
        row_pos = int(positions[np.argmax(ridge_totals[positions])])

    return row_pos, pos_test.iloc[row_pos], None


# ---------------------------------------------------------------------------
# Contamination stats
# ---------------------------------------------------------------------------


def _receiving_contamination(pos_train: pd.DataFrame):
    stats = []
    for col in ["receiving_tds", "receptions", "receiving_yards"]:
        if col not in pos_train.columns:
            continue
        nonzero = (pos_train[col].fillna(0) > 0).sum()
        total = len(pos_train)
        stats.append({"column": col, "nonzero_rows": int(nonzero), "total": int(total)})

    if "receiving_tds" in pos_train.columns:
        td_contam = pos_train[pos_train["receiving_tds"].fillna(0) > 0]
        by_player = (
            td_contam.groupby("player_display_name")["receiving_tds"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        top_td_contributors = [
            {"player": p, "receiving_tds_total": float(v)} for p, v in by_player.items()
        ]
    else:
        top_td_contributors = []

    return {"column_summary": stats, "top_receiving_td_contributors": top_td_contributors}


# ---------------------------------------------------------------------------
# Player history dump
# ---------------------------------------------------------------------------


RAW_STAT_COLS = [
    "season",
    "week",
    "position",
    "recent_team",
    "opponent_team",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "carries",
    "receiving_yards",
    "receiving_tds",
    "receptions",
    "snap_pct",
    "fantasy_points",
]

ROLLING_INSPECT_COLS = [
    "rolling_mean_fantasy_points_L3",
    "rolling_max_fantasy_points_L3",
    "rolling_mean_carries_L3",
    "rolling_mean_rushing_yards_L3",
    "rolling_mean_passing_yards_L3",
    "prior_season_max_fantasy_points",
    "prior_season_mean_fantasy_points",
    "prior_season_mean_carries",
    "prior_season_mean_rushing_yards",
    "opp_recv_pts_allowed_to_pos",
]


def _player_history(all_rows: pd.DataFrame, player_name: str, k=8):
    """Return the last k rows for a player across all splits, sorted by season/week."""
    mask = all_rows["player_display_name"].str.lower() == player_name.lower()
    if not mask.any():
        return pd.DataFrame()
    sub = all_rows[mask].sort_values(["season", "week"]).tail(k)
    keep = [c for c in RAW_STAT_COLS + ROLLING_INSPECT_COLS if c in sub.columns]
    return sub[keep].copy()


def _unfiltered_history(train_df, val_df, test_df, player_name: str, k=12):
    """Last k rows for a player across all positions (shows TE/RB games too).

    Rolling features are computed over all games regardless of position, so to
    see whether non-QB games contaminated a QB row's rolling window we must
    look at the unfiltered feature-built frame.
    """
    frames = []
    for df in [train_df, val_df, test_df]:
        if df is None or "player_display_name" not in df.columns:
            continue
        sub = df[df["player_display_name"].str.lower() == player_name.lower()]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True).sort_values(["season", "week"]).tail(k)
    keep = [c for c in RAW_STAT_COLS + ROLLING_INSPECT_COLS if c in full.columns]
    return full[keep].copy()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_ridge_table(attr):
    lines = [
        f"#### Ridge — target `{attr['target']}`  "
        f"(intercept={attr['intercept']:.3f}, raw_pred={attr['raw_prediction']:.3f}, "
        f"clamped={attr['clamped_prediction']:.3f})",
        "",
        "| Feature | Raw | Scaled | Coef | Contribution |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in attr["top_contributions"]:
        lines.append(
            f"| `{r['feature']}` | {r['raw_value']:.3f} | {r['scaled_value']:.3f} "
            f"| {r['coef']:.3f} | {r['contribution']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_nn_table(attr):
    lines = [
        f"#### NN IG — target `{attr['target']}`  "
        f"(f(baseline)={attr['f_baseline']:.3f}, f(x)={attr['f_x']:.3f}, "
        f"Σ attributions={attr['sum_attributions']:.3f}, recon_err={attr['ig_reconciliation_error']:+.3f})",
        "",
        "| Feature | Raw | Scaled | Attribution |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in attr["top_attributions"]:
        lines.append(
            f"| `{r['feature']}` | {r['raw_value']:.3f} | {r['scaled_value']:.3f} "
            f"| {r['attribution']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_history(df):
    if df.empty:
        return "_(no rows found)_\n"
    # Render manually so we don't depend on tabulate.
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        parts = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                parts.append(f"{v:.2f}")
            else:
                parts.append(str(v))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Training Ridge + NN on QB splits ...")
    bundle = _train_models()

    pos_test = bundle["pos_test"]
    pos_train = bundle["pos_train"]
    pos_val = bundle["pos_val"]
    feature_cols = bundle["feature_cols"]
    targets = bundle["targets"]
    X_test = bundle["X_test"]
    ridge_preds = bundle["ridge_preds"]
    nn_preds = bundle["nn_preds"]
    ridge_model = bundle["ridge_model"]
    nn_model = bundle["nn_model"]
    nn_scaler = bundle["nn_scaler"]
    device = bundle["device"]

    ridge_total = sum(ridge_preds[t] for t in targets)
    nn_total = sum(nn_preds[t] for t in targets)

    md = ["# QB Outlier Diagnostic\n"]
    json_out = {"rows": [], "contamination": None, "summary": {}}

    # --- Receiving contamination stats ---
    contam = _receiving_contamination(pos_train)
    json_out["contamination"] = contam
    md.append("## Receiving contamination in QB training rows\n")
    md.append("| Column | Nonzero rows | Total rows | % |\n| --- | ---: | ---: | ---: |\n")
    for s in contam["column_summary"]:
        pct = 100 * s["nonzero_rows"] / max(s["total"], 1)
        md.append(f"| `{s['column']}` | {s['nonzero_rows']} | {s['total']} | {pct:.2f}% |\n")
    md.append("\n")
    if contam["top_receiving_td_contributors"]:
        md.append("**Top-10 QBs by total `receiving_tds` in training set:**\n\n")
        md.append("| Player | Receiving TDs |\n| --- | ---: |\n")
        for r in contam["top_receiving_td_contributors"]:
            md.append(f"| {r['player']} | {r['receiving_tds_total']:.0f} |\n")
        md.append("\n")

    # --- Combined splits for player history (prior-season features already in test split) ---
    combined = pd.concat([pos_train, pos_val, pos_test], ignore_index=True)

    # --- Per-player diagnostic ---
    for target in TARGETS:
        md.append(f"## {target.name}\n")
        row_pos, row, err = _pick_row(pos_test, ridge_total, target)
        if err:
            md.append(f"_{err}_\n\n")
            json_out["rows"].append({"player": target.name, "error": err})
            continue

        actual_fp = row.get("fantasy_points", float("nan"))
        header = (
            f"- season={int(row['season'])}, week={int(row['week'])}, "
            f"team={row.get('recent_team', '?')}, opp={row.get('opponent_team', '?')}\n"
            f"- ridge_total = {ridge_total[row_pos]:.2f}, nn_total = {nn_total[row_pos]:.2f}, "
            f"actual_fp = {actual_fp:.2f}\n"
            f"- raw stat line: "
            f"pass_yd={row.get('passing_yards', 0):.0f}, pass_td={row.get('passing_tds', 0):.0f}, "
            f"rush_yd={row.get('rushing_yards', 0):.0f}, rush_td={row.get('rushing_tds', 0):.0f}, "
            f"recv_yd={row.get('receiving_yards', 0):.0f}, recv_td={row.get('receiving_tds', 0):.0f}, "
            f"rec={row.get('receptions', 0):.0f}, int={row.get('interceptions', 0):.0f}, "
            f"snap_pct={row.get('snap_pct', 0):.2f}\n"
        )
        md.append(header + "\n")

        x_raw = X_test[row_pos]
        row_json = {
            "player": target.name,
            "season": int(row["season"]),
            "week": int(row["week"]),
            "ridge_total": float(ridge_total[row_pos]),
            "nn_total": float(nn_total[row_pos]),
            "actual_fp": float(actual_fp) if pd.notna(actual_fp) else None,
            "ridge_by_target": {},
            "nn_by_target": {},
        }

        md.append("### Ridge attribution\n")
        for t in targets:
            attr = ridge_attribution(ridge_model, feature_cols, x_raw, t)
            md.append(_fmt_ridge_table(attr))
            row_json["ridge_by_target"][t] = attr

        md.append("### NN integrated gradients\n")
        for t in targets:
            attr = nn_integrated_gradients(nn_model, nn_scaler, feature_cols, x_raw, t, device)
            md.append(_fmt_nn_table(attr))
            row_json["nn_by_target"][t] = attr

        md.append("### Last 8 QB-only games (filtered splits)\n")
        hist = _player_history(combined, target.name, k=8)
        md.append(_fmt_history(hist))

        md.append("### Last 12 games across ALL positions (rolling-window source)\n")
        raw_hist = _unfiltered_history(
            bundle["raw_train"], bundle["raw_val"], bundle["raw_test"], target.name, k=12
        )
        md.append(_fmt_history(raw_hist))

        json_out["rows"].append(row_json)

    with open(MD_PATH, "w") as f:
        f.write("\n".join(md))
    with open(JSON_PATH, "w") as f:
        json.dump(json_out, f, indent=2, default=str)

    print(f"Wrote {MD_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()

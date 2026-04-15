"""Error stratification analysis for fantasy point prediction models.

Provides functions to slice prediction errors by game context, player usage,
opponent quality, and scoring patterns. Position-agnostic — works for any
position with the standard pipeline output.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Stratification column definitions
# ---------------------------------------------------------------------------

SNAP_BINS = [0, 40, 70, 100]
SNAP_LABELS = ["backup", "timeshare", "starter"]

OPP_BINS = [0, 10, 22, 32]
OPP_LABELS = ["top10_def", "mid_def", "bottom10_def"]

WEEK_BINS = [0, 6, 12, 18]
WEEK_LABELS = ["early", "mid", "late"]


def add_stratification_columns(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Add bucketed columns to the test DataFrame for stratified error analysis.

    Modifies df in-place and returns it. Handles missing columns gracefully.
    """
    # Snap percentage buckets — handle both 0-1 (decimal) and 0-100 (percent) formats
    if "snap_pct" in df.columns and df["snap_pct"].notna().any():
        snap = df["snap_pct"].fillna(0)
        if snap.max() <= 1.5:  # decimal format
            bins = [b / 100 for b in SNAP_BINS]
        else:
            bins = SNAP_BINS
        df["snap_bucket"] = pd.cut(snap, bins=bins, labels=SNAP_LABELS, include_lowest=True)
    else:
        df["snap_bucket"] = "unknown"

    # Opponent defense tier
    if "opp_def_rank_vs_pos" in df.columns:
        df["opp_tier"] = pd.cut(
            df["opp_def_rank_vs_pos"].fillna(16), bins=OPP_BINS, labels=OPP_LABELS, include_lowest=True,
        )
    else:
        df["opp_tier"] = "unknown"

    # Season phase
    df["week_phase"] = pd.cut(
        df["week"], bins=WEEK_BINS, labels=WEEK_LABELS, include_lowest=True,
    )

    # TD bucket (zero-inflated analysis)
    td_col = None
    for t in targets:
        if "td" in t.lower():
            td_col = t
            break
    if td_col and td_col in df.columns:
        df["td_bucket"] = np.where(df[td_col] > 0, "has_td", "zero_td")
    else:
        df["td_bucket"] = "unknown"

    # Volatility quartiles
    vol_col = "rolling_std_fantasy_points_L3"
    if vol_col in df.columns:
        df["volatility_q"] = pd.qcut(
            df[vol_col].fillna(0), q=4, labels=["Q1_stable", "Q2", "Q3", "Q4_volatile"],
            duplicates="drop",
        )
    else:
        df["volatility_q"] = "unknown"

    # Home/away
    if "is_home" in df.columns:
        df["home_away"] = np.where(df["is_home"] == 1, "home", "away")
    else:
        df["home_away"] = "unknown"

    return df


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_stratum_metrics(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str,
) -> pd.DataFrame:
    """Compute MAE, RMSE, and signed bias per stratum.

    Args:
        df: DataFrame with actual and predicted columns.
        y_true_col: Column name for actual values.
        y_pred_col: Column name for predicted values.
        group_col: Column to group by (a stratification bucket).

    Returns:
        DataFrame with columns: [group_col, n, mae, rmse, bias].
        bias = mean(pred - actual), positive means over-prediction.
    """
    tmp = df[[group_col, y_true_col, y_pred_col]].dropna()
    tmp = tmp.copy()
    tmp["_error"] = tmp[y_pred_col] - tmp[y_true_col]
    tmp["_abs_error"] = tmp["_error"].abs()
    tmp["_sq_error"] = tmp["_error"] ** 2

    grouped = tmp.groupby(group_col, observed=True).agg(
        n=("_error", "size"),
        mae=("_abs_error", "mean"),
        rmse=("_sq_error", "mean"),
        bias=("_error", "mean"),
    ).reset_index()
    grouped["rmse"] = np.sqrt(grouped["rmse"])
    return grouped


def run_stratified_analysis(
    df: pd.DataFrame,
    model_pred_cols: dict,
    target_cols: dict,
    strata_cols: list[str],
) -> dict:
    """Run stratified error analysis for every (stratum, model, target) combo.

    Args:
        df: Test DataFrame with stratification and prediction columns.
        model_pred_cols: {model_name: {target_name: pred_col_name, ...}, ...}
        target_cols: {target_name: actual_col_name, ...}
        strata_cols: List of stratification column names.

    Returns:
        Nested dict: {stratum_col: {model_name: {target_name: DataFrame}}}
    """
    results = {}
    for stratum in strata_cols:
        if stratum not in df.columns or df[stratum].nunique() < 2:
            continue
        results[stratum] = {}
        for model_name, pred_map in model_pred_cols.items():
            results[stratum][model_name] = {}
            for target_name, actual_col in target_cols.items():
                pred_col = pred_map.get(target_name)
                if pred_col is None or pred_col not in df.columns:
                    continue
                if actual_col not in df.columns:
                    continue
                metrics = compute_stratum_metrics(df, actual_col, pred_col, stratum)
                results[stratum][model_name][target_name] = metrics
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_stratified_table(
    results: dict, model_name: str, target_name: str,
) -> None:
    """Print a stratified error table for one model and target across all strata."""
    print(f"\n{'=' * 80}")
    print(f"Error Stratification: {model_name} — {target_name}")
    print(f"{'=' * 80}")
    print(f"{'Stratum':<20} {'Bucket':<16} {'N':>5} {'MAE':>7} {'RMSE':>7} {'Bias':>8}")
    print("-" * 80)

    for stratum, model_dict in results.items():
        if model_name not in model_dict:
            continue
        if target_name not in model_dict[model_name]:
            continue
        metrics_df = model_dict[model_name][target_name]
        for _, row in metrics_df.iterrows():
            print(f"{stratum:<20} {str(row.iloc[0]):<16} {int(row['n']):>5} "
                  f"{row['mae']:>7.3f} {row['rmse']:>7.3f} {row['bias']:>+8.3f}")
        print("-" * 80)


def find_top_error_sources(
    results: dict, model_name: str, metric: str = "mae", top_k: int = 10, min_n: int = 20,
) -> list[dict]:
    """Find the strata with the highest error for a given model.

    Returns list of dicts sorted by metric descending, filtered to n >= min_n.
    """
    rows = []
    for stratum, model_dict in results.items():
        if model_name not in model_dict:
            continue
        for target_name, metrics_df in model_dict[model_name].items():
            for _, row in metrics_df.iterrows():
                if row["n"] >= min_n:
                    rows.append({
                        "stratum": stratum,
                        "bucket": str(row.iloc[0]),
                        "target": target_name,
                        "n": int(row["n"]),
                        "mae": row["mae"],
                        "rmse": row["rmse"],
                        "bias": row["bias"],
                    })
    rows.sort(key=lambda r: r[metric], reverse=True)
    return rows[:top_k]


def print_top_error_sources(sources: list[dict], model_name: str) -> None:
    """Print the top error sources table."""
    print(f"\n{'=' * 90}")
    print(f"Top Error Sources: {model_name}")
    print(f"{'=' * 90}")
    print(f"{'#':<4} {'Stratum':<16} {'Bucket':<16} {'Target':<18} "
          f"{'N':>5} {'MAE':>7} {'Bias':>8}")
    print("-" * 90)
    for i, s in enumerate(sources, 1):
        print(f"{i:<4} {s['stratum']:<16} {s['bucket']:<16} {s['target']:<18} "
              f"{s['n']:>5} {s['mae']:>7.3f} {s['bias']:>+8.3f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_by_stratum(
    results: dict,
    model_name: str,
    stratum_name: str,
    target_names: list[str],
    save_path: str,
) -> None:
    """Grouped bar chart: MAE by stratum bucket for each target."""
    if stratum_name not in results or model_name not in results[stratum_name]:
        return

    model_results = results[stratum_name][model_name]
    buckets = None
    data = {}
    for target in target_names:
        if target not in model_results:
            continue
        df = model_results[target]
        if buckets is None:
            buckets = [str(b) for b in df.iloc[:, 0]]
        data[target] = df["mae"].values

    if buckets is None or not data:
        return

    x = np.arange(len(buckets))
    width = 0.8 / len(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (target, maes) in enumerate(data.items()):
        ax.bar(x + i * width, maes, width, label=target.replace("_", " ").title())

    ax.set_xlabel(stratum_name.replace("_", " ").title())
    ax.set_ylabel("MAE")
    ax.set_title(f"Error by {stratum_name.replace('_', ' ').title()} — {model_name}")
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.set_xticklabels(buckets)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_bias_heatmap(
    results: dict,
    model_name: str,
    strata_cols: list[str],
    target_names: list[str],
    save_path: str,
) -> None:
    """Heatmap of signed bias: rows = strata buckets, cols = targets.

    Red = over-prediction, blue = under-prediction.
    """
    row_labels = []
    bias_matrix = []

    for stratum in strata_cols:
        if stratum not in results or model_name not in results[stratum]:
            continue
        model_results = results[stratum][model_name]
        for target in target_names:
            if target not in model_results:
                continue
            df = model_results[target]
            for _, row in df.iterrows():
                label = f"{stratum}:{row.iloc[0]}"
                if label not in row_labels:
                    row_labels.append(label)

    # Build the matrix
    bias_data = {t: {} for t in target_names}
    for stratum in strata_cols:
        if stratum not in results or model_name not in results[stratum]:
            continue
        model_results = results[stratum][model_name]
        for target in target_names:
            if target not in model_results:
                continue
            df = model_results[target]
            for _, row in df.iterrows():
                label = f"{stratum}:{row.iloc[0]}"
                bias_data[target][label] = row["bias"]

    if not row_labels:
        return

    matrix = np.zeros((len(row_labels), len(target_names)))
    for j, target in enumerate(target_names):
        for i, label in enumerate(row_labels):
            matrix[i, j] = bias_data[target].get(label, 0)

    fig, ax = plt.subplots(figsize=(8, max(6, len(row_labels) * 0.4)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.5)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels([t.replace("_", " ").title() for t in target_names], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"Prediction Bias by Stratum — {model_name}\n(Red=over-predict, Blue=under-predict)")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(target_names)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, label="Bias (pred - actual)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_td_zero_vs_scored(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    save_path: str,
    title: str = "",
) -> None:
    """Two-panel error histogram: 0-TD games vs 1+TD games."""
    if actual_col not in df.columns or pred_col not in df.columns:
        return

    zero_mask = df[actual_col] == 0
    has_td_mask = df[actual_col] > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mask, label in [
        (ax1, zero_mask, "0-TD Games"),
        (ax2, has_td_mask, "1+ TD Games"),
    ]:
        errors = df.loc[mask, pred_col] - df.loc[mask, actual_col]
        if len(errors) == 0:
            continue
        ax.hist(errors, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
        mae = errors.abs().mean()
        bias = errors.mean()
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(bias, color="red", linestyle="-", linewidth=2, label=f"Bias: {bias:+.2f}")
        ax.set_title(f"{label} (n={mask.sum()}, MAE={mae:.2f})")
        ax.set_xlabel("Prediction Error (pred - actual)")
        ax.legend()

    fig.suptitle(title or f"TD Prediction Errors: {pred_col}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

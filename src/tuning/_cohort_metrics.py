"""Subgroup reports shared by offline A/B specifications."""

import numpy as np


def receiving_boom_metrics(result, position):
    """Per-model overall MAE/bias + boom-subgroup bias/RMSE/correlation.

    The gap is the boom tier, so judge there, not on overall MAE (a feature on ~1% of
    rows dilutes to noise). Subgroups are defined on *actuals* — ``q4`` = top fantasy-point
    quartile, ``rztd`` = receiving-TD games — so the slice is identical across arms with no
    baseline injection. ``correlation`` (Pearson pred-vs-actual on the slice) is the
    decomposition's closable edge. ``mae`` (overall) feeds the harness Ridge sentinel.
    """
    from src.evaluation.metrics import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)

    cuts: dict = {}
    if len(df):
        q75 = float(np.quantile(df["fantasy_points"].to_numpy(dtype=float), 0.75))
        cuts["q4"] = df[df["fantasy_points"] >= q75]
        if "receiving_tds" in df.columns:
            cuts["rztd"] = df[df["receiving_tds"] >= 1]
    sub_m = {k: per_model_metrics(v, models) for k, v in cuts.items()}

    def _corr(sub, col):
        if len(sub) < 2:
            return float("nan")
        a = sub[col].to_numpy(dtype=float)
        b = sub["fantasy_points"].to_numpy(dtype=float)
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out: dict = {}
    for name, col in models.items():
        row = {"mae": float(overall[name]["mae"]), "bias": float(overall[name]["bias"])}
        for k, sub in cuts.items():
            row[f"{k}_bias"] = float(sub_m[k][name]["bias"])
            row[f"{k}_rmse"] = float(sub_m[k][name]["rmse"])
            row[f"{k}_corr"] = _corr(sub, col)
            row[f"{k}_n"] = float(sub_m[k][name]["n"])
        out[name] = row
    return out


def inheritance_metrics(result, position):
    """Per-model overall MAE/bias PLUS inheritor-subgroup (``inherited_opportunity > 0``).

    Judge the targeted subgroup, not overall MAE (the feature fires on few rows). The column
    is injected into *every* arm (baseline carries it un-whitelisted), so the slice is identical
    across arms. ``inh_n`` flags whether the TE cohort is large enough to read.
    """
    from src.evaluation.metrics import available_models, per_model_metrics

    df = result["test_df"]
    models = available_models(df)
    overall = per_model_metrics(df, models)
    sub = (
        df[df["inherited_opportunity"] > 0]
        if "inherited_opportunity" in df.columns
        else df.iloc[0:0]
    )
    sub_m = per_model_metrics(sub, models) if len(sub) else {}
    out: dict = {}
    for m, mv in overall.items():
        row = {"mae": float(mv["mae"]), "bias": float(mv["bias"])}
        if m in sub_m:
            row["inh_mae"] = float(sub_m[m]["mae"])
            row["inh_bias"] = float(sub_m[m]["bias"])
            row["inh_n"] = float(sub_m[m]["n"])
        out[m] = row
    return out

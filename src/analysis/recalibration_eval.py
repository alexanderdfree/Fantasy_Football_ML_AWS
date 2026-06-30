"""P0 diagnostic: validate per-position monotone recalibration + head selection (zero retrain).

⚠️ CONCLUSION (do not act on the single-season numbers below): the recalibration lever this script
evaluates was **REJECTED on multi-season replication** — it reduces elite *bias* but not *RMSE* and
*hurts* RB/WR lineup regret every season (2022-2025). The dramatic "95.9% QB / 100% TE closure" first
read was a **stale-QB-artifact bug** in ``artifact_eval`` (now caught by ``validate_reconstruction``).
This module is kept as a reusable LOWO-isotonic diagnostic; the verdict lives in
``todo/expert-gap-investigation-2026-06.md`` (and ``src/tuning/ab_rolling_origin_rotowire.py`` is the
authoritative fresh-trained, multi-season comparison).

For each of QB/RB/WR/TE this scores the **served** model heads (no retraining — predictions come
from the validated 2025 substrate built off ``src/{pos}/outputs/models`` via
``artifact_eval.build_test_df_from_artifacts``) and asks two questions the expert-gap benchmark
raised:

1. **Recalibration** — does a leave-one-week-out (LOWO) isotonic recalibration of a head's fantasy
   total close the elite RMSE gap vs the experts (the workflow's E4 claim: 81-97% QB / 80-90% TE)?
   LOWO = for each week w, fit ``IsotonicRegression`` on every *other* week and apply to w, so the
   reported closure is on held-out folds (no in-fold overfit). Monotone single-head transform — not
   an ensemble (ADR-0003 clean), post-hoc (no retrain, no loss-coupling change).

2. **Head selection** — does our own LightGBM rank better than the served attn_nn for RB/WR (the
   "served-head artifact"), and does recalibrating attn_nn close that? Compares the four candidates
   {attn_nn, lgbm, recal(attn_nn), recal(lgbm)} on weekly precision@N and optimal-lineup regret.

Also reconciles the QB-slice question: our −3.46 FP under-level is on ``depth_chart_rank==1``
starters; the scoring-tier doc's "QB needs nothing (+0.47)" is the a-priori prior-season-rank elite.

Reads ``scratchpad/substrate/{pos}.parquet`` (built + validated by
``scratchpad/substrate/build_substrate.py``); falls back to rebuilding from artifacts if absent.
Writes ``scratchpad/recal/recal_eval.{json,md}``. Run: ``python -m src.analysis.recalibration_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SUBSTRATE = Path("scratchpad/substrate")
OUT = Path("scratchpad/recal")
POSITIONS = ["QB", "RB", "WR", "TE"]
KEYS = ["player_id", "season", "week"]
# Weekly startable depth per position (1QB/1TE, ~2RB+flex, ~3WR+flex in a 12-team league).
LINEUP_N = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
HEADS = ["pred_attn_nn_total", "pred_lgbm_total"]
EXPERTS = [("nflcom", "nflcom_pred"), ("rotowire", "rotowire_pred")]


def _load_pos(pos: str) -> pd.DataFrame:
    p = SUBSTRATE / f"{pos}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["player_id"] = df["player_id"].astype(str)
        return df
    raise FileNotFoundError(
        f"{p} missing — run `python -m scratchpad.substrate.build_substrate` first"
    )


def _top30_ids(pos: str, df: pd.DataFrame) -> set[str]:
    """Elite cohort = top-30 by actual season-total FP (the comparison_experts.json basis)."""
    cj = Path("src/serving/comparison_experts.json")
    if cj.exists():
        with open(cj) as f:
            ids = json.load(f).get("top30_ids", {}).get(pos)
        if ids:
            return {str(x) for x in ids}
    tot = df.groupby("player_id")["fantasy_points"].sum().nlargest(30)
    return set(tot.index)


def lowo_isotonic(df: pd.DataFrame, predcol: str, ycol: str = "fantasy_points") -> np.ndarray:
    """Leave-one-week-out isotonic recalibration of ``predcol`` → ``ycol`` (held-out per week)."""
    out = np.full(len(df), np.nan)
    pred = df[predcol].to_numpy()
    y = df[ycol].to_numpy()
    wk = df["week"].to_numpy()
    mask_valid = ~np.isnan(pred) & ~np.isnan(y)
    for w in np.unique(wk):
        tr = mask_valid & (wk != w)
        te = (wk == w) & ~np.isnan(pred)
        if tr.sum() < 20 or te.sum() == 0:
            out[te] = pred[te]  # too little to fit — pass through
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(pred[tr], y[tr])
        out[te] = iso.predict(pred[te])
    return out


def _rmse(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _bias(p, y):
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def _weekly_precision_at_n(df: pd.DataFrame, predcol: str, n: int) -> float:
    """Mean over weeks of |actual_topN ∩ pred_topN| / N."""
    rates = []
    for _, g in df.groupby("week"):
        gg = g[g[predcol].notna()]
        if len(gg) < n:
            continue
        a = set(gg.nlargest(n, "fantasy_points")["player_id"])
        p = set(gg.nlargest(n, predcol)["player_id"])
        rates.append(len(a & p) / n)
    return float(np.mean(rates)) if rates else float("nan")


def _weekly_regret(df: pd.DataFrame, predcol: str, n: int) -> float:
    """Mean per-week points left on bench: (sum actual of true topN) − (sum actual of proj topN)."""
    regrets = []
    for _, g in df.groupby("week"):
        gg = g[g[predcol].notna()]
        if len(gg) < n:
            continue
        opt = gg.nlargest(n, "fantasy_points")["fantasy_points"].sum()
        got = gg.nlargest(n, predcol)["fantasy_points"].sum()
        regrets.append(opt - got)
    return float(np.mean(regrets)) if regrets else float("nan")


def evaluate_position(pos: str) -> dict:
    df = _load_pos(pos)
    df = df[df["fantasy_points"].notna()].copy()
    n = LINEUP_N[pos]

    # --- recalibrated heads (LOWO isotonic, fit on the full position slate) ---
    recal = {}
    for h in HEADS:
        df[f"recal_{h}"] = lowo_isotonic(df, h)
        recal[h] = f"recal_{h}"

    elite_ids = _top30_ids(pos, df)
    elite = df[df["player_id"].isin(elite_ids)].copy()

    res: dict = {
        "pos": pos,
        "n_rows": int(len(df)),
        "n_elite_rows": int(len(elite)),
        "lineup_n": n,
        "experts": {},
        "head_selection": {},
        "qb_slice": {},
    }

    # --- 1. recalibration: elite RMSE-gap closure vs each expert (intersection cohort) ---
    for ename, ecol in EXPERTS:
        sub = elite[elite[ecol].notna()]
        if sub.empty:
            continue
        y = sub["fantasy_points"].to_numpy()
        rmse_exp = _rmse(sub[ecol], y)
        block = {"n": int(len(sub)), "expert_rmse": round(rmse_exp, 3), "heads": {}}
        for h in HEADS:
            rmse_raw = _rmse(sub[h], y)
            rmse_rec = _rmse(sub[recal[h]], y)
            gap_raw = rmse_raw - rmse_exp
            gap_rec = rmse_rec - rmse_exp
            closure = (1 - gap_rec / gap_raw) if gap_raw > 1e-9 else None
            block["heads"][h] = {
                "rmse_raw": round(rmse_raw, 3),
                "rmse_recal": round(rmse_rec, 3),
                "bias_raw": round(_bias(sub[h], y), 3),
                "bias_recal": round(_bias(sub[recal[h]], y), 3),
                "gap_raw": round(gap_raw, 3),
                "gap_recal": round(gap_rec, 3),
                "pct_gap_closed": None if closure is None else round(100 * closure, 1),
            }
        res["experts"][ename] = block

    # --- 2. head selection: decision metrics on the FULL slate for all 4 candidates + experts ---
    cands = {
        "attn_nn": "pred_attn_nn_total",
        "lgbm": "pred_lgbm_total",
        "recal_attn": "recal_pred_attn_nn_total",
        "recal_lgbm": "recal_pred_lgbm_total",
    }
    for ename, ecol in EXPERTS:
        cands[ename] = ecol
    for name, col in cands.items():
        scope = df if not name.startswith(("nflcom", "rotowire")) else df[df[col].notna()]
        res["head_selection"][name] = {
            f"precision@{n}": round(_weekly_precision_at_n(scope, col, n), 4),
            "precision@12": round(_weekly_precision_at_n(scope, col, 12), 4),
            f"regret@{n}": round(_weekly_regret(scope, col, n), 2),
            "n_scope": int(len(scope)),
        }

    # --- 3. QB-slice reconciliation: bias on dcr1 starters vs a-priori prior-season elite ---
    if "depth_chart_rank" in df.columns:
        dcr1 = df[df["depth_chart_rank"] == 1]
        res["qb_slice"]["dcr1_starters"] = _slice_bias(dcr1, recal)
    if "prior_season_mean_fantasy_points" in df.columns:
        thr = df["prior_season_mean_fantasy_points"].quantile(0.80)  # top-quintile a-priori elite
        apriori = df[df["prior_season_mean_fantasy_points"] >= thr]
        res["qb_slice"]["apriori_prior_elite_top20pct"] = _slice_bias(apriori, recal)

    return res


def _slice_bias(sub: pd.DataFrame, recal: dict) -> dict:
    if sub.empty:
        return {"n": 0}
    y = sub["fantasy_points"].to_numpy()
    out = {"n": int(len(sub))}
    for h in HEADS:
        out[h.replace("pred_", "").replace("_total", "")] = round(_bias(sub[h], y), 2)
        out["recal_" + h.replace("pred_", "").replace("_total", "")] = round(
            _bias(sub[recal[h]], y), 2
        )
    for ename, ecol in EXPERTS:
        s = sub[sub[ecol].notna()]
        if not s.empty:
            out[ename] = round(_bias(s[ecol], s["fantasy_points"]), 2)
    return out


def _render_md(results: list[dict]) -> str:
    L = ["# P0 — Recalibration + head-selection validation (LOWO isotonic, served heads, 2025)\n"]
    L.append(
        "All numbers are on held-out leave-one-week-out folds. `recal_*` = LOWO-isotonic-recalibrated head.\n"
    )
    L.append("## 1. Elite RMSE-gap closure vs experts (top-30-by-actual cohort, intersection)\n")
    L.append(
        "| pos | expert | head | rmse_raw | rmse_recal | expert_rmse | bias_raw→recal | % gap closed |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        for ename, b in r["experts"].items():
            for h, hb in b["heads"].items():
                hn = h.replace("pred_", "").replace("_total", "")
                L.append(
                    f"| {r['pos']} | {ename} | {hn} | {hb['rmse_raw']} | {hb['rmse_recal']} | "
                    f"{b['expert_rmse']} | {hb['bias_raw']}→{hb['bias_recal']} | {hb['pct_gap_closed']} |"
                )
    L.append("\n## 2. Head selection — decision metrics, full slate (which head to serve)\n")
    L.append("| pos | candidate | precision@N | precision@12 | regret@N (pts/wk) |")
    L.append("|---|---|---|---|---|")
    for r in results:
        n = r["lineup_n"]
        for name, m in r["head_selection"].items():
            L.append(
                f"| {r['pos']} | {name} | {m.get(f'precision@{n}')} | {m['precision@12']} | {m.get(f'regret@{n}')} |"
            )
    L.append("\n## 3. QB-slice reconciliation — bias (mean pred−actual; negative = under-level)\n")
    for r in results:
        if not r["qb_slice"]:
            continue
        L.append(f"### {r['pos']}")
        for slname, sb in r["qb_slice"].items():
            L.append(
                f"- **{slname}** (n={sb.get('n')}): "
                + ", ".join(f"{k}={v}" for k, v in sb.items() if k != "n")
            )
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [evaluate_position(p) for p in POSITIONS]
    with open(OUT / "recal_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    md = _render_md(results)
    (OUT / "recal_eval.md").write_text(md)
    print(md)
    print(f"\n[recal] wrote {OUT / 'recal_eval.json'} and {OUT / 'recal_eval.md'}")


if __name__ == "__main__":
    main()

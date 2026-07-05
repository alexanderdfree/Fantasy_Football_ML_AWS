/* Comparison — our four model architectures (live) vs expert projection sources
 * (NFL.com, RotoWire), by position, on three player subsets (all + top-30 +
 * top-12/position). One /api/comparison fetch (module-level cache, mirroring the
 * vanilla comparisonLoaded flag); the MAE/RMSE/R² toggle re-renders from the
 * cached payload. Lower is better for MAE/RMSE, higher for R²; best cell per row
 * is highlighted. */
import { useEffect, useState } from "react";
import { fetchJSON } from "../api.js";
import { PillGroup, ApproachBanner } from "../components/common.jsx";

const COMPARISON_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"];
// Our four model architectures, then the static expert sources. Keys match the
// per-model blocks in the /api/comparison payload (model prefixes) and the expert
// cell keys. Shared (via COMPARISON_SOURCES) by the accuracy tables and the
// quartile-bias table.
const MODEL_SOURCES = [
    { key: "ridge", label: "Ridge" },
    { key: "nn", label: "Neural Net" },
    { key: "attn_nn", label: "Attention NN" },
    { key: "lgbm", label: "LightGBM" },
];
const EXPERT_SOURCES = [
    { key: "nflcom", label: "NFL.com" },
    { key: "rotowire", label: "RotoWire" },
];
const COMPARISON_SOURCES = [...MODEL_SOURCES, ...EXPERT_SOURCES];
const COMPARISON_METRIC_HINTS = {
    mae: "Mean absolute error — lower is better",
    rmse: "Root mean squared error — lower is better",
    r2: "R² (coefficient of determination) — higher is better",
};
const METRIC_OPTIONS = [
    { value: "mae", label: "MAE" },
    { value: "rmse", label: "RMSE" },
    { value: "r2", label: "R²" },
];
const QUARTILE_LABELS = { Q1: "Q1 · lowest", Q2: "Q2", Q3: "Q3", Q4: "Q4 · highest" };

// Module-level cache so tab revisits don't refetch (mirrors prod comparisonLoaded;
// only a successful payload is cached — an error retries on the next visit).
let cachedComparison = null;

function comparisonCellValue(cell, metric) {
    if (!cell) return null;
    const v = cell[metric];
    return v === null || v === undefined || Number.isNaN(v) ? null : v;
}

function formatComparisonValue(v, metric) {
    return metric === "r2" ? v.toFixed(3) : v.toFixed(2);
}

/* Shared 7-column header: first label ("Position" / "Quartile"), then the six
 * source columns in COMPARISON_SOURCES order. */
function ComparisonTableHead({ firstLabel }) {
    return (
        <thead>
            <tr>
                <th>{firstLabel}</th>
                {COMPARISON_SOURCES.map((s) => (
                    <th key={s.key} className="comparison-num">{s.label}</th>
                ))}
            </tr>
        </thead>
    );
}

/* Port of renderComparisonRows: one row per position, best value per row
 * highlighted (max for R², min otherwise); missing cells render an em dash. */
function ComparisonRows({ posMap, metric }) {
    const higherBetter = metric === "r2";
    return COMPARISON_POSITIONS.map((pos) => {
        const cells = posMap[pos] || {};
        const values = COMPARISON_SOURCES.map((s) => comparisonCellValue(cells[s.key], metric)).filter(
            (v) => v !== null
        );
        const best = values.length ? (higherBetter ? Math.max(...values) : Math.min(...values)) : null;
        return (
            <tr key={pos}>
                <td className="comparison-pos">{pos}</td>
                {COMPARISON_SOURCES.map((s) => {
                    const v = comparisonCellValue(cells[s.key], metric);
                    if (v === null) {
                        return <td key={s.key} className="comparison-num comparison-empty">{"—"}</td>;
                    }
                    const isBest = best !== null && Math.abs(v - best) < 1e-9;
                    return (
                        <td key={s.key} className={"comparison-num" + (isBest ? " comparison-best" : "")}>
                            {formatComparisonValue(v, metric)}
                        </td>
                    );
                })}
            </tr>
        );
    });
}

/* One of the three accuracy tables (all / top-30 / top-12). */
function ComparisonSubsetBlock({ header, bodyId, posMap, metric, error }) {
    return (
        <div className="comparison-table-block">
            <div className="section-header">{header}</div>
            <div className="table-container">
                <table className="comparison-table">
                    <ComparisonTableHead firstLabel="Position" />
                    <tbody id={bodyId}>
                        {error ? (
                            <tr><td colSpan={7} className="arch-error">Failed to load: {error}</td></tr>
                        ) : posMap ? (
                            <ComparisonRows posMap={posMap} metric={metric} />
                        ) : (
                            <tr><td colSpan={7} className="arch-loading">Loading comparison…</td></tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

/* Port of quartileBiasCell: signed value, background tinted by magnitude (red =
 * over-, blue = under-prediction); MAE + n on hover. Empty when the source has
 * no rows in the bin. ~6 pts saturates the tint. */
function QuartileBiasCell({ cell }) {
    if (!cell || cell.bias === null || cell.bias === undefined || Number.isNaN(cell.bias)) {
        return <td className="comparison-num comparison-empty">{"—"}</td>;
    }
    const bias = cell.bias;
    const sign = bias >= 0 ? "+" : "";
    const dir = bias > 0 ? "over" : bias < 0 ? "under" : "even";
    const mag = Math.min(Math.abs(bias) / 6, 1);
    const alpha = (0.08 + 0.42 * mag).toFixed(3);
    const rgb = bias >= 0 ? "220,38,38" : "37,99,235";
    const style = dir === "even" ? undefined : { background: `rgba(${rgb},${alpha})` };
    const title = `bias ${sign}${bias.toFixed(2)} pts (${dir}-predicts) · MAE ${cell.mae.toFixed(2)} · n=${cell.n}`;
    return (
        <td className="comparison-num" style={style} title={title}>
            {sign}{bias.toFixed(2)}
        </td>
    );
}

export function ComparisonView({ scoring, search, theme, onPlayer, activateView }) {
    const [data, setData] = useState(cachedComparison);
    const [error, setError] = useState(null);
    const [metric, setMetric] = useState("mae");
    const [quartilePos, setQuartilePos] = useState("QB");

    useEffect(() => {
        if (cachedComparison) return;
        let cancelled = false;
        (async () => {
            try {
                const payload = await fetchJSON("/api/comparison");
                if (payload.error) throw new Error(payload.error);
                cachedComparison = payload;
                if (!cancelled) setData(payload);
            } catch (e) {
                console.error("Failed to load comparison:", e);
                if (!cancelled) setError(e.message);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const subsets = (data && data.subsets) || {};

    // Quartile bias — default to the first position that has data; disable any without.
    const qb = (data && data.quartile_bias) || {};
    const hasQuartile = COMPARISON_POSITIONS.some((p) => qb[p]);
    const activeQuartilePos = qb[quartilePos]
        ? quartilePos
        : (COMPARISON_POSITIONS.find((p) => qb[p]) || quartilePos);
    const quartileByPos = qb[activeQuartilePos] || {};
    const quartilePosOptions = data
        ? COMPARISON_POSITIONS.map((pos) => ({ value: pos, label: pos, disabled: !qb[pos] }))
        : [];

    // Notes (port of renderComparisonNotes).
    const meta = (data && data.experts_meta) || {};
    const date = ((data && data.generated_at) || "").slice(0, 10);
    const unavailable = data && data.model_source === "unavailable";
    const nflNote = (meta.nflcom && meta.nflcom.note) || "";
    const rwNote = (meta.rotowire && meta.rotowire.note) || "";
    const modelLine = unavailable
        ? "Currently unavailable (models not loaded). "
        : "Each of our four architectures is computed live from the deployed models, one column per architecture, so they track the latest retrain. ";

    const onWikiLink = (ev) => {
        ev.preventDefault();
        const hash = "#wiki:expert-comparison";
        if ((location.hash || "") !== hash) {
            history.pushState(null, "", location.pathname + location.search + hash);
        }
        activateView("wiki");
    };

    return (
        <section id="view-comparison" className="view active">
            <ApproachBanner icon="chart" title="Our Models vs Expert Projections">
                Weekly fantasy-point accuracy on the 2025 test season, by position. Lower MAE / RMSE is better; higher R² is better. Each of our four model architectures has its own column, updated live from the deployed models; the expert columns are scored offline against the same actuals. The best cell in each row is highlighted.
            </ApproachBanner>

            <div className="comparison-controls">
                <span className="comparison-metric-label">Metric</span>
                <PillGroup
                    id="comparison-metric-toggle"
                    className="pill-group comparison-metric-toggle"
                    options={METRIC_OPTIONS}
                    value={metric}
                    onChange={setMetric}
                />
                <span className="comparison-metric-hint" id="comparison-metric-hint">
                    {COMPARISON_METRIC_HINTS[metric] || ""}
                </span>
            </div>

            <ComparisonSubsetBlock
                header="All rostered players (2025)"
                bodyId="comparison-all-body"
                posMap={data ? (subsets.all || {}) : null}
                metric={metric}
                error={error}
            />
            <ComparisonSubsetBlock
                header="Top 30 per position (2025)"
                bodyId="comparison-top30-body"
                posMap={data ? (subsets.top30 || {}) : null}
                metric={metric}
                error={error}
            />
            <ComparisonSubsetBlock
                header="Top 12 per position (2025)"
                bodyId="comparison-top12-body"
                posMap={data ? (subsets.top12 || {}) : null}
                metric={metric}
                error={error}
            />

            {(!data || hasQuartile) && (
                <div className="comparison-table-block" id="comparison-quartile-block">
                    <div className="section-header">Bias by scoring quartile (2025)</div>
                    <div className="comparison-reliability-sub">
                        Players are split into quartiles by their <strong>actual</strong> fantasy points for the
                        selected position — Q1 = lowest scorers, Q4 = the highest / boom weeks. Each cell is the
                        source's <strong>signed bias</strong>, mean(prediction − actual), in that quartile:{" "}
                        <span style={{ color: "#dc2626", fontWeight: 600 }}>red over-predicts (+)</span>,{" "}
                        <span style={{ color: "#2563eb", fontWeight: 600 }}>blue under-predicts (−)</span>. This exposes
                        systematic miss patterns the overall MAE hides — most notably regression-to-the-mean
                        under-prediction of the Q4 boom tier. Computed live on the 2025 test season; hover a cell
                        for its MAE and sample size.
                    </div>
                    <div className="intervals-examples-controls">
                        <span className="comparison-metric-label">Position</span>
                        <PillGroup
                            id="quartile-pos-toggle"
                            options={quartilePosOptions}
                            value={activeQuartilePos}
                            onChange={setQuartilePos}
                        />
                    </div>
                    <div className="table-container">
                        <table className="comparison-table">
                            <ComparisonTableHead firstLabel="Quartile" />
                            <tbody id="quartile-bias-body">
                                {data ? (
                                    ["Q1", "Q2", "Q3", "Q4"].map((q) => {
                                        const row = quartileByPos[q] || {};
                                        return (
                                            <tr key={q}>
                                                <td className="comparison-pos">{QUARTILE_LABELS[q]}</td>
                                                {COMPARISON_SOURCES.map((s) => (
                                                    <QuartileBiasCell key={s.key} cell={row[s.key]} />
                                                ))}
                                            </tr>
                                        );
                                    })
                                ) : (
                                    <tr><td colSpan={7} className="arch-loading">Loading quartile bias…</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            <div className="comparison-notes" id="comparison-notes">
                {data && (
                    <>
                        <div className="section-header">About this comparison</div>
                        <ul className="comparison-note-list">
                            <li><strong>Seasons.</strong> Our model trains on 2012–2023, validates on 2024, and is tested on <strong>2025</strong>; every number here is on the held-out 2025 season, and the experts are scored on 2025 too.</li>
                            <li><strong>Scoring.</strong> Full PPR (1 pt / reception). Projections and actuals run through the same scoring formula, so it's apples-to-apples. RMSE is shown alongside MAE because expert projections implicitly target squared error.</li>
                            <li><strong>Our models.</strong> {modelLine}MAE/RMSE/R² are on weekly fantasy-point totals; the best cell in each row is highlighted.</li>
                            <li><strong>NFL.com.</strong> {nflNote}</li>
                            <li><strong>RotoWire.</strong> {rwNote}</li>
                            <li><strong>Top 30.</strong> The second table restricts to the top 30 players per position by actual 2025 fantasy points — the fantasy-relevant starters.</li>
                            <li><strong>Top 12.</strong> The third table tightens further to the top 12 per position by actual 2025 fantasy points — roughly a standard league's starters at each spot.</li>
                            <li>
                                <strong>Caveat.</strong> Each source is scored on the players it actually projects, so this is an approximate scoreboard rather than a strictly paired test. For the rigorous paired, significance-tested head-to-heads, see the{" "}
                                <a
                                    href="#wiki:expert-comparison"
                                    className="comparison-link"
                                    data-slug="expert-comparison"
                                    onClick={onWikiLink}
                                >
                                    Expert Projection Comparison
                                </a>{" "}
                                wiki page.
                            </li>
                            {date && <li className="comparison-note-meta">Expert data generated {date}.</li>}
                        </ul>
                    </>
                )}
            </div>
        </section>
    );
}

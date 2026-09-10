/* Comparison — models and archived expert forecasts on shared player-weeks.
 * Expected starters use a fixed pregame reference; seasonal leaders and weekly
 * leader capture are separate diagnostics. One /api/comparison fetch (mirroring the
 * vanilla comparisonLoaded flag); the MAE/RMSE/R² toggle re-renders from the
 * cached payload. Lower is better for MAE/RMSE, higher for R²; best cell per row
 * is highlighted. */
import { useEffect, useState } from "react";
import { fetchJSON } from "../api.js";
import { PillGroup, ApproachBanner } from "../components/common.jsx";

const COMPARISON_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"];
// Our four model architectures, then the archived expert sources. Keys match the
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
    { key: "espn", label: "ESPN" },
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
    if (metric === "hit_rate") return `${(v * 100).toFixed(1)}%`;
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
function ComparisonRows({ posMap, metric, coverage }) {
    const higherBetter = metric === "r2" || metric === "hit_rate";
    return COMPARISON_POSITIONS.map((pos) => {
        const cells = posMap[pos] || {};
        const values = COMPARISON_SOURCES.map((s) => comparisonCellValue(cells[s.key], metric)).filter(
            (v) => v !== null
        );
        const best = values.length ? (higherBetter ? Math.max(...values) : Math.min(...values)) : null;
        return (
            <tr key={pos}>
                <td className="comparison-pos">
                    {pos}
                    {coverage?.[pos] && (
                        <div style={{ fontSize: "0.72rem", fontWeight: 400 }}>
                            {coverage[pos].status === "unavailable"
                                ? "Unavailable"
                                : `${coverage[pos].n} player-weeks${coverage[pos].status === "partial" ? " · partial reference" : ""}`}
                        </div>
                    )}
                </td>
                {COMPARISON_SOURCES.map((s) => {
                    const v = comparisonCellValue(cells[s.key], metric);
                    if (v === null) {
                        return <td key={s.key} className="comparison-num comparison-empty">{"—"}</td>;
                    }
                    const isBest = best !== null && Math.abs(v - best) < 1e-9;
                    return (
                        <td key={s.key} className={"comparison-num" + (isBest ? " comparison-best" : "")}
                            title={cells[s.key]?.n_weeks != null ? `${cells[s.key].n_weeks} comparable weeks` : undefined}>
                            {formatComparisonValue(v, metric)}
                        </td>
                    );
                })}
            </tr>
        );
    });
}

/* One of the three accuracy tables (all / top-30 / top-12). */
function ComparisonSubsetBlock({ header, bodyId, posMap, metric, error, coverage }) {
    return (
        <div className="comparison-table-block">
            <div className="section-header">{header}</div>
            <div className="table-container">
                <table className="comparison-table">
                    <ComparisonTableHead firstLabel="Position" />
                    <tbody id={bodyId}>
                        {error ? (
                            <tr><td colSpan={COMPARISON_SOURCES.length + 1} className="arch-error">Failed to load: {error}</td></tr>
                        ) : posMap ? (
                            <ComparisonRows posMap={posMap} metric={metric} coverage={coverage} />
                        ) : (
                            <tr><td colSpan={COMPARISON_SOURCES.length + 1} className="arch-loading">Loading comparison…</td></tr>
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
    const espnNote = (meta.espn && meta.espn.note) || "";
    const modelLine = unavailable
        ? "Evaluation is currently unavailable. "
        : "Each architecture uses its deployed forecasts, and all sources are graded on identical player-weeks. ";

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
                Weekly fantasy-point accuracy on the 2025 regular season. Every source is graded against full PPR actuals on identical player-weeks. Expected starters are selected before kickoff using a shared expert reference. Lower MAE / RMSE is better; higher R² is better.
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
                header="Expected starters · weekly top 24"
                bodyId="comparison-weekly-top24"
                posMap={data?.subsets?.weekly_reference_top24}
                coverage={data?.coverage?.weekly_reference_top24}
                metric={metric}
                error={error}
            />

            <ComparisonSubsetBlock
                header="All comparable player-weeks (2025)"
                coverage={data?.coverage?.all}
                bodyId="comparison-all-body"
                posMap={data ? (subsets.all || {}) : null}
                metric={metric}
                error={error}
            />
            <ComparisonSubsetBlock
                header="Season leaders · top 30 (2025)"
                coverage={data?.coverage?.top30}
                bodyId="comparison-top30-body"
                posMap={data ? (subsets.top30 || {}) : null}
                metric={metric}
                error={error}
            />
            <ComparisonSubsetBlock
                header="Season leaders · top 12 (2025)"
                coverage={data?.coverage?.top12}
                bodyId="comparison-top12-body"
                posMap={data ? (subsets.top12 || {}) : null}
                metric={metric}
                error={error}
            />

            <ComparisonSubsetBlock
                header="Weekly top-24 leader capture · higher is better"
                bodyId="comparison-weekly-capture"
                posMap={data?.weekly_ranking}
                metric="hit_rate"
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
                        patterns hidden by the overall MAE. Selecting high-scoring weeks after the fact naturally
                        produces negative bias, even for sensible forecasts; it is not a target for raising every
                        projection. Hover a cell for its MAE and sample size.
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
                                    <tr><td colSpan={COMPARISON_SOURCES.length + 1} className="arch-loading">Loading quartile bias…</td></tr>
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
                            <li><strong>Seasons.</strong> Our model trains on 2013–2023 (2012 is loaded for prior-season context only), validates on 2024, and is tested on <strong>2025</strong>; every number here is on the held-out 2025 season, and the experts are scored on 2025 too.</li>
                            <li><strong>Scoring.</strong> Every source is graded against full regular-season PPR actuals, including rushing points for receivers and receiving points for quarterbacks.</li>
                            <li><strong>Our models.</strong> {modelLine}MAE/RMSE/R² are on weekly fantasy-point totals; the best cell in each row is highlighted.</li>
                            <li><strong>NFL.com.</strong> {nflNote}</li>
                            <li><strong>RotoWire.</strong> {rwNote}</li>
                            <li><strong>ESPN.</strong> {espnNote}</li>
                            <li><strong>Expected starters.</strong> The weekly top 24 uses a fixed average of archived NFL.com and RotoWire forecasts, with NFL.com alone for K and RotoWire alone for DST. Selection happens before filtering for recorded outcomes or model coverage. Missing reference weeks are reported explicitly.</li>
                            <li><strong>Season leaders.</strong> Top 30 and top 12 use total actual regular-season points, excluding playoffs. These are retrospective diagnostics, not pregame starter lists.</li>
                            <li><strong>Weekly leader capture.</strong> The fraction of actual weekly top-24 scorers selected by each source's own forecasts. Only weeks with at least 24 comparable players count; hover for the number of weeks.</li>
                            <li>
                                <strong>Coverage.</strong> Every displayed source in a position is scored on the same player-weeks. Missing forecasts are excluded, never treated as zero. Sample sizes appear beside each position. Historical investigations and uncertainty estimates are available in the{" "}
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
                            {date && <li className="comparison-note-meta">Evaluation calculated {date}.</li>}
                        </ul>
                    </>
                )}
            </div>
        </section>
    );
}

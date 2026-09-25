/* Comparison — models and archived expert forecasts on shared player-weeks.
 * Expected starters come from pregame depth charts and prior-season importance,
 * which no graded source selects; the archived expert reference is a secondary
 * view, and seasonal leaders and weekly leader capture are separate diagnostics.
 * One /api/comparison fetch (mirroring the vanilla comparisonLoaded flag); the
 * metric toggle re-renders from the cached payload. A row names a winner only
 * when the server's paired bootstrap interval for best model minus best expert
 * excludes zero under both MAE and RMSE; otherwise it reads "≈ tie". Sources that
 * are graded nowhere (NFL.com offense) are omitted from the tables. */
import { useEffect, useState } from "react";
import { fetchJSON } from "../api.js";
import { contract } from "../api-contract.js";
import { PillGroup, ApproachBanner } from "../components/common.jsx";

const COMPARISON_POSITIONS = contract.positions;
// Our four model architectures, then the archived expert sources. Keys match the
// per-model blocks in the /api/comparison payload (model prefixes) and the expert
// cell keys. Shared (via COMPARISON_SOURCES) by the accuracy tables and the
// quartile-bias table.
const SOURCE_LABELS = { ridge: "Ridge", nn: "Neural Net", attn_nn: "Attention NN", lgbm: "LightGBM", nflcom: "NFL.com", rotowire: "RotoWire", espn: "ESPN" };
const MODEL_SOURCES = contract.model_sources.map((key) => ({ key, label: SOURCE_LABELS[key] || key }));
const EXPERT_SOURCES = contract.expert_sources.map((key) => ({ key, label: SOURCE_LABELS[key] || key }));
const COMPARISON_SOURCES = [...MODEL_SOURCES, ...EXPERT_SOURCES];
const MODEL_KEYS = new Set(contract.model_sources);
const EXPERT_KEYS = new Set(contract.expert_sources);
const COMPARISON_METRIC_HINTS = {
    mae: "Mean absolute error — lower is better; it favors median-like forecasts",
    rmse: "Root mean squared error — lower is better; it rewards accurate expected points",
    r2: "R² (coefficient of determination) — higher is better",
    bias: "Mean of prediction − actual — positive over-predicts; not ranked",
};
const METRIC_OPTIONS = [
    { value: "mae", label: "MAE" },
    { value: "rmse", label: "RMSE" },
    { value: "r2", label: "R²" },
    { value: "bias", label: "Bias" },
];
// R² on one common sample orders sources exactly as RMSE does.
const GAP_METRIC = { mae: "mae", rmse: "rmse", r2: "rmse" };
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
    if (metric === "bias") return `${v > 0 ? "+" : ""}${v.toFixed(2)}`;
    return metric === "r2" ? v.toFixed(3) : v.toFixed(2);
}

function signed(value) {
    return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

/* The server's paired interval for this row: which group (if any) wins, and the
 * best-model-minus-best-expert gap with its 95% interval for the shown metric. */
function rowGap(coverageCell, metric) {
    const uncertainty = coverageCell?.uncertainty;
    const key = GAP_METRIC[metric];
    if (!key || !uncertainty || uncertainty.status !== "available" || !uncertainty[key]) return null;
    const gap = uncertainty[key];
    const group = (verdict) => (verdict === "models" ? "Models" : "Experts");
    // A decided row needs both metrics; one decided metric alone stays a tie.
    const label = uncertainty.winner === "models" || uncertainty.winner === "experts"
        ? `${group(uncertainty.winner)} ahead`
        : gap.verdict === "models" || gap.verdict === "experts"
            ? `≈ tie (${group(gap.verdict).toLowerCase()} ahead on ${key.toUpperCase()} only)`
            : "≈ tie";
    return {
        winner: uncertainty.winner,
        text: `${label} · best model − best expert ${signed(gap.best_model_minus_best_expert)} [${signed(gap.ci[0])}, ${signed(gap.ci[1])}] ${key.toUpperCase()}`,
    };
}

/* Shared header: first label ("Position" / "Quartile"), then the graded
 * source columns in COMPARISON_SOURCES order. */
function ComparisonTableHead({ firstLabel, sources }) {
    return (
        <thead>
            <tr>
                <th>{firstLabel}</th>
                {sources.map((s) => (
                    <th key={s.key} className="comparison-num">{s.label}</th>
                ))}
            </tr>
        </thead>
    );
}

/* One row per position. The best cell of the winning group is highlighted only
 * when the paired interval excludes zero under both MAE and RMSE; a statistical
 * tie highlights nothing. Missing cells render an em dash. */
function ComparisonRows({ posMap, metric, coverage, sources }) {
    const higherBetter = metric === "r2" || metric === "hit_rate";
    return COMPARISON_POSITIONS.map((pos) => {
        const cells = posMap[pos] || {};
        const gap = rowGap(coverage?.[pos], metric);
        const group = gap?.winner === "models" ? MODEL_KEYS : gap?.winner === "experts" ? EXPERT_KEYS : null;
        const values = group
            ? sources.filter((s) => group.has(s.key)).map((s) => comparisonCellValue(cells[s.key], metric)).filter((v) => v !== null)
            : [];
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
                    {gap && <span className={"comparison-gap" + (gap.winner === "tie" ? "" : " comparison-gap-decided")}>{gap.text}</span>}
                </td>
                {sources.map((s) => {
                    const v = comparisonCellValue(cells[s.key], metric);
                    if (v === null) {
                        return <td key={s.key} className="comparison-num comparison-empty">{"—"}</td>;
                    }
                    const isBest = best !== null && group.has(s.key) && Math.abs(v - best) < 1e-9;
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

/* One accuracy table for a cohort. */
function ComparisonSubsetBlock({ header, bodyId, posMap, metric, error, coverage, definition, sources }) {
    return (
        <div className="comparison-table-block">
            <div className="section-header">{header}</div>
            {definition && <p className="comparison-notes">{definition}</p>}
            <div className="table-container">
                <table className="comparison-table">
                    <ComparisonTableHead firstLabel="Position" sources={sources} />
                    <tbody id={bodyId}>
                        {error ? (
                            <tr><td colSpan={sources.length + 1} className="arch-error">Failed to load: {error}</td></tr>
                        ) : posMap ? (
                            <ComparisonRows posMap={posMap} metric={metric} coverage={coverage} sources={sources} />
                        ) : (
                            <tr><td colSpan={sources.length + 1} className="arch-loading">Loading comparison…</td></tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

/* One note per (source, reason), listing every position it applies to. */
function groupedExclusions(excluded) {
    const groups = new Map();
    for (const [position, sources] of Object.entries(excluded || {})) {
        for (const [source, reason] of Object.entries(sources || {})) {
            const key = `${source}\u0000${reason}`;
            if (!groups.has(key)) groups.set(key, { source, reason, positions: [] });
            groups.get(key).positions.push(position);
        }
    }
    return [...groups.values()];
}

/* A source column stays only if some cohort grades it somewhere. NFL.com offense
 * is excluded server-side (and has no K/DST forecasts), so its column would be
 * dashes everywhere; the excluded-source notes explain why it is absent. */
function gradedSources(data) {
    if (!data) return COMPARISON_SOURCES;
    const cohorts = Object.values(data.subsets || {});
    const graded = COMPARISON_SOURCES.filter((s) => cohorts.some((byPos) => COMPARISON_POSITIONS.some(
        (pos) => comparisonCellValue(byPos?.[pos]?.[s.key], "n") !== null,
    )));
    return graded.length ? graded : COMPARISON_SOURCES;
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
    const sources = gradedSources(data);

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
        : data?.sample_basis === "shared_player_weeks"
            ? "Each architecture uses its deployed forecasts, and all sources are graded on identical player-weeks. "
            : "The response does not establish a shared player-week sample. ";

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
                Model and expert forecast accuracy. Scoring components, sample coverage, and cohort definitions are supplied by the server below. Lower MAE / RMSE is better; higher R² is better.
            </ApproachBanner>

            {data && <div className="comparison-notes" id="comparison-contract">
                <p>{data.sample_basis === "shared_player_weeks"
                    ? "Every displayed source is scored on the same regular-season player-weeks. Missing forecasts are excluded. A projected zero is retained, but a provider row with every shared component at zero is an unprojected placeholder and counts as missing."
                    : `Sample basis: ${data.sample_basis || "not supplied by this response"}.`}</p>
                <p>{["shared_projected_components_v1", "shared_projected_components_v2"].includes(data.actual_basis)
                    ? "Predictions and actuals use only the shared projected components below. Stats outside those sets are excluded from actuals too."
                    : `Actual basis: ${data.actual_basis || "not supplied by this response"}.`}</p>
                {Object.entries(data.scoring_components || {}).map(([position, components]) => (
                    <p key={position}><strong>{position}.</strong> {components.map((name) => name.replaceAll("_", " ")).join(", ")}</p>
                ))}
                {data.coverage?.all && <p><strong>Graded sources.</strong> {COMPARISON_POSITIONS.map((pos) => {
                    const cell = data.coverage.all[pos] || {};
                    const graded = (cell.sources || []).map((s) => SOURCE_LABELS[s] || s).join(", ") || "unavailable";
                    const missing = (cell.unavailable_sources || []).map((s) => SOURCE_LABELS[s] || s).join(", ");
                    return `${pos}: ${graded}${missing ? ` (no usable forecasts: ${missing})` : ""}`;
                }).join(" · ")}. The Timeline tab grades the same source groups.</p>}
                {groupedExclusions(data.excluded_sources).map(({ positions, source, reason }) => (
                    <p key={`${source}-${reason}`}>{positions.join(", ")} · {SOURCE_LABELS[source] || source}: {reason}</p>
                ))}
                {Object.entries(data.excluded_components || {}).flatMap(([position, components]) =>
                    Object.entries(components).map(([component, reason]) => <p key={`${position}-${component}`}>{position} · {component.replaceAll("_", " ")}: {reason}</p>))}
            </div>}

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
                header="Expected starters · pregame depth chart"
                bodyId="comparison-depth-starters"
                posMap={data ? (subsets.weekly_depth_starters || {}) : null}
                coverage={data?.coverage?.weekly_depth_starters}
                definition={data?.cohort_definitions?.weekly_depth_starters}
                metric={metric}
                error={error}
                sources={sources}
            />

            <ComparisonSubsetBlock
                header="All comparable player-weeks"
                coverage={data?.coverage?.all}
                bodyId="comparison-all-body"
                posMap={data ? (subsets.all || {}) : null}
                metric={metric}
                error={error}
                sources={sources}
            />

            <ComparisonSubsetBlock
                header="Prior-season elite · top 24"
                bodyId="comparison-elite-top24"
                posMap={data ? (subsets.elite_top24 || {}) : null}
                coverage={data?.coverage?.elite_top24}
                definition={data?.cohort_definitions?.elite_top24}
                metric={metric}
                error={error}
                sources={sources}
            />

            <ComparisonSubsetBlock
                header="Expert-reference top 24 · secondary view"
                bodyId="comparison-weekly-top24"
                posMap={data ? (subsets.weekly_reference_top24 || {}) : null}
                coverage={data?.coverage?.weekly_reference_top24}
                definition={data?.cohort_definitions?.weekly_reference_top24}
                metric={metric}
                error={error}
                sources={sources}
            />

            <ComparisonSubsetBlock
                header="Season leaders · top 30"
                coverage={data?.coverage?.top30}
                definition={data?.cohort_definitions?.top30}
                bodyId="comparison-top30-body"
                posMap={data ? (subsets.top30 || {}) : null}
                metric={metric}
                error={error}
                sources={sources}
            />
            <ComparisonSubsetBlock
                header="Season leaders · top 12"
                coverage={data?.coverage?.top12}
                definition={data?.cohort_definitions?.top12}
                bodyId="comparison-top12-body"
                posMap={data ? (subsets.top12 || {}) : null}
                metric={metric}
                error={error}
                sources={sources}
            />

            <ComparisonSubsetBlock
                header="Weekly top-24 leader capture · higher is better · not ranked"
                bodyId="comparison-weekly-capture"
                posMap={data ? (data.weekly_ranking || {}) : null}
                metric="hit_rate"
                error={error}
                sources={sources}
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
                            <ComparisonTableHead firstLabel="Quartile" sources={sources} />
                            <tbody id="quartile-bias-body">
                                {data ? (
                                    ["Q1", "Q2", "Q3", "Q4"].map((q) => {
                                        const row = quartileByPos[q] || {};
                                        return (
                                            <tr key={q}>
                                                <td className="comparison-pos">{QUARTILE_LABELS[q]}</td>
                                                {sources.map((s) => (
                                                    <QuartileBiasCell key={s.key} cell={row[s.key]} />
                                                ))}
                                            </tr>
                                        );
                                    })
                                ) : (
                                    <tr><td colSpan={sources.length + 1} className="arch-loading">Loading quartile bias…</td></tr>
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
                            {data.quartile_bias_meta?.seasons?.length > 0 && <li><strong>Evaluation seasons.</strong> {data.quartile_bias_meta.seasons.join(", ")}.{data.evaluation_season_note ? ` ${data.evaluation_season_note}` : ""}</li>}
                            <li><strong>Scoring.</strong> {data.scoring}. {["shared_projected_components_v1", "shared_projected_components_v2"].includes(data.actual_basis) ? "Predictions and regular-season actuals include only the shared projected components listed above. These component scores differ from full fantasy totals." : "Refer to the response's actual basis above."}</li>
                            <li><strong>Our models.</strong> {modelLine}MAE/RMSE/R² are on weekly shared-component point totals. A row names a winner (its best cell highlighted) only when the 95% interval for the best model minus the best expert excludes zero under both MAE and RMSE; otherwise it reads “≈ tie”. {data.uncertainty_meta ? `Intervals resample whole players (${data.uncertainty_meta.replicates} paired draws), and the best of each group is chosen inside every draw, so picking the best of four models after the fact is accounted for.` : ""}</li>
                            <li><strong>Metrics.</strong> MAE rewards median-like forecasts on these right-skewed points; RMSE rewards accurate expected points, which is what published projections estimate. Bias is shown for context and never ranked.</li>
                            <li><strong>NFL.com.</strong> {nflNote}</li>
                            <li><strong>RotoWire.</strong> {rwNote}</li>
                            <li><strong>ESPN.</strong> {espnNote}</li>
                            {data.cohort_definitions?.weekly_depth_starters && <li><strong>Expected starters.</strong> {data.cohort_definitions.weekly_depth_starters}.</li>}
                            {data.cohort_definitions?.elite_top24 && <li><strong>Prior-season elite.</strong> {data.cohort_definitions.elite_top24}.</li>}
                            {data.cohort_definitions?.weekly_reference_top24 && <li><strong>Expert reference.</strong> {data.cohort_definitions.weekly_reference_top24}. Selection happens before filtering for recorded outcomes or model coverage. Missing reference weeks are reported explicitly.</li>}
                            {(data.cohort_definitions?.top30 || data.cohort_definitions?.top12) && <li><strong>Season leaders.</strong> {data.cohort_definitions.top30 || data.cohort_definitions.top12}. These are retrospective diagnostics, not pregame starter lists.</li>}
                            <li><strong>Weekly leader capture.</strong> The fraction of actual weekly top-24 scorers selected by each source's own forecasts. Only weeks with at least 24 comparable players count; hover for the number of weeks.</li>
                            <li>
                                <strong>Coverage.</strong> {data.sample_basis === "shared_player_weeks" ? "Every displayed source in a position is scored on the same player-weeks. Missing forecasts are excluded, never treated as zero." : "The response does not establish whether source samples are paired."} Sample sizes and each row’s interval appear beside each position. Historical investigations are available in the{" "}
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

/* Model Performance — overall accuracy cards, per-position model breakdown,
 * position accuracy bar charts, and the weekly MAE line chart. Faithful port
 * of the vanilla loadMetrics / renderPositionModelDetail / renderPositionCharts
 * / renderWeeklyChart; markup, ids, and copy mirror templates/index.html. */
import { useEffect, useMemo, useState } from "react";
import { fetchJSON } from "../api.js";
import { formatTargetMae } from "../lib/format.js";
import { modelColors } from "../lib/chartTheme.js";
import { useChart } from "../hooks/useChart.js";
import { PillGroup, PosBadge, ApproachBanner } from "../components/common.jsx";

const POSITION_OPTIONS = ["QB", "RB", "WR", "TE", "K", "DST"].map((v) => ({ value: v, label: v }));

/* Overall metric cards — one per model, keyed by the API's display names. */
const METRIC_CARDS = [
    { key: "Ridge Regression", prefix: "ridge" },
    { key: "Neural Network", prefix: "nn" },
    { key: "Attention NN", prefix: "attn-nn" },
    { key: "LightGBM", prefix: "lgbm" },
];

function MetricCard({ title, prefix, overall }) {
    return (
        <div className="metric-card">
            <div className="metric-card-header">{title}</div>
            <div className="metric-card-body" id={`${prefix}-metrics`}>
                <div className="metric-row">
                    <span className="metric-label">MAE</span>
                    <span className="metric-value" id={`${prefix}-mae`}>{overall ? overall.mae.toFixed(3) : "--"}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">RMSE</span>
                    <span className="metric-value" id={`${prefix}-rmse`}>{overall ? overall.rmse.toFixed(3) : "--"}</span>
                </div>
                <div className="metric-row">
                    <span className="metric-label">R&sup2;</span>
                    <span className="metric-value" id={`${prefix}-r2`}>{overall ? overall.r2.toFixed(3) : "--"}</span>
                </div>
            </div>
        </div>
    );
}

/* Per-position model detail card (target MAE table, feature badges, NN arch). */
function PositionModelDetail({ pos, detail, scoring }) {
    if (!detail) {
        return <p className="pos-model-empty">Loading...</p>;
    }
    const d = detail;
    const tm = d.target_metrics || {};

    // Total row is always in fantasy points (aggregator output), so no unit formatting.
    const totalM = tm["total"] || {};
    const totalCell = (v) => (v != null ? <strong>{v.toFixed(2)}</strong> : <strong>--</strong>);

    const arch = d.architecture || {};
    const backbone = (arch.backbone || []).join(" > ");

    return (
        <div className="pos-model-card">
            <div className="pos-model-header">
                <PosBadge position={pos} />
                <span className="pos-model-name">{d.label} Model</span>
                <span className="pos-model-meta">{d.n_features || "?"} features &middot; {d.n_samples_test || "?"} test samples</span>
            </div>

            <div className="pos-model-section-label">Raw-Stat Targets</div>
            <div className="table-container pos-model-table-wrap">
                <table className="pos-model-table">
                    <thead>
                        <tr>
                            <th>Target</th>
                            <th>Formula</th>
                            <th>Ridge MAE</th>
                            <th>NN MAE</th>
                            <th>Attn NN MAE</th>
                            <th>LGBM MAE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {(d.targets || []).map((t) => {
                            // Per-target rows render MAE in the target's native unit
                            // (yards / TDs / receptions), plus a fantasy-point
                            // equivalent for count-style targets (formatTargetMae).
                            const m = tm[t.key] || {};
                            return (
                                <tr key={t.key}>
                                    <td className="tm-name">{t.label}</td>
                                    <td className="tm-formula">{t.formula}</td>
                                    <td className="tm-val">{formatTargetMae(m.ridge_mae, t.key, m.unit, scoring)}</td>
                                    <td className="tm-val">{formatTargetMae(m.nn_mae, t.key, m.unit, scoring)}</td>
                                    <td className="tm-val">{formatTargetMae(m.attn_nn_mae, t.key, m.unit, scoring)}</td>
                                    <td className="tm-val">{formatTargetMae(m.lgbm_mae, t.key, m.unit, scoring)}</td>
                                </tr>
                            );
                        })}
                        <tr className="tm-total-row">
                            <td className="tm-name"><strong>Total (fantasy points)</strong></td>
                            <td className="tm-formula">{d.adjustments || ""}</td>
                            <td className="tm-val">{totalCell(totalM.ridge_mae)}</td>
                            <td className="tm-val">{totalCell(totalM.nn_mae)}</td>
                            <td className="tm-val">{totalCell(totalM.attn_nn_mae)}</td>
                            <td className="tm-val">{totalCell(totalM.lgbm_mae)}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div className="pos-model-section-label">Position-Specific Features</div>
            <div className="feature-badges">
                {(d.specific_features || []).map((f) => <span key={f} className="feature-badge">{f}</span>)}
            </div>

            <div className="pos-model-section-label">Neural Network Architecture</div>
            <div className="arch-info">Shared backbone <span className="arch-val">[{backbone}]</span> &rarr; {(d.targets || []).length} heads (hidden: <span className="arch-val">{arch.head_hidden || "?"}</span>)</div>
        </div>
    );
}

/* Collect every position that appears in any model's by_position — union so
 * charts render the full set even if one model is missing a row. */
function buildPositionChartData(metrics) {
    const COLORS = modelColors();
    const modelSeries = [
        { key: "Ridge Regression", label: "Ridge", color: COLORS.ridge, bg: COLORS.ridgeBg },
        { key: "Neural Network", label: "Neural Net", color: COLORS.nn, bg: COLORS.nnBg },
        { key: "Attention NN", label: "Attention NN", color: COLORS.attn_nn, bg: COLORS.attn_nnBg },
        { key: "LightGBM", label: "LightGBM", color: COLORS.lgbm, bg: COLORS.lgbmBg },
    ];
    const positionsSet = new Set();
    for (const { key } of modelSeries) {
        const m = metrics[key];
        if (!m || !m.by_position) continue;
        m.by_position.forEach((p) => positionsSet.add(p.position));
    }
    const positions = ["QB", "RB", "WR", "TE", "K", "DST"].filter((p) => positionsSet.has(p));

    const buildDatasets = (metricName) => modelSeries
        .map(({ key, label, color, bg }) => {
            const m = metrics[key];
            if (!m || !m.by_position || m.by_position.length === 0) return null;
            const byPos = Object.fromEntries(m.by_position.map((p) => [p.position, p]));
            // null entries let Chart.js leave gaps where this model has no
            // prediction for that position (e.g. LightGBM for K/DST).
            const data = positions.map((p) => (byPos[p] != null ? byPos[p][metricName] : null));
            return { label, data, backgroundColor: bg, borderColor: color, borderWidth: 1.5 };
        })
        .filter(Boolean);

    return { positions, maeDatasets: buildDatasets("mae"), r2Datasets: buildDatasets("r2") };
}

function PositionBarChart({ id, labels, datasets, title, theme }) {
    const canvasRef = useChart((palette) => {
        if (!labels || !datasets) return null;
        return {
            type: "bar",
            data: { labels, datasets },
            options: {
                responsive: true,
                plugins: { title: { display: true, text: title, color: palette.heading } },
                scales: { y: { beginAtZero: true, grid: { color: palette.grid } }, x: { grid: { display: false } } },
            },
        };
    }, [labels, datasets, title, theme]);
    return <canvas ref={canvasRef} id={id} />;
}

function WeeklyChart({ weekly, theme }) {
    const canvasRef = useChart((palette) => {
        if (!weekly) return null;
        const COLORS = modelColors();
        const series = [
            { label: "Ridge MAE", data: weekly.ridge_mae, color: COLORS.ridge, bg: COLORS.ridgeBg },
            { label: "Neural Net MAE", data: weekly.nn_mae, color: COLORS.nn, bg: COLORS.nnBg },
            { label: "Attention NN MAE", data: weekly.attn_nn_mae, color: COLORS.attn_nn, bg: COLORS.attn_nnBg },
            { label: "LightGBM MAE", data: weekly.lgbm_mae, color: COLORS.lgbm, bg: COLORS.lgbmBg },
        ];
        const datasets = series
            .filter((s) => Array.isArray(s.data) && s.data.some((v) => v != null))
            .map((s) => ({
                label: s.label,
                data: s.data,
                borderColor: s.color,
                backgroundColor: s.bg,
                fill: false,
                tension: 0.3,
                pointRadius: 3,
                pointHoverRadius: 5,
                spanGaps: true,
            }));
        return {
            type: "line",
            data: { labels: weekly.weeks.map((w) => `Wk ${w}`), datasets },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: "Weekly MAE Across Test Season (Lower is Better)", color: palette.heading },
                },
                scales: {
                    y: { beginAtZero: true, grid: { color: palette.grid }, title: { display: true, text: "MAE", color: palette.text } },
                    x: { grid: { color: palette.grid } },
                },
            },
        };
    }, [weekly, theme]);
    return <canvas ref={canvasRef} id="weekly-mae-chart" />;
}

export function ModelPerformanceView({ scoring, search, theme, onPlayer, activateView }) {
    const [position, setPosition] = useState("QB");
    const [{ state, metrics, weekly, posDetails }, setData] = useState({
        state: "loading", metrics: null, weekly: null, posDetails: null,
    });

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const q = `?scoring=${scoring}`;
                const [metricsData, weeklyData, posDetailsData] = await Promise.all([
                    fetchJSON(`/api/metrics${q}`),
                    fetchJSON(`/api/weekly_accuracy${q}`),
                    fetchJSON(`/api/position_details${q}`),
                ]);
                if (!cancelled) setData({ state: "ready", metrics: metricsData, weekly: weeklyData, posDetails: posDetailsData });
            } catch (e) {
                console.error("Failed to load metrics:", e);
                if (!cancelled) setData({ state: "error", metrics: null, weekly: null, posDetails: null });
            }
        })();
        return () => { cancelled = true; };
    }, [scoring]);

    const chartData = useMemo(() => (metrics ? buildPositionChartData(metrics) : null), [metrics]);

    return (
        <section id="view-model-performance" className="view active">
            <ApproachBanner icon="layers" title="Position-Specific Modeling">
                Each position has a dedicated multi-target model that predicts raw NFL stats (yards, TDs, receptions, etc.); a shared aggregator converts predictions to fantasy points. Features and architecture are tuned per position.
            </ApproachBanner>

            <div className="section-header">Overall Accuracy (All Positions)</div>
            <div className="metrics-grid">
                {state === "error"
                    ? <p className="error-message">Failed to load model metrics.</p>
                    : METRIC_CARDS.map(({ key, prefix }) => (
                        <MetricCard key={prefix} title={key} prefix={prefix} overall={metrics && metrics[key] && metrics[key].overall} />
                    ))}
            </div>

            <div className="section-header">Position Model Breakdown</div>
            <div className="pos-model-section">
                <PillGroup id="perf-position-filter" options={POSITION_OPTIONS} value={position} onChange={setPosition} />
                <div id="pos-model-detail" className="pos-model-detail">
                    {state !== "error" && (
                        <PositionModelDetail pos={position} detail={posDetails && posDetails[position]} scoring={scoring} />
                    )}
                </div>
            </div>

            <div className="section-header">Accuracy by Position</div>
            <div className="charts-row">
                <div className="chart-box">
                    <PositionBarChart
                        id="position-mae-chart"
                        labels={chartData && chartData.positions}
                        datasets={chartData && chartData.maeDatasets}
                        title="MAE by Position (Lower is Better)"
                        theme={theme}
                    />
                </div>
                <div className="chart-box">
                    <PositionBarChart
                        id="position-r2-chart"
                        labels={chartData && chartData.positions}
                        datasets={chartData && chartData.r2Datasets}
                        title={"R² by Position (Higher is Better)"}
                        theme={theme}
                    />
                </div>
            </div>

            <div className="section-header">Weekly Prediction Accuracy</div>
            <div className="chart-box wide">
                <WeeklyChart weekly={weekly} theme={theme} />
            </div>
        </section>
    );
}

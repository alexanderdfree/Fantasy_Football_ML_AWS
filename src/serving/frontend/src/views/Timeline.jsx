/* Changelog & Timeline — the weekly re-scoring log and the model release
 * changelog (design-system HistoryView pattern). Every completed week is
 * ground truth; all four models plus the two expert baselines are re-scored
 * weekly server-side (/api/timeline), and the release rail renders the
 * committed release_changelog.json entries, filterable by model family. */
import { useEffect, useMemo, useState } from "react";
import { fetchJSON } from "../api.js";
import { fmt } from "../lib/format.js";
import { modelColors, chartTheme } from "../lib/chartTheme.js";
import { useChart } from "../hooks/useChart.js";
import { SortableTh } from "../components/common.jsx";

const MODELS = ["ridge", "nn", "attn_nn", "lgbm"];
const FAMILY_LABELS = { ALL: "All", ridge: "Ridge", nn: "Neural Net", lgbm: "LightGBM", attn_nn: "Attention NN" };

function familyColor(family) {
    const COLORS = modelColors();
    return COLORS[family] || COLORS.actual;
}

/* Weekly MAE for all four of our models across the test season. */
function WeeklyTrendChart({ weekly, labels, theme }) {
    const ref = useChart((t) => {
        const COLORS = modelColors();
        return {
            type: "line",
            data: {
                labels: weekly.map((w) => `Wk ${w.week}`),
                datasets: MODELS.map((m) => ({
                    label: labels[m],
                    data: weekly.map((w) => w[m]),
                    borderColor: COLORS[m],
                    backgroundColor: COLORS[m],
                    tension: 0.3,
                    pointRadius: 2,
                    borderWidth: m === "attn_nn" ? 2.6 : 1.4,
                    spanGaps: true,
                })),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
                    title: { display: true, text: "Weekly MAE — our four models", color: t.heading, font: { size: 12, weight: "600" }, padding: { bottom: 8 } },
                },
                scales: {
                    y: { title: { display: true, text: "MAE (fantasy pts)" }, grid: { color: t.grid } },
                    x: { grid: { display: false } },
                },
            },
        };
    }, [weekly, theme]);
    return <canvas ref={ref} />;
}

/* Our strongest model vs the two expert baselines — the edge over time. */
function VsExpertsChart({ weekly, champion, labels, theme }) {
    const ref = useChart((t) => {
        const COLORS = modelColors();
        const line = (key, label, color, dash) => ({
            label,
            data: weekly.map((w) => w[key]),
            borderColor: color,
            backgroundColor: color,
            borderDash: dash || [],
            tension: 0.3,
            pointRadius: 2,
            borderWidth: dash ? 1.4 : 2.6,
            fill: false,
            spanGaps: true,
        });
        const champ = champion || "attn_nn";
        return {
            type: "line",
            data: {
                labels: weekly.map((w) => `Wk ${w.week}`),
                datasets: [
                    line(champ, `${labels[champ]} (ours)`, COLORS[champ]),
                    line("nflcom", "NFL.com", COLORS.nflcom, [6, 4]),
                    line("rotowire", "RotoWire", COLORS.rotowire, [6, 4]),
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
                    title: { display: true, text: "Our best vs the experts", color: t.heading, font: { size: 12, weight: "600" }, padding: { bottom: 8 } },
                },
                scales: {
                    y: { title: { display: true, text: "MAE (fantasy pts)" }, grid: { color: t.grid } },
                    x: { grid: { display: false } },
                },
            },
        };
    }, [weekly, champion, theme]);
    return <canvas ref={ref} />;
}

function EdgeValue({ value }) {
    if (value == null) return <span className="delta-neutral">--</span>;
    const cls = value >= 0.02 ? "delta-positive" : value <= -0.02 ? "delta-negative" : "delta-neutral";
    const sign = value > 0 ? "+" : "";
    return <span className={cls}>{`${sign}${value.toFixed(2)}`}</span>;
}

// Module-level cache keyed by scoring so tab revisits don't refetch.
const timelineCache = new Map();

export function TimelineView({ scoring, theme }) {
    const [payload, setPayload] = useState(() => timelineCache.get(scoring) || null);
    const [error, setError] = useState(null);
    const [family, setFamily] = useState("ALL");
    const [sort, setSort] = useState({ key: "week", order: "desc" });

    useEffect(() => {
        const cached = timelineCache.get(scoring);
        if (cached) { setPayload(cached); return undefined; }
        let cancelled = false;
        setPayload(null);
        setError(null);
        fetchJSON(`/api/timeline?scoring=${scoring}`)
            .then((data) => {
                timelineCache.set(scoring, data);
                if (!cancelled) setPayload(data);
            })
            .catch((e) => {
                console.error("Failed to load timeline:", e);
                if (!cancelled) setError(e.message);
            });
        return () => { cancelled = true; };
    }, [scoring]);

    const labels = (payload && payload.model_labels) || FAMILY_LABELS;
    const weekly = (payload && payload.weekly) || [];
    const releases = (payload && payload.releases) || [];
    const summary = (payload && payload.summary) || null;

    const families = useMemo(
        () => ["ALL", ...Array.from(new Set(releases.map((r) => r.family)))],
        [releases],
    );
    const shownReleases = family === "ALL" ? releases : releases.filter((r) => r.family === family);

    const onSort = (k) => setSort((s) => (
        s.key === k ? { key: k, order: s.order === "desc" ? "asc" : "desc" } : { key: k, order: "desc" }
    ));
    const sortedWeekly = useMemo(() => {
        const rows = [...weekly];
        rows.sort((a, b) => {
            const va = a[sort.key];
            const vb = b[sort.key];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            const cmp = typeof va === "string" ? String(va).localeCompare(vb) : va - vb;
            return sort.order === "desc" ? -cmp : cmp;
        });
        return rows;
    }, [weekly, sort]);

    const winnerTag = (m) => (
        <span className="winner-tag">
            <span className="winner-dot" style={{ background: familyColor(m) }} />
            {labels[m] || m}
        </span>
    );

    const modelCell = (w, m) => {
        const win = w.winner === m;
        return (
            <span style={win ? { fontWeight: 700, color: familyColor(m) } : undefined}>
                {w[m] == null ? "--" : w[m].toFixed(2)}
            </span>
        );
    };

    return (
        <section id="view-timeline" className="view active">
            <div className="callout secondary">
                <span className="callout-icon">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 3v5h5" /><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" /><path d="M12 7v5l3 2" />
                    </svg>
                </span>
                <div>
                    <div className="callout-title">Changelog &amp; Timeline</div>
                    <div className="callout-desc">
                        Every week, the season's completed games become new ground truth and all four models plus the
                        two expert baselines are re-scored. This is the running log — the accuracy trend, the
                        head-to-head weekly record, and the release changelog behind each step down in error.
                    </div>
                </div>
            </div>

            {error && <p className="error-message">Failed to load timeline: {error}</p>}
            {!error && !payload && <p className="arch-loading">Loading timeline…</p>}

            {payload && summary && (
                <>
                    <div className="section-header">Track Record</div>
                    <div className="timeline-track-card">
                        <div className="stat-block-row">
                            <div className="stat-block">
                                <span className="stat-block-label">Current Champion</span>
                                <span className="stat-block-value neutral">{summary.champion ? labels[summary.champion] : "--"}</span>
                            </div>
                            <div className="stat-block">
                                <span className="stat-block-label">Champion Weeks</span>
                                <span className="stat-block-value">{`${summary.champion_weeks} / ${summary.total_weeks}`}</span>
                            </div>
                            <div className="stat-block">
                                <span className="stat-block-label">{summary.best_week != null ? `Best MAE · Wk ${summary.best_week}` : "Best MAE"}</span>
                                <span className="stat-block-value">{summary.best_mae != null ? summary.best_mae.toFixed(2) : "--"}</span>
                            </div>
                            <div className="stat-block">
                                <span className="stat-block-label">Beat Both Experts</span>
                                <span className="stat-block-value">{`${summary.beat_experts} / ${summary.total_weeks}`}</span>
                            </div>
                            {payload.season && <span className="meta-badge">{`${payload.season} Season`}</span>}
                        </div>
                    </div>

                    <div className="section-header">Season Accuracy Trend</div>
                    <div className="charts-row">
                        <div className="chart-box"><WeeklyTrendChart weekly={weekly} labels={labels} theme={theme} /></div>
                        <div className="chart-box"><VsExpertsChart weekly={weekly} champion={summary.champion} labels={labels} theme={theme} /></div>
                    </div>

                    {releases.length > 0 && (
                        <>
                            <div className="section-header">Model Release Changelog</div>
                            <div className="release-controls">
                                <div className="pill-group">
                                    {families.map((f) => (
                                        <button
                                            key={f}
                                            type="button"
                                            className={`pill${family === f ? " active" : ""}`}
                                            onClick={() => setFamily(f)}
                                        >
                                            {FAMILY_LABELS[f] || labels[f] || f}
                                        </button>
                                    ))}
                                </div>
                            </div>
                            <div className="release-list">
                                {shownReleases.map((r) => {
                                    const gain = r.prev_mae != null ? +(r.prev_mae - r.mae).toFixed(2) : null;
                                    return (
                                        <div className="release-item" key={r.version}>
                                            <div className="release-rail">
                                                <span className="release-dot" style={{ background: familyColor(r.family) }} />
                                                <span className="release-line" />
                                            </div>
                                            <div className="release-card">
                                                <div className="release-head">
                                                    <span className="meta-badge">{r.version}</span>
                                                    <span className="release-title">{r.title}</span>
                                                    <span className="release-model">{r.model}</span>
                                                    <span className="release-date">{r.date}</span>
                                                </div>
                                                <p className="release-summary">{r.summary}</p>
                                                <div className="release-metrics">
                                                    <div className="release-metric">
                                                        <span className="release-metric-label">MAE</span>
                                                        <span className="release-metric-value">{fmt(r.mae, 2)}</span>
                                                    </div>
                                                    <div className="release-metric">
                                                        <span className="release-metric-label">R²</span>
                                                        <span className="release-metric-value">{r.r2 != null ? Number(r.r2).toFixed(3) : "--"}</span>
                                                    </div>
                                                    <div className="release-metric">
                                                        <span className="release-metric-label">vs Prev</span>
                                                        <span className="release-metric-value">
                                                            {gain == null
                                                                ? <span className="delta-neutral">baseline</span>
                                                                : <EdgeValue value={gain} />}
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </>
                    )}

                    <div className="section-header">Weekly Benchmark Log</div>
                    <div className="results-info">
                        {`${weekly.length} weeks benchmarked · lower MAE is better · green edge means our best model beat both experts`}
                    </div>
                    <div className="table-container">
                        <table id="timeline-table">
                            <thead>
                                <tr>
                                    <SortableTh label="Wk" sortKey="week" className="col-week" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <th>Best Model</th>
                                    <SortableTh label="Ridge" sortKey="ridge" className="col-pred ridge-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="NN" sortKey="nn" className="col-pred nn-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="Attn NN" sortKey="attn_nn" className="col-pred attn-nn-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="LGBM" sortKey="lgbm" className="col-pred lgbm-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="NFL.com" sortKey="nflcom" className="col-pred nflcom-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="RotoWire" sortKey="rotowire" className="col-pred rotowire-col" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="Edge vs Exp." sortKey="edge" className="col-delta" sort={sort.key} order={sort.order} onSort={onSort} />
                                </tr>
                            </thead>
                            <tbody>
                                {sortedWeekly.map((w) => (
                                    <tr key={w.week}>
                                        <td className="col-week"><strong>{w.week}</strong></td>
                                        <td>{w.winner ? winnerTag(w.winner) : "--"}</td>
                                        {MODELS.map((m) => (
                                            <td key={m} className="col-pred">{modelCell(w, m)}</td>
                                        ))}
                                        <td className="col-pred">{w.nflcom == null ? "--" : w.nflcom.toFixed(2)}</td>
                                        <td className="col-pred">{w.rotowire == null ? "--" : w.rotowire.toFixed(2)}</td>
                                        <td className="col-delta"><EdgeValue value={w.edge} /></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </section>
    );
}

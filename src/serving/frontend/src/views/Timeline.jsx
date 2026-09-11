/* Matched weekly evaluation, fixed per-model records, and historical releases. */
import { useEffect, useMemo, useState } from "react";
import { fetchJSON } from "../api.js";
import { fmt } from "../lib/format.js";
import { modelColors } from "../lib/chartTheme.js";
import { useChart } from "../hooks/useChart.js";
import { SortableTh } from "../components/common.jsx";

const MODELS = ["ridge", "nn", "attn_nn", "lgbm"];
const FAMILY_LABELS = { ALL: "All", ridge: "Ridge", nn: "Neural Net", lgbm: "LightGBM", attn_nn: "Attention NN" };

function familyColor(family) {
    const COLORS = modelColors();
    return COLORS[family] || COLORS.actual;
}

const GROUPS = [{ id: "offense", label: "Offense" }, { id: "k", label: "Kickers" }, { id: "dst", label: "D/ST" }];
const REASONS = {
    no_regular_season_rows: "No regular-season rows are available for this selection.",
    shared_actual_components_missing: "The observed stats needed for this comparison are unavailable.",
    required_forecasts_missing: "A required source has no matching forecasts.",
    no_common_source_rows: "The sources have no player-weeks in common.",
};

function TimelineChart({ weekly, sources, labels, theme, edges = false }) {
    const ref = useChart((t) => {
        const COLORS = modelColors();
        return {
            type: "line",
            data: {
                labels: weekly.map((w) => `Wk ${w.week}`),
                datasets: (edges ? MODELS : sources).map((m) => ({
                    label: labels[m],
                    data: weekly.map((w) => (edges ? w.edges : w.mae)[m]),
                    borderColor: COLORS[m] || "#f97316",
                    backgroundColor: COLORS[m] || "#f97316",
                    borderDash: MODELS.includes(m) ? [] : [6, 4],
                    tension: 0,
                    pointRadius: 2,
                    borderWidth: 2,
                    spanGaps: false,
                })),
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } },
                    title: { display: true, text: edges ? "Each model’s edge vs experts" : "Weekly MAE — same player-weeks", color: t.heading, font: { size: 12, weight: "600" }, padding: { bottom: 8 } },
                },
                scales: {
                    y: { title: { display: true, text: edges ? "Expert MAE − model MAE (positive is better)" : "MAE (shared-component points)" }, grid: { color: t.grid } },
                    x: { grid: { display: false } },
                },
            },
        };
    }, [weekly, sources, labels, theme, edges]);
    return <canvas ref={ref} />;
}

function EdgeValue({ value }) {
    if (value == null) return <span className="delta-neutral">--</span>;
    const cls = value >= 0.02 ? "delta-positive" : value <= -0.02 ? "delta-negative" : "delta-neutral";
    const sign = value > 0 ? "+" : "";
    return <span className={cls}>{`${sign}${value.toFixed(2)}`}</span>;
}

// Every filter participates in the cache key; responses never relabel old data.
const timelineCache = new Map();

export function TimelineView({ scoring, theme }) {
    const [group, setGroup] = useState("offense");
    const [season, setSeason] = useState("");
    const query = `/api/timeline?${new URLSearchParams({ scoring, group, ...(season ? { season } : {}) })}`;
    const [response, setResponse] = useState(null);
    const payload = response?.query === query ? response.data : timelineCache.get(query);
    const error = response?.query === query ? response.error : null;
    const [family, setFamily] = useState("ALL");
    const [sort, setSort] = useState({ key: "week", order: "desc" });

    useEffect(() => {
        const cached = timelineCache.get(query);
        if (cached) { setResponse({ query, data: cached }); return undefined; }
        let cancelled = false;
        fetchJSON(query)
            .then((data) => {
                if (data.schema_version !== 2) throw new Error("Timeline data is updating. Reload this page to try again.");
                timelineCache.set(query, data);
                if (!cancelled) setResponse({ query, data });
            })
            .catch((e) => {
                console.error("Failed to load timeline:", e);
                if (!cancelled) setResponse({ query, error: e.message });
            });
        return () => { cancelled = true; };
    }, [query]);

    const labels = (payload && payload.model_labels) || FAMILY_LABELS;
    const weekly = (payload && payload.weekly) || [];
    const releases = (payload && payload.releases) || [];
    const summary = (payload && payload.summary) || null;
    const sources = payload?.sources || [];
    const expertNames = (payload?.experts || []).map((key) => labels[key]).join(" and ");

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
            const va = a.mae[sort.key] ?? a[sort.key];
            const vb = b.mae[sort.key] ?? b[sort.key];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            const cmp = typeof va === "string" ? String(va).localeCompare(vb) : va - vb;
            return sort.order === "desc" ? -cmp : cmp;
        });
        return rows;
    }, [weekly, sort]);

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
                        Retrospective evaluation of the current forecasts on completed regular-season games.
                        Every source is graded on the same player-weeks and projected stats.
                        Each model keeps its own record across the season.
                    </div>
                </div>
            </div>

            <div className="release-controls">
                <div className="pill-group" aria-label="Timeline comparison group">
                    {(payload?.groups || GROUPS).map((option) => (
                        <button key={option.id} type="button" aria-pressed={group === option.id}
                            className={`pill${group === option.id ? " active" : ""}`}
                            onClick={() => setGroup(option.id)}>{option.label}</button>
                    ))}
                </div>
                <label>Season {" "}
                    <select aria-label="Timeline season" value={season} onChange={(e) => setSeason(e.target.value)}>
                        <option value="">Latest</option>
                        {(payload?.seasons || (season ? [Number(season)] : [])).map((year) => <option key={year} value={year}>{year}</option>)}
                    </select>
                </label>
            </div>
            {error && <p className="error-message">Failed to load timeline: {error}</p>}
            {!error && !payload && <p className="arch-loading">Loading timeline…</p>}

            {payload && summary && (
                <>
                    <div className="section-header">Each Model’s Season Record</div>
                    <p className="results-info">
                        {payload.season ? `${payload.season} · ` : ""}{payload.positions.join(" / ")} · {expertNames} · {summary.n} common player-weeks
                    </p>
                    {summary.reason && <p className="error-message">{REASONS[summary.reason]}</p>}
                    <div className="timeline-track-card">
                        <div className="stat-block-row">
                            {MODELS.map((model) => (
                                <div className="stat-block" key={model}>
                                    <span className="stat-block-label">{labels[model]} · MAE</span>
                                    <span className="stat-block-value neutral">{fmt(summary.models[model].mae, 2)}</span>
                                    <span>Beat {payload.experts.length > 1 ? "both experts" : expertNames}: {summary.models[model].beat_experts} / {summary.models[model].evaluated_weeks} weeks</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <p className="results-info">
                        Season MAE weights each evaluated player-week equally. {summary.evaluated_weeks} / {summary.total_weeks} weeks evaluable.
                        Positive edge means that model beat every required expert on the common sample.
                    </p>
                    <details className="results-info">
                        <summary>Scoring and coverage</summary>
                        <p>Common rows / eligible rows: {summary.n} / {summary.cohort_n}. Matching observed stats: {summary.actual_n}.</p>
                        <p>Forecast coverage with matching actuals: {sources.map((source) => `${labels[source]} ${summary.source_n[source]}`).join(" · ")}.</p>
                        {Object.entries(payload.scoring_components).map(([position, components]) => (
                            <p key={position}>{position}: {components.map((name) => name.replaceAll("_", " ")).join(", ")}.</p>
                        ))}
                        {Object.entries(payload.excluded_sources).map(([source, reason]) => <p key={source}>{reason}</p>)}
                        <p>Missing stats or forecasts reduce every source’s sample equally. An unavailable required source leaves a gap in the charts.</p>
                    </details>

                    <div className="section-header">Season Accuracy Trend</div>
                    <div className="charts-row">
                        <div className="chart-box"><TimelineChart weekly={weekly} sources={sources} labels={labels} theme={theme} /></div>
                        <div className="chart-box"><TimelineChart weekly={weekly} sources={sources} labels={labels} theme={theme} edges /></div>
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
                        Common rows / eligible rows · lower MAE is better · expand a week’s coverage for missing data
                    </div>
                    <div className="table-container">
                        <table id="timeline-table">
                            <thead>
                                <tr>
                                    <SortableTh label="Wk" sortKey="week" className="col-week" sort={sort.key} order={sort.order} onSort={onSort} />
                                    <SortableTh label="Common rows" sortKey="n" sort={sort.key} order={sort.order} onSort={onSort} />
                                    {sources.map((source) => <SortableTh key={source} label={labels[source]} sortKey={source} className="col-pred" sort={sort.key} order={sort.order} onSort={onSort} />)}
                                </tr>
                            </thead>
                            <tbody>
                                {sortedWeekly.map((w) => (
                                    <tr key={w.week}>
                                        <td className="col-week"><strong>{w.week}</strong></td>
                                        <td>
                                            <details>
                                                <summary>{w.n} / {w.cohort_n}</summary>
                                                {w.reason && <p>{REASONS[w.reason]}</p>}
                                                <p>Matching observed stats: {w.actual_n}</p>
                                                {sources.map((source) => <p key={source}>{labels[source]}: {w.source_n[source]}</p>)}
                                            </details>
                                        </td>
                                        {sources.map((source) => (
                                            <td key={source} className="col-pred" title={MODELS.includes(source) && w.edges[source] != null ? `Edge vs experts: ${w.edges[source].toFixed(3)}` : undefined}>{fmt(w.mae[source], 2)}</td>
                                        ))}
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

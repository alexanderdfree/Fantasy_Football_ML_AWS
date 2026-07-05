/* Benchmark History — reads /api/benchmark_history, one row per training run,
 * newest first. Fidelity port of the vanilla app.js History section: markup,
 * class names, pill tint math, delta annotation, and timestamp formatting are
 * unchanged. Each MAE cell renders a list of per-position pills since a run may
 * only retrain a subset of positions (CI's `detect` job scopes by changed
 * paths). PR numbers come from a top-level field that CI writes when it can
 * resolve the merge commit to a PR; for runs where the lookup returned empty
 * (manual dispatches, force pushes) we fall back to a commit-SHA link. The
 * fetched payload is cached at module level so tab revisits re-render without
 * re-fetching. */
import { Fragment, useEffect, useMemo, useState } from "react";
import { fetchJSON } from "../api.js";
import { fmt, formatTargetMae } from "../lib/format.js";
import { ApproachBanner, PosBadge } from "../components/common.jsx";

// Layout constants for the History table. Mirror the backend's
// _BENCHMARK_MODELS / _BENCHMARK_POSITIONS ordering so a row's per-model pill
// arrays line up by index.
const HISTORY_MODELS = ["ridge", "nn", "attn_nn", "lgbm"];
const HISTORY_MODEL_LABELS = { ridge: "Ridge", nn: "NN", attn_nn: "Attn NN", lgbm: "LGBM" };
const HISTORY_MODEL_COL_CLASS = { ridge: "ridge-col", nn: "nn-col", attn_nn: "attn-nn-col", lgbm: "lgbm-col" };
const HISTORY_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"];

// Module-level cache of the last fetch so tab revisits re-render without
// re-fetching. { rows, repoSlug, targetLabels, targetUnits } | null.
let historyCache = null;

function formatTrainingTime(seconds) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "--";
    const total = Math.round(seconds);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

function formatHistoryDelta(delta) {
    // Signed 2-decimal delta vs the prior run, as plain tooltip text (the caller
    // wraps it in the title attribute and may append an "all-time best" note);
    // U+2212 minus matches the displayed glyphs. Lower MAE is better. Returns ""
    // when there's no baseline (first appearance / metric absent on prior run).
    if (delta == null || !Number.isFinite(delta)) return "";
    const sign = delta < 0 ? "−" : "+";
    return `${sign}${fmt(Math.abs(delta), 2)} vs prev run`;
}

// Magnitude→fill-intensity scaling for the History pills. A relative (percent)
// change of HISTORY_INTENSITY_FULL_PCT or more saturates the green/red tint;
// smaller changes fade toward HISTORY_PILL_MIN_ALPHA so a barely-over-threshold
// delta is still faintly visible. Percent (not absolute) so positions sitting on
// different fantasy-point scales weight comparably.
const HISTORY_INTENSITY_FULL_PCT = 0.05;
const HISTORY_PILL_MIN_ALPHA = 0.12;
const HISTORY_PILL_MAX_ALPHA = 0.5;
// Fixed fill for an all-time-best pill: a record is binary, so it gets one clear,
// consistent blue regardless of the margin it won by.
const HISTORY_RECORD_ALPHA = 0.32;

function historyPillAlpha(intensity) {
    const t = Math.min(1, Math.max(0, intensity || 0));
    return (HISTORY_PILL_MIN_ALPHA + (HISTORY_PILL_MAX_ALPHA - HISTORY_PILL_MIN_ALPHA) * t).toFixed(3);
}

function historyPillTint(e) {
    // Resolve a pill's fill class + inline background. An all-time best is solid
    // blue and overrides the delta tint; otherwise green (improve) / red
    // (regress) with alpha scaled by the change magnitude. The background is set
    // inline (mirrors the vanilla band-fill pattern) so the class only carries
    // the border color + legend hook.
    if (e.value == null) return { cls: null, style: null };
    if (e.isRecord) {
        return { cls: "history-pill-record", style: { backgroundColor: `rgba(59,130,246,${HISTORY_RECORD_ALPHA})` } };
    }
    if (e.deltaClass === "history-pill-improve") {
        return { cls: "history-pill-improve", style: { backgroundColor: `rgba(34,197,94,${historyPillAlpha(e.intensity)})` } };
    }
    if (e.deltaClass === "history-pill-regress") {
        return { cls: "history-pill-regress", style: { backgroundColor: `rgba(239,68,68,${historyPillAlpha(e.intensity)})` } };
    }
    return { cls: null, style: null };
}

function historyPillTitle(e) {
    // Tooltip: signed delta vs prior run, plus an "all-time best" note on records.
    const parts = [];
    const d = formatHistoryDelta(e.delta);
    if (d) parts.push(d);
    if (e.isRecord) parts.push("all-time best");
    return parts.length ? parts.join(" · ") : "";
}

/* Generic pill list: each entry is
 * {label, value, deltaClass?, delta?, intensity?, isRecord?, isBest?}.
 * `value` is the active metric (MAE or RMSE). label is a position
 * (group-by-model layout) or a model name (group-by-position). value=null
 * renders as "--" (that position-model pair didn't train in this run, or has
 * no value for the active metric); empty list renders an em-dash. The pill
 * fill encodes change vs the prior run — green (improve) / red (regress) with
 * a deeper tint for a larger relative change — or solid blue when the value
 * is an all-time best (isRecord, which overrides the green/red tint). isBest
 * bolds the best model in a group-by-position cell. */
function SummaryPills({ entries }) {
    if (!Array.isArray(entries) || entries.length === 0) return <span className="history-empty">—</span>;
    return entries.map((e, i) => {
        const display = e.value == null
            ? <span className="history-pill-skip">--</span>
            : fmt(e.value, 2);
        const tint = historyPillTint(e);
        const cls = ["history-pill", tint.cls, e.isBest ? "history-pill-best" : null]
            .filter(Boolean)
            .join(" ");
        const title = e.value == null ? "" : historyPillTitle(e);
        return (
            <span key={i} className={cls} title={title || undefined} style={tint.style || undefined}>
                <span className="history-pill-pos">{e.label}</span>{" "}{display}
            </span>
        );
    });
}

function historyColumns(groupByPosition, metric) {
    // Group-by-position (default): one column per position. These carry no
    // .{model}-col class, so the page-wide #model-display hide rule is
    // intentionally inert here (model becomes an inner dimension). Group-by-model:
    // one column per model, keeping .{model}-col so #model-display still filters.
    if (groupByPosition) {
        return HISTORY_POSITIONS.map(pos => ({ key: pos, label: pos, cls: "col-history-mae" }));
    }
    const metricLabel = metric === "rmse" ? "RMSE" : "MAE";
    return HISTORY_MODELS.map(m => ({
        key: m,
        label: `${HISTORY_MODEL_LABELS[m]} ${metricLabel}`,
        cls: `col-history-mae ${HISTORY_MODEL_COL_CLASS[m]}`,
    }));
}

function historyCellEntries(row, columnKey, groupByPosition, metric) {
    if (groupByPosition) {
        // Column is a position; inner entries are the four models at that position.
        // The four values share a scale here, so we can flag the best (lowest) one.
        const posIdx = HISTORY_POSITIONS.indexOf(columnKey);
        const entries = HISTORY_MODELS.map(m => {
            const p = row[m] && row[m][posIdx];
            return {
                label: HISTORY_MODEL_LABELS[m],
                value: p ? (p[metric] ?? null) : null,
                deltaClass: p ? p.deltaClass : null,
                delta: p ? p.delta : null,
                intensity: p ? p.intensity : 0,
                isRecord: p ? !!p.isRecord : false,
            };
        });
        let bestIdx = -1;
        entries.forEach((e, i) => {
            if (e.value != null && (bestIdx < 0 || e.value < entries[bestIdx].value)) bestIdx = i;
        });
        if (bestIdx >= 0) entries[bestIdx].isBest = true;
        return entries;
    }
    // Column is a model; inner entries are the six positions in canonical order.
    // No best-flag: positions sit on different raw-stat scales (not comparable).
    return (row[columnKey] || []).map(p => ({
        label: p.position,
        value: p[metric] ?? null,
        deltaClass: p.deltaClass,
        delta: p.delta,
        intensity: p.intensity,
        isRecord: !!p.isRecord,
    }));
}

function historyRowHasDetail(row) {
    // A run is expandable only when some (position, model) cell carries
    // per-target detail — skipped/sentinel runs and old totals-only runs aren't.
    return HISTORY_MODELS.some(m => (row[m] || []).some(p => p && p.per_target));
}

/* PR/commit identifier cell. If the API didn't return a slug (test scenario,
 * misconfigured env), skip the link and render the identifier as plain text
 * rather than emitting a broken https:///pull/N URL that 404s on click. */
function HistoryIdCell({ repoSlug, row }) {
    const slug = (repoSlug || "").trim();
    if (row.pr_number != null) {
        if (!slug) return <span className="history-link-disabled">#{row.pr_number}</span>;
        const href = `https://github.com/${slug}/pull/${row.pr_number}`;
        return <a className="history-link" href={href} target="_blank" rel="noopener">#{row.pr_number}</a>;
    }
    if (row.git_hash) {
        if (!slug) return <span className="history-link-disabled"><code>{row.git_hash}</code></span>;
        const href = `https://github.com/${slug}/commit/${encodeURIComponent(row.git_hash)}`;
        return <a className="history-link" href={href} target="_blank" rel="noopener"><code>{row.git_hash}</code></a>;
    }
    return "—";
}

// Lazily-built, reused across rows/renders: constructing an Intl.DateTimeFormat
// is comparatively expensive and a render formats one timestamp per row. Built
// on first use rather than at module load so a (theoretical) missing-IANA-zone
// throw stays contained to a single cell render instead of breaking the module.
let _historyTsFormatter = null;
function historyTsFormatter() {
    if (!_historyTsFormatter) {
        _historyTsFormatter = new Intl.DateTimeFormat("en-CA", {
            timeZone: "America/New_York",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            hourCycle: "h23",
        });
    }
    return _historyTsFormatter;
}

function formatHistoryTimestamp(ts) {
    if (!ts) return "--";
    // Stored as "2026-05-19T22:47:20" (no tz marker, always UTC per
    // utc_now_iso) — the canonical value stays UTC behind the scenes. For
    // display we convert to US Eastern (ET, auto EST/EDT via the IANA zone)
    // since that's the operator's locale. Keep the compact 24-hour
    // "YYYY-MM-DD HH:MM" shape at minute resolution so same-PR reruns stay
    // distinguishable.
    const raw = String(ts);
    const match = raw.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
    if (!match) return raw;
    // Parse the naive timestamp as UTC: strip any sub-second / tz suffix the
    // source might carry, then append "Z" so Date interprets it as UTC rather
    // than the viewer's local zone.
    const isoUtc = raw.replace(/(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$/, "") + "Z";
    const d = new Date(isoUtc);
    if (Number.isNaN(d.getTime())) return `${match[1]} ${match[2]}`;
    // Rebuild from parts (en-CA + America/New_York would emit a comma between
    // date and time); hourCycle h23 keeps midnight as "00", not "24".
    const parts = {};
    for (const p of historyTsFormatter().formatToParts(d)) {
        parts[p.type] = p.value;
    }
    return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
}

const HISTORY_DELTA_EPS = 0.005; // only color a change the 2-decimal display reflects

function annotateHistoryDeltas(rows, metric) {
    // Tag each pill on the active metric with (a) a deltaClass (improve/regress)
    // + tint `intensity` vs the most recent EARLIER run that trained the same
    // position+model, and (b) `isRecord` when its value is the all-time best
    // (lowest) for that position+model across the whole history. Runs only
    // retrain a subset of positions, so the delta baseline is not the adjacent
    // table row — a lastSeen map keyed by `${position}|${model}` carries it
    // across the gaps. Walk oldest→newest (rows arrive newest-first) so lastSeen
    // is always the prior value. First appearance of a pos+model stays neutral
    // (no baseline) but can still be a record. Recomputed on every metric change
    // so the marks track the metric toggle; pills with no value for the active
    // metric (e.g. RMSE on a run that predates it) get delta/deltaClass/
    // intensity/isRecord cleared so a prior metric's mark doesn't linger.
    if (!Array.isArray(rows)) return;
    // Pass 1: all-time minimum per pos+model on this metric. Records compare
    // against the entire history, not just the rows walked so far, so this must
    // precede the delta walk.
    const minVal = {};
    for (const row of rows) {
        for (const m of HISTORY_MODELS) {
            for (const p of row[m] || []) {
                if (!p) continue;
                const v = p[metric];
                if (v == null) continue;
                const key = `${p.position}|${m}`;
                if (minVal[key] == null || v < minVal[key]) minVal[key] = v;
            }
        }
    }
    // Pass 2: per-run delta direction/magnitude + the record flag.
    const lastSeen = {};
    for (let i = rows.length - 1; i >= 0; i--) {
        const row = rows[i];
        for (const m of HISTORY_MODELS) {
            for (const p of row[m] || []) {
                if (!p) continue;
                const cur = p[metric];
                if (cur == null) {
                    p.delta = null;
                    p.deltaClass = null;
                    p.intensity = 0;
                    p.isRecord = false;
                    continue;
                }
                const key = `${p.position}|${m}`;
                const prev = lastSeen[key];
                if (prev != null) {
                    const delta = cur - prev;
                    p.delta = delta;
                    // Relative magnitude (guard a ~0 baseline → treat as full).
                    const den = Math.abs(prev);
                    const pct = den > 1e-9 ? Math.abs(delta) / den : 1;
                    p.intensity = Math.min(1, pct / HISTORY_INTENSITY_FULL_PCT);
                    p.deltaClass =
                        delta <= -HISTORY_DELTA_EPS
                            ? "history-pill-improve"
                            : delta >= HISTORY_DELTA_EPS
                              ? "history-pill-regress"
                              : null;
                } else {
                    p.delta = null;
                    p.deltaClass = null;
                    p.intensity = 0;
                }
                // All-time best for this pos+model on the active metric. minVal is
                // an observed value, so the holder(s) match within a float
                // epsilon; an exact tie flags both. A record overrides the
                // green/red delta tint at render time.
                p.isRecord = cur <= minVal[key] + 1e-9;
                lastSeen[key] = cur;
            }
        }
    }
}

/* One block per trained position: a target(rows) x model(cols) table that
 * mirrors the Model Performance tab's per-position detail, reusing
 * formatTargetMae for units + fantasy-point equivalents. Orientation is fixed
 * (per-position blocks) regardless of the group-by-position toggle — targets
 * are position-specific, so model-as-column is the only clean layout. */
function HistoryDetailRow({ row, colSpan, metric, hidden, targetLabels, targetUnits, scoring }) {
    // Values come from the metric-specific per-target map; the RMSE map is
    // absent on runs predating it, so those cells render "--".
    const ptKey = metric === "rmse" ? "per_target_rmse" : "per_target";
    return (
        <tr className="history-detail-row" hidden={hidden}>
            <td colSpan={colSpan}>
                {HISTORY_POSITIONS.map((pos, posIdx) => {
                    // Target set/order always comes from the MAE map (per_target),
                    // which is present whenever a cell has detail — so a run with
                    // no per-target RMSE still lists its targets (rendered "--"
                    // under the RMSE view) rather than collapsing the block.
                    let targets = null;
                    for (const m of HISTORY_MODELS) {
                        const pt = row[m] && row[m][posIdx] && row[m][posIdx].per_target;
                        if (pt) { targets = Object.keys(pt); break; }
                    }
                    if (!targets || !targets.length) return null;
                    return (
                        <div className="history-detail-block" key={pos}>
                            <div className="history-detail-pos"><PosBadge position={pos} /></div>
                            <div className="table-container">
                                <table className="pos-model-table">
                                    <thead><tr><th>Target</th><th>Ridge</th><th>NN</th><th>Attn NN</th><th>LGBM</th></tr></thead>
                                    <tbody>
                                        {targets.map(tkey => (
                                            <tr key={tkey}>
                                                <td className="tm-name">{targetLabels[tkey] || tkey}</td>
                                                {HISTORY_MODELS.map(m => {
                                                    const pt = row[m] && row[m][posIdx] && row[m][posIdx][ptKey];
                                                    const val = pt ? pt[tkey] : null;
                                                    return (
                                                        <td className="tm-val" key={m}>
                                                            {formatTargetMae(val, tkey, targetUnits[tkey], scoring)}
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    );
                })}
            </td>
        </tr>
    );
}

export function HistoryView({ scoring, search, theme, onPlayer, activateView }) {
    const [{ status, data }, setHistory] = useState(() => (
        historyCache ? { status: "ready", data: historyCache } : { status: "loading", data: null }
    ));
    const [detailed, setDetailed] = useState(false);
    // Checkbox is "Group by model"; default (unchecked) groups by position.
    const [groupByModel, setGroupByModel] = useState(false);
    // Active error metric (MAE/RMSE toggle, mirrors the Comparison tab).
    // Re-renders from cached data on change — never refetches.
    const [metric, setMetric] = useState("mae");
    const [expanded, setExpanded] = useState(() => new Set());

    useEffect(() => {
        if (status !== "loading") return;
        if (historyCache) {
            setHistory({ status: "ready", data: historyCache });
            return;
        }
        let cancelled = false;
        (async () => {
            try {
                const payload = await fetchJSON("/api/benchmark_history");
                historyCache = {
                    rows: payload.rows || [],
                    repoSlug: payload.repo_slug || "",
                    targetLabels: payload.target_labels || {},
                    targetUnits: payload.target_units || {},
                };
                if (!cancelled) setHistory({ status: "ready", data: historyCache });
            } catch (e) {
                console.error("Failed to load benchmark history:", e);
                historyCache = null;
                if (!cancelled) setHistory({ status: "error", data: null });
            }
        })();
        return () => { cancelled = true; };
    }, [status]);

    const groupByPosition = !groupByModel;
    const rows = data ? data.rows : null;

    // Deltas (and their green/red tints) are metric-specific, so re-annotate on
    // metric change rather than once at fetch time. Mutates the module-cached
    // rows (plain data, not React state) exactly like the vanilla renderer.
    const visibleRows = useMemo(() => {
        if (!rows) return [];
        annotateHistoryDeltas(rows, metric);
        // Hide commits that didn't retrain (training-skipped sentinels): they
        // carry no MAE data and only add noise. training_skipped is set per row
        // by the backend (explicit sentinel flag, or empty results).
        return rows.filter(row => !row.training_skipped);
    }, [rows, metric]);

    const columns = historyColumns(groupByPosition, metric);
    const colSpan = columns.length + 3; // PR + Timestamp + variable cols + Training time

    // Any layout-affecting toggle rebuilds the table in the vanilla app
    // (innerHTML), which collapses every expanded row — mirror that here.
    const onDetailed = e => {
        setDetailed(e.target.checked);
        setExpanded(new Set());
    };
    const onGroupByModel = e => {
        setGroupByModel(e.target.checked);
        setExpanded(new Set());
    };
    const onMetric = m => {
        setMetric(m);
        setExpanded(new Set());
    };
    // Click a detailed-mode row to toggle its detail row. Clicks on the
    // PR/commit link still work.
    const toggleRow = (e, idx) => {
        if (e.target.closest("a")) return;
        setExpanded(prev => {
            const next = new Set(prev);
            if (next.has(idx)) next.delete(idx);
            else next.add(idx);
            return next;
        });
    };
    const retry = () => setHistory({ status: "loading", data: null });

    return (
        <section id="view-history" className="view active">
            <ApproachBanner icon="clock" title="Benchmark History">
                One row per CI training run, newest first. Each cell lists every position trained in that run, on the position's raw-stat scale; toggle MAE/RMSE to switch the error metric. PRs that only touch a subset of positions retrain just those.
            </ApproachBanner>
            <div className="history-controls">
                <label className="history-check">
                    <input type="checkbox" id="history-detailed-toggle" checked={detailed} onChange={onDetailed} />
                    {" "}Detailed mode <span className="history-check-hint">(click a run for the per-target breakdown)</span>
                </label>
                <label className="history-check">
                    <input type="checkbox" id="history-group-by-model-toggle" checked={groupByModel} onChange={onGroupByModel} />
                    {" "}Group by model
                </label>
                <span className="pill-group history-metric-toggle" id="history-metric-toggle">
                    <button className={`pill${metric === "mae" ? " active" : ""}`} type="button" data-metric="mae" onClick={() => onMetric("mae")}>MAE</button>
                    <button className={`pill${metric === "rmse" ? " active" : ""}`} type="button" data-metric="rmse" onClick={() => onMetric("rmse")}>RMSE</button>
                </span>
                <span className="history-legend">
                    <span className="history-legend-swatch history-pill-improve"></span> better than prior run{" "}
                    <span className="history-legend-swatch history-pill-regress"></span> worse{" "}
                    <span className="history-legend-swatch history-pill-record"></span> all-time best{" "}
                    <span className="history-legend-hint">deeper tint = larger change</span>{" "}
                    {/* The "bold = best model" legend only applies when grouping by
                        position (the group-by-model layout sets no best-flag —
                        positions sit on different raw-stat scales), so hide that
                        legend swatch when grouped by model. */}
                    <span id="history-legend-best-wrap" style={groupByPosition ? undefined : { display: "none" }}>
                        <span className="history-legend-best">bold</span> = best model
                    </span>
                </span>
            </div>
            <div className={`table-container${status === "loading" ? " loading" : ""}`} id="history-table-container">
                <table id="history-table">
                    <thead id="history-head">
                        {status === "ready" && (
                            <tr>
                                <th className="col-history-pr">PR</th>
                                <th className="col-history-ts">Timestamp (ET)</th>
                                {columns.map(c => <th key={c.key} className={c.cls}>{c.label}</th>)}
                                <th className="col-history-time">Training time</th>
                            </tr>
                        )}
                    </thead>
                    <tbody id="history-body">
                        {status === "loading" && (
                            <tr><td colSpan={9} className="arch-loading">Loading benchmark history…</td></tr>
                        )}
                        {status === "error" && (
                            // colSpan 9 = the widest layout (group-by-position); harmless when fewer.
                            <tr>
                                <td colSpan={9} className="error-message">
                                    Failed to load benchmark history.{" "}
                                    <button id="history-retry" className="history-retry" type="button" onClick={retry}>Retry</button>
                                </td>
                            </tr>
                        )}
                        {status === "ready" && !visibleRows.length && (
                            <tr><td colSpan={colSpan} className="arch-loading">No benchmark runs yet.</td></tr>
                        )}
                        {status === "ready" && visibleRows.map((row, i) => {
                            const expandable = detailed && historyRowHasDetail(row);
                            const isOpen = expandable && expanded.has(i);
                            return (
                                <Fragment key={i}>
                                    <tr
                                        className={expandable ? `history-row-expandable${isOpen ? " expanded" : ""}` : undefined}
                                        onClick={expandable ? e => toggleRow(e, i) : undefined}
                                    >
                                        <td className="col-history-pr">
                                            {expandable && <span className="history-caret">▸</span>}
                                            <HistoryIdCell repoSlug={data.repoSlug} row={row} />
                                        </td>
                                        <td className="col-history-ts">{formatHistoryTimestamp(row.timestamp)}</td>
                                        {columns.map(c => (
                                            <td key={c.key} className={c.cls}>
                                                <SummaryPills entries={historyCellEntries(row, c.key, groupByPosition, metric)} />
                                            </td>
                                        ))}
                                        <td className="col-history-time">{formatTrainingTime(row.total_elapsed_sec)}</td>
                                    </tr>
                                    {expandable && (
                                        <HistoryDetailRow
                                            row={row}
                                            colSpan={colSpan}
                                            metric={metric}
                                            hidden={!isOpen}
                                            targetLabels={data.targetLabels}
                                            targetUnits={data.targetUnits}
                                            scoring={scoring}
                                        />
                                    )}
                                </Fragment>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

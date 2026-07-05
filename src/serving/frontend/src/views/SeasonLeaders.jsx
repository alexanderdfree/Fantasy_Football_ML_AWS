/* Season Leaders (predictions) — the full player-week table. Snapshot mode
 * filters/sorts/paginates client-side over the preloaded rows; live-API mode
 * (no snapshot on this container) refetches /api/predictions on server-side
 * filter changes. Rows expand (via the ▸ caret) into a lazily-fetched
 * per-target breakdown; clicking elsewhere opens the week-trend modal. */
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchJSON } from "../api.js";
import { fmt, errDelta } from "../lib/format.js";
import { PillGroup, PosBadge, PlayerCell, DeltaCell, Pagination } from "../components/common.jsx";

const PAGE_SIZE = 50;
const COLUMN_FILTER_STORAGE_KEY = "seasonLeaderColumns";

const POSITION_OPTIONS = ["ALL", "QB", "RB", "WR", "TE", "K", "DST"].map((v) => ({
    value: v, label: v === "ALL" ? "All" : v,
}));

/* Column registry — mirrors the vanilla TABLE_COLUMNS (keys, classes, sort
 * fields, default visibility, and the localStorage contract). */
const TABLE_COLUMNS = [
    { key: "rank", label: "#", cls: "col-rank", always: true },
    { key: "player", label: "Player", cls: "col-player", sort: "name", always: true },
    { key: "position", label: "Pos", cls: "col-pos", sort: "position", defaultVisible: true },
    { key: "team", label: "Team", cls: "col-team", sort: "team", defaultVisible: true },
    { key: "week", label: "Wk", cls: "col-week", sort: "week", defaultVisible: true },
    { key: "actual", label: "Actual", cls: "col-actual", sort: "actual", defaultVisible: true },
    { key: "ridge_pred", label: "Ridge", cls: "col-pred ridge-col", sort: "ridge_pred", defaultVisible: true },
    { key: "nn_pred", label: "NN", cls: "col-pred nn-col", sort: "nn_pred", defaultVisible: true },
    { key: "attn_nn_pred", label: "Attn NN", cls: "col-pred attn-nn-col", sort: "attn_nn_pred", defaultVisible: true },
    { key: "lgbm_pred", label: "LGBM", cls: "col-pred lgbm-col", sort: "lgbm_pred", defaultVisible: true },
    { key: "nflcom_pred", label: "NFL.com", cls: "col-pred nflcom-col", sort: "nflcom_pred", defaultVisible: true },
    { key: "rotowire_pred", label: "RotoWire", cls: "col-pred rotowire-col", sort: "rotowire_pred", defaultVisible: true },
    { key: "ridge_err", label: "Ridge Err", cls: "col-delta ridge-col", sort: "ridge_err", defaultVisible: true },
    { key: "nn_err", label: "NN Err", cls: "col-delta nn-col", sort: "nn_err", defaultVisible: true },
    { key: "attn_err", label: "Attn Err", cls: "col-delta attn-nn-col", sort: "attn_err", defaultVisible: true },
    { key: "lgbm_err", label: "LGBM Err", cls: "col-delta lgbm-col", sort: "lgbm_err", defaultVisible: true },
];
const TOGGLEABLE_COLUMNS = TABLE_COLUMNS.filter((c) => !c.always);
const COLUMN_KEYS = new Set(TABLE_COLUMNS.map((c) => c.key));

function loadVisibleColumnKeys() {
    try {
        const raw = localStorage.getItem(COLUMN_FILTER_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : null;
        if (Array.isArray(parsed)) {
            const valid = parsed.filter((k) => COLUMN_KEYS.has(k));
            if (valid.length) return new Set(valid);
        }
    } catch (_e) { /* storage may be disabled or stale JSON may be present */ }
    return new Set(TOGGLEABLE_COLUMNS.filter((c) => c.defaultVisible).map((c) => c.key));
}

function saveVisibleColumnKeys(keys) {
    try {
        localStorage.setItem(COLUMN_FILTER_STORAGE_KEY, JSON.stringify([...keys]));
    } catch (_e) { /* storage may be disabled — non-fatal */ }
}

function sortValue(p, key) {
    switch (key) {
        case "ridge_err": return errDelta(p.ridge_pred, p.actual);
        case "nn_err": return errDelta(p.nn_pred, p.actual);
        case "attn_err": return errDelta(p.attn_nn_pred, p.actual);
        case "lgbm_err": return errDelta(p.lgbm_pred, p.actual);
        default: return p[key];
    }
}

function renderCell(col, p) {
    switch (col.key) {
        case "player": return <PlayerCell player={p} />;
        case "position": return <PosBadge position={p.position} />;
        case "team": return p.team;
        case "week": return p.week;
        case "actual": return <strong>{fmt(p.actual)}</strong>;
        case "ridge_pred": return fmt(p.ridge_pred);
        case "nn_pred": return fmt(p.nn_pred);
        case "attn_nn_pred": return fmt(p.attn_nn_pred);
        case "lgbm_pred": return fmt(p.lgbm_pred);
        case "nflcom_pred": return fmt(p.nflcom_pred);
        case "rotowire_pred": return fmt(p.rotowire_pred);
        case "ridge_err": return <DeltaCell pred={p.ridge_pred} actual={p.actual} />;
        case "nn_err": return <DeltaCell pred={p.nn_pred} actual={p.actual} />;
        case "attn_err": return <DeltaCell pred={p.attn_nn_pred} actual={p.actual} />;
        case "lgbm_err": return <DeltaCell pred={p.lgbm_pred} actual={p.actual} />;
        default: return null;
    }
}

/* Per-stat breakdown drill-down (lazy fetch, cached per player-week). */
const BREAKDOWN_MODELS = [
    { key: "actual", label: "Actual", cls: "" },
    { key: "ridge", label: "Ridge", cls: "ridge-col" },
    { key: "nn", label: "NN", cls: "nn-col" },
    { key: "attn_nn", label: "Attn NN", cls: "attn-nn-col" },
    { key: "lgbm", label: "LGBM", cls: "lgbm-col" },
];

function BreakdownRow({ playerId, week, colSpan }) {
    const [state, setState] = useState({ status: "loading", data: null });
    useEffect(() => {
        let cancelled = false;
        setState({ status: "loading", data: null });
        const params = new URLSearchParams({ player_id: playerId, week });
        fetchJSON(`/api/predictions/breakdown?${params}`)
            .then((data) => { if (!cancelled) setState({ status: "ready", data }); })
            .catch((e) => {
                console.error("Failed to load breakdown:", e);
                if (!cancelled) setState({ status: "error", data: null });
            });
        return () => { cancelled = true; };
    }, [playerId, week]);

    let body;
    if (state.status === "loading") {
        body = <span className="breakdown-msg">Loading…</span>;
    } else if (state.status === "error") {
        body = <span className="breakdown-msg">Failed to load breakdown.</span>;
    } else if (state.data.unavailable || !state.data.components || !state.data.components.length) {
        body = <span className="breakdown-msg">Per-stat breakdown unavailable for this snapshot.</span>;
    } else {
        body = (
            <table className="breakdown-table">
                <thead>
                    <tr>
                        <th className="bd-stat">Stat</th>
                        {BREAKDOWN_MODELS.map((m) => <th key={m.key} className={m.cls}>{m.label}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {state.data.components.map((c) => (
                        <tr key={c.key}>
                            <td className="bd-stat">{c.label}</td>
                            {BREAKDOWN_MODELS.map((m) => {
                                const v = c[m.key];
                                return (
                                    <td key={m.key} className={m.cls}>
                                        {v == null ? "--" : `${fmt(v, 1)}${c.unit ? ` ${c.unit}` : ""}`}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        );
    }
    return (
        <tr className="predictions-detail-row">
            <td colSpan={colSpan}>{body}</td>
        </tr>
    );
}

/* Columns show/hide dropdown (checkbox popover; closes on outside click). */
function ColumnFilter({ visibleKeys, onToggle }) {
    const [open, setOpen] = useState(false);
    const wrapRef = useRef(null);
    useEffect(() => {
        if (!open) return undefined;
        const onDoc = (e) => {
            if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
        };
        document.addEventListener("click", onDoc);
        return () => document.removeEventListener("click", onDoc);
    }, [open]);
    const selected = TOGGLEABLE_COLUMNS.filter((c) => visibleKeys.has(c.key)).length;
    return (
        <div className="column-filter" id="column-filter" ref={wrapRef}>
            <button
                type="button"
                id="column-filter-button"
                className="column-filter-button"
                aria-haspopup="true"
                aria-expanded={open}
                onClick={() => setOpen((v) => !v)}
            >
                {`Columns (${selected})`}
            </button>
            <div className="column-filter-menu" id="column-filter-menu" hidden={!open}>
                {TOGGLEABLE_COLUMNS.map((c) => (
                    <label key={c.key} className="column-filter-option">
                        <input
                            type="checkbox"
                            value={c.key}
                            checked={visibleKeys.has(c.key)}
                            onChange={() => onToggle(c.key)}
                        />
                        <span>{c.label}</span>
                    </label>
                ))}
            </div>
        </div>
    );
}

export function SeasonLeadersView({ scoring, search, bootstrap, onPlayer }) {
    const { usingSnapshot, snapshotData, weeks, teams } = bootstrap;
    const [position, setPosition] = useState("ALL");
    const [week, setWeek] = useState("ALL");
    const [team, setTeam] = useState("ALL");
    const [minPts, setMinPts] = useState("");
    const [sort, setSort] = useState({ key: "actual", order: "desc" });
    const [page, setPage] = useState(1);
    const [visibleKeys, setVisibleKeys] = useState(loadVisibleColumnKeys);
    const [expanded, setExpanded] = useState(() => new Set());
    // Live-API mode state (snapshot rows come straight from `bootstrap`).
    const [liveRows, setLiveRows] = useState([]);
    const [liveDegraded, setLiveDegraded] = useState([]);
    const [loading, setLoading] = useState(false);
    const [loadError, setLoadError] = useState(false);
    const tableRef = useRef(null);

    // Live-API fallback: the server filters position/week/search/sort/scoring;
    // team + min-points remain client-side (same as the vanilla app).
    useEffect(() => {
        if (usingSnapshot || !bootstrap.ready) return undefined;
        let cancelled = false;
        setLoading(true);
        setLoadError(false);
        const params = new URLSearchParams({
            position, week, search,
            sort: sort.key, order: sort.order, scoring,
        });
        fetchJSON(`/api/predictions?${params}`)
            .then((data) => {
                if (cancelled) return;
                setLiveRows(data.players || []);
                setLiveDegraded(data.degraded_positions || []);
            })
            .catch((e) => {
                console.error("Failed to load predictions:", e);
                if (!cancelled) { setLiveRows([]); setLoadError(true); }
            })
            .finally(() => { if (!cancelled) setLoading(false); });
        return () => { cancelled = true; };
    }, [usingSnapshot, bootstrap.ready, position, week, search, sort, scoring]);

    const allPlayers = usingSnapshot
        ? ((snapshotData && snapshotData.scoring[scoring]) || [])
        : liveRows;
    const degraded = usingSnapshot ? (bootstrap.degraded || []) : liveDegraded;

    // Keep the active sort on a visible column (hiding it falls back to Actual).
    const visibleColumns = TABLE_COLUMNS.filter((c) => c.always || visibleKeys.has(c.key));
    const sortCol = TABLE_COLUMNS.find((c) => c.sort === sort.key);
    const effectiveSort = sortCol && !(sortCol.always || visibleKeys.has(sortCol.key))
        ? { key: "actual", order: "desc" }
        : sort;

    const sorted = useMemo(() => {
        const q = (search || "").trim().toLowerCase();
        const minVal = parseFloat(minPts);
        const filtered = allPlayers.filter((p) => {
            if (position !== "ALL" && p.position !== position) return false;
            if (week !== "ALL" && String(p.week) !== String(week)) return false;
            if (team !== "ALL" && p.team !== team) return false;
            if (q && !(p.name || "").toLowerCase().includes(q)) return false;
            if (!isNaN(minVal)) {
                const preds = [p.ridge_pred, p.nn_pred, p.attn_nn_pred, p.lgbm_pred, p.nflcom_pred, p.rotowire_pred]
                    .filter((v) => v != null);
                if (!preds.length || Math.max(...preds) < minVal) return false;
            }
            return true;
        });
        return filtered.sort((a, b) => {
            const va = sortValue(a, effectiveSort.key);
            const vb = sortValue(b, effectiveSort.key);
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            const cmp = (typeof va === "string" || typeof vb === "string")
                ? String(va).localeCompare(String(vb))
                : va - vb;
            return effectiveSort.order === "desc" ? -cmp : cmp;
        });
    }, [allPlayers, position, week, team, search, minPts, effectiveSort.key, effectiveSort.order]);

    const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
    const safePage = Math.min(page, totalPages || 1);
    const start = (safePage - 1) * PAGE_SIZE;
    const pageRows = sorted.slice(start, start + PAGE_SIZE);
    const colSpan = visibleColumns.length;

    const onSort = (key) => {
        setPage(1);
        setSort((s) => (s.key === key ? { key, order: s.order === "desc" ? "asc" : "desc" } : { key, order: "desc" }));
    };

    const toggleColumn = (key) => {
        setVisibleKeys((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            saveVisibleColumnKeys(next);
            return next;
        });
    };

    const toggleExpanded = (rowKey) => {
        setExpanded((prev) => {
            const next = new Set(prev);
            if (next.has(rowKey)) next.delete(rowKey);
            else next.add(rowKey);
            return next;
        });
    };

    const onPageChange = (p) => {
        setPage(p);
        if (tableRef.current) tableRef.current.scrollIntoView({ behavior: "smooth" });
    };

    return (
        <section id="view-predictions" className="view active">
            <div className="filters-bar">
                <div className="filters-row-top">
                    <div className="filter-group">
                        <label>Position</label>
                        <PillGroup
                            id="position-filter"
                            options={POSITION_OPTIONS}
                            value={position}
                            onChange={(v) => { setPosition(v); setPage(1); }}
                        />
                    </div>
                </div>
                <div className="filters-row-bottom">
                    <div className="filter-group">
                        <label>Week</label>
                        <select id="week-filter" value={week} onChange={(e) => { setWeek(e.target.value); setPage(1); }}>
                            <option value="ALL">All Weeks</option>
                            {(weeks || []).map((w) => <option key={w} value={w}>Week {w}</option>)}
                        </select>
                    </div>
                    <div className="filter-group">
                        <label>Team</label>
                        <select id="team-filter" value={team} onChange={(e) => { setTeam(e.target.value); setPage(1); }}>
                            <option value="ALL">All Teams</option>
                            {(teams || []).map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>
                    <div className="filter-group">
                        <label>Min Proj. Pts</label>
                        <input
                            type="number"
                            id="min-points-filter"
                            min="0"
                            step="any"
                            placeholder="0"
                            inputMode="decimal"
                            value={minPts}
                            onChange={(e) => { setMinPts(e.target.value); setPage(1); }}
                        />
                    </div>
                    <div className="filter-group column-filter-group">
                        <label>Columns</label>
                        <ColumnFilter visibleKeys={visibleKeys} onToggle={toggleColumn} />
                    </div>
                    <div className="filter-meta">
                        <span className="meta-badge">2025 Season</span>
                    </div>
                </div>
            </div>

            <div className="results-info">
                <span id="results-count">
                    {`${sorted.length.toLocaleString()} player-week${sorted.length !== 1 ? "s" : ""}`}
                </span>
            </div>

            {degraded.length > 0 && (
                <div id="degraded-banner" className="degraded-banner" role="status" aria-live="polite">
                    {`Heads up: predictions unavailable for ${degraded.join(", ")}. Showing last updated data for the other positions.`}
                </div>
            )}

            <div className={`table-container${loading ? " loading" : ""}`} ref={tableRef}>
                <table id="predictions-table">
                    <thead id="predictions-head">
                        <tr>
                            {visibleColumns.map((c) => {
                                if (!c.sort) return <th key={c.key} className={c.cls}>{c.label}</th>;
                                const active = c.sort === effectiveSort.key;
                                const arrow = active ? (effectiveSort.order === "desc" ? "▼" : "▲") : "";
                                return (
                                    <th
                                        key={c.key}
                                        className={`${c.cls} sortable${active ? " active-sort" : ""}`}
                                        data-sort={c.sort}
                                        onClick={() => onSort(c.sort)}
                                    >
                                        {c.label} <span className="sort-arrow">{arrow}</span>
                                    </th>
                                );
                            })}
                        </tr>
                    </thead>
                    <tbody id="predictions-body">
                        {loadError && (
                            <tr><td colSpan={colSpan} className="error-message">Failed to load predictions.</td></tr>
                        )}
                        {!loadError && pageRows.map((p, i) => {
                            const rowKey = `${p.player_id}|${p.week}`;
                            const isExpanded = expanded.has(rowKey);
                            return [
                                <tr
                                    key={rowKey}
                                    className={`predictions-row-expandable${isExpanded ? " expanded" : ""}`}
                                    data-player-id={p.player_id}
                                    data-week={p.week}
                                    onClick={(e) => {
                                        if (e.target.closest(".row-caret")) {
                                            e.stopPropagation();
                                            toggleExpanded(rowKey);
                                        } else {
                                            onPlayer(p.player_id, null);
                                        }
                                    }}
                                >
                                    {visibleColumns.map((c) => (
                                        <td key={c.key} className={c.cls}>
                                            {c.key === "rank"
                                                ? <><span className="row-caret">▸</span>{start + i + 1}</>
                                                : renderCell(c, p)}
                                        </td>
                                    ))}
                                </tr>,
                                isExpanded && (
                                    <BreakdownRow
                                        key={`${rowKey}-detail`}
                                        playerId={p.player_id}
                                        week={p.week}
                                        colSpan={colSpan}
                                    />
                                ),
                            ];
                        })}
                    </tbody>
                </table>
            </div>

            <Pagination page={safePage} totalPages={totalPages} onChange={onPageChange} />
        </section>
    );
}

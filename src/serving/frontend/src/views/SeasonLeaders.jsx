/* Season Leaders (predictions) — the full player-week table. Snapshot mode
 * filters/sorts/paginates client-side over the preloaded rows; live-API mode
 * (no snapshot on this container) refetches /api/predictions on server-side
 * filter changes. Rows expand (via the ▸ caret) into a lazily-fetched
 * per-target breakdown; clicking elsewhere opens the week-trend modal.
 *
 * Filter bar v2 (design system): auto-fit one-row bar with Position (incl.
 * FLEX = RB/WR/TE), Week, Team, Age, Class (Rookies), and Min Proj. Pts,
 * plus pinned Columns / Filters menus and a live filtered-slice stat readout.
 * Age/Class only appear when the loaded rows carry `age` (stale snapshot
 * artifacts predating the roster-meta fields degrade to the classic bar). */
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchJSON } from "../api.js";
import { fmt, errDelta } from "../lib/format.js";
import { PillGroup, PosBadge, PlayerCell, DeltaCell, Pagination } from "../components/common.jsx";
import { AutoFitFilterBar, FilterSliceStats, AGE_BUCKETS, ageBucketFor } from "../components/FilterBar.jsx";
import { TeamLabel } from "../components/TeamLabel.jsx";
import { DropdownMenu } from "../ds/controls/DropdownMenu.jsx";

const PAGE_SIZE = 50;
const COLUMN_FILTER_STORAGE_KEY = "seasonLeaderColumns";

const POSITION_OPTIONS = ["ALL", "QB", "RB", "WR", "TE", "FLEX", "K", "DST"].map((v) => ({
    value: v, label: v === "ALL" ? "All" : v,
}));
const FLEX_POSITIONS = new Set(["RB", "WR", "TE"]);

const STAT_SOURCES = [
    { key: "ridge_pred", label: "Ridge" },
    { key: "nn_pred", label: "NN" },
    { key: "attn_nn_pred", label: "Attn NN" },
    { key: "lgbm_pred", label: "LGBM" },
    { key: "nflcom_pred", label: "NFL.com" },
    { key: "rotowire_pred", label: "RotoWire" },
];

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
        case "team": return <TeamLabel abbr={p.team} />;
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

export function SeasonLeadersView({ scoring, search, bootstrap, onPlayer }) {
    const { usingSnapshot, snapshotData, weeks, teams } = bootstrap;
    const [position, setPosition] = useState("ALL");
    const [week, setWeek] = useState("ALL");
    const [team, setTeam] = useState("ALL");
    const [age, setAge] = useState("ALL");
    const [rookieOnly, setRookieOnly] = useState(false);
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
    // team/age/class/min-points remain client-side. FLEX is a client-side
    // union, so the server query falls back to ALL for it.
    useEffect(() => {
        if (usingSnapshot || !bootstrap.ready) return undefined;
        let cancelled = false;
        setLoading(true);
        setLoadError(false);
        const params = new URLSearchParams({
            position: position === "FLEX" ? "ALL" : position,
            week, search,
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

    // Stale artifacts predating the roster-meta fields carry no `age` — hide
    // the Age/Class filters entirely rather than showing dead controls.
    const hasAge = useMemo(() => allPlayers.some((p) => p.age != null), [allPlayers]);

    const FILTER_ITEMS = useMemo(() => {
        const items = [
            { value: "position", label: "Position" },
            { value: "week", label: "Week" },
            { value: "team", label: "Team" },
        ];
        if (hasAge) {
            items.push({ value: "age", label: "Age" });
            items.push({ value: "class", label: "Class" });
        }
        items.push({ value: "minpts", label: "Min Proj. Pts" });
        return items;
    }, [hasAge]);

    // Keep the active sort on a visible column (hiding it falls back to Actual).
    const visibleColumns = TABLE_COLUMNS.filter((c) => c.always || visibleKeys.has(c.key));
    const sortCol = TABLE_COLUMNS.find((c) => c.sort === sort.key);
    const effectiveSort = sortCol && !(sortCol.always || visibleKeys.has(sortCol.key))
        ? { key: "actual", order: "desc" }
        : sort;

    const filtered = useMemo(() => {
        const q = (search || "").trim().toLowerCase();
        const minVal = parseFloat(minPts);
        const bucket = ageBucketFor(age);
        return allPlayers.filter((p) => {
            if (position === "FLEX") {
                if (!FLEX_POSITIONS.has(p.position)) return false;
            } else if (position !== "ALL" && p.position !== position) {
                return false;
            }
            if (week !== "ALL" && String(p.week) !== String(week)) return false;
            if (team !== "ALL" && p.team !== team) return false;
            if (age !== "ALL" && !bucket.test(p.age)) return false;
            if (rookieOnly && p.is_rookie !== true) return false;
            if (q && !(p.name || "").toLowerCase().includes(q)) return false;
            if (!isNaN(minVal)) {
                const preds = [p.ridge_pred, p.nn_pred, p.attn_nn_pred, p.lgbm_pred, p.nflcom_pred, p.rotowire_pred]
                    .filter((v) => v != null);
                if (!preds.length || Math.max(...preds) < minVal) return false;
            }
            return true;
        });
    }, [allPlayers, position, week, team, age, rookieOnly, search, minPts]);

    const sorted = useMemo(() => {
        return [...filtered].sort((a, b) => {
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
    }, [filtered, effectiveSort.key, effectiveSort.order]);

    const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
    const safePage = Math.min(page, totalPages || 1);
    const start = (safePage - 1) * PAGE_SIZE;
    const pageRows = sorted.slice(start, start + PAGE_SIZE);
    const colSpan = visibleColumns.length;

    const onSort = (key) => {
        setPage(1);
        setSort((s) => (s.key === key ? { key, order: s.order === "desc" ? "asc" : "desc" } : { key, order: "desc" }));
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

    const resetFilter = (key) => {
        if (key === "position") setPosition("ALL");
        else if (key === "week") setWeek("ALL");
        else if (key === "team") setTeam("ALL");
        else if (key === "age") setAge("ALL");
        else if (key === "class") setRookieOnly(false);
        else if (key === "minpts") setMinPts("");
    };

    const renderControl = (key, measure) => {
        const idSuffix = measure ? "-m" : "";
        switch (key) {
            case "position":
                return (
                    <div className="filter-group">
                        <label className="field-label">Position</label>
                        <PillGroup
                            id={measure ? undefined : "position-filter"}
                            options={POSITION_OPTIONS}
                            value={position}
                            onChange={(v) => { setPosition(v); setPage(1); }}
                        />
                    </div>
                );
            case "week":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`week-filter${idSuffix}`}>Week</label>
                        <select
                            id={`week-filter${idSuffix}`}
                            className="field-select"
                            value={week}
                            onChange={(e) => { setWeek(e.target.value); setPage(1); }}
                        >
                            <option value="ALL">All Weeks</option>
                            {(weeks || []).map((w) => <option key={w} value={w}>Week {w}</option>)}
                        </select>
                    </div>
                );
            case "team":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`team-filter${idSuffix}`}>Team</label>
                        <select
                            id={`team-filter${idSuffix}`}
                            className="field-select"
                            value={team}
                            onChange={(e) => { setTeam(e.target.value); setPage(1); }}
                        >
                            <option value="ALL">All Teams</option>
                            {(teams || []).map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>
                );
            case "age":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`age-filter${idSuffix}`}>Age</label>
                        <select
                            id={`age-filter${idSuffix}`}
                            className="field-select"
                            value={age}
                            onChange={(e) => { setAge(e.target.value); setPage(1); }}
                        >
                            {AGE_BUCKETS.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
                        </select>
                    </div>
                );
            case "class":
                return (
                    <div className="filter-group">
                        <label className="field-label">Class</label>
                        <div className="pill-group">
                            <button
                                type="button"
                                className={`pill${rookieOnly ? " active" : ""}`}
                                onClick={() => { setRookieOnly((v) => !v); setPage(1); }}
                            >
                                Rookies
                            </button>
                        </div>
                    </div>
                );
            case "minpts":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`min-points-filter${idSuffix}`}>Min Proj. Pts</label>
                        <input
                            type="number"
                            id={`min-points-filter${idSuffix}`}
                            className="text-field"
                            min="0"
                            step="any"
                            placeholder="0"
                            inputMode="decimal"
                            value={minPts}
                            onChange={(e) => { setMinPts(e.target.value); setPage(1); }}
                        />
                    </div>
                );
            default:
                return null;
        }
    };

    // Columns menu: `locked` columns are always shown (rendered checked but
    // disabled); rank never appears. Persisted under the legacy storage key.
    const columnItems = [
        { value: "player", label: "Player", disabled: true },
        ...TOGGLEABLE_COLUMNS.map((c) => ({ value: c.key, label: c.label })),
    ];
    const columnValue = ["player", ...TOGGLEABLE_COLUMNS.filter((c) => visibleKeys.has(c.key)).map((c) => c.key)];
    const onColumnsChange = (next) => {
        const nextSet = new Set(TOGGLEABLE_COLUMNS.filter((c) => next.includes(c.key)).map((c) => c.key));
        saveVisibleColumnKeys(nextSet);
        setVisibleKeys(nextSet);
    };

    const renderMenus = ({ visibleFilters, onFiltersChange, measure }) => (
        <>
            <span className="meta-badge">2025 Season</span>
            <DropdownMenu
                fieldLabel="Columns"
                label="Columns"
                panelTitle="Show columns"
                align="right"
                id={measure ? undefined : "column-filter-button"}
                items={columnItems}
                value={columnValue}
                onChange={onColumnsChange}
            />
            <DropdownMenu
                fieldLabel="Filters"
                label="Filters"
                panelTitle="Show filters"
                align="right"
                items={FILTER_ITEMS}
                value={visibleFilters}
                onChange={onFiltersChange}
            />
        </>
    );

    return (
        <section id="view-predictions" className="view active">
            <AutoFitFilterBar
                items={FILTER_ITEMS}
                renderControl={renderControl}
                renderMenus={renderMenus}
                onResetFilter={resetFilter}
                stats={<FilterSliceStats rows={filtered} sources={STAT_SOURCES} />}
            />

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

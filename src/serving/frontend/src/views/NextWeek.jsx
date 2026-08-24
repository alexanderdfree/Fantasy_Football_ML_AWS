/* Next Week (homepage) — live upcoming-week projections off /api/upcoming_week.
 * States: loading | warming (503) | offseason (available:false) | ready | error.
 * K/DST pills stay disabled: the upcoming artifact serves QB/RB/WR/TE only.
 *
 * Filter bar v2 (design system): auto-fit one-row bar with Position (incl.
 * FLEX), Team, Age, Class (Rookies), Min Proj. Pts, plus pinned Columns /
 * Filters menus. Age/Class appear only when the artifact rows carry `age`
 * (older artifacts degrade to the classic bar). */
import { useEffect, useMemo, useState } from "react";
import { fmt } from "../lib/format.js";
import { PillGroup, PosBadge, PlayerCell, SortableTh } from "../components/common.jsx";
import { AutoFitFilterBar, AGE_BUCKETS, ageBucketFor } from "../components/FilterBar.jsx";
import { TeamLabel, MatchupLabel } from "../components/TeamLabel.jsx";
import { DropdownMenu } from "../ds/controls/DropdownMenu.jsx";

const POSITION_OPTIONS = [
    { value: "ALL", label: "All" },
    { value: "QB", label: "QB" },
    { value: "RB", label: "RB" },
    { value: "WR", label: "WR" },
    { value: "TE", label: "TE" },
    { value: "FLEX", label: "FLEX" },
    { value: "K", label: "K", disabled: true, title: "Coming soon" },
    { value: "DST", label: "DST", disabled: true, title: "Coming soon" },
];
const FLEX_POSITIONS = new Set(["RB", "WR", "TE"]);

const COLUMNS = [
    { key: "rank", label: "#", cls: "col-rank", sort: "rank", always: true },
    { key: "player", label: "Player", cls: "col-player", sort: "name", always: true },
    { key: "position", label: "Pos", cls: "col-pos", sort: "position", defaultVisible: true },
    { key: "team", label: "Team", cls: "col-team", sort: "team", defaultVisible: true },
    { key: "matchup", label: "Matchup", cls: "col-matchup", sort: "matchup", defaultVisible: true },
    { key: "nn_pred", label: "NN", cls: "col-pred nn-col", sort: "nn_pred", defaultVisible: true },
    { key: "attn_nn_pred", label: "Attn NN", cls: "col-pred attn-nn-col", sort: "attn_nn_pred", defaultVisible: true },
    { key: "lgbm_pred", label: "LGBM", cls: "col-pred lgbm-col", sort: "lgbm_pred", defaultVisible: true },
    // Expert columns are feature-detected per source: each renders only when the
    // artifact carries data for that source (NFL.com publishes in-season only, so
    // it can be absent while RotoWire is live; older artifacts predate both fields).
    { key: "nflcom_pred", label: "NFL.com", cls: "col-pred nflcom-col", sort: "nflcom_pred", defaultVisible: true, expert: true },
    { key: "rotowire_pred", label: "RotoWire", cls: "col-pred rotowire-col", sort: "rotowire_pred", defaultVisible: true, expert: true },
    { key: "espn_pred", label: "ESPN", cls: "col-pred espn-col", sort: "espn_pred", defaultVisible: true, expert: true },
];
const TOGGLEABLE_COLUMNS = COLUMNS.filter((c) => !c.always);

// Best-ranking model projection for a row, used for the default sort. Per-position
// head *selection* (ADR-0003-compatible — selection, not ensembling): LightGBM
// ranks RB/WR better than the attention head (beats it on lineup regret in 4/4
// rolling-origin seasons vs RotoWire — todo/expert-gap-investigation-2026-06.md §3);
// other positions keep the attention-first chain. Display columns are unaffected.
// Ridge is deliberately absent: it's hidden on this view.
const LGBM_RANKED_POSITIONS = new Set(["RB", "WR"]);
function upcomingProjection(p) {
    const order = LGBM_RANKED_POSITIONS.has(p.position)
        ? [p.lgbm_pred, p.attn_nn_pred, p.nn_pred]
        : [p.attn_nn_pred, p.lgbm_pred, p.nn_pred];
    const best = order.find((v) => v != null);
    return best != null ? best : null;
}

function sortValue(p, key) {
    switch (key) {
        case "rank": return upcomingProjection(p);
        case "matchup": return p.opponent || null;
        default: return p[key];
    }
}

// Module-level cache so tab switches don't refetch a ready artifact.
let cachedState = null; // { state, data }

export function NextWeekView({ scoring, search, onPlayer }) {
    const [position, setPosition] = useState("ALL");
    const [team, setTeam] = useState("ALL");
    const [age, setAge] = useState("ALL");
    const [rookieOnly, setRookieOnly] = useState(false);
    const [minPts, setMinPts] = useState("");
    const [hiddenCols, setHiddenCols] = useState(() => new Set());
    const [{ state, data }, setUpcoming] = useState(cachedState || { state: "loading", data: null });
    const [sort, setSort] = useState({ key: "rank", order: "desc" });

    useEffect(() => {
        if (cachedState && cachedState.state === "ready") return;
        let cancelled = false;
        (async () => {
            try {
                const resp = await fetch("/api/upcoming_week");
                if (resp.status === 503) {
                    if (!cancelled) setUpcoming({ state: "warming", data: null });
                    return;
                }
                // Any other non-ok status (e.g. Flask's JSON 500 error handler)
                // is a real failure, not an artifact — reject it here so it hits
                // the catch and renders the error state, instead of parsing an
                // error body as data (#1436).
                if (!resp.ok) throw new Error(`API error: ${resp.status}`);
                const payload = await resp.json();
                if (!payload || payload.available === false) {
                    if (!cancelled) setUpcoming({ state: "offseason", data: null });
                    return;
                }
                cachedState = { state: "ready", data: payload };
                if (!cancelled) setUpcoming(cachedState);
            } catch (e) {
                console.error("Failed to load upcoming week:", e);
                if (!cancelled) setUpcoming({ state: "error", data: null });
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const onSort = (key) => setSort((s) => (
        s.key === key ? { key, order: s.order === "desc" ? "asc" : "desc" } : { key, order: "desc" }
    ));

    const allRows = useMemo(() => {
        if (state !== "ready" || !data) return [];
        return (data.scoring && data.scoring[scoring]) || [];
    }, [state, data, scoring]);

    const teams = useMemo(() => [...new Set(allRows.map((p) => p.team).filter(Boolean))].sort(), [allRows]);
    const hasAge = useMemo(() => allRows.some((p) => p.age != null), [allRows]);
    const expertHasData = useMemo(() => Object.fromEntries(
        COLUMNS.filter((c) => c.expert).map((c) => [c.key, allRows.some((p) => p[c.key] != null)]),
    ), [allRows]);

    const FILTER_ITEMS = useMemo(() => {
        const items = [
            { value: "position", label: "Position" },
            { value: "team", label: "Team" },
        ];
        if (hasAge) {
            items.push({ value: "age", label: "Age" });
            items.push({ value: "class", label: "Class" });
        }
        items.push({ value: "minpts", label: "Min Proj. Pts" });
        return items;
    }, [hasAge]);

    const rows = useMemo(() => {
        const q = (search || "").trim().toLowerCase();
        const minVal = parseFloat(minPts);
        const bucket = ageBucketFor(age);
        const filtered = allRows.filter((p) => {
            if (position === "FLEX") {
                if (!FLEX_POSITIONS.has(p.position)) return false;
            } else if (position !== "ALL" && p.position !== position) {
                return false;
            }
            if (team !== "ALL" && p.team !== team) return false;
            if (age !== "ALL" && !bucket.test(p.age)) return false;
            if (rookieOnly && p.is_rookie !== true) return false;
            if (q && !(p.name || "").toLowerCase().includes(q)) return false;
            if (!isNaN(minVal)) {
                const preds = [p.nn_pred, p.attn_nn_pred, p.lgbm_pred, p.nflcom_pred, p.rotowire_pred, p.espn_pred]
                    .filter((v) => v != null);
                if (!preds.length || Math.max(...preds) < minVal) return false;
            }
            return true;
        });
        return filtered.slice().sort((a, b) => {
            const va = sortValue(a, sort.key);
            const vb = sortValue(b, sort.key);
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            const cmp = (typeof va === "string" || typeof vb === "string")
                ? String(va).localeCompare(String(vb))
                : va - vb;
            return sort.order === "desc" ? -cmp : cmp;
        });
    }, [allRows, position, team, age, rookieOnly, search, minPts, sort]);

    const visibleColumns = COLUMNS.filter(
        (c) => (c.always || !hiddenCols.has(c.key)) && (!c.expert || expertHasData[c.key]),
    );

    const resetFilter = (key) => {
        if (key === "position") setPosition("ALL");
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
                            id={measure ? undefined : "homepage-position-filter"}
                            options={POSITION_OPTIONS}
                            value={position}
                            onChange={setPosition}
                        />
                    </div>
                );
            case "team":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`homepage-team-filter${idSuffix}`}>Team</label>
                        <select
                            id={`homepage-team-filter${idSuffix}`}
                            className="field-select"
                            value={team}
                            onChange={(e) => setTeam(e.target.value)}
                        >
                            <option value="ALL">All Teams</option>
                            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>
                );
            case "age":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`homepage-age-filter${idSuffix}`}>Age</label>
                        <select
                            id={`homepage-age-filter${idSuffix}`}
                            className="field-select"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
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
                                onClick={() => setRookieOnly((v) => !v)}
                            >
                                Rookies
                            </button>
                        </div>
                    </div>
                );
            case "minpts":
                return (
                    <div className="filter-group">
                        <label className="field-label" htmlFor={`homepage-min-points${idSuffix}`}>Min Proj. Pts</label>
                        <input
                            type="number"
                            id={`homepage-min-points${idSuffix}`}
                            className="text-field"
                            min="0"
                            step="any"
                            placeholder="0"
                            inputMode="decimal"
                            value={minPts}
                            onChange={(e) => setMinPts(e.target.value)}
                        />
                    </div>
                );
            default:
                return null;
        }
    };

    const columnItems = [
        { value: "player", label: "Player", disabled: true },
        ...TOGGLEABLE_COLUMNS.filter((c) => !c.expert || expertHasData[c.key]).map((c) => ({ value: c.key, label: c.label })),
    ];
    const columnValue = ["player", ...TOGGLEABLE_COLUMNS.filter((c) => !hiddenCols.has(c.key)).map((c) => c.key)];
    const onColumnsChange = (next) => {
        setHiddenCols(new Set(TOGGLEABLE_COLUMNS.filter((c) => !next.includes(c.key)).map((c) => c.key)));
    };

    const renderMenus = ({ visibleFilters, onFiltersChange }) => (
        <>
            {state === "ready" && data && <span className="meta-badge">{data.week_label}</span>}
            <DropdownMenu
                fieldLabel="Columns"
                label="Columns"
                panelTitle="Show columns"
                align="right"
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

    const message = {
        loading: "Loading next week's projections…",
        warming: "Building this week's projections… check back in a minute.",
        offseason: "No upcoming games scheduled — live projections resume when the next slate is posted.",
        error: "Failed to load next-week projections.",
    }[state];

    const renderCell = (col, p, i) => {
        switch (col.key) {
            case "rank": return i + 1;
            case "player": return <PlayerCell player={p} />;
            case "position": return <PosBadge position={p.position} />;
            case "team": return <TeamLabel abbr={p.team} />;
            case "matchup": return <MatchupLabel opponent={p.opponent} isHome={p.is_home} />;
            case "nn_pred": return fmt(p.nn_pred);
            case "attn_nn_pred": return fmt(p.attn_nn_pred);
            case "lgbm_pred": return fmt(p.lgbm_pred);
            case "nflcom_pred": return fmt(p.nflcom_pred);
            case "rotowire_pred": return fmt(p.rotowire_pred);
            case "espn_pred": return fmt(p.espn_pred);
            default: return null;
        }
    };

    return (
        <section id="view-homepage" className="view active">
            <AutoFitFilterBar
                items={FILTER_ITEMS}
                renderControl={renderControl}
                renderMenus={renderMenus}
                onResetFilter={resetFilter}
            />

            {state === "ready" && data && (
                <div id="homepage-banner" className="homepage-banner" role="status" aria-live="polite">
                    {`${data.week_label} — projected fantasy points (no games played yet)`}
                </div>
            )}

            <div className="results-info">
                <span id="homepage-count">
                    {state === "ready" ? `${rows.length.toLocaleString()} player${rows.length !== 1 ? "s" : ""}` : ""}
                </span>
            </div>

            <div className="table-container">
                <table id="homepage-table">
                    <thead>
                        <tr>
                            {visibleColumns.map((c) => (
                                <SortableTh
                                    key={c.key}
                                    label={c.label}
                                    sortKey={c.sort}
                                    className={c.cls}
                                    sort={sort.key}
                                    order={sort.order}
                                    onSort={onSort}
                                />
                            ))}
                        </tr>
                    </thead>
                    <tbody id="homepage-body">
                        {state !== "ready" ? (
                            <tr><td colSpan={visibleColumns.length} className="arch-loading">{message}</td></tr>
                        ) : rows.map((p, i) => (
                            <tr
                                key={`${p.player_id}-${i}`}
                                data-player-id={p.player_id}
                                onClick={() => onPlayer(p.player_id, {
                                    name: p.name, position: p.position, team: p.team, headshot: p.headshot,
                                })}
                            >
                                {visibleColumns.map((c) => (
                                    <td key={c.key} className={c.cls}>{renderCell(c, p, i)}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

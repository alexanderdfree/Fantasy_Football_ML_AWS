/* Next Week (homepage) — live upcoming-week projections off /api/upcoming_week.
 * States: loading | warming (503) | offseason (available:false) | ready | error.
 * K/DST pills stay disabled: the upcoming artifact serves QB/RB/WR/TE only. */
import { useEffect, useMemo, useState } from "react";
import { fmt } from "../lib/format.js";
import { PillGroup, PosBadge, PlayerCell, SortableTh } from "../components/common.jsx";

const POSITION_OPTIONS = [
    { value: "ALL", label: "All" },
    { value: "QB", label: "QB" },
    { value: "RB", label: "RB" },
    { value: "WR", label: "WR" },
    { value: "TE", label: "TE" },
    { value: "K", label: "K", disabled: true, title: "Coming soon" },
    { value: "DST", label: "DST", disabled: true, title: "Coming soon" },
];

// Best available model projection for a row (Attention NN preferred — best
// WR/RB model — falling back across the others), used for the default sort.
// Ridge is deliberately absent: it's hidden on this view.
function upcomingProjection(p) {
    if (p.attn_nn_pred != null) return p.attn_nn_pred;
    if (p.lgbm_pred != null) return p.lgbm_pred;
    return p.nn_pred;
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

    const rows = useMemo(() => {
        if (state !== "ready" || !data) return [];
        const all = (data.scoring && data.scoring[scoring]) || [];
        const q = (search || "").trim().toLowerCase();
        const filtered = all.filter((p) => {
            if (position !== "ALL" && p.position !== position) return false;
            if (q && !(p.name || "").toLowerCase().includes(q)) return false;
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
    }, [state, data, scoring, search, position, sort]);

    const message = {
        loading: "Loading next week's projections…",
        warming: "Building this week's projections… check back in a minute.",
        offseason: "No upcoming games scheduled — live projections resume when the next slate is posted.",
        error: "Failed to load next-week projections.",
    }[state];

    return (
        <section id="view-homepage" className="view active">
            <div className="filters-bar">
                <div className="filters-row-top">
                    <div className="filter-group">
                        <label>Position</label>
                        <PillGroup id="homepage-position-filter" options={POSITION_OPTIONS} value={position} onChange={setPosition} />
                    </div>
                </div>
            </div>

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
                            <SortableTh label="#" sortKey="rank" className="col-rank" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="Player" sortKey="name" className="col-player" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="Pos" sortKey="position" className="col-pos" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="Team" sortKey="team" className="col-team" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="Matchup" sortKey="matchup" className="col-matchup" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="NN" sortKey="nn_pred" className="col-pred nn-col" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="Attn NN" sortKey="attn_nn_pred" className="col-pred attn-nn-col" sort={sort.key} order={sort.order} onSort={onSort} />
                            <SortableTh label="LGBM" sortKey="lgbm_pred" className="col-pred lgbm-col" sort={sort.key} order={sort.order} onSort={onSort} />
                        </tr>
                    </thead>
                    <tbody id="homepage-body">
                        {state !== "ready" ? (
                            <tr><td colSpan={8} className="arch-loading">{message}</td></tr>
                        ) : rows.map((p, i) => (
                            <tr
                                key={`${p.player_id}-${i}`}
                                data-player-id={p.player_id}
                                onClick={() => onPlayer(p.player_id, {
                                    name: p.name, position: p.position, team: p.team, headshot: p.headshot,
                                })}
                            >
                                <td className="col-rank">{i + 1}</td>
                                <td className="col-player"><PlayerCell player={p} /></td>
                                <td className="col-pos"><PosBadge position={p.position} /></td>
                                <td className="col-team">{p.team}</td>
                                <td className="col-matchup">
                                    {p.opponent ? (p.is_home === 1 ? `vs ${p.opponent}` : `@ ${p.opponent}`) : "—"}
                                </td>
                                <td className="col-pred nn-col">{fmt(p.nn_pred)}</td>
                                <td className="col-pred attn-nn-col">{fmt(p.attn_nn_pred)}</td>
                                <td className="col-pred lgbm-col">{fmt(p.lgbm_pred)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}

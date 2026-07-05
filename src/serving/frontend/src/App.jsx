/* Top-level app: header, hash-routed tabs, active view, player modal.
 *
 * URL state mirrors the active tab so refresh, share-links, and the browser
 * back/forward buttons all behave as expected. Tab clicks pushState; wiki
 * sub-page navigation replaceState (inside WikiView) so intra-wiki link clicks
 * don't pile up history entries. Deep-link forms: #predictions, #wiki,
 * #wiki:slug[:anchor], #history, … (unknown hashes normalize to /).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchJSON } from "./api.js";
import { VALID_SCORING } from "./lib/format.js";
import { applyChartTheme } from "./lib/chartTheme.js";
import { Header } from "./Header.jsx";
import { PlayerModal } from "./PlayerModal.jsx";
import { NextWeekView } from "./views/NextWeek.jsx";
import { SeasonLeadersView } from "./views/SeasonLeaders.jsx";
import { ModelPerformanceView } from "./views/ModelPerformance.jsx";
import { ComparisonView } from "./views/Comparison.jsx";
import { ArchitectureView } from "./views/Architecture.jsx";
import { WikiView } from "./views/Wiki.jsx";
import { HistoryView } from "./views/History.jsx";

const TAB_VIEWS = ["homepage", "predictions", "model-performance", "comparison", "model-architecture", "wiki", "history"];
const TAB_LABELS = {
    homepage: "Next Week",
    predictions: "Season Leaders",
    "model-performance": "Model Performance",
    comparison: "Comparison",
    "model-architecture": "Model Architecture",
    wiki: "Wiki",
    history: "History",
};

function viewFromHash(hash) {
    if (!hash || hash === "#") return "homepage";
    if (hash.startsWith("#wiki:") || hash === "#wiki") return "wiki";
    const v = hash.slice(1);
    return TAB_VIEWS.includes(v) ? v : "homepage";
}

function hashForView(view) {
    if (view === "homepage") return "";
    return `#${view}`;
}

export function App() {
    const [view, setView] = useState(() => viewFromHash(location.hash));
    const [scoring, setScoring] = useState(() => {
        try {
            const stored = localStorage.getItem("scoringFormat");
            return VALID_SCORING.includes(stored) ? stored : "ppr";
        } catch (_e) { return "ppr"; }
    });
    const [searchInput, setSearchInput] = useState("");
    const [search, setSearch] = useState("");
    const [theme, setTheme] = useState(() => (
        document.documentElement.getAttribute("data-theme") === "oled" ? "oled" : "midnight"
    ));
    const [modal, setModal] = useState(null); // { playerId, fallback }
    // Snapshot-first bootstrap (zero server compute; falls back to the live API).
    const [bootstrap, setBootstrap] = useState({
        ready: false, usingSnapshot: false, snapshotData: null, weeks: [], teams: [], degraded: [],
    });

    // Debounce the header search (300ms, matching the vanilla app).
    const searchTimer = useRef(null);
    const onSearch = (value) => {
        setSearchInput(value);
        clearTimeout(searchTimer.current);
        searchTimer.current = setTimeout(() => setSearch(value), 300);
    };

    // Hydrate the first paint from the precomputed static snapshot when
    // available. On a miss (404 / no snapshot yet on this container) fall back
    // to the live API: weeks + teams load here, predictions load in the view.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const snap = await fetchJSON("/api/snapshot");
                if (snap && snap.scoring) {
                    // Teams aren't carried in the snapshot payload — derive them
                    // from the rows we already have (scoring-invariant).
                    const teams = [...new Set((snap.scoring.ppr || []).map((p) => p.team).filter(Boolean))].sort();
                    if (!cancelled) {
                        setBootstrap({
                            ready: true,
                            usingSnapshot: true,
                            snapshotData: snap,
                            weeks: snap.weeks || [],
                            teams,
                            degraded: snap.degraded_positions || [],
                        });
                    }
                    return;
                }
            } catch (_e) {
                // 404 or network error — fall through to the live API path.
            }
            try {
                const [weeksData, teamsData] = await Promise.all([
                    fetchJSON("/api/weeks").catch(() => ({ weeks: [] })),
                    fetchJSON("/api/teams").catch(() => ({ teams: [] })),
                ]);
                if (!cancelled) {
                    setBootstrap({
                        ready: true,
                        usingSnapshot: false,
                        snapshotData: null,
                        weeks: weeksData.weeks || [],
                        teams: teamsData.teams || [],
                        degraded: [],
                    });
                }
            } catch (_e) {
                if (!cancelled) setBootstrap((b) => ({ ...b, ready: true }));
            }
        })();
        return () => { cancelled = true; };
    }, []);

    // Routing: normalize the initial hash, follow back/forward.
    useEffect(() => {
        const expected = view === "wiki" && location.hash.startsWith("#wiki") ? location.hash : hashForView(view);
        if ((location.hash || "") !== expected) {
            history.replaceState(null, "", location.pathname + location.search + expected);
        }
        const onPop = () => setView(viewFromHash(location.hash));
        window.addEventListener("popstate", onPop);
        return () => window.removeEventListener("popstate", onPop);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const activateView = useCallback((next) => {
        const newHash = next === "wiki" && location.hash.startsWith("#wiki:") ? location.hash : hashForView(next);
        if ((location.hash || "") !== newHash) {
            history.pushState(null, "", location.pathname + location.search + newHash);
        }
        setView(next);
    }, []);

    const onToggleTheme = useCallback(() => {
        setTheme((prev) => {
            const next = prev === "oled" ? "midnight" : "oled";
            if (next === "oled") document.documentElement.setAttribute("data-theme", "oled");
            else document.documentElement.removeAttribute("data-theme");
            try { localStorage.setItem("ffp-theme", next); } catch (_e) { /* private mode */ }
            applyChartTheme();
            return next;
        });
    }, []);

    const onScoring = useCallback((next) => {
        setScoring(next);
        try { localStorage.setItem("scoringFormat", next); } catch (_e) { /* non-fatal */ }
    }, []);

    const openPlayer = useCallback((playerId, fallback = null) => {
        setModal({ playerId, fallback });
    }, []);
    const closeModal = useCallback(() => setModal(null), []);

    const viewProps = useMemo(() => ({
        scoring, search, theme, onPlayer: openPlayer, activateView,
    }), [scoring, search, theme, openPlayer, activateView]);

    return (
        <>
            <Header
                theme={theme}
                onToggleTheme={onToggleTheme}
                scoring={scoring}
                onScoring={onScoring}
                search={searchInput}
                onSearch={onSearch}
            />
            <nav className="nav-tabs">
                {TAB_VIEWS.map((v) => (
                    <button
                        key={v}
                        type="button"
                        className={`tab${v === view ? " active" : ""}`}
                        data-view={v}
                        onClick={() => activateView(v)}
                    >
                        {TAB_LABELS[v]}
                    </button>
                ))}
            </nav>
            <main className="main-content">
                {view === "homepage" && <NextWeekView {...viewProps} />}
                {view === "predictions" && <SeasonLeadersView {...viewProps} bootstrap={bootstrap} />}
                {view === "model-performance" && <ModelPerformanceView {...viewProps} />}
                {view === "comparison" && <ComparisonView {...viewProps} />}
                {view === "model-architecture" && <ArchitectureView {...viewProps} />}
                {view === "wiki" && <WikiView {...viewProps} />}
                {view === "history" && <HistoryView {...viewProps} />}
            </main>
            {modal && (
                <PlayerModal
                    playerId={modal.playerId}
                    fallback={modal.fallback}
                    scoring={scoring}
                    theme={theme}
                    onClose={closeModal}
                />
            )}
        </>
    );
}

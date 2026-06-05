/* Fantasy Football Predictor - Frontend */

const PAGE_SIZE = 50;
const VALID_SCORING = ["ppr", "half_ppr", "standard"];
let allPlayers = [];
// When true, predictions were hydrated from the precomputed /api/snapshot
// (zero server compute); filtering/sorting/scoring all happen client-side over
// `snapshotData`. When false we fell back to the live /api/predictions API.
let usingSnapshot = false;
let snapshotData = null;
let currentPage = 1;
let currentSort = "actual";
let currentOrder = "desc";
let playerChart = null;
let positionMaeChart = null;
let positionR2Chart = null;
let weeklyMaeChart = null;
let positionDetailsData = null;
let perfFilterInitialized = false;
let currentScoring = (() => {
    const stored = localStorage.getItem("scoringFormat");
    return VALID_SCORING.includes(stored) ? stored : "ppr";
})();
let currentPlayerId = null;
let modalOpen = false;

function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function fmt(n, d = 1) {
    return (n == null || isNaN(n)) ? "--" : Number(n).toFixed(d);
}

// Chart.js defaults
Chart.defaults.color = "#9aa0b0";
Chart.defaults.borderColor = "#2e3347";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}

const COLORS = {
    ridge: "#3b82f6",
    nn: "#22c55e",
    attn_nn: "#a855f7",
    lgbm: "#f59e0b",
    actual: "#e8eaed",
    ridgeBg: "rgba(59, 130, 246, 0.2)",
    nnBg: "rgba(34, 197, 94, 0.2)",
    attn_nnBg: "rgba(168, 85, 247, 0.2)",
    lgbmBg: "rgba(245, 158, 11, 0.2)",
};

// Per-target fantasy-point-equivalent multipliers. Mirror of
// shared/aggregate_targets.py:POINT_EQUIVALENT_MULTIPLIER, but `receptions`
// depends on scoring format (1.0 / 0.5 / 0.0). Applied only to count-style
// targets where the raw MAE would be dominated by the scoring coefficient
// (e.g. 0.4 TDs = 2.4 points).
const RECEPTION_WEIGHT = { ppr: 1.0, half_ppr: 0.5, standard: 0.0 };
const BASE_POINT_EQUIVALENT = {
    passing_tds: 4.0,
    rushing_tds: 6.0,
    receiving_tds: 6.0,
    interceptions: 2.0,
    fumbles_lost: 2.0,
};
function pointEquivMultiplier(targetKey, scoring) {
    if (targetKey === "receptions") return RECEPTION_WEIGHT[scoring] ?? 1.0;
    return BASE_POINT_EQUIVALENT[targetKey];
}

// Format a per-target MAE with its raw unit, and — for targets with a known
// point-equivalent multiplier — also show the implied fantasy-point delta.
// The API serializes a `unit` field per target (from shared/aggregate_targets.py:TARGET_UNITS).
function formatTargetMae(val, targetKey, unit, scoring) {
    if (val == null) return "--";
    const raw = unit ? `${fmt(val, 2)} ${unit}` : fmt(val, 2);
    const mult = pointEquivMultiplier(targetKey, scoring);
    return mult != null ? `${raw} (${fmt(val * mult, 2)} pts)` : raw;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);

async function init() {
    setupNavTabs();
    setupPositionFilters();
    setupSortHeaders();
    setupModal();
    setupSearch();
    setupModelToggle();
    setupScoringToggle();
    setupHistoryControls();
    setupWikiClickHandler();
    applyInitialRoute();

    // Attach filter change listeners
    document.getElementById("week-filter").addEventListener("change", () => { currentPage = 1; loadPredictions(); });
    // Team and Min-Proj-Pts are pure client-side filters (applied in
    // getFilteredPlayers over the already-loaded rows), so they re-render
    // without a refetch in both the snapshot and live-API paths.
    document.getElementById("team-filter").addEventListener("change", () => { currentPage = 1; renderTable(); });
    document.getElementById("min-points-filter").addEventListener("input", () => { currentPage = 1; renderTable(); });

    // The page shell is ready — drop the blocking full-screen overlay now
    // instead of holding it until data arrives. Predictions stream into the
    // table afterward with the lighter in-table loading indicator, so the
    // chrome (tabs, filters, search) is interactive immediately even when the
    // server is cold.
    document.getElementById("loading-overlay").classList.add("hidden");

    await bootstrapPredictions();
}

// Hydrate the first paint from the precomputed static snapshot (/api/snapshot)
// when available — zero model-load on the server, instant in all cases incl.
// cold containers. On a miss (404 / network / no snapshot yet) fall back to the
// live API, loading weeks + predictions in parallel rather than sequentially.
async function bootstrapPredictions() {
    try {
        const snap = await fetchJSON("/api/snapshot");
        if (snap && snap.scoring) {
            usingSnapshot = true;
            snapshotData = snap;
            populateWeeks(snap.weeks || []);
            // Teams aren't carried in the snapshot payload — derive them from the
            // rows we already have (scoring-invariant, so any format works).
            const teams = [...new Set((snap.scoring.ppr || []).map(p => p.team).filter(Boolean))].sort();
            populateTeams(teams);
            renderDegradedBanner(snap.degraded_positions || []);
            loadPredictions();  // snapshot branch: client-side filter + render
            return;
        }
    } catch (e) {
        // 404 (this container has no snapshot yet) or a network error — fall
        // through to the live API path below.
    }
    usingSnapshot = false;
    await Promise.all([loadWeeks(), loadTeams(), loadPredictions()]);
}

async function loadWeeks() {
    try {
        const weeksData = await fetchJSON("/api/weeks");
        populateWeeks(weeksData.weeks);
    } catch (e) {
        console.error("Failed to load weeks:", e);
    }
}

async function loadTeams() {
    try {
        const teamsData = await fetchJSON("/api/teams");
        populateTeams(teamsData.teams);
    } catch (e) {
        console.error("Failed to load teams:", e);
    }
}

function populateWeeks(weeks) {
    const weekSelect = document.getElementById("week-filter");
    (weeks || []).forEach(w => {
        const opt = document.createElement("option");
        opt.value = w;
        opt.textContent = `Week ${w}`;
        weekSelect.appendChild(opt);
    });
}

function populateTeams(teams) {
    const teamSelect = document.getElementById("team-filter");
    (teams || []).forEach(t => {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        teamSelect.appendChild(opt);
    });
}

// ---------------------------------------------------------------------------
// Navigation
//
// URL state mirrors the active tab so refresh, share-links, and the browser
// back/forward buttons all behave as expected. Tab clicks pushState; wiki
// sub-page navigation (inside loadWikiPage) replaceState so intra-wiki link
// clicks don't pile up history entries.
// ---------------------------------------------------------------------------
const TAB_VIEWS = new Set(["predictions", "standings", "model-performance", "comparison", "model-architecture", "wiki", "history"]);

function viewFromHash(hash) {
    if (!hash || hash === "#") return "predictions";
    if (hash.startsWith("#wiki:") || hash === "#wiki") return "wiki";
    const v = hash.slice(1);
    return TAB_VIEWS.has(v) ? v : "predictions";
}

function hashForView(view) {
    if (view === "predictions") return "";
    if (view === "wiki") return wikiCurrentSlug ? `#wiki:${wikiCurrentSlug}` : "#wiki";
    return `#${view}`;
}

function activateTab(view) {
    document.querySelectorAll(".nav-tabs .tab").forEach(t => {
        t.classList.toggle("active", t.dataset.view === view);
    });
    document.querySelectorAll(".view").forEach(v => {
        v.classList.toggle("active", v.id === `view-${view}`);
    });
    if (view === "model-performance") loadMetrics();
    else if (view === "standings") loadStandings();
    else if (view === "comparison") loadComparison();
    else if (view === "model-architecture") loadModelArchitecture();
    else if (view === "wiki") loadWiki();
    else if (view === "history") loadHistory();
}

function setupNavTabs() {
    document.querySelectorAll(".nav-tabs .tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const view = tab.dataset.view;
            // Push history BEFORE activating: loadWikiPage's cached-path
            // replaceState runs synchronously inside activateTab() and would
            // otherwise replace the previous tab's entry instead of letting
            // us push a new one on top.
            const newHash = hashForView(view);
            if ((location.hash || "") !== newHash) {
                history.pushState(null, "", location.pathname + location.search + newHash);
            }
            activateTab(view);
        });
    });
    window.addEventListener("popstate", () => {
        activateTab(viewFromHash(location.hash));
    });
}

function applyInitialRoute() {
    // Predictions is already `.active` in the HTML, so we only need to switch
    // when the hash routes elsewhere. Normalize unknown hashes so a paste of
    // /#bogus lands cleanly on /.
    const view = viewFromHash(location.hash);
    if (view !== "predictions") activateTab(view);
    const expectedHash = hashForView(view);
    if ((location.hash || "") !== expectedHash) {
        history.replaceState(null, "", location.pathname + location.search + expectedHash);
    }
}

// ---------------------------------------------------------------------------
// Position Filters
// ---------------------------------------------------------------------------
function setupPositionFilters() {
    setupPillGroup("position-filter", () => { currentPage = 1; loadPredictions(); });
    setupPillGroup("standings-position-filter", () => loadStandings());
}

function setupPillGroup(containerId, callback) {
    const container = document.getElementById(containerId);
    container.querySelectorAll(".pill").forEach(pill => {
        pill.addEventListener("click", () => {
            container.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            callback();
        });
    });
}

function getActivePosition(containerId) {
    const active = document.querySelector(`#${containerId} .pill.active`);
    return active ? active.dataset.value : "ALL";
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
function setupSearch() {
    let timeout;
    document.getElementById("search-input").addEventListener("input", e => {
        clearTimeout(timeout);
        timeout = setTimeout(() => { currentPage = 1; loadPredictions(); }, 300);
    });
}

// ---------------------------------------------------------------------------
// Model Toggle
// ---------------------------------------------------------------------------
function setupModelToggle() {
    document.getElementById("model-display").addEventListener("change", e => {
        const val = e.target.value;
        document.body.classList.remove("model-ridge", "model-nn", "model-attn_nn", "model-lgbm");
        if (val === "ridge") document.body.classList.add("model-ridge");
        else if (val === "nn") document.body.classList.add("model-nn");
        else if (val === "attn_nn") document.body.classList.add("model-attn_nn");
        else if (val === "lgbm") document.body.classList.add("model-lgbm");
    });
}

// ---------------------------------------------------------------------------
// Scoring Toggle (PPR / Half-PPR / Standard)
//
// Lives in the global header so it persists across tabs. Persisted in
// localStorage so the choice survives reloads. On change we always reload
// the predictions table (the underlying view) and then refresh whichever
// secondary view is currently active. Model Architecture and the Wiki are
// scoring-invariant so we don't reload them.
// ---------------------------------------------------------------------------
function setupScoringToggle() {
    const container = document.getElementById("scoring-filter");
    if (!container) return;
    // Apply saved value: mark the matching pill active and clear the others.
    container.querySelectorAll(".pill").forEach(pill => {
        pill.classList.toggle("active", pill.dataset.value === currentScoring);
    });
    container.querySelectorAll(".pill").forEach(pill => {
        pill.addEventListener("click", () => {
            const next = pill.dataset.value;
            if (!VALID_SCORING.includes(next) || next === currentScoring) {
                container.querySelectorAll(".pill").forEach(p =>
                    p.classList.toggle("active", p.dataset.value === currentScoring),
                );
                return;
            }
            container.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            currentScoring = next;
            try {
                localStorage.setItem("scoringFormat", currentScoring);
            } catch (_e) { /* storage may be disabled — non-fatal */ }
            onScoringChanged();
        });
    });
}

function onScoringChanged() {
    // Always reload the predictions table — it's the default landing view.
    currentPage = 1;
    loadPredictions();
    // Refresh whichever secondary tab is currently visible.
    const activeTab = document.querySelector(".nav-tabs .tab.active");
    const view = activeTab ? activeTab.dataset.view : null;
    if (view === "standings") loadStandings();
    if (view === "model-performance") loadMetrics();
    // History detail rows show per-target fantasy-point equivalents, so refresh
    // them too (collapses any open detail; re-opening uses the new format).
    if (view === "history") loadHistory();
    // If the player modal is open, re-fetch with the new format.
    if (modalOpen && currentPlayerId) openPlayerModal(currentPlayerId);
}

// ---------------------------------------------------------------------------
// Sort Headers
// ---------------------------------------------------------------------------
function setupSortHeaders() {
    document.querySelectorAll("th.sortable").forEach(th => {
        th.addEventListener("click", () => {
            const sort = th.dataset.sort;
            if (currentSort === sort) {
                currentOrder = currentOrder === "desc" ? "asc" : "desc";
            } else {
                currentSort = sort;
                currentOrder = "desc";
            }
            // Update UI
            document.querySelectorAll("th.sortable").forEach(t => {
                t.classList.remove("active-sort");
                t.querySelector(".sort-arrow").textContent = "";
            });
            th.classList.add("active-sort");
            th.querySelector(".sort-arrow").textContent = currentOrder === "desc" ? "\u25BC" : "\u25B2";
            currentPage = 1;
            renderTable();
        });
    });
}

// ---------------------------------------------------------------------------
// Predictions
// ---------------------------------------------------------------------------
async function loadPredictions() {
    if (usingSnapshot) {
        // The snapshot already holds every player-week for each scoring format.
        // Pick the active format; renderTable() filters (position/week/search),
        // sorts, and paginates client-side — no network round trip.
        allPlayers = (snapshotData && snapshotData.scoring[currentScoring]) || [];
        renderTable();
        return;
    }

    const position = getActivePosition("position-filter");
    const week = document.getElementById("week-filter").value;
    const search = document.getElementById("search-input").value;

    const params = new URLSearchParams({
        position, week, search,
        sort: currentSort,
        order: currentOrder,
        scoring: currentScoring,
    });

    const container = document.querySelector("#view-predictions .table-container");
    container.classList.add("loading");
    try {
        const data = await fetchJSON(`/api/predictions?${params}`);
        allPlayers = data.players || [];
        renderDegradedBanner(data.degraded_positions || []);
        renderTable();
    } catch (e) {
        console.error("Failed to load predictions:", e);
        allPlayers = [];
        document.getElementById("predictions-body").innerHTML =
            '<tr><td colspan="14" class="error-message">Failed to load predictions.</td></tr>';
    } finally {
        container.classList.remove("loading");
    }
}

function renderDegradedBanner(degraded) {
    const banner = document.getElementById("degraded-banner");
    if (!banner) return;
    if (!degraded || degraded.length === 0) {
        banner.classList.add("hidden");
        banner.textContent = "";
        return;
    }
    banner.textContent =
        `Heads up: predictions unavailable for ${degraded.join(", ")}. ` +
        `Showing last updated data for the other positions.`;
    banner.classList.remove("hidden");
}

// Client-side position/week/search filter over `allPlayers`. In snapshot mode
// `allPlayers` is the full dataset for the active scoring format, so this is the
// real filter. In live-API (fallback) mode the server already filtered by the
// same current criteria, so re-applying them here is an idempotent no-op.
function getFilteredPlayers() {
    const position = getActivePosition("position-filter");
    const week = document.getElementById("week-filter").value;
    const team = document.getElementById("team-filter").value;
    const search = document.getElementById("search-input").value.trim().toLowerCase();
    // Min projected points: a row passes if ANY model projects >= the threshold
    // (max across the available model predictions). NaN (empty input) disables it.
    const minPts = parseFloat(document.getElementById("min-points-filter").value);
    return allPlayers.filter(p => {
        if (position !== "ALL" && p.position !== position) return false;
        if (week !== "ALL" && String(p.week) !== String(week)) return false;
        if (team !== "ALL" && p.team !== team) return false;
        if (search && !(p.name || "").toLowerCase().includes(search)) return false;
        if (!isNaN(minPts)) {
            const preds = [p.ridge_pred, p.nn_pred, p.attn_nn_pred, p.lgbm_pred].filter(v => v != null);
            if (!preds.length || Math.max(...preds) < minPts) return false;
        }
        return true;
    });
}

function renderTable() {
    // Sort locally for instant re-sorting
    const sorted = [...getFilteredPlayers()].sort((a, b) => {
        const va = a[currentSort] ?? 0;
        const vb = b[currentSort] ?? 0;
        return currentOrder === "desc" ? vb - va : va - vb;
    });

    const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
    if (currentPage > totalPages) currentPage = totalPages || 1;
    const start = (currentPage - 1) * PAGE_SIZE;
    const page = sorted.slice(start, start + PAGE_SIZE);

    document.getElementById("results-count").textContent =
        `${sorted.length.toLocaleString()} player-week${sorted.length !== 1 ? "s" : ""}`;

    const tbody = document.getElementById("predictions-body");
    tbody.innerHTML = page.map((p, i) => {
        const rank = start + i + 1;

        const delta = (pred) => (pred != null && p.actual != null) ? (pred - p.actual).toFixed(1) : null;
        const ridgeDelta = delta(p.ridge_pred);
        const nnDelta = delta(p.nn_pred);
        const attnDelta = delta(p.attn_nn_pred);
        const lgbmDelta = delta(p.lgbm_pred);
        const cls = (d) => d != null ? deltaClass(d) : "delta-neutral";

        const headshot = p.headshot
            ? `<img class="player-headshot" src="${escapeHtml(p.headshot)}" alt="" loading="lazy">`
            : `<div class="player-headshot"></div>`;

        // Main row is followed by a hidden detail row (the per-stat breakdown,
        // fetched lazily on first expand). The caret toggles the breakdown;
        // clicking elsewhere on the row still opens the week-trend modal.
        return `<tr class="predictions-row-expandable" data-player-id="${escapeHtml(p.player_id)}" data-week="${p.week}">
            <td class="col-rank"><span class="row-caret">▸</span>${rank}</td>
            <td class="col-player"><div class="player-cell">${headshot}<span class="player-name">${escapeHtml(p.name)}</span></div></td>
            <td class="col-pos"><span class="pos-badge pos-${escapeHtml(p.position)}">${escapeHtml(p.position)}</span></td>
            <td class="col-team">${escapeHtml(p.team)}</td>
            <td class="col-week">${p.week}</td>
            <td class="col-actual"><strong>${fmt(p.actual)}</strong></td>
            <td class="col-pred ridge-col">${fmt(p.ridge_pred)}</td>
            <td class="col-pred nn-col">${fmt(p.nn_pred)}</td>
            <td class="col-pred attn-nn-col">${fmt(p.attn_nn_pred)}</td>
            <td class="col-pred lgbm-col">${fmt(p.lgbm_pred)}</td>
            <td class="col-delta ridge-col ${cls(ridgeDelta)}">${ridgeDelta != null ? fmtDelta(ridgeDelta) : "--"}</td>
            <td class="col-delta nn-col ${cls(nnDelta)}">${nnDelta != null ? fmtDelta(nnDelta) : "--"}</td>
            <td class="col-delta attn-nn-col ${cls(attnDelta)}">${attnDelta != null ? fmtDelta(attnDelta) : "--"}</td>
            <td class="col-delta lgbm-col ${cls(lgbmDelta)}">${lgbmDelta != null ? fmtDelta(lgbmDelta) : "--"}</td>
        </tr><tr class="predictions-detail-row" hidden><td colspan="14"></td></tr>`;
    }).join("");

    // Caret toggles the inline per-stat breakdown; clicking elsewhere on the row
    // opens the week-trend modal (unchanged behavior). Scope to main rows so the
    // detail rows don't get a modal handler.
    tbody.querySelectorAll("tr.predictions-row-expandable").forEach(row => {
        row.addEventListener("click", (e) => {
            if (e.target.closest(".row-caret")) {
                e.stopPropagation();
                toggleBreakdown(row);
            } else {
                openPlayerModal(row.dataset.playerId);
            }
        });
    });

    renderPagination(totalPages);
}

// Per-stat breakdown drill-down. Columns reuse the model col classes so the
// "Model Display" selector hides sub-table columns in lockstep with the main
// table (CSS body.model-* rules in style.css are descendant selectors).
const BREAKDOWN_MODELS = [
    { key: "actual", label: "Actual", cls: "" },
    { key: "ridge", label: "Ridge", cls: "ridge-col" },
    { key: "nn", label: "NN", cls: "nn-col" },
    { key: "attn_nn", label: "Attn NN", cls: "attn-nn-col" },
    { key: "lgbm", label: "LGBM", cls: "lgbm-col" },
];

function toggleBreakdown(row) {
    const detail = row.nextElementSibling;
    if (!detail || !detail.classList.contains("predictions-detail-row")) return;
    const willShow = detail.hidden;
    detail.hidden = !detail.hidden;
    row.classList.toggle("expanded", willShow);
    // Lazy: fetch + render only on first expand. Cached in the DOM thereafter.
    if (willShow && !detail.dataset.loaded) {
        detail.dataset.loaded = "1";
        loadBreakdown(row.dataset.playerId, row.dataset.week, detail.querySelector("td"));
    }
}

async function loadBreakdown(playerId, week, cellEl) {
    cellEl.innerHTML = '<span class="breakdown-msg">Loading…</span>';
    try {
        const params = new URLSearchParams({ player_id: playerId, week });
        const data = await fetchJSON(`/api/predictions/breakdown?${params}`);
        cellEl.innerHTML = renderBreakdownTable(data);
    } catch (e) {
        console.error("Failed to load breakdown:", e);
        cellEl.innerHTML = '<span class="breakdown-msg">Failed to load breakdown.</span>';
        // Allow a retry on next expand.
        cellEl.parentElement.dataset.loaded = "";
    }
}

function renderBreakdownTable(data) {
    if (data.unavailable || !data.components || !data.components.length) {
        return '<span class="breakdown-msg">Per-stat breakdown unavailable for this snapshot.</span>';
    }
    const head = BREAKDOWN_MODELS.map(m => `<th class="${m.cls}">${m.label}</th>`).join("");
    const body = data.components.map(c => {
        const cells = BREAKDOWN_MODELS.map(m => {
            const v = c[m.key];
            const txt = v == null
                ? "--"
                : `${fmt(v, 1)}${c.unit ? " " + escapeHtml(c.unit) : ""}`;
            return `<td class="${m.cls}">${txt}</td>`;
        }).join("");
        return `<tr><td class="bd-stat">${escapeHtml(c.label)}</td>${cells}</tr>`;
    }).join("");
    return `<table class="breakdown-table">
        <thead><tr><th class="bd-stat">Stat</th>${head}</tr></thead>
        <tbody>${body}</tbody>
    </table>`;
}

function deltaClass(d) {
    const n = parseFloat(d);
    if (Math.abs(n) < 1) return "delta-neutral";
    return n > 0 ? "delta-positive" : "delta-negative";
}

function fmtDelta(d) {
    const n = parseFloat(d);
    const sign = n > 0 ? "+" : "";
    return `${sign}${n}`;
}

function renderPagination(totalPages) {
    const container = document.getElementById("pagination");
    if (totalPages <= 1) { container.innerHTML = ""; return; }

    let html = `<button class="page-btn" ${currentPage === 1 ? "disabled" : ""} data-page="${currentPage - 1}">&laquo;</button>`;

    const maxVisible = 7;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);

    if (startPage > 1) html += `<button class="page-btn" data-page="1">1</button><span style="color:var(--text-muted)">...</span>`;

    for (let p = startPage; p <= endPage; p++) {
        html += `<button class="page-btn ${p === currentPage ? "active" : ""}" data-page="${p}">${p}</button>`;
    }

    if (endPage < totalPages) html += `<span style="color:var(--text-muted)">...</span><button class="page-btn" data-page="${totalPages}">${totalPages}</button>`;

    html += `<button class="page-btn" ${currentPage === totalPages ? "disabled" : ""} data-page="${currentPage + 1}">&raquo;</button>`;

    container.innerHTML = html;
    container.querySelectorAll(".page-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.disabled) return;
            currentPage = parseInt(btn.dataset.page);
            renderTable();
            document.querySelector(".table-container").scrollIntoView({ behavior: "smooth" });
        });
    });
}

// ---------------------------------------------------------------------------
// Season Leaders
// ---------------------------------------------------------------------------
async function loadStandings() {
    const position = getActivePosition("standings-position-filter");
    const container = document.querySelector("#view-standings .table-container");
    container.classList.add("loading");

    try {
        const data = await fetchJSON(
            `/api/top_players?position=${position}&scoring=${currentScoring}`,
        );

        const tbody = document.getElementById("standings-body");
        tbody.innerHTML = data.players.map((p, i) => `
            <tr data-player-id="${escapeHtml(p.player_id)}">
                <td class="col-rank">${i + 1}</td>
                <td class="col-player"><span class="player-name">${escapeHtml(p.name)}</span></td>
                <td class="col-pos"><span class="pos-badge pos-${escapeHtml(p.position)}">${escapeHtml(p.position)}</span></td>
                <td class="col-team">${escapeHtml(p.team)}</td>
                <td class="col-games">${p.games}</td>
                <td class="col-actual"><strong>${fmt(p.avg_actual)}</strong></td>
                <td class="col-pred">${fmt(p.avg_ridge)}</td>
                <td class="col-pred">${fmt(p.avg_nn)}</td>
                <td class="col-pred">${fmt(p.avg_attn_nn)}</td>
                <td class="col-pred">${fmt(p.avg_lgbm)}</td>
            </tr>
        `).join("");

        tbody.querySelectorAll("tr").forEach(row => {
            row.addEventListener("click", () => openPlayerModal(row.dataset.playerId));
        });
    } catch (e) {
        console.error("Failed to load standings:", e);
        document.getElementById("standings-body").innerHTML =
            '<tr><td colspan="10" class="error-message">Failed to load standings.</td></tr>';
    } finally {
        container.classList.remove("loading");
    }
}

// ---------------------------------------------------------------------------
// Model Performance
// ---------------------------------------------------------------------------
async function loadMetrics() {
    try {
        const q = `?scoring=${currentScoring}`;
        const [metrics, weekly, posDetails] = await Promise.all([
            fetchJSON(`/api/metrics${q}`),
            fetchJSON(`/api/weekly_accuracy${q}`),
            fetchJSON(`/api/position_details${q}`),
        ]);
        positionDetailsData = posDetails;

        // Overall metrics cards — populate each from its model entry, gracefully
        // falling back to "--" when a model has no overall (e.g. only K/DST rows).
        const cards = [
            { key: "Ridge Regression", prefix: "ridge" },
            { key: "Neural Network", prefix: "nn" },
            { key: "Attention NN", prefix: "attn-nn" },
            { key: "LightGBM", prefix: "lgbm" },
        ];
        for (const { key, prefix } of cards) {
            const m = metrics[key];
            const overall = m && m.overall;
            document.getElementById(`${prefix}-mae`).textContent = overall ? overall.mae.toFixed(3) : "--";
            document.getElementById(`${prefix}-rmse`).textContent = overall ? overall.rmse.toFixed(3) : "--";
            document.getElementById(`${prefix}-r2`).textContent = overall ? overall.r2.toFixed(3) : "--";
        }

        // Position model breakdown
        setupPerfPositionFilter();
        renderPositionModelDetail(getActivePosition("perf-position-filter"));

        // Position charts — pass the full metrics object so charts can render up to 4 series
        renderPositionCharts(metrics);

        // Weekly MAE chart
        renderWeeklyChart(weekly);
    } catch (e) {
        console.error("Failed to load metrics:", e);
        document.querySelector("#view-model-performance .metrics-grid").innerHTML =
            '<p class="error-message">Failed to load model metrics.</p>';
    }
}

function setupPerfPositionFilter() {
    if (perfFilterInitialized) return;
    perfFilterInitialized = true;
    setupPillGroup("perf-position-filter", () => {
        renderPositionModelDetail(getActivePosition("perf-position-filter"));
    });
}

function renderPositionModelDetail(pos) {
    const container = document.getElementById("pos-model-detail");
    if (!positionDetailsData || !positionDetailsData[pos]) {
        container.innerHTML = '<p class="pos-model-empty">Loading...</p>';
        return;
    }

    const d = positionDetailsData[pos];
    const tm = d.target_metrics || {};

    // Per-target rows render MAE in the target's native unit (yards / TDs / receptions).
    // For TD/INT/fumble/reception targets, the raw MAE is also shown in
    // fantasy-point-equivalent terms via the active scoring multiplier (so
    // 0.40 TDs renders as "0.40 TDs (2.40 pts)" and 0.50 receptions as
    // "(0.50 pts)" PPR / "(0.25 pts)" half / "(0.00 pts)" standard).
    const targetRows = (d.targets || []).map(t => {
        const m = tm[t.key] || {};
        const unit = m.unit;
        return `<tr>
            <td class="tm-name">${escapeHtml(t.label)}</td>
            <td class="tm-formula">${escapeHtml(t.formula)}</td>
            <td class="tm-val">${escapeHtml(formatTargetMae(m.ridge_mae, t.key, unit, currentScoring))}</td>
            <td class="tm-val">${escapeHtml(formatTargetMae(m.nn_mae, t.key, unit, currentScoring))}</td>
            <td class="tm-val">${escapeHtml(formatTargetMae(m.attn_nn_mae, t.key, unit, currentScoring))}</td>
            <td class="tm-val">${escapeHtml(formatTargetMae(m.lgbm_mae, t.key, unit, currentScoring))}</td>
        </tr>`;
    }).join("");

    // Total row is always in fantasy points (aggregator output), so no unit formatting.
    const totalM = tm["total"] || {};
    const totalCell = (v) => v != null ? `<strong>${v.toFixed(2)}</strong>` : '<strong>--</strong>';
    const totalRow = `<tr class="tm-total-row">
        <td class="tm-name"><strong>Total (fantasy points)</strong></td>
        <td class="tm-formula">${escapeHtml(d.adjustments || '')}</td>
        <td class="tm-val">${totalCell(totalM.ridge_mae)}</td>
        <td class="tm-val">${totalCell(totalM.nn_mae)}</td>
        <td class="tm-val">${totalCell(totalM.attn_nn_mae)}</td>
        <td class="tm-val">${totalCell(totalM.lgbm_mae)}</td>
    </tr>`;

    // Feature badges
    const featureBadges = (d.specific_features || []).map(f =>
        `<span class="feature-badge">${escapeHtml(f)}</span>`
    ).join("");

    // Architecture
    const arch = d.architecture || {};
    const backbone = (arch.backbone || []).join(" > ");

    container.innerHTML = `
        <div class="pos-model-card">
            <div class="pos-model-header">
                <span class="pos-badge pos-${escapeHtml(pos)}">${escapeHtml(pos)}</span>
                <span class="pos-model-name">${escapeHtml(d.label)} Model</span>
                <span class="pos-model-meta">${escapeHtml(d.n_features || '?')} features &middot; ${escapeHtml(d.n_samples_test || '?')} test samples</span>
            </div>

            <div class="pos-model-section-label">Raw-Stat Targets</div>
            <div class="table-container pos-model-table-wrap">
                <table class="pos-model-table">
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
                        ${targetRows}
                        ${totalRow}
                    </tbody>
                </table>
            </div>

            <div class="pos-model-section-label">Position-Specific Features</div>
            <div class="feature-badges">${featureBadges}</div>

            <div class="pos-model-section-label">Neural Network Architecture</div>
            <div class="arch-info">Shared backbone <span class="arch-val">[${escapeHtml(backbone)}]</span> &rarr; ${(d.targets || []).length} heads (hidden: <span class="arch-val">${escapeHtml(arch.head_hidden || '?')}</span>)</div>
        </div>
    `;
}

function renderPositionCharts(metrics) {
    // Collect every position that appears in any model's by_position — union so
    // charts render the full set even if one model is missing a row.
    const positionsSet = new Set();
    const modelSeries = [
        { key: "Ridge Regression", label: "Ridge", color: COLORS.ridge, bg: COLORS.ridgeBg },
        { key: "Neural Network", label: "Neural Net", color: COLORS.nn, bg: COLORS.nnBg },
        { key: "Attention NN", label: "Attention NN", color: COLORS.attn_nn, bg: COLORS.attn_nnBg },
        { key: "LightGBM", label: "LightGBM", color: COLORS.lgbm, bg: COLORS.lgbmBg },
    ];
    for (const { key } of modelSeries) {
        const m = metrics[key];
        if (!m || !m.by_position) continue;
        m.by_position.forEach(p => positionsSet.add(p.position));
    }
    const positions = ["QB", "RB", "WR", "TE", "K", "DST"].filter(p => positionsSet.has(p));

    const buildDataset = (metricName) => modelSeries
        .map(({ key, label, color, bg }) => {
            const m = metrics[key];
            if (!m || !m.by_position || m.by_position.length === 0) return null;
            const byPos = Object.fromEntries(m.by_position.map(p => [p.position, p]));
            // null entries let Chart.js leave gaps where this model has no
            // prediction for that position (e.g. LightGBM for K/DST).
            const data = positions.map(p => byPos[p] != null ? byPos[p][metricName] : null);
            return { label, data, backgroundColor: bg, borderColor: color, borderWidth: 1.5 };
        })
        .filter(Boolean);

    const maeDatasets = buildDataset("mae");
    const r2Datasets = buildDataset("r2");
    if (positionMaeChart) positionMaeChart.destroy();
    positionMaeChart = new Chart(document.getElementById("position-mae-chart"), {
        type: "bar",
        data: { labels: positions, datasets: maeDatasets },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: "MAE by Position (Lower is Better)", color: "#e8eaed" } },
            scales: { y: { beginAtZero: true, grid: { color: "#2e3347" } }, x: { grid: { display: false } } },
        },
    });

    if (positionR2Chart) positionR2Chart.destroy();
    positionR2Chart = new Chart(document.getElementById("position-r2-chart"), {
        type: "bar",
        data: { labels: positions, datasets: r2Datasets },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: "R\u00B2 by Position (Higher is Better)", color: "#e8eaed" } },
            scales: { y: { beginAtZero: true, grid: { color: "#2e3347" } }, x: { grid: { display: false } } },
        },
    });
}

function renderWeeklyChart(weekly) {
    if (weeklyMaeChart) weeklyMaeChart.destroy();
    const series = [
        { label: "Ridge MAE", data: weekly.ridge_mae, color: COLORS.ridge, bg: COLORS.ridgeBg },
        { label: "Neural Net MAE", data: weekly.nn_mae, color: COLORS.nn, bg: COLORS.nnBg },
        { label: "Attention NN MAE", data: weekly.attn_nn_mae, color: COLORS.attn_nn, bg: COLORS.attn_nnBg },
        { label: "LightGBM MAE", data: weekly.lgbm_mae, color: COLORS.lgbm, bg: COLORS.lgbmBg },
    ];
    const datasets = series
        .filter(s => Array.isArray(s.data) && s.data.some(v => v != null))
        .map(s => ({
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
    weeklyMaeChart = new Chart(document.getElementById("weekly-mae-chart"), {
        type: "line",
        data: { labels: weekly.weeks.map(w => `Wk ${w}`), datasets },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: "Weekly MAE Across Test Season (Lower is Better)", color: "#e8eaed" },
            },
            scales: {
                y: { beginAtZero: true, grid: { color: "#2e3347" }, title: { display: true, text: "MAE", color: "#9aa0b0" } },
                x: { grid: { color: "#2e3347" } },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Player Modal
// ---------------------------------------------------------------------------
function setupModal() {
    document.getElementById("modal-close").addEventListener("click", closeModal);
    document.getElementById("player-modal").addEventListener("click", e => {
        if (e.target === e.currentTarget) closeModal();
    });
    document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
}

async function openPlayerModal(playerId) {
    currentPlayerId = playerId;
    try {
        const data = await fetchJSON(
            `/api/player/${encodeURIComponent(playerId)}?scoring=${currentScoring}`,
        );

        document.getElementById("modal-name").textContent = data.name;
        document.getElementById("modal-pos-team").textContent = `${data.position} - ${data.team}`;
        document.getElementById("modal-avg").textContent = fmt(data.season_avg);
        document.getElementById("modal-total").textContent = fmt(data.season_total);

        const img = document.getElementById("modal-headshot");
        if (data.headshot) {
            img.src = data.headshot;
            img.alt = data.name;
            img.style.display = "block";
        } else {
            img.removeAttribute("src");
            img.alt = "";
            img.style.display = "none";
        }

        // Chart — Actual plus up to 4 predictions (null entries where a model isn't available)
        const weeks = data.weekly.map(w => `Wk ${w.week}`);
        const actual = data.weekly.map(w => w.actual);
        const predSeries = [
            { label: "Ridge Pred", key: "ridge_pred", color: COLORS.ridge },
            { label: "NN Pred", key: "nn_pred", color: COLORS.nn },
            { label: "Attn NN Pred", key: "attn_nn_pred", color: COLORS.attn_nn },
            { label: "LGBM Pred", key: "lgbm_pred", color: COLORS.lgbm },
        ];
        const chartDatasets = [
            { label: "Actual", data: actual, borderColor: COLORS.actual, borderWidth: 2.5, tension: 0.3, pointRadius: 5, pointHoverRadius: 7 },
        ];
        for (const { label, key, color } of predSeries) {
            const series = data.weekly.map(w => w[key] != null ? w[key] : null);
            if (series.some(v => v != null)) {
                chartDatasets.push({
                    label,
                    data: series,
                    borderColor: color,
                    borderWidth: 2,
                    borderDash: [6, 3],
                    tension: 0.3,
                    pointRadius: 4,
                    spanGaps: true,
                });
            }
        }

        if (playerChart) playerChart.destroy();
        playerChart = new Chart(document.getElementById("player-chart"), {
            type: "line",
            data: { labels: weeks, datasets: chartDatasets },
            options: {
                responsive: true,
                // The modal chart lives in a fixed-height container (.modal .chart-box);
                // let the canvas fill it instead of Chart.js's default 2:1 aspect ratio,
                // which renders too short/squished in the modal (and worse on mobile).
                maintainAspectRatio: false,
                plugins: { title: { display: true, text: "Weekly Fantasy Points: Actual vs Predicted", color: "#e8eaed" } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "#2e3347" }, title: { display: true, text: "Fantasy Points", color: "#9aa0b0" } },
                    x: { grid: { color: "#2e3347" } },
                },
            },
        });

        document.getElementById("player-modal").classList.add("open");
        modalOpen = true;
    } catch (e) {
        console.error("Failed to load player:", e);
        document.getElementById("modal-name").textContent = "Error loading player";
        document.getElementById("modal-pos-team").textContent = "";
        document.getElementById("player-modal").classList.add("open");
        modalOpen = true;
    }
}

function closeModal() {
    document.getElementById("player-modal").classList.remove("open");
    modalOpen = false;
    currentPlayerId = null;
}

// ---------------------------------------------------------------------------
// Model Architecture
// ---------------------------------------------------------------------------
let modelArchitectureLoaded = false;

const ARCH_CATEGORY_LABELS = {
    specific: "Position-specific",
    rolling: "Rolling windows (L3 / L5 / L8)",
    prior_season: "Prior season",
    ewma: "EWMA",
    trend: "Trend",
    share: "Share / HHI",
    matchup: "Matchup vs opponent",
    defense: "Opponent defense",
    contextual: "Contextual",
    weather_vegas: "Weather / Vegas",
    attention_history: "Attention history (per-game inputs)",
    other: "Other",
};

const ARCH_POSITION_ORDER = ["QB", "RB", "WR", "TE", "K", "DST"];

function fmtList(arr) {
    return (arr || []).join(", ") || "—";
}

function fmtLayers(arr) {
    return Array.isArray(arr) && arr.length ? `[${arr.join(", ")}]` : "—";
}

function fmtNum(v, digits = 4) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") {
        if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
        return Number(v.toFixed(digits)).toString();
    }
    return String(v);
}

function renderArchConfigTable(positions) {
    const cols = [
        { key: "targets",            label: "Targets",    render: p => fmtList(p.targets) },
        { key: "backbone_layers",    label: "Backbone",   render: p => fmtLayers(p.backbone_layers) },
        { key: "head_hidden",        label: "Head",       render: p => fmtNum(p.head_hidden) },
        { key: "dropout",            label: "Dropout",    render: p => fmtNum(p.dropout) },
        { key: "lr",                 label: "LR",         render: p => fmtNum(p.lr) },
        { key: "weight_decay",       label: "WD",         render: p => fmtNum(p.weight_decay) },
        { key: "batch_size",         label: "Batch",      render: p => fmtNum(p.batch_size) },
        { key: "epochs",             label: "Epochs",     render: p => fmtNum(p.epochs) },
        { key: "patience",           label: "Patience",   render: p => fmtNum(p.patience) },
        { key: "scheduler",          label: "Scheduler",  render: p => p.scheduler || "—" },
        { key: "attention_enabled",  label: "Attn",       render: p => p.attention_enabled ? "✓" : "—" },
        { key: "lightgbm_enabled",   label: "LGBM",       render: p => p.lightgbm_enabled ? "✓" : "—" },
        { key: "feature_count",      label: "# Features", render: p => fmtNum(p.feature_count) },
    ];

    const head = `<tr><th>Position</th>${cols.map(c => `<th>${c.label}</th>`).join("")}</tr>`;
    const rows = ARCH_POSITION_ORDER.map(pos => {
        const p = positions[pos];
        if (!p) return "";
        const cells = cols.map(c => `<td>${c.render(p)}</td>`).join("");
        return `<tr><td class="arch-pos-cell">${pos}</td>${cells}</tr>`;
    }).join("");

    return `<table class="arch-table"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

function renderArchFeatureAccordions(positions) {
    return ARCH_POSITION_ORDER.map(pos => {
        const p = positions[pos];
        if (!p) return "";
        const features = p.features || {};
        const overrides = p.head_hidden_overrides || {};
        const overrideStr = Object.keys(overrides).length
            ? Object.entries(overrides).map(([k, v]) => `${k}: ${v}`).join(", ")
            : null;

        const sections = Object.keys(ARCH_CATEGORY_LABELS)
            .filter(key => features[key] && features[key].length)
            .map(key => {
                const chips = features[key].map(f => `<span class="feature-chip">${f}</span>`).join("");
                return `<div class="feature-category">
                    <div class="feature-category-title">${ARCH_CATEGORY_LABELS[key]} <span class="feature-category-count">(${features[key].length})</span></div>
                    <div class="feature-chip-row">${chips}</div>
                </div>`;
            }).join("");

        const meta = [
            `<span><strong>Targets:</strong> ${fmtList(p.targets)}</span>`,
            `<span><strong>Huber δ (total):</strong> ${fmtNum(p.huber_delta_total)}</span>`,
            overrideStr ? `<span><strong>Head overrides:</strong> ${overrideStr}</span>` : "",
        ].filter(Boolean).join(" · ");

        return `<details class="arch-accordion">
            <summary>
                <span class="arch-pos-label">${pos}</span>
                <span class="arch-pos-count">${p.feature_count} features</span>
            </summary>
            <div class="arch-accordion-body">
                <div class="arch-accordion-meta">${meta}</div>
                ${sections}
            </div>
        </details>`;
    }).join("");
}

async function loadModelArchitecture() {
    if (modelArchitectureLoaded) return;
    const tableEl = document.getElementById("arch-config-table");
    const accEl = document.getElementById("arch-feature-accordions");
    try {
        const data = await fetchJSON("/api/model_architecture");
        if (data.error) throw new Error(data.error);
        tableEl.innerHTML = renderArchConfigTable(data.positions || {});
        accEl.innerHTML = renderArchFeatureAccordions(data.positions || {});
        modelArchitectureLoaded = true;
    } catch (e) {
        console.error("Failed to load model architecture:", e);
        tableEl.innerHTML = `<p class="arch-error">Failed to load: ${e.message}</p>`;
        accEl.innerHTML = "";
    }
}

// ---------------------------------------------------------------------------
// Comparison — our model (live) vs expert projection sources (NFL.com,
// RotoWire), by position, on two player subsets (all + top-30/position). One
// /api/comparison fetch; the MAE/RMSE/R² toggle re-renders from the cached
// payload. Lower is better for MAE/RMSE, higher for R².
// ---------------------------------------------------------------------------
let comparisonLoaded = false;
let comparisonToggleWired = false;
let comparisonData = null;
let comparisonMetric = "mae";

const COMPARISON_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"];
const COMPARISON_SOURCES = [
    { key: "model", label: "Our Model" },
    { key: "nflcom", label: "NFL.com" },
    { key: "rotowire", label: "RotoWire" },
];
const COMPARISON_METRIC_HINTS = {
    mae: "Mean absolute error — lower is better",
    rmse: "Root mean squared error — lower is better",
    r2: "R² (coefficient of determination) — higher is better",
};

async function loadComparison() {
    setupComparisonToggle();
    if (comparisonLoaded) return;
    const allBody = document.getElementById("comparison-all-body");
    const top30Body = document.getElementById("comparison-top30-body");
    try {
        comparisonData = await fetchJSON("/api/comparison");
        if (comparisonData.error) throw new Error(comparisonData.error);
        comparisonLoaded = true;
        renderComparisonTables();
        renderComparisonReliability();
        renderComparisonNotes();
        renderIntervals();
    } catch (e) {
        console.error("Failed to load comparison:", e);
        const msg = `<tr><td colspan="4" class="arch-error">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
        if (allBody) allBody.innerHTML = msg;
        if (top30Body) top30Body.innerHTML = msg;
    }
}

function setupComparisonToggle() {
    if (comparisonToggleWired) return;
    const toggle = document.getElementById("comparison-metric-toggle");
    if (!toggle) return;
    toggle.querySelectorAll(".pill").forEach(pill => {
        pill.addEventListener("click", () => {
            comparisonMetric = pill.dataset.metric;
            toggle.querySelectorAll(".pill").forEach(p => p.classList.toggle("active", p === pill));
            const hint = document.getElementById("comparison-metric-hint");
            if (hint) hint.textContent = COMPARISON_METRIC_HINTS[comparisonMetric] || "";
            if (comparisonData) renderComparisonTables();
        });
    });
    comparisonToggleWired = true;
}

function comparisonCellValue(cell) {
    if (!cell) return null;
    const v = cell[comparisonMetric];
    return v === null || v === undefined || Number.isNaN(v) ? null : v;
}

function formatComparisonValue(v) {
    return comparisonMetric === "r2" ? v.toFixed(3) : v.toFixed(2);
}

function renderComparisonRows(posMap) {
    const higherBetter = comparisonMetric === "r2";
    return COMPARISON_POSITIONS.map(pos => {
        const cells = posMap[pos] || {};
        const values = COMPARISON_SOURCES.map(s => comparisonCellValue(cells[s.key])).filter(
            v => v !== null
        );
        const best = values.length ? (higherBetter ? Math.max(...values) : Math.min(...values)) : null;

        const tds = COMPARISON_SOURCES.map(s => {
            const cell = cells[s.key];
            const v = comparisonCellValue(cell);
            if (v === null) return `<td class="comparison-num comparison-empty">—</td>`;
            const isBest = best !== null && Math.abs(v - best) < 1e-9;
            const cls = "comparison-num" + (isBest ? " comparison-best" : "");
            const arch =
                s.key === "model" && cell && cell.best_arch
                    ? `<span class="comparison-arch">${escapeHtml(cell.best_arch)}</span>`
                    : "";
            return `<td class="${cls}">${formatComparisonValue(v)}${arch}</td>`;
        }).join("");
        return `<tr><td class="comparison-pos">${pos}</td>${tds}</tr>`;
    }).join("");
}

function renderComparisonTables() {
    if (!comparisonData) return;
    const subsets = comparisonData.subsets || {};
    const allBody = document.getElementById("comparison-all-body");
    const top30Body = document.getElementById("comparison-top30-body");
    if (allBody) allBody.innerHTML = renderComparisonRows(subsets.all || {});
    if (top30Body) top30Body.innerHTML = renderComparisonRows(subsets.top30 || {});
}

// Source reliability — residual σ per source on the 2025 test season. The model
// side is live (auto-updates on retrain); experts are scored on the same season,
// with their full 2018–2025 archive σ shown on hover. One table, always σ
// (independent of the MAE/RMSE/R² toggle); lower σ = steadier.
const RELIABILITY_COLS = [
    { key: "model", label: "Our Model" },
    { key: "nflcom", label: "NFL.com" },
    { key: "rotowire", label: "RotoWire" },
];

function _validSigma(v) {
    return v !== null && v !== undefined && !Number.isNaN(v);
}

// Resolve one (position, source) reliability cell on the 2025 basis. The model
// comes from the live model_reliability map; experts come from the committed
// block's per_season["2025"] slice, carrying the pooled multi-season σ for hover.
function reliabilityCell(pos, key) {
    if (key === "model") {
        const m = comparisonData.model_reliability && comparisonData.model_reliability[pos];
        if (!m || !_validSigma(m.sigma)) return null;
        return { sigma: m.sigma, bias: m.bias, n: m.n, arch: m.best_arch, kind: "model" };
    }
    const rel = comparisonData.expert_reliability;
    const pooled = rel && rel.positions && rel.positions[pos] && rel.positions[pos][key];
    if (!pooled) return null;
    const s = pooled.per_season && pooled.per_season["2025"];
    if (!s || !_validSigma(s.sigma)) return null;
    return {
        sigma: s.sigma,
        bias: s.bias,
        n: s.n,
        kind: "expert",
        totals_only: !!pooled.totals_only,
        archiveSigma: pooled.sigma,
        archiveN: pooled.n,
    };
}

function renderComparisonReliability() {
    const block = document.getElementById("comparison-reliability-block");
    const body = document.getElementById("comparison-reliability-body");
    if (!block || !body) return;
    const rel = comparisonData && comparisonData.expert_reliability;
    if (!rel || !rel.positions) {
        block.style.display = "none";
        return;
    }
    block.style.display = "";

    body.innerHTML = COMPARISON_POSITIONS.map(pos => {
        const cells = RELIABILITY_COLS.map(c => reliabilityCell(pos, c.key));
        const sigmas = cells.filter(Boolean).map(c => c.sigma);
        const best = sigmas.length ? Math.min(...sigmas) : null;
        const tds = cells.map(c => {
            if (!c) return `<td class="comparison-num comparison-empty">—</td>`;
            const isBest = best !== null && Math.abs(c.sigma - best) < 1e-9;
            const cls = "comparison-num" + (isBest ? " comparison-best" : "");
            const star = c.totals_only ? `<span class="comparison-arch">totals-only*</span>` : "";
            const biasTxt = (c.bias >= 0 ? "+" : "") + c.bias.toFixed(2);
            const dir = c.bias >= 0 ? "over" : "under";
            const verb = c.kind === "model" ? "predicts" : "projects";
            let title = `bias ${biasTxt} pts (${dir}-${verb}) · n=${c.n} · 2025`;
            if (c.kind === "model" && c.arch) title += ` · ${c.arch}`;
            if (c.kind === "expert" && _validSigma(c.archiveSigma)) {
                title += ` · 2018–2025 σ ${c.archiveSigma.toFixed(2)} (n=${c.archiveN})`;
            }
            return `<td class="${cls}" title="${escapeHtml(title)}">${c.sigma.toFixed(2)}${star}</td>`;
        }).join("");
        return `<tr><td class="comparison-pos">${pos}</td>${tds}</tr>`;
    }).join("");

    const noteEl = document.getElementById("comparison-reliability-note");
    if (noteEl) noteEl.innerHTML = rel.note ? `<p>${escapeHtml(rel.note)}</p>` : "";
}

function renderComparisonNotes() {
    const el = document.getElementById("comparison-notes");
    if (!el || !comparisonData) return;
    const meta = comparisonData.experts_meta || {};
    const date = (comparisonData.generated_at || "").slice(0, 10);
    const unavailable = comparisonData.model_source === "unavailable";
    const nflNote = (meta.nflcom && meta.nflcom.note) || "";
    const rwNote = (meta.rotowire && meta.rotowire.note) || "";
    const modelLine = unavailable
        ? "Currently unavailable (models not loaded). "
        : "Computed live from the deployed models (best architecture per position), so it tracks the latest retrain. ";
    el.innerHTML = `
        <div class="section-header">About this comparison</div>
        <ul class="comparison-note-list">
            <li><strong>Seasons.</strong> Our model trains on 2012–2023, validates on 2024, and is tested on <strong>2025</strong>; every number here is on the held-out 2025 season, and the experts are scored on 2025 too.</li>
            <li><strong>Scoring.</strong> Full PPR (1 pt / reception). Projections and actuals run through the same scoring formula, so it's apples-to-apples. RMSE is shown alongside MAE because expert projections implicitly target squared error.</li>
            <li><strong>Our Model.</strong> ${modelLine}MAE/RMSE/R² are on weekly fantasy-point totals.</li>
            <li><strong>NFL.com.</strong> ${escapeHtml(nflNote)}</li>
            <li><strong>RotoWire.</strong> ${escapeHtml(rwNote)}</li>
            <li><strong>Top 30.</strong> The second table restricts to the top 30 players per position by actual 2025 fantasy points — the fantasy-relevant starters.</li>
            <li><strong>Caveat.</strong> Each source is scored on the players it actually projects, so this is an approximate scoreboard rather than a strictly paired test. For the rigorous paired, significance-tested head-to-heads, see the <a href="#wiki:expert-comparison" class="comparison-link" data-slug="expert-comparison">Expert Projection Comparison</a> wiki page.</li>
            ${date ? `<li class="comparison-note-meta">Expert data generated ${escapeHtml(date)}.</li>` : ""}
        </ul>`;
    el.querySelectorAll(".comparison-link").forEach(a => {
        a.addEventListener("click", ev => {
            ev.preventDefault();
            const hash = `#wiki:${a.dataset.slug}`;
            if ((location.hash || "") !== hash) {
                history.pushState(null, "", location.pathname + location.search + hash);
            }
            activateTab("wiki");
        });
    });
}

// ---------------------------------------------------------------------------
// Prediction intervals — per-projection 80% floor–ceiling bands for the expert
// sources. Fit offline (src/analysis/expert_intervals.py) and committed; ride
// along on the /api/comparison payload. The calibration table reports held-out
// coverage (does the 80% band contain ~80% of actuals?); the example bands show
// real test-season player-weeks with floor/median/ceiling and where the actual
// landed. Optional block — hidden when the committed JSON is absent.
// ---------------------------------------------------------------------------
const INTERVAL_SOURCES = [
    { key: "nflcom", label: "NFL.com" },
    { key: "rotowire", label: "RotoWire" },
];
let intervalsPos = "QB";
let intervalsPosWired = false;

function intervalsBlockFor(intervals, source, pos) {
    const s = intervals && intervals.intervals && intervals.intervals[source];
    if (!s) return null;
    const b = s[pos];
    if (!b || b.skipped || !b.calibration) return null;
    return b;
}

function renderIntervals() {
    const block = document.getElementById("comparison-intervals-block");
    if (!block) return;
    const intervals = comparisonData && comparisonData.intervals;
    if (!intervals || !intervals.intervals) {
        block.style.display = "none";
        return;
    }
    block.style.display = "";

    const nominalPct = Math.round((intervals.nominal_coverage || 0.8) * 100);
    const intro = document.getElementById("intervals-intro");
    if (intro) {
        intro.innerHTML = `Each expert gives a single number per player-week. We quantile-regress the
            actual fantasy points on that projection (per position) to attach an <strong>${nominalPct}%
            band</strong> — a floor (10th percentile) and ceiling (90th percentile) — around every
            projection. A well-calibrated band contains the actual outcome about ${nominalPct}% of the time.`;
    }
    renderIntervalsCalibration(intervals);
    setupIntervalsPosToggle(intervals);
    renderIntervalsNotes(intervals);
}

function renderIntervalsCalibration(intervals) {
    const body = document.getElementById("intervals-calibration-body");
    if (!body) return;
    body.innerHTML = COMPARISON_POSITIONS.map(pos => {
        const tds = INTERVAL_SOURCES.map(s => {
            const b = intervalsBlockFor(intervals, s.key, pos);
            if (!b) return `<td class="comparison-num comparison-empty">—</td>`;
            const cal = b.calibration;
            const covPct = (cal.coverage * 100).toFixed(0);
            const flag = cal.flag || "ok";
            const std = b.totals_only ? ` <span class="interval-totals" title="standard scoring">std</span>` : "";
            const fit = (b.fit_seasons || []).join(", ");
            const tip = fit ? ` title="fit on ${escapeHtml(fit)}"` : "";
            return `<td class="comparison-num"${tip}>
                <span class="interval-cov interval-cov-${flag}">${covPct}%</span>${std}
                <span class="interval-cov-sub">n=${cal.n_eval} · band ${cal.mean_width.toFixed(1)}</span></td>`;
        }).join("");
        return `<tr><td class="comparison-pos">${pos}</td>${tds}</tr>`;
    }).join("");
}

function setupIntervalsPosToggle(intervals) {
    const toggle = document.getElementById("intervals-pos-toggle");
    if (!toggle) return;
    // Default to the first position that actually has example bands.
    const hasEx = pos => INTERVAL_SOURCES.some(s => {
        const b = intervalsBlockFor(intervals, s.key, pos);
        return b && b.examples && b.examples.length;
    });
    if (!hasEx(intervalsPos)) intervalsPos = COMPARISON_POSITIONS.find(hasEx) || intervalsPos;
    toggle.innerHTML = COMPARISON_POSITIONS.map(
        pos => `<button class="pill${pos === intervalsPos ? " active" : ""}" data-pos="${pos}">${pos}</button>`
    ).join("");
    if (!intervalsPosWired) {
        toggle.addEventListener("click", ev => {
            const pill = ev.target.closest(".pill");
            if (!pill) return;
            intervalsPos = pill.dataset.pos;
            toggle.querySelectorAll(".pill").forEach(p => p.classList.toggle("active", p === pill));
            renderIntervalsExamples(comparisonData.intervals, intervalsPos);
        });
        intervalsPosWired = true;
    }
    renderIntervalsExamples(intervals, intervalsPos);
}

function renderIntervalsExamples(intervals, pos) {
    const host = document.getElementById("intervals-examples");
    if (!host) return;
    const cols = INTERVAL_SOURCES.map(s => {
        const b = intervalsBlockFor(intervals, s.key, pos);
        const examples = (b && b.examples) || [];
        const head = `<div class="intervals-col-head">${s.label}</div>`;
        if (!examples.length) {
            return `<div class="intervals-col">${head}<div class="intervals-empty">No ${escapeHtml(pos)} projections.</div></div>`;
        }
        return `<div class="intervals-col">${head}${examples.map(renderBandBar).join("")}</div>`;
    }).join("");
    host.innerHTML = `<div class="intervals-cols">${cols}</div>`;
}

function renderBandBar(ex) {
    // Per-row scale spans the band AND the actual, so the actual marker is always
    // visible even when it lands outside the band.
    const lo = Math.min(ex.floor, ex.actual);
    const hi = Math.max(ex.ceiling, ex.actual);
    const span = hi - lo || 1;
    const pct = x => ((x - lo) / span) * 100;
    const fillL = pct(ex.floor);
    const fillW = pct(ex.ceiling) - fillL;
    const cls = ex.in_band ? "in" : "out";
    const mark = ex.in_band ? "✓" : "✗";
    return `<div class="band-row">
        <div class="band-head">
            <span class="band-player">${escapeHtml(ex.player_name)}</span>
            <span class="band-week">Wk ${ex.week}</span>
            <span class="band-proj">proj ${ex.projection.toFixed(1)}</span>
        </div>
        <div class="band-track">
            <div class="band-fill" style="left:${fillL}%;width:${fillW}%"></div>
            <div class="band-median" style="left:${pct(ex.median)}%" title="median ${ex.median.toFixed(1)}"></div>
            <div class="band-actual band-actual-${cls}" style="left:${pct(ex.actual)}%"></div>
        </div>
        <div class="band-foot">
            <span class="band-end">${ex.floor.toFixed(1)}</span>
            <span class="band-actual-label band-actual-${cls}">actual ${ex.actual.toFixed(1)} ${mark}</span>
            <span class="band-end">${ex.ceiling.toFixed(1)}</span>
        </div>
    </div>`;
}

function renderIntervalsNotes(intervals) {
    const el = document.getElementById("intervals-notes");
    if (!el) return;
    const meta = intervals.sources_meta || {};
    const evalSeasons = (intervals.eval_seasons || []).join(", ");
    const rwUnverified = meta.rotowire && meta.rotowire.provenance_unverified;
    const nflLeak = (meta.nflcom && meta.nflcom.look_ahead_seasons) || [];
    el.innerHTML = `
        <ul class="comparison-note-list">
            <li><strong>Method.</strong> ${escapeHtml(intervals.method || "")}. Fit on genuine pre-${escapeHtml(evalSeasons)} player-weeks (which seasons varies by position — hover a coverage cell); the coverage above is measured on the held-out ${escapeHtml(evalSeasons)} season, so it reflects the bands as shipped.</li>
            <li><strong>Reading coverage.</strong> Near 80% is well-calibrated. Well below means the band is too tight (over-confident); well above means it is too wide.</li>
            ${nflLeak.length ? `<li><strong>Look-ahead guard.</strong> Some NFL.com "projected" files (${escapeHtml(nflLeak.join(", "))}) are backfilled with realized stats — implausibly accurate, so auto-excluded from the fit. NFL.com offense bands therefore fit on the recent genuine season(s).</li>` : ""}
            <li><strong>NFL.com K.</strong> Totals-only ("std") — the band is on the standard-scoring kicker scale, not PPR.</li>
            ${rwUnverified ? `<li><strong>RotoWire caveat.</strong> Provenance is unverified, but its error spread is stable across every season and matches the held-out season, so its bands sanity-check clean (no look-ahead detected).</li>` : ""}
            <li><strong>Coverage holes.</strong> NFL.com has no DST; RotoWire has no K.</li>
        </ul>`;
}

// ---------------------------------------------------------------------------
// Wiki — render committed markdown docs in the Wiki tab. The sidebar is
// fetched once from /api/wiki/index; doc bodies are fetched lazily and
// cached server-side. Intra-wiki links arrive as `#wiki:slug[:anchor]` and
// are intercepted to swap content without a page reload.
// ---------------------------------------------------------------------------
const WIKI_DEFAULT_SLUG = "architecture";
let wikiIndexLoaded = false;
let wikiCurrentSlug = null;
const wikiPageCache = new Map();

function parseWikiHash(hash) {
    if (!hash || !hash.startsWith("#wiki:")) return null;
    const rest = hash.slice("#wiki:".length);
    const [slug, ...anchorParts] = rest.split(":");
    return { slug, anchor: anchorParts.join(":") || null };
}

function renderWikiSidebar(items) {
    const groups = new Map();
    items.forEach(item => {
        if (!groups.has(item.group)) groups.set(item.group, []);
        groups.get(item.group).push(item);
    });
    const html = Array.from(groups.entries()).map(([group, entries]) => {
        const links = entries.map(item =>
            `<li><a href="#wiki:${escapeHtml(item.slug)}" class="wiki-sidebar-link" data-slug="${escapeHtml(item.slug)}">${escapeHtml(item.name)}</a></li>`
        ).join("");
        return `<div class="wiki-sidebar-group">
            <h3 class="wiki-sidebar-heading">${escapeHtml(group)}</h3>
            <ul class="wiki-sidebar-list">${links}</ul>
        </div>`;
    }).join("");
    document.getElementById("wiki-sidebar").innerHTML = html;
}

function setWikiActiveSidebar(slug) {
    document.querySelectorAll(".wiki-sidebar-link").forEach(a => {
        a.classList.toggle("active", a.dataset.slug === slug);
    });
}

async function loadWikiPage(slug, anchor = null) {
    const contentEl = document.getElementById("wiki-content");
    const cached = wikiPageCache.get(slug);
    if (cached) {
        contentEl.innerHTML = cached;
    } else {
        contentEl.innerHTML = `<p class="arch-loading">Loading…</p>`;
        try {
            const data = await fetchJSON(`/api/wiki/${encodeURIComponent(slug)}`);
            if (data.error) throw new Error(data.error);
            wikiPageCache.set(slug, data.html);
            contentEl.innerHTML = data.html;
        } catch (e) {
            console.error("Failed to load wiki page:", e);
            contentEl.innerHTML = `<p class="arch-error">Failed to load: ${escapeHtml(e.message)}</p>`;
            return;
        }
    }
    wikiCurrentSlug = slug;
    setWikiActiveSidebar(slug);
    const newHash = anchor ? `#wiki:${slug}:${anchor}` : `#wiki:${slug}`;
    if (location.hash !== newHash) {
        history.replaceState(null, "", newHash);
    }
    if (anchor) {
        const target = document.getElementById(anchor);
        if (target) target.scrollIntoView({ behavior: "auto", block: "start" });
        else contentEl.scrollTop = 0;
    } else {
        contentEl.scrollTop = 0;
    }
}

// ---------------------------------------------------------------------------
// Benchmark History
//
// Reads /api/benchmark_history — one row per training run, newest first. Each
// MAE cell renders a list of per-position pills since a run may only retrain
// a subset of positions (CI's `detect` job scopes by changed paths). PR
// numbers come from a top-level field that CI writes when it can resolve the
// merge commit to a PR; for runs where the lookup returned empty (manual
// dispatches, force pushes) we fall back to a commit-SHA link.
// ---------------------------------------------------------------------------
// Layout constants for the History table. Mirror the backend's
// _BENCHMARK_MODELS / _BENCHMARK_POSITIONS ordering so a row's per-model pill
// arrays line up by index. historyData caches the last fetch so the two
// checkboxes (detailed mode, group-by-model) re-render without re-fetching.
const HISTORY_MODELS = ["ridge", "nn", "attn_nn", "lgbm"];
const HISTORY_MODEL_LABELS = { ridge: "Ridge", nn: "NN", attn_nn: "Attn NN", lgbm: "LGBM" };
const HISTORY_MODEL_COL_CLASS = { ridge: "ridge-col", nn: "nn-col", attn_nn: "attn-nn-col", lgbm: "lgbm-col" };
const HISTORY_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"];
let historyData = null;

function formatTrainingTime(seconds) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "--";
    const total = Math.round(seconds);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
}

function renderSummaryPills(entries) {
    // Generic pill list: each entry is {label, mae}. label is a position
    // (group-by-model layout) or a model name (group-by-position). mae=null
    // renders as "--" (that position-model pair didn't train in this run);
    // empty list renders an em-dash. The optional per_target field carried on
    // pills is ignored here — it only drives the detailed-mode expansion.
    if (!Array.isArray(entries) || entries.length === 0) return '<span class="history-empty">—</span>';
    return entries
        .map(e => {
            const value = e.mae == null
                ? '<span class="history-pill-skip">--</span>'
                : fmt(e.mae, 2);
            return `<span class="history-pill"><span class="history-pill-pos">${escapeHtml(e.label)}</span> ${value}</span>`;
        })
        .join("");
}

function historyColumns(groupByPosition) {
    // Group-by-position (default): one column per position. These carry no
    // .{model}-col class, so the page-wide #model-display hide rule is
    // intentionally inert here (model becomes an inner dimension). Group-by-model:
    // one column per model, keeping .{model}-col so #model-display still filters.
    if (groupByPosition) {
        return HISTORY_POSITIONS.map(pos => ({ key: pos, label: pos, cls: "col-history-mae" }));
    }
    return HISTORY_MODELS.map(m => ({
        key: m,
        label: `${HISTORY_MODEL_LABELS[m]} MAE`,
        cls: `col-history-mae ${HISTORY_MODEL_COL_CLASS[m]}`,
    }));
}

function historyCellEntries(row, columnKey, groupByPosition) {
    if (groupByPosition) {
        // Column is a position; inner entries are the four models at that position.
        const posIdx = HISTORY_POSITIONS.indexOf(columnKey);
        return HISTORY_MODELS.map(m => ({
            label: HISTORY_MODEL_LABELS[m],
            mae: row[m] && row[m][posIdx] ? row[m][posIdx].mae : null,
        }));
    }
    // Column is a model; inner entries are the six positions in canonical order.
    return (row[columnKey] || []).map(p => ({ label: p.position, mae: p.mae }));
}

function historyRowHasDetail(row) {
    // A run is expandable only when some (position, model) cell carries
    // per-target detail — skipped/sentinel runs and old totals-only runs aren't.
    return HISTORY_MODELS.some(m => (row[m] || []).some(p => p && p.per_target));
}

function renderHistoryIdCell(repoSlug, row) {
    // If the API didn't return a slug (test scenario, misconfigured env),
    // skip the link and render the identifier as plain text rather than
    // emitting a broken https:///pull/N URL that 404s on click.
    const slug = (repoSlug || "").trim();
    if (row.pr_number != null) {
        if (!slug) return `<span class="history-link-disabled">#${row.pr_number}</span>`;
        const href = `https://github.com/${slug}/pull/${row.pr_number}`;
        return `<a class="history-link" href="${href}" target="_blank" rel="noopener">#${row.pr_number}</a>`;
    }
    if (row.git_hash) {
        if (!slug) return `<span class="history-link-disabled"><code>${escapeHtml(row.git_hash)}</code></span>`;
        const href = `https://github.com/${slug}/commit/${encodeURIComponent(row.git_hash)}`;
        return `<a class="history-link" href="${href}" target="_blank" rel="noopener"><code>${escapeHtml(row.git_hash)}</code></a>`;
    }
    return "—";
}

function formatHistoryTimestamp(ts) {
    if (!ts) return "--";
    // Stored as "2026-05-19T22:47:20" (no tz marker, always UTC per
    // utc_now_iso). Replace the T with a space and trim seconds for a
    // compact display; keep minute resolution so same-PR reruns are
    // distinguishable.
    const match = String(ts).match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
    return match ? `${match[1]} ${match[2]}` : escapeHtml(String(ts));
}

async function loadHistory() {
    const tbody = document.getElementById("history-body");
    try {
        const data = await fetchJSON("/api/benchmark_history");
        historyData = {
            rows: data.rows || [],
            repoSlug: data.repo_slug || "",
            targetLabels: data.target_labels || {},
            targetUnits: data.target_units || {},
        };
        renderHistory();
    } catch (e) {
        console.error("Failed to load benchmark history:", e);
        historyData = null;
        const head = document.getElementById("history-head");
        if (head) head.innerHTML = "";
        // colspan 9 = the widest layout (group-by-position); harmless when fewer.
        tbody.innerHTML = '<tr><td colspan="9" class="error-message">Failed to load benchmark history.</td></tr>';
    }
}

function renderHistory() {
    // Re-render from cached historyData using the live checkbox state. Called by
    // loadHistory (after fetch) and on every checkbox change — never re-fetches.
    // Rebuilds both <thead> and <tbody> since the column set differs between
    // group-by-model (4 cols) and group-by-position (6 cols).
    const head = document.getElementById("history-head");
    const tbody = document.getElementById("history-body");
    if (!historyData || !head || !tbody) return;
    // Checkbox is "Group by model"; default (unchecked) groups by position.
    const groupByPosition = !document.getElementById("history-group-by-model-toggle").checked;
    const detailed = document.getElementById("history-detailed-toggle").checked;
    const columns = historyColumns(groupByPosition);
    const colSpan = columns.length + 3; // PR + Timestamp + variable cols + Training time

    head.innerHTML = `<tr>
        <th class="col-history-pr">PR</th>
        <th class="col-history-ts">Timestamp (UTC)</th>
        ${columns.map(c => `<th class="${c.cls}">${escapeHtml(c.label)}</th>`).join("")}
        <th class="col-history-time">Training time</th>
    </tr>`;

    const { repoSlug } = historyData;
    // Hide commits that didn't retrain (training-skipped sentinels): they carry
    // no MAE data and only add noise. training_skipped is set per row by the
    // backend (explicit sentinel flag, or empty results).
    const rows = historyData.rows.filter(row => !row.training_skipped);
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="${colSpan}" class="arch-loading">No benchmark runs yet.</td></tr>`;
        return;
    }

    tbody.innerHTML = rows
        .map(row => {
            const expandable = detailed && historyRowHasDetail(row);
            const cells = columns
                .map(c => `<td class="${c.cls}">${renderSummaryPills(historyCellEntries(row, c.key, groupByPosition))}</td>`)
                .join("");
            const caret = expandable ? '<span class="history-caret">▸</span>' : "";
            const mainRow = `<tr${expandable ? ' class="history-row-expandable"' : ""}>
                <td class="col-history-pr">${caret}${renderHistoryIdCell(repoSlug, row)}</td>
                <td class="col-history-ts">${formatHistoryTimestamp(row.timestamp)}</td>
                ${cells}
                <td class="col-history-time">${formatTrainingTime(row.total_elapsed_sec)}</td>
            </tr>`;
            return mainRow + (expandable ? renderHistoryDetail(row, colSpan) : "");
        })
        .join("");
}

function renderHistoryDetail(row, colSpan) {
    // One block per trained position: a target(rows) x model(cols) table that
    // mirrors the Model Performance tab's renderPositionModelDetail, reusing
    // formatTargetMae for units + fantasy-point equivalents. Orientation is
    // fixed (per-position blocks) regardless of the group-by-position toggle —
    // targets are position-specific, so model-as-column is the only clean layout.
    const { targetLabels, targetUnits } = historyData;
    const blocks = HISTORY_POSITIONS.map((pos, posIdx) => {
        // Target set/order comes from the first model that has detail for this
        // position (every model shares a position's targets).
        let targets = null;
        for (const m of HISTORY_MODELS) {
            const pt = row[m] && row[m][posIdx] && row[m][posIdx].per_target;
            if (pt) { targets = Object.keys(pt); break; }
        }
        if (!targets || !targets.length) return "";
        const trows = targets
            .map(tkey => {
                const label = targetLabels[tkey] || tkey;
                const unit = targetUnits[tkey];
                const cells = HISTORY_MODELS.map(m => {
                    const pt = row[m] && row[m][posIdx] && row[m][posIdx].per_target;
                    const val = pt ? pt[tkey] : null;
                    return `<td class="tm-val">${escapeHtml(formatTargetMae(val, tkey, unit, currentScoring))}</td>`;
                }).join("");
                return `<tr><td class="tm-name">${escapeHtml(label)}</td>${cells}</tr>`;
            })
            .join("");
        return `
            <div class="history-detail-block">
                <div class="history-detail-pos"><span class="pos-badge pos-${escapeHtml(pos)}">${escapeHtml(pos)}</span></div>
                <div class="table-container">
                    <table class="pos-model-table">
                        <thead><tr><th>Target</th><th>Ridge</th><th>NN</th><th>Attn NN</th><th>LGBM</th></tr></thead>
                        <tbody>${trows}</tbody>
                    </table>
                </div>
            </div>`;
    }).join("");
    return `<tr class="history-detail-row" hidden><td colspan="${colSpan}">${blocks}</td></tr>`;
}

function setupHistoryControls() {
    ["history-detailed-toggle", "history-group-by-model-toggle"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", renderHistory);
    });
    // Delegated expand/collapse: click a detailed-mode row to toggle its detail
    // row (the immediate next sibling). Clicks on the PR/commit link still work.
    const tbody = document.getElementById("history-body");
    if (tbody) {
        tbody.addEventListener("click", e => {
            if (e.target.closest("a")) return;
            const row = e.target.closest("tr.history-row-expandable");
            if (!row) return;
            const detail = row.nextElementSibling;
            if (detail && detail.classList.contains("history-detail-row")) {
                detail.hidden = !detail.hidden;
                row.classList.toggle("expanded", !detail.hidden);
            }
        });
    }
}

async function loadWiki() {
    if (!wikiIndexLoaded) {
        try {
            const items = await fetchJSON("/api/wiki/index");
            renderWikiSidebar(items);
            wikiIndexLoaded = true;
        } catch (e) {
            console.error("Failed to load wiki index:", e);
            document.getElementById("wiki-sidebar").innerHTML =
                `<p class="arch-error">Failed to load index: ${escapeHtml(e.message)}</p>`;
            return;
        }
    }
    const fromHash = parseWikiHash(location.hash);
    const slug = (fromHash && fromHash.slug) || wikiCurrentSlug || WIKI_DEFAULT_SLUG;
    const anchor = fromHash ? fromHash.anchor : null;
    await loadWikiPage(slug, anchor);
}

function setupWikiClickHandler() {
    // One delegated listener on the wiki view catches both sidebar clicks and
    // intra-content `#wiki:` links produced by the server-side link rewriter.
    const view = document.getElementById("view-wiki");
    if (!view) return;
    view.addEventListener("click", e => {
        const a = e.target.closest("a");
        if (!a) return;
        const href = a.getAttribute("href") || "";
        if (!href.startsWith("#wiki:")) return;
        e.preventDefault();
        const parsed = parseWikiHash(href);
        if (parsed) loadWikiPage(parsed.slug, parsed.anchor);
    });
}


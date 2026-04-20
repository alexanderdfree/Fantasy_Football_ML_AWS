/* Fantasy Football Predictor - Frontend */

const PAGE_SIZE = 50;
let allPlayers = [];
let currentPage = 1;
let currentSort = "actual";
let currentOrder = "desc";
let playerChart = null;
let positionMaeChart = null;
let positionR2Chart = null;
let weeklyMaeChart = null;
let positionDetailsData = null;
let perfFilterInitialized = false;

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

    // Load weeks dropdown
    try {
        const weeksData = await fetchJSON("/api/weeks");
        const weekSelect = document.getElementById("week-filter");
        weeksData.weeks.forEach(w => {
            const opt = document.createElement("option");
            opt.value = w;
            opt.textContent = `Week ${w}`;
            weekSelect.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load weeks:", e);
    }

    // Attach filter change listeners
    document.getElementById("week-filter").addEventListener("change", () => { currentPage = 1; loadPredictions(); });

    // Initial data load
    try {
        await loadPredictions();
    } catch (e) {
        console.error("Failed to load predictions:", e);
    } finally {
        document.getElementById("loading-overlay").classList.add("hidden");
    }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function setupNavTabs() {
    document.querySelectorAll(".nav-tabs .tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".nav-tabs .tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const view = tab.dataset.view;
            document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
            document.getElementById(`view-${view}`).classList.add("active");

            if (view === "model-performance") loadMetrics();
            if (view === "standings") loadStandings();
            if (view === "model-architecture") loadModelArchitecture();
        });
    });
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
    const position = getActivePosition("position-filter");
    const week = document.getElementById("week-filter").value;
    const search = document.getElementById("search-input").value;

    const params = new URLSearchParams({
        position, week, search,
        sort: currentSort,
        order: currentOrder,
    });

    const container = document.querySelector("#view-predictions .table-container");
    container.classList.add("loading");
    try {
        const data = await fetchJSON(`/api/predictions?${params}`);
        allPlayers = data.players || [];
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

function renderTable() {
    // Sort locally for instant re-sorting
    const sorted = [...allPlayers].sort((a, b) => {
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

        return `<tr data-player-id="${escapeHtml(p.player_id)}">
            <td class="col-rank">${rank}</td>
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
        </tr>`;
    }).join("");

    // Click rows to open modal
    tbody.querySelectorAll("tr").forEach(row => {
        row.addEventListener("click", () => openPlayerModal(row.dataset.playerId));
    });

    renderPagination(totalPages);
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
        const data = await fetchJSON(`/api/top_players?position=${position}`);

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
        const [metrics, weekly, posDetails] = await Promise.all([
            fetchJSON("/api/metrics"),
            fetchJSON("/api/weekly_accuracy"),
            fetchJSON("/api/position_details"),
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

    const maeCell = (v) => v != null ? v.toFixed(2) : '--';

    // Target decomposition rows — 4 MAE columns, "--" where a model isn't available (K/DST)
    const targetRows = (d.targets || []).map(t => {
        const m = tm[t.key] || {};
        return `<tr>
            <td class="tm-name">${t.label}</td>
            <td class="tm-formula">${t.formula}</td>
            <td class="tm-val">${maeCell(m.ridge_mae)}</td>
            <td class="tm-val">${maeCell(m.nn_mae)}</td>
            <td class="tm-val">${maeCell(m.attn_nn_mae)}</td>
            <td class="tm-val">${maeCell(m.lgbm_mae)}</td>
        </tr>`;
    }).join("");

    const totalM = tm["total"] || {};
    const totalCell = (v) => v != null ? `<strong>${v.toFixed(2)}</strong>` : '<strong>--</strong>';
    const totalRow = `<tr class="tm-total-row">
        <td class="tm-name"><strong>Total (with adjustments)</strong></td>
        <td class="tm-formula">${d.adjustments || ''}</td>
        <td class="tm-val">${totalCell(totalM.ridge_mae)}</td>
        <td class="tm-val">${totalCell(totalM.nn_mae)}</td>
        <td class="tm-val">${totalCell(totalM.attn_nn_mae)}</td>
        <td class="tm-val">${totalCell(totalM.lgbm_mae)}</td>
    </tr>`;

    // Feature badges
    const featureBadges = (d.specific_features || []).map(f =>
        `<span class="feature-badge">${f}</span>`
    ).join("");

    // Architecture
    const arch = d.architecture || {};
    const backbone = (arch.backbone || []).join(" > ");

    container.innerHTML = `
        <div class="pos-model-card">
            <div class="pos-model-header">
                <span class="pos-badge pos-${pos}">${pos}</span>
                <span class="pos-model-name">${d.label} Model</span>
                <span class="pos-model-meta">${d.n_features || '?'} features &middot; ${d.n_samples_test || '?'} test samples</span>
            </div>

            <div class="pos-model-section-label">Target Decomposition</div>
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
            <div class="arch-info">Shared backbone <span class="arch-val">[${backbone}]</span> &rarr; ${(d.targets || []).length} heads (hidden: <span class="arch-val">${arch.head_hidden || '?'}</span>)</div>
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
    try {
        const data = await fetchJSON(`/api/player/${playerId}`);

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
                plugins: { title: { display: true, text: "Weekly Fantasy Points: Actual vs Predicted", color: "#e8eaed" } },
                scales: {
                    y: { beginAtZero: true, grid: { color: "#2e3347" }, title: { display: true, text: "Fantasy Points", color: "#9aa0b0" } },
                    x: { grid: { color: "#2e3347" } },
                },
            },
        });

        document.getElementById("player-modal").classList.add("open");
    } catch (e) {
        console.error("Failed to load player:", e);
        document.getElementById("modal-name").textContent = "Error loading player";
        document.getElementById("modal-pos-team").textContent = "";
        document.getElementById("player-modal").classList.add("open");
    }
}

function closeModal() {
    document.getElementById("player-modal").classList.remove("open");
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

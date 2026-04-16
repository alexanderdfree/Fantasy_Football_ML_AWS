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
    actual: "#e8eaed",
    ridgeBg: "rgba(59, 130, 246, 0.2)",
    nnBg: "rgba(34, 197, 94, 0.2)",
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
        document.body.classList.remove("model-ridge", "model-nn");
        if (val === "ridge") document.body.classList.add("model-ridge");
        else if (val === "nn") document.body.classList.add("model-nn");
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

    try {
        const data = await fetchJSON(`/api/predictions?${params}`);
        allPlayers = data.players || [];
        renderTable();
    } catch (e) {
        console.error("Failed to load predictions:", e);
        allPlayers = [];
        document.getElementById("predictions-body").innerHTML =
            '<tr><td colspan="12" class="error-message">Failed to load predictions.</td></tr>';
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
        const ridgeDelta = (p.ridge_pred - p.actual).toFixed(1);
        const nnDelta = (p.nn_pred - p.actual).toFixed(1);
        const ridgeCls = deltaClass(ridgeDelta);
        const nnCls = deltaClass(nnDelta);
        const headshot = p.headshot
            ? `<img class="player-headshot" src="${escapeHtml(p.headshot)}" alt="" loading="lazy">`
            : `<div class="player-headshot"></div>`;

        return `<tr data-player-id="${escapeHtml(p.player_id)}">
            <td class="col-rank">${rank}</td>
            <td class="col-player"><div class="player-cell">${headshot}<span class="player-name">${escapeHtml(p.name)}</span></div></td>
            <td class="col-pos"><span class="pos-badge pos-${escapeHtml(p.position)}">${escapeHtml(p.position)}</span></td>
            <td class="col-team">${escapeHtml(p.team)}</td>
            <td class="col-week">${p.week}</td>
            <td class="col-actual"><strong>${p.actual.toFixed(1)}</strong></td>
            <td class="col-pred ridge-col">${p.ridge_pred.toFixed(1)}</td>
            <td class="col-pred nn-col">${p.nn_pred.toFixed(1)}</td>
            <td class="col-delta ridge-col ${ridgeCls}">${fmtDelta(ridgeDelta)}</td>
            <td class="col-delta nn-col ${nnCls}">${fmtDelta(nnDelta)}</td>
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
                <td class="col-actual"><strong>${p.avg_actual.toFixed(1)}</strong></td>
                <td class="col-pred">${p.avg_ridge.toFixed(1)}</td>
                <td class="col-pred">${p.avg_nn.toFixed(1)}</td>
            </tr>
        `).join("");

        tbody.querySelectorAll("tr").forEach(row => {
            row.addEventListener("click", () => openPlayerModal(row.dataset.playerId));
        });
    } catch (e) {
        console.error("Failed to load standings:", e);
        document.getElementById("standings-body").innerHTML =
            '<tr><td colspan="9" class="error-message">Failed to load standings.</td></tr>';
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

        // Overall metrics cards
        const ridge = metrics["Ridge Regression"];
        const nn = metrics["Neural Network"];
        document.getElementById("ridge-mae").textContent = ridge.overall.mae.toFixed(3);
        document.getElementById("ridge-rmse").textContent = ridge.overall.rmse.toFixed(3);
        document.getElementById("ridge-r2").textContent = ridge.overall.r2.toFixed(3);
        document.getElementById("nn-mae").textContent = nn.overall.mae.toFixed(3);
        document.getElementById("nn-rmse").textContent = nn.overall.rmse.toFixed(3);
        document.getElementById("nn-r2").textContent = nn.overall.r2.toFixed(3);

        // Position model breakdown
        setupPerfPositionFilter();
        renderPositionModelDetail(getActivePosition("perf-position-filter"));

        // Position charts
        renderPositionCharts(ridge.by_position, nn.by_position);

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

    // Target decomposition rows
    const targetRows = (d.targets || []).map(t => {
        const m = tm[t.key] || {};
        return `<tr>
            <td class="tm-name">${t.label}</td>
            <td class="tm-formula">${t.formula}</td>
            <td class="tm-val">${m.ridge_mae != null ? m.ridge_mae.toFixed(2) : '--'}</td>
            <td class="tm-val">${m.nn_mae != null ? m.nn_mae.toFixed(2) : '--'}</td>
        </tr>`;
    }).join("");

    const totalM = tm["total"] || {};
    const totalRow = `<tr class="tm-total-row">
        <td class="tm-name"><strong>Total (with adjustments)</strong></td>
        <td class="tm-formula">${d.adjustments || ''}</td>
        <td class="tm-val"><strong>${totalM.ridge_mae != null ? totalM.ridge_mae.toFixed(2) : '--'}</strong></td>
        <td class="tm-val"><strong>${totalM.nn_mae != null ? totalM.nn_mae.toFixed(2) : '--'}</strong></td>
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

function renderPositionCharts(ridgePos, nnPos) {
    const positions = ridgePos.map(p => p.position);
    const ridgeMAE = ridgePos.map(p => p.mae);
    const nnMAE = nnPos.map(p => p.mae);
    const ridgeR2 = ridgePos.map(p => p.r2);
    const nnR2 = nnPos.map(p => p.r2);

    const maeDatasets = [
        { label: "Ridge", data: ridgeMAE, backgroundColor: COLORS.ridgeBg, borderColor: COLORS.ridge, borderWidth: 1.5 },
        { label: "Neural Net", data: nnMAE, backgroundColor: COLORS.nnBg, borderColor: COLORS.nn, borderWidth: 1.5 },
    ];
    const r2Datasets = [
        { label: "Ridge", data: ridgeR2, backgroundColor: COLORS.ridgeBg, borderColor: COLORS.ridge, borderWidth: 1.5 },
        { label: "Neural Net", data: nnR2, backgroundColor: COLORS.nnBg, borderColor: COLORS.nn, borderWidth: 1.5 },
    ];
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
    const datasets = [
        {
            label: "Ridge MAE",
            data: weekly.ridge_mae,
            borderColor: COLORS.ridge,
            backgroundColor: COLORS.ridgeBg,
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6,
        },
        {
            label: "Neural Net MAE",
            data: weekly.nn_mae,
            borderColor: COLORS.nn,
            backgroundColor: COLORS.nnBg,
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6,
        },
    ];
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
        if (data.error) return;

        document.getElementById("modal-name").textContent = data.name;
        document.getElementById("modal-pos-team").textContent = `${data.position} - ${data.team}`;
        document.getElementById("modal-avg").textContent = data.season_avg.toFixed(1);
        document.getElementById("modal-total").textContent = data.season_total.toFixed(1);

        const img = document.getElementById("modal-headshot");
        if (data.headshot) { img.src = data.headshot; img.style.display = "block"; }
        else { img.style.display = "none"; }

        // Chart
        const weeks = data.weekly.map(w => `Wk ${w.week}`);
        const actual = data.weekly.map(w => w.actual);
        const ridge = data.weekly.map(w => w.ridge_pred);
        const nn = data.weekly.map(w => w.nn_pred);
        const chartDatasets = [
            { label: "Actual", data: actual, borderColor: COLORS.actual, borderWidth: 2.5, tension: 0.3, pointRadius: 5, pointHoverRadius: 7 },
            { label: "Ridge Pred", data: ridge, borderColor: COLORS.ridge, borderWidth: 2, borderDash: [6, 3], tension: 0.3, pointRadius: 4 },
            { label: "NN Pred", data: nn, borderColor: COLORS.nn, borderWidth: 2, borderDash: [6, 3], tension: 0.3, pointRadius: 4 },
        ];

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

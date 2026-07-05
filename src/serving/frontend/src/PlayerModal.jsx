/* Player detail modal: headshot, season stats, weekly actual-vs-predicted
 * trend chart. A /api/player 404 with a row-supplied fallback degrades to a
 * name+headshot card (rookies / backups absent from the backtest cache). */
import { useEffect, useState } from "react";
import { fetchJSON } from "./api.js";
import { fmt, sizedHeadshot } from "./lib/format.js";
import { modelColors, chartTheme } from "./lib/chartTheme.js";
import { useChart } from "./hooks/useChart.js";

function TrendChart({ weekly, theme }) {
    const canvasRef = useChart(() => {
        const COLORS = modelColors();
        const weeks = weekly.map((w) => `Wk ${w.week}`);
        const predSeries = [
            { label: "Ridge Pred", key: "ridge_pred", color: COLORS.ridge },
            { label: "NN Pred", key: "nn_pred", color: COLORS.nn },
            { label: "Attn NN Pred", key: "attn_nn_pred", color: COLORS.attn_nn },
            { label: "LGBM Pred", key: "lgbm_pred", color: COLORS.lgbm },
            { label: "NFL.com Pred", key: "nflcom_pred", color: COLORS.nflcom },
            { label: "RotoWire Pred", key: "rotowire_pred", color: COLORS.rotowire },
        ];
        const datasets = [
            { label: "Actual", data: weekly.map((w) => w.actual), borderColor: COLORS.actual, borderWidth: 2.5, tension: 0.3, pointRadius: 5, pointHoverRadius: 7 },
        ];
        for (const { label, key, color } of predSeries) {
            const series = weekly.map((w) => (w[key] != null ? w[key] : null));
            if (series.some((v) => v != null)) {
                datasets.push({ label, data: series, borderColor: color, borderWidth: 2, borderDash: [6, 3], tension: 0.3, pointRadius: 4, spanGaps: true });
            }
        }
        return {
            type: "line",
            data: { labels: weeks, datasets },
            options: {
                responsive: true,
                // The modal chart lives in a fixed-height container (.modal .chart-box);
                // let the canvas fill it instead of Chart.js's default 2:1 aspect ratio.
                maintainAspectRatio: false,
                plugins: { title: { display: true, text: "Weekly Fantasy Points: Actual vs Predicted", color: chartTheme().heading } },
                scales: {
                    y: { beginAtZero: true, grid: { color: chartTheme().grid }, title: { display: true, text: "Fantasy Points", color: chartTheme().text } },
                    x: { grid: { color: chartTheme().grid } },
                },
            },
        };
    }, [weekly, theme]);
    return <canvas ref={canvasRef} id="player-chart" />;
}

export function PlayerModal({ playerId, fallback, scoring, theme, onClose }) {
    const [data, setData] = useState(null);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        if (!playerId) return undefined;
        let cancelled = false;
        setData(null);
        setFailed(false);
        fetchJSON(`/api/player/${encodeURIComponent(playerId)}?scoring=${scoring}`)
            .then((d) => { if (!cancelled) setData(d); })
            .catch((e) => {
                console.error("Failed to load player:", e);
                if (!cancelled) setFailed(true);
            });
        return () => { cancelled = true; };
    }, [playerId, scoring]);

    useEffect(() => {
        if (!playerId) return undefined;
        const onKey = (e) => { if (e.key === "Escape") onClose(); };
        document.addEventListener("keydown", onKey);
        return () => document.removeEventListener("keydown", onKey);
    }, [playerId, onClose]);

    if (!playerId) return null;

    // Three render states: loaded, failed-with-fallback (identity card), failed.
    const identity = data || (failed && fallback
        ? { name: fallback.name || "—", position: fallback.position, team: fallback.team, headshot: fallback.headshot }
        : failed
            ? { name: "Error loading player", position: "", team: "", headshot: null }
            : null);

    const posTeam = identity ? [identity.position, identity.team].filter(Boolean).join(" - ") : "";
    const note = failed && fallback ? "No prior-season game log to chart yet." : "";

    return (
        <div id="player-modal" className="modal open" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
            <div className="modal-content">
                <button type="button" className="modal-close" id="modal-close" onClick={onClose}>&times;</button>
                <div className="modal-header">
                    {identity && identity.headshot
                        ? <img id="modal-headshot" className="modal-headshot" src={sizedHeadshot(identity.headshot, 400)} alt={identity.name || ""} loading="lazy" decoding="async" />
                        : <img id="modal-headshot" className="modal-headshot" alt="" style={{ display: "none" }} />}
                    <div>
                        <h2 id="modal-name">{identity ? identity.name : "Loading…"}</h2>
                        <span id="modal-pos-team" className="modal-pos-team">{posTeam}</span>
                    </div>
                </div>
                <div className="modal-stats">
                    <div className="modal-stat">
                        <span className="modal-stat-label">Season Avg</span>
                        <span className="modal-stat-value" id="modal-avg">{data ? fmt(data.season_avg) : "--"}</span>
                    </div>
                    <div className="modal-stat">
                        <span className="modal-stat-label">Season Total</span>
                        <span className="modal-stat-value" id="modal-total">{data ? fmt(data.season_total) : "--"}</span>
                    </div>
                </div>
                <div className="chart-box">
                    {data && data.weekly && data.weekly.length > 0 && <TrendChart weekly={data.weekly} theme={theme} />}
                    {note
                        ? <p id="modal-note" className="modal-note">{note}</p>
                        : <p id="modal-note" className="modal-note hidden"></p>}
                </div>
            </div>
        </div>
    );
}

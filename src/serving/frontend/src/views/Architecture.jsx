/* Model Architecture — static NN diagram + training-loop notes, plus the
 * live per-position config table and feature accordions off
 * /api/model_architecture. Faithful port of the vanilla loadModelArchitecture
 * / renderArchConfigTable / renderArchFeatureAccordions; markup, ids, and
 * copy mirror templates/index.html. */
import { useEffect, useState } from "react";
import { fetchJSON } from "../api.js";
import { ApproachBanner } from "../components/common.jsx";

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

const ARCH_COLUMNS = [
    { key: "targets",            label: "Targets",    render: (p) => fmtList(p.targets) },
    { key: "backbone_layers",    label: "Backbone",   render: (p) => fmtLayers(p.backbone_layers) },
    { key: "head_hidden",        label: "Head",       render: (p) => fmtNum(p.head_hidden) },
    { key: "dropout",            label: "Dropout",    render: (p) => fmtNum(p.dropout) },
    { key: "lr",                 label: "LR",         render: (p) => fmtNum(p.lr) },
    { key: "weight_decay",       label: "WD",         render: (p) => fmtNum(p.weight_decay) },
    { key: "batch_size",         label: "Batch",      render: (p) => fmtNum(p.batch_size) },
    { key: "epochs",             label: "Epochs",     render: (p) => fmtNum(p.epochs) },
    { key: "patience",           label: "Patience",   render: (p) => fmtNum(p.patience) },
    { key: "scheduler",          label: "Scheduler",  render: (p) => p.scheduler || "—" },
    { key: "attention_enabled",  label: "Attn",       render: (p) => (p.attention_enabled ? "✓" : "—") },
    { key: "lightgbm_enabled",   label: "LGBM",       render: (p) => (p.lightgbm_enabled ? "✓" : "—") },
    { key: "feature_count",      label: "# Features", render: (p) => fmtNum(p.feature_count) },
];

/* The <pre> ASCII diagram is kept in a template literal so JSX whitespace
 * handling can't disturb the box-drawing alignment. Verbatim from index.html. */
const ARCH_DIAGRAM = `MultiHeadNet (dense)
    Input (static features)
      └─ Shared backbone: [Linear → BatchNorm1d → ReLU → Dropout] × N
         └─ Per-target head: Linear → ReLU → Linear → (optional clamp ≥ 0)
      └─ "total" output = sum of target heads

MultiHeadNetWithHistory (attention — QB / RB / WR / TE)
    Static features ─────────────────────────────┐
    Game history [B, seq_len, game_dim]          │
     └─ GameEncoder: Linear → ReLU               │
     └─ (optional) Positional embedding          │
     └─ AttentionPool: learned queries,          │
        multi-head scaled dot-product attn ──────┤
                                                 ▼
                    Concat [static_dim + n_heads * d_model]
                     └─ Shared backbone (BatchNorm + Dropout)
                     └─ Per-target heads (GatedTDHead on TD-count targets)`;

function ArchConfigTable({ positions }) {
    return (
        <table className="arch-table">
            <thead>
                <tr>
                    <th>Position</th>
                    {ARCH_COLUMNS.map((c) => <th key={c.key}>{c.label}</th>)}
                </tr>
            </thead>
            <tbody>
                {ARCH_POSITION_ORDER.map((pos) => {
                    const p = positions[pos];
                    if (!p) return null;
                    return (
                        <tr key={pos}>
                            <td className="arch-pos-cell">{pos}</td>
                            {ARCH_COLUMNS.map((c) => <td key={c.key}>{c.render(p)}</td>)}
                        </tr>
                    );
                })}
            </tbody>
        </table>
    );
}

function ArchFeatureAccordion({ pos, p }) {
    const features = p.features || {};
    const overrides = p.head_hidden_overrides || {};
    const overrideStr = Object.keys(overrides).length
        ? Object.entries(overrides).map(([k, v]) => `${k}: ${v}`).join(", ")
        : null;

    return (
        <details className="arch-accordion">
            <summary>
                <span className="arch-pos-label">{pos}</span>
                <span className="arch-pos-count">{p.feature_count} features</span>
            </summary>
            <div className="arch-accordion-body">
                <div className="arch-accordion-meta">
                    <span><strong>Targets:</strong> {fmtList(p.targets)}</span>
                    {overrideStr != null && <>{" · "}<span><strong>Head overrides:</strong> {overrideStr}</span></>}
                </div>
                {Object.keys(ARCH_CATEGORY_LABELS)
                    .filter((key) => features[key] && features[key].length)
                    .map((key) => (
                        <div key={key} className="feature-category">
                            <div className="feature-category-title">{ARCH_CATEGORY_LABELS[key]} <span className="feature-category-count">({features[key].length})</span></div>
                            <div className="feature-chip-row">
                                {features[key].map((f) => <span key={f} className="feature-chip">{f}</span>)}
                            </div>
                        </div>
                    ))}
            </div>
        </details>
    );
}

/* Module-level cache: the architecture payload is static per process, so
 * fetch it once per session (mirrors the vanilla modelArchitectureLoaded —
 * only a successful load is cached; an error retries on the next visit). */
let cachedArchPositions = null;

export function ArchitectureView(props) {
    const [{ state, positions, message }, setArch] = useState(() => (
        cachedArchPositions
            ? { state: "ready", positions: cachedArchPositions, message: null }
            : { state: "loading", positions: null, message: null }
    ));

    useEffect(() => {
        if (cachedArchPositions) return undefined;
        let cancelled = false;
        (async () => {
            try {
                const data = await fetchJSON("/api/model_architecture");
                if (data.error) throw new Error(data.error);
                cachedArchPositions = data.positions || {};
                if (!cancelled) setArch({ state: "ready", positions: cachedArchPositions, message: null });
            } catch (e) {
                console.error("Failed to load model architecture:", e);
                if (!cancelled) setArch({ state: "error", positions: null, message: e.message });
            }
        })();
        return () => { cancelled = true; };
    }, []);

    return (
        <section id="view-model-architecture" className="view active">
            <ApproachBanner icon="layers" title="Per-Position Multi-Head Neural Networks">
                Each position has a dedicated multi-target model that predicts raw NFL stats (yards, TD counts, receptions, etc.), with a deterministic aggregator converting predictions to fantasy points under any scoring format. Compared against Ridge regression, an attention-based game-history variant, and LightGBM where applicable. Config and features below are loaded live from the Python config modules.
            </ApproachBanner>

            <div className="section-header">Neural Network Architecture</div>
            <pre className="arch-diagram">{ARCH_DIAGRAM}</pre>

            <div className="section-header">Training Loop</div>
            <ul className="arch-bullets">
                <li><strong>Optimizer:</strong> <code>AdamW</code> with per-position LR and weight decay.</li>
                <li><strong>Loss:</strong> <code>MultiTargetLoss</code> — per-target Huber or Poisson NLL (DST's four rare counts) + optional BCE on the TD gate logit.</li>
                <li><strong>Scaling:</strong> <code>StandardScaler</code> fit on train, applied and clipped to <code>[-4, 4]</code>.</li>
                <li><strong>Gradient clipping:</strong> <code>clip_grad_norm_(max_norm=1.0)</code> each step.</li>
                <li><strong>Schedulers:</strong> <code>CosineAnnealingWarmRestarts</code>, <code>OneCycleLR</code>, or <code>ReduceLROnPlateau</code> (per-position).</li>
                <li><strong>Early stopping:</strong> tracks <code>val_mae_total</code>; restores the best <code>state_dict</code> when patience expires.</li>
                <li><strong>Data split:</strong> Train 2013–2023, Val 2024, Test 2025 (2012 loaded for prior-season context only; Kickers: 2015+ only, post-PAT rule change).</li>
                <li><strong>Artifacts:</strong> <code>{"{pos}_multihead_nn.pt"}</code>, <code>{"{pos}_attention_nn.pt"}</code>, <code>nn_scaler.pkl</code>, Ridge/LightGBM models — tarred and uploaded to S3.</li>
            </ul>

            <div className="section-header">Per-Position Configuration</div>
            <div id="arch-config-table" className="arch-table-wrap">
                {state === "loading" && <p className="arch-loading">Loading configuration…</p>}
                {state === "error" && <p className="arch-error">Failed to load: {message}</p>}
                {state === "ready" && <ArchConfigTable positions={positions} />}
            </div>

            <div className="section-header">Features by Position</div>
            <div id="arch-feature-accordions">
                {state === "loading" && <p className="arch-loading">Loading features…</p>}
                {state === "ready" && ARCH_POSITION_ORDER.map((pos) => (
                    positions[pos] ? <ArchFeatureAccordion key={pos} pos={pos} p={positions[pos]} /> : null
                ))}
            </div>
        </section>
    );
}

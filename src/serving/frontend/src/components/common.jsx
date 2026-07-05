/* Small shared building blocks. Markup + class names mirror the vanilla app
 * exactly so the (design-system-tokened) style.css styles them unchanged. */
import { fmt, sizedHeadshot, errDelta, deltaClass, fmtDelta } from "../lib/format.js";

/* Single-select pill row (position filters, metric toggles). `options` is
 * [{value, label, disabled?, title?}]; parent owns the active value. */
export function PillGroup({ id, options, value, onChange, className = "pill-group" }) {
    return (
        <div className={className} id={id}>
            {options.map((o) => (
                <button
                    key={o.value}
                    type="button"
                    className={`pill${o.value === value ? " active" : ""}`}
                    data-value={o.value}
                    disabled={o.disabled || undefined}
                    title={o.title || undefined}
                    onClick={() => { if (!o.disabled && o.value !== value) onChange(o.value); }}
                >
                    {o.label}
                </button>
            ))}
        </div>
    );
}

export function PosBadge({ position }) {
    return <span className={`pos-badge pos-${position}`}>{position}</span>;
}

/* Headshot + name cell. DST is a team unit with no player photo; a missing
 * headshot falls back to the empty-circle placeholder. */
export function PlayerCell({ player }) {
    const headshot = player.position === "DST"
        ? null
        : player.headshot
            ? <img className="player-headshot" src={sizedHeadshot(player.headshot, 400)} alt="" loading="lazy" decoding="async" />
            : <div className="player-headshot" />;
    return (
        <div className="player-cell">
            {headshot}
            <span className="player-name">{player.name}</span>
        </div>
    );
}

/* Signed prediction-minus-actual delta, colored by direction (±1 reads neutral). */
export function DeltaCell({ pred, actual }) {
    const d = errDelta(pred, actual);
    if (d == null) return "--";
    return <span className={deltaClass(d)}>{fmtDelta(d.toFixed(1))}</span>;
}

/* Sortable table header cell: click cycles desc → asc; shows ▼/▲ when active. */
export function SortableTh({ label, sortKey, sort, order, onSort, className = "" }) {
    const active = sort === sortKey;
    const arrow = active ? (order === "desc" ? "▼" : "▲") : "";
    return (
        <th
            className={`${className} sortable${active ? " active-sort" : ""}`.trim()}
            data-sort={sortKey}
            onClick={() => onSort(sortKey)}
        >
            {label} <span className="sort-arrow">{arrow}</span>
        </th>
    );
}

/* Numbered pagination with «/» steppers and edge-ellipsis, mirroring the
 * vanilla renderPagination (7 visible page buttons max). */
export function Pagination({ page, totalPages, onChange }) {
    if (totalPages <= 1) return <div className="pagination" />;
    const maxVisible = 7;
    let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
    const endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);
    const pages = [];
    for (let p = startPage; p <= endPage; p++) pages.push(p);
    const ellipsis = { color: "var(--text-muted)" };
    return (
        <div className="pagination">
            <button type="button" className="page-btn" disabled={page === 1} onClick={() => onChange(page - 1)}>&laquo;</button>
            {startPage > 1 && (
                <>
                    <button type="button" className="page-btn" onClick={() => onChange(1)}>1</button>
                    <span style={ellipsis}>...</span>
                </>
            )}
            {pages.map((p) => (
                <button
                    key={p}
                    type="button"
                    className={`page-btn${p === page ? " active" : ""}`}
                    onClick={() => onChange(p)}
                >
                    {p}
                </button>
            ))}
            {endPage < totalPages && (
                <>
                    <span style={ellipsis}>...</span>
                    <button type="button" className="page-btn" onClick={() => onChange(totalPages)}>{totalPages}</button>
                </>
            )}
            <button type="button" className="page-btn" disabled={page === totalPages} onClick={() => onChange(page + 1)}>&raquo;</button>
        </div>
    );
}

/* The layered-cube / bar-chart / clock approach banners shared by the report views. */
export function ApproachBanner({ icon, title, children }) {
    const icons = {
        layers: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="24" height="24">
                <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
            </svg>
        ),
        chart: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="24" height="24">
                <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
            </svg>
        ),
        clock: (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="24" height="24">
                <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
            </svg>
        ),
    };
    return (
        <div className="approach-banner">
            <div className="approach-icon">{icons[icon] || icons.layers}</div>
            <div>
                <div className="approach-title">{title}</div>
                <div className="approach-desc">{children}</div>
            </div>
        </div>
    );
}

export { fmt };

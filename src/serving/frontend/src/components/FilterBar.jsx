/* Adaptive one-row filter bar (design-system "filter bar v2").
 *
 * Auto-fit: each view shows as many filter controls as fit on one line, in
 * list order (the first — Position — is always shown), with the badge +
 * Columns + Filters menus pinned right. A hidden off-screen copy of every
 * control is measured to compute the fit; editing the Filters checklist
 * switches to manual picks. A control that leaves the bar has its state reset
 * (via onResetFilter) so it can't invisibly constrain the table. */
import { Fragment, useEffect, useState } from "react";
import { useFilterFit } from "../hooks/useFilterFit.js";

export const AGE_BUCKETS = [
    { value: "ALL", label: "All Ages", test: () => true },
    { value: "u25", label: "Under 25", test: (a) => a != null && a < 25 },
    { value: "25_28", label: "25–28", test: (a) => a != null && a >= 25 && a <= 28 },
    { value: "29_31", label: "29–31", test: (a) => a != null && a >= 29 && a <= 31 },
    { value: "32p", label: "32 & up", test: (a) => a != null && a >= 32 },
];

export function ageBucketFor(value) {
    return AGE_BUCKETS.find((b) => b.value === value) || AGE_BUCKETS[0];
}

export function AutoFitFilterBar({ items, renderControl, renderMenus, onResetFilter, stats }) {
    const keys = items.map((i) => i.value);
    const { rowRef, measureRef, fit } = useFilterFit(keys.length);
    // null → auto-fit (show what the row can hold); an array → manual picks.
    const [manual, setManual] = useState(null);
    const visible = manual || keys.slice(0, fit);

    useEffect(() => {
        if (manual) return;
        keys.slice(fit).forEach(onResetFilter);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [fit, manual]);

    const onFiltersChange = (next) => {
        const removed = visible.filter((k) => !next.includes(k));
        removed.forEach(onResetFilter);
        setManual(keys.filter((k) => next.includes(k)));
    };

    return (
        <div className="filters-bar">
            <div className="filters-row" ref={rowRef}>
                {items.filter((i) => visible.includes(i.value)).map((i) => (
                    <Fragment key={i.value}>{renderControl(i.value, false)}</Fragment>
                ))}
                <div className="filter-menus">{renderMenus({ visibleFilters: visible, onFiltersChange, measure: false })}</div>
            </div>
            {/* Each control renders twice — live, and inside the hidden measure
                row (the `measure` flag keeps input ids unique there). */}
            <div className="filters-row filters-row--measure" ref={measureRef} aria-hidden="true">
                {items.map((i) => (
                    <Fragment key={i.value}>{renderControl(i.value, true)}</Fragment>
                ))}
                <div className="filter-menus">{renderMenus({ visibleFilters: visible, onFiltersChange, measure: true })}</div>
            </div>
            {stats || null}
        </div>
    );
}

/* Live readout of the filtered slice — average actual output and which
 * source best predicts these exact rows (updates with every filter change). */
export function FilterSliceStats({ rows, sources }) {
    const withActual = rows.filter((p) => p.actual != null);
    if (!withActual.length) return null;
    const avgActual = withActual.reduce((s, p) => s + p.actual, 0) / withActual.length;
    let best = null;
    for (const src of sources) {
        let sum = 0;
        let n = 0;
        for (const p of withActual) {
            const v = p[src.key];
            if (v != null) { sum += Math.abs(v - p.actual); n += 1; }
        }
        if (n) {
            const mae = sum / n;
            if (!best || mae < best.mae) best = { label: src.label, mae };
        }
    }
    return (
        <div className="filters-stats-row">
            <div className="stat-block-row">
                <div className="stat-block">
                    <span className="stat-block-label">Avg Actual</span>
                    <span className="stat-block-value neutral">{avgActual.toFixed(1)}</span>
                </div>
                {best && (
                    <div className="stat-block">
                        <span className="stat-block-label">Most Accurate</span>
                        <span className="stat-block-value">{best.label}</span>
                    </div>
                )}
                {best && (
                    <div className="stat-block">
                        <span className="stat-block-label">Best MAE</span>
                        <span className="stat-block-value neutral">{best.mae.toFixed(2)}</span>
                    </div>
                )}
            </div>
        </div>
    );
}

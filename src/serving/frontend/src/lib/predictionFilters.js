const PROJECTION_KEYS = [
    "ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred",
    "nflcom_pred", "rotowire_pred", "espn_pred",
];

export function meetsMinimumProjection(player, minimum) {
    if (Number.isNaN(minimum)) return true;
    return PROJECTION_KEYS.some((key) => player[key] != null && player[key] >= minimum);
}

// Every candidate is graded on the same rows and the server's declared truth.
// Legacy snapshots lack comparison_actual and cannot establish this comparison.
export function sliceAccuracy(rows, sources) {
    const allowed = (row, source) => !(row.comparison_excluded_sources || [])
        .includes(source.key.replace(/_pred$/, ""));
    const candidates = sources.filter((source) => rows.some((row) => (
        allowed(row, source) && Number.isFinite(row[source.key])
    )));
    const common = candidates.length ? rows.filter((row) => (
        row.comparison_actual_basis === "shared_projected_components_v1"
        && Number.isFinite(row.comparison_actual)
        && candidates.every((source) => allowed(row, source) && Number.isFinite(row[source.key]))
    )) : [];
    let best = null;
    if (common.length) {
        for (const source of candidates) {
            const mae = common.reduce((sum, row) => (
                sum + Math.abs(row[source.key] - row.comparison_actual)
            ), 0) / common.length;
            if (!best || mae < best.mae) best = { label: source.label, mae };
        }
    }
    return { best, n: common.length, cohortN: rows.length };
}

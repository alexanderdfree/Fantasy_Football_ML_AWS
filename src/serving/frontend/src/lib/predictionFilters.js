/* Prediction-table filters and the filtered-slice accuracy readout.
 *
 * The contract JSON is imported with an attribute so this module also loads
 * under `node --test` (Node requires it for JSON modules; esbuild inlines it). */
import contract from "../api-contract.json" with { type: "json" };

const PROJECTION_KEYS = [
    "ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred",
    "nflcom_pred", "rotowire_pred", "espn_pred",
];

/* The minimum-points threshold keeps a row when any available source —
 * including Ridge, the kicker's primary model — projects at least `minimum`. */
export function meetsMinimumProjection(player, minimum) {
    if (Number.isNaN(minimum)) return true;
    return PROJECTION_KEYS.some((key) => player[key] != null && player[key] >= minimum);
}

/* The server's declared basis for row-level comparison truth (`comparison_actual`). */
export const COMPARISON_ACTUAL_BASIS = contract.comparison.actual_basis;

/* `<source>_pred` is the full display forecast; `<source>_comparison_pred` is the
 * shared-projected-component total that may be graded against `comparison_actual`
 * (ADR-0024). A display forecast never substitutes for a missing comparison one. */
export function comparisonForecastKey(sourceKey) {
    return sourceKey.replace(/_pred$/, "_comparison_pred");
}

// Every candidate is graded on the same rows and the server's declared truth.
// Rows without comparison truth (legacy snapshots, another basis) cannot
// establish this comparison; they count toward cohortN only.
export function sliceAccuracy(rows, sources) {
    const excluded = (row, source) => (row.comparison_excluded_sources || [])
        .includes(source.key.replace(/_pred$/, ""));
    const forecast = (row, source) => row[comparisonForecastKey(source.key)];
    const usable = (row, source) => !excluded(row, source) && Number.isFinite(forecast(row, source));
    const candidates = sources.filter((source) => rows.some((row) => usable(row, source)));
    const common = candidates.length ? rows.filter((row) => (
        row.comparison_actual_basis === COMPARISON_ACTUAL_BASIS
        && Number.isFinite(row.comparison_actual)
        && candidates.every((source) => usable(row, source))
    )) : [];
    let best = null;
    if (common.length) {
        for (const source of candidates) {
            const mae = common.reduce((sum, row) => (
                sum + Math.abs(forecast(row, source) - row.comparison_actual)
            ), 0) / common.length;
            if (!best || mae < best.mae) best = { label: source.label, mae };
        }
    }
    return { best, n: common.length, cohortN: rows.length };
}

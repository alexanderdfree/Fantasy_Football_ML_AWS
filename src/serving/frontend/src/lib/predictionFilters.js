/* Prediction-table filters. */

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

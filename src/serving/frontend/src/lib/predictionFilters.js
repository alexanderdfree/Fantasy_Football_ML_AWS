const PROJECTION_KEYS = [
    "ridge_pred", "nn_pred", "attn_nn_pred", "lgbm_pred",
    "nflcom_pred", "rotowire_pred", "espn_pred",
];

export function meetsMinimumProjection(player, minimum) {
    if (Number.isNaN(minimum)) return true;
    return PROJECTION_KEYS.some((key) => player[key] != null && player[key] >= minimum);
}

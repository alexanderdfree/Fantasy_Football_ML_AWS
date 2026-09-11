import contract from "./api-contract.json";

/** @typedef {'ppr'|'half_ppr'|'standard'} ScoringFormat */
/** @typedef {{player_id:string, position:string, name:string, actual:number|null, ridge_pred:number|null, nn_pred:number|null, attn_nn_pred:number|null, lgbm_pred:number|null}} PredictionRow */
/** @typedef {{weeks:number[], scoring:Record<ScoringFormat, PredictionRow[]>, degraded_positions:string[], generated_at?:string}} SnapshotResponse */
/** @typedef {{players:PredictionRow[], total:number, scoring:ScoringFormat, degraded_positions?:string[]}} PredictionsResponse */
/** @typedef {{scoring:ScoringFormat, model_source:string, subsets:Object, coverage?:Object, sample_basis?:string, actual_basis?:string, scoring_components?:Object, cohort_definitions?:Object}} ComparisonResponse */

export { contract };

/** Check the transport boundary before views interpret values. Legacy servers
 * without version headers remain readable; unknown major versions fail visibly.
 * Added fields/source IDs are compatible. Zero is never a missing prediction.
 */
export function validateAPIResponse(url, payload, version = null) {
    if (version && version.split(".")[0] !== contract.version.split(".")[0]) {
        throw new Error(`Unsupported API contract version: ${version}`);
    }
    const path = new URL(url, "http://localhost").pathname;
    let envelope = contract.endpoints[path];
    if (!envelope) return payload;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error(`Invalid ${path} response`);
    }
    if (path === "/api/upcoming_week") {
        envelope = payload.status === "warming" ? "warming"
            : payload.available === true ? "upcoming_available" : "upcoming_unavailable";
    }
    for (const [field, kind] of Object.entries(contract.envelopes[envelope])) {
        const value = payload[field];
        const valid = kind === "array" ? Array.isArray(value)
            : kind === "object" ? value && typeof value === "object" && !Array.isArray(value)
                : kind === "integer" ? Number.isInteger(value) : typeof value === kind;
        if (!valid) throw new Error(`Invalid ${path}.${field}`);
    }
    if (["predictions", "comparison"].includes(envelope) && !contract.scoring_formats.includes(payload.scoring)) {
        throw new Error(`Unknown scoring format: ${payload.scoring}`);
    }
    let rows = envelope === "predictions" ? payload.players : [];
    if (["snapshot", "upcoming_available"].includes(envelope)) {
        rows = contract.scoring_formats.flatMap((format) => {
            if (!Array.isArray(payload.scoring[format])) throw new Error(`Missing scoring format: ${format}`);
            return payload.scoring[format];
        });
    }
    for (const row of rows) {
        if (!row || typeof row.player_id !== "string" || !contract.positions.includes(row.position)) {
            throw new Error("Invalid prediction row identity");
        }
        for (const field of ["actual", ...contract.nullable_prediction_fields]) {
            if (row[field] != null && (typeof row[field] !== "number" || !Number.isFinite(row[field]))) {
                throw new Error(`Invalid prediction: ${field}`);
            }
        }
    }
    return payload;
}

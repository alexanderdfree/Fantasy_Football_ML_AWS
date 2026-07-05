/* Shared formatting + URL helpers, ported from the vanilla app.js. */

export const VALID_SCORING = ["ppr", "half_ppr", "standard"];

export function fmt(n, d = 1) {
    return (n == null || isNaN(n)) ? "--" : Number(n).toFixed(d);
}

// Request a small, face-cropped headshot instead of the full-size source image.
// The source photos are huge (NFL Cloudinary ~318KB, ESPN ~234KB) and the
// browser downloads them in full only to shrink to 32–64px; both CDNs can serve
// a sized variant (~90×/25× smaller), so resize at the URL for a much faster load.
// NOTE: tests/test_app.py pins the exact `combiner/i?img=${m[1]}&w=${size}`
// literal in the built bundle (width-only — forcing height too stretches the
// photo); keep this function's body byte-stable under bundling.
export function sizedHeadshot(url, size) {
    if (!url) return url;
    // NFL Cloudinary: insert a named transform (f_auto,q_auto,w_N)
    const nfl = url.match(/^https:\/\/static\.www\.nfl\.com\/image\/upload\/(.+)$/);
    if (nfl) {
        return `https://static.www.nfl.com/image/upload/f_auto,q_auto,w_${size}/${nfl[1]}`;
    }
    // ESPN combiner: resize via query params (width only, keep aspect ratio)
    const m = url.match(/^https:\/\/a\.espncdn\.com(\/i\/headshots\/.+\.png)$/);
    if (m) {
        return `https://a.espncdn.com/combiner/i?img=${m[1]}&w=${size}`;
    }
    return url;
}

// Per-target fantasy-point-equivalent multipliers. Display-only, JS-local
// (the Python POINT_EQUIVALENT_MULTIPLIER constant this once mirrored has been
// removed); `receptions` depends on scoring format (1.0 / 0.5 / 0.0). Applied only to count-style
// targets where the raw MAE would be dominated by the scoring coefficient
// (e.g. 0.4 TDs = 2.4 points).
const RECEPTION_WEIGHT = { ppr: 1.0, half_ppr: 0.5, standard: 0.0 };
const BASE_POINT_EQUIVALENT = {
    passing_tds: 4.0,
    rushing_tds: 6.0,
    receiving_tds: 6.0,
    interceptions: 2.0,
    fumbles_lost: 2.0,
};

export function pointEquivMultiplier(targetKey, scoring) {
    if (targetKey === "receptions") return RECEPTION_WEIGHT[scoring] ?? 1.0;
    return BASE_POINT_EQUIVALENT[targetKey];
}

// Format a per-target MAE with its raw unit, and — for targets with a known
// point-equivalent multiplier — also show the implied fantasy-point delta.
// The API serializes a `unit` field per target (from shared/aggregate_targets.py:TARGET_UNITS).
export function formatTargetMae(val, targetKey, unit, scoring) {
    if (val == null) return "--";
    const raw = unit ? `${fmt(val, 2)} ${unit}` : fmt(val, 2);
    const mult = pointEquivMultiplier(targetKey, scoring);
    return mult != null ? `${raw} (${fmt(val * mult, 2)} pts)` : raw;
}

export function errDelta(pred, actual) {
    return (pred != null && actual != null) ? pred - actual : null;
}

export function deltaClass(d) {
    const n = parseFloat(d);
    if (Math.abs(n) < 1) return "delta-neutral";
    return n > 0 ? "delta-positive" : "delta-negative";
}

export function fmtDelta(d) {
    const n = parseFloat(d);
    const sign = n > 0 ? "+" : "";
    return `${sign}${n}`;
}
